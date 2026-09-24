"""The ComfyUI nodes: make a world, edit it, read what the user said about it.

They are the same operations the MCP server and the CLI offer, so a workflow
(or an agent that builds workflows) can drive the loop too.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from .bepic_worlds import OPS
from . import server_routes as routes


def _ui_images(paths):
    """The `ui` block that puts files in ComfyUI's history and Assets panel —
    which is where an agent reading /history looks for a node's output."""
    out = []
    for p in paths:
        ref = routes.view_ref(p)
        if ref and os.path.isfile(p):
            out.append(ref)
    return {"images": out} if out else {}


class bEpicWorldFromReference:
    """A walkable world from a reference image.

    The picture sets the sky, the sun, the fog, the ground textures and the
    camera the world is seen from; the spec (plain words or JSON) sets what
    grows there and the time of day. A depth map adds the picture itself as a
    3D hero view; a heightmap or a panorama replaces the made-up terrain or sky.
    The world is saved under output/worlds/<name> and opened in the viewer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "name": ("STRING", {"default": "world"}),
                "spec": ("STRING", {"default": "", "multiline": True,
                                    "placeholder": "e.g. misty pine forest, late afternoon — or JSON"}),
                "world_size": ("FLOAT", {"default": 240.0, "min": 20.0, "max": 4000.0, "step": 10.0}),
                "fov": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 120.0, "step": 1.0,
                                  "tooltip": "Vertical field of view of the reference picture, in degrees"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
                "overwrite": ("BOOLEAN", {"default": True,
                                          "tooltip": "Replace a world of this name (its versions and feedback stay)"}),
                "open_in_viewer": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "heightmap": ("IMAGE",),
                "panorama": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("world_name", "world_json", "summary")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "bEpic/worlds"

    def run(self, reference, name, spec, world_size, fov, seed, overwrite, open_in_viewer,
            depth=None, heightmap=None, panorama=None):
        s = routes.store()
        summary = s.create(reference, name=name or "world", overwrite=overwrite, spec=spec,
                           depth=depth, heightmap=heightmap, panorama=panorama,
                           fov=fov, world_size=world_size, seed=seed, note="made by node")
        if open_in_viewer:
            routes.open_in_viewer(summary["name"])
        # This version's files (a rebuild keeps them in a folder of its own).
        scene = s.load(summary["name"])
        cam = next((i for i in scene["items"] if i.get("id") == "refcam"), None)
        ref_path = ((cam or {}).get("reference") or {}).get("src", {}).get("path")
        assets = os.path.dirname(ref_path) if ref_path else os.path.join(summary["folder"], "assets")
        ui = _ui_images([os.path.join(assets, f) for f in
                         ("reference.png", "heightmap_preview.png", "tex_ground.png")])
        return {"ui": ui, "result": (summary["name"], summary["world_json"], json.dumps(summary, indent=1))}


class bEpicWorldEdit:
    """Change a world by operations (JSON), as one new version."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "world"}),
            "ops": ("STRING", {"default": '[{"op": "scale_scatter", "factor": 0.5}]', "multiline": True,
                               "tooltip": "\n".join(f"{k}: {v}" for k, v in OPS.items())}),
            "note": ("STRING", {"default": ""}),
            "open_in_viewer": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("world_name", "summary")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "bEpic/worlds"

    def run(self, name, ops, note, open_in_viewer):
        summary = routes.store().edit(name, ops, note=note)
        if open_in_viewer:
            routes.open_in_viewer(summary["name"])
        return (summary["name"], json.dumps(summary, indent=1))


class bEpicWorldFeedback:
    """What the user said about a world in the viewer: the notes as text, and
    the snapshots taken with them as images (for a vision model)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "world"}),
            "status": (["open", "resolved", "all"],),
        }}

    RETURN_TYPES = ("STRING", "INT", "IMAGE")
    RETURN_NAMES = ("feedback", "count", "snapshots")
    FUNCTION = "run"
    CATEGORY = "bEpic/worlds"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")          # feedback arrives between runs

    def run(self, name, status):
        entries = routes.store().feedback(name, None if status == "all" else status)
        lines = []
        for f in entries:
            where = f" at {[round(v, 1) for v in f['point']]}" if f.get("point") else ""
            lines.append(f"#{f.get('n')} [{f['id']}] ({f['status']}, v{f.get('version')}){where}: {f['text']}")
        shots = []
        for f in entries:
            p = f.get("snapshot")
            if p and os.path.isfile(p):
                with Image.open(p) as im:
                    shots.append(np.asarray(im.convert("RGB").resize((512, 288)), dtype=np.float32) / 255.0)
        images = torch.from_numpy(np.stack(shots)) if shots else torch.zeros((1, 64, 64, 3))
        return ("\n".join(lines) or "no feedback", len(entries), images)


class bEpicWorldDepth:
    """Depth for a world, at 16 bits.

    Depth Anything V2 (the model comfyui_controlnet_aux uses) kept in float and
    saved as a 16-bit PNG under output/worlds_depth. The usual depth nodes hand
    on 8 bits, which leaves the far end of a picture — a hall, a valley — with
    only a few steps of depth; the world's hero view tears into terraces there.
    Feed the file to bEpic World From Reference (or `create … depth`)."""

    CKPTS = ["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth",
             "depth_anything_v2_vits.pth", "depth_anything_v2_vitg.pth"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "ckpt_name": (cls.CKPTS, {"default": "depth_anything_v2_vitl.pth"}),
            "input_size": ("INT", {"default": 770, "min": 266, "max": 1540, "step": 14,
                                   "tooltip": "The model's working size (a multiple of 14); larger = finer, slower"}),
            "filename_prefix": ("STRING", {"default": "depth"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_preview",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "bEpic/worlds"

    def run(self, image, ckpt_name, input_size, filename_prefix):
        try:
            from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
        except ImportError:
            raise RuntimeError("bEpic World Depth needs comfyui_controlnet_aux (its Depth Anything V2 model)")
        import comfy.model_management as mm
        import folder_paths
        det = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(mm.get_torch_device())
        maps = []
        try:
            for frame in image:
                rgb = np.clip(frame.cpu().numpy()[..., :3] * 255.0, 0, 255).astype(np.uint8)
                with torch.no_grad():
                    maps.append(det.model.infer_image(np.ascontiguousarray(rgb[..., ::-1]), input_size=int(input_size)))
        finally:
            del det
        paths, previews = _save_depth16(maps, filename_prefix)
        return {"ui": _ui_images(paths), "result": (torch.from_numpy(np.stack(previews)),)}


def _save_depth16(maps, filename_prefix, metric=False, size=None):
    """Depth maps (float arrays) as 16-bit PNGs under output/worlds_depth,
    normalised to 0..1 inverse depth (bright = near, as worlds read it).
    `metric` maps hold distances in metres and are inverted first; `size`
    (w, h) resizes each to the picture it belongs to."""
    import folder_paths
    folder = os.path.join(folder_paths.get_output_directory(), "worlds_depth")
    os.makedirs(folder, exist_ok=True)
    stem = "".join(c for c in filename_prefix if c.isalnum() or c in "-_") or "depth"
    n = 1 + max([int(f[len(stem) + 1:-4]) for f in os.listdir(folder)
                 if f.startswith(stem + "_") and f.endswith(".png") and f[len(stem) + 1:-4].isdigit()] or [0])
    paths, previews = [], []
    for d in maps:
        d = np.asarray(d, dtype=np.float32)
        if metric:
            d = 1.0 / np.maximum(d, 1e-3)
        if size and tuple(size) != (d.shape[1], d.shape[0]):
            d = np.asarray(Image.fromarray(d, "F").resize(tuple(size), Image.BICUBIC), dtype=np.float32)
        if metric:
            # A stray near pixel shouldn't flatten everything else.
            lo, hi = float(np.percentile(d, 0.1)), float(np.percentile(d, 99.9))
        else:
            lo, hi = float(d.min()), float(d.max())
        d = np.clip((d - lo) / max(1e-8, hi - lo), 0.0, 1.0)
        path = os.path.join(folder, f"{stem}_{n:05d}.png")
        n += 1
        Image.fromarray(np.round(d * 65535).astype(np.uint16)).save(path)
        paths.append(path)
        previews.append(np.repeat(d[..., None], 3, axis=2).astype(np.float32))
    return paths, previews


class bEpicSaveDepth16:
    """Any depth model's output, saved the way worlds want it: 16-bit PNG,
    bright = near, at the size of the picture it belongs to. For depth nodes
    other than bEpic World Depth. Metric ones (distances in metres, like
    ComfyUI-Sharp) are inverted; relative ones (already bright = near) are
    only normalised. Feed it the model's raw float output: a node that has
    already rounded to 8 bits has lost what this keeps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE",),
            "kind": (["relative (bright = near)", "metric distance (metres)"],),
            "filename_prefix": ("STRING", {"default": "depth"}),
        }, "optional": {
            "match_size_of": ("IMAGE", {"tooltip": "The picture the depth belongs to; the map is resized to it"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_preview",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "bEpic/worlds"

    def run(self, depth, kind, filename_prefix, match_size_of=None):
        size = (int(match_size_of.shape[2]), int(match_size_of.shape[1])) if match_size_of is not None else None
        maps = [frame.cpu().numpy()[..., 0] for frame in depth]
        paths, previews = _save_depth16(maps, filename_prefix, metric=kind.startswith("metric"), size=size)
        return {"ui": _ui_images(paths), "result": (torch.from_numpy(np.stack(previews)),)}


class bEpicSeamlessModel:
    """Make any diffusion model draw pictures that tile. At every denoising
    step the latent is rolled by a random offset before the model sees it and
    rolled back after, so the wrap-around edge sits somewhere inside the
    model's view each time and comes out as continuous as everything else.
    Works for text-to-image and img2img, for UNet and DiT models alike; decode
    with bEpic Seamless VAE Decode so the decoder's edges tile too. `axes` "x"
    is for panoramas, which wrap sideways only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "axes": (["xy", "x"],)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "bEpic/worlds"

    def run(self, model, axes):
        import random
        rng = random.Random(0)
        both = axes == "xy"

        def wrapper(apply_model, args):
            x = args["input"]
            h, w = x.shape[-2], x.shape[-1]
            dy = rng.randrange(0, h, 2) if both else 0     # even: DiTs patch in 2x2
            dx = rng.randrange(0, w, 2)
            c = dict(args["c"])
            for k, v in list(c.items()):                    # spatial conditioning moves with it
                if torch.is_tensor(v) and v.dim() >= 4 and v.shape[-2:] == x.shape[-2:]:
                    c[k] = torch.roll(v, (dy, dx), dims=(-2, -1))
            out = apply_model(torch.roll(x, (dy, dx), dims=(-2, -1)), args["timestep"], **c)
            return torch.roll(out, (-dy, -dx), dims=(-2, -1))

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


class bEpicSeamlessVAEDecode:
    """VAE Decode for a latent that tiles: it is padded with its own opposite
    edges first and the padding cut off after, so the decoder never sees a
    hard border and the picture tiles as the latent does."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "axes": (["xy", "x"],)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "bEpic/worlds"

    def run(self, samples, vae, axes):
        lat = samples["samples"]
        p = 8
        py = p if axes == "xy" else 0
        padded = torch.nn.functional.pad(lat, (p, p, py, py), mode="circular") if lat.dim() == 4 else lat
        img = vae.decode(padded)
        if img.dim() == 5:                                   # video VAEs hand back frames
            img = img.reshape(-1, *img.shape[-3:])
        try:
            f = int(vae.spacial_compression_decode())
        except Exception:
            f = img.shape[2] // padded.shape[-1]
        h, w = img.shape[1], img.shape[2]
        return (img[:, py * f:h - py * f, p * f:w - p * f, :].contiguous(),)


class bEpicLoopFrames:
    """Frames that loop without a jump, whatever made them: the last
    `crossfade` frames are faded into the first ones, and the clip gets that
    much shorter. A video model asked to end where it began seldom quite does;
    this makes sure."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",),
                             "crossfade": ("INT", {"default": 12, "min": 1, "max": 256})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "bEpic/worlds"

    def run(self, images, crossfade):
        n = images.shape[0]
        k = min(int(crossfade), n // 3)
        if k < 1:
            return (images,)
        # Cut where the clip itself changes least from one frame to the next
        # (video models jump a little at their latent chunk boundaries), so
        # the frame before the cut and the one after it are a plain step.
        small = images[:, ::8, ::8, :3]
        steps = (small[1:] - small[:-1]).abs().mean(dim=(1, 2, 3))      # steps[c-1]: frame c-1 -> c
        lo, hi = n // 2, n - k
        c = lo + int(torch.argmin(steps[lo - 1:hi - 1]).item())
        out = images[:c].clone()
        for i in range(k):
            t = i / k                                   # 0: all tail, 1: all head
            out[i] = images[c + i] * (1 - t) + images[i] * t
        return (out,)


class bEpicWrapPad:
    """Pad an image with its own opposite edges (as a tile repeats), so a
    model run on it sees across the seams. Crop the padding off afterwards
    (ImageCrop at x = y = pad) and the maps it made tile too."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "pad": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "bEpic/worlds"

    def run(self, image, pad):
        if pad <= 0:
            return (image,)
        x = image.permute(0, 3, 1, 2)
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="circular")
        return (x.permute(0, 2, 3, 1).contiguous(),)


class bEpicUnpad:
    """Take off what bEpic Wrap Pad added, scaled to the image as it is now
    (an upscaler or a model may have changed its size). Takes single-channel
    maps too (roughness, metalness) and hands every map on as an RGB image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "pad": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8}),
                             "padded_size": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8,
                                                     "tooltip": "Width of the padded image the pad was measured "
                                                                "on (0 = the image's own width)"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "bEpic/worlds"

    def run(self, image, pad, padded_size):
        x = image if image.dim() == 4 else image.unsqueeze(-1)
        if x.shape[-1] == 1:
            x = x.repeat(1, 1, 1, 3)
        h, w = x.shape[1], x.shape[2]
        k = w / float(padded_size) if padded_size else 1.0
        py, px = int(round(pad * k * h / w)), int(round(pad * k))
        return (x[:, py:h - py, px:w - px, :3].contiguous(),)


NODE_CLASS_MAPPINGS = {
    "bEpicWorldDepth": bEpicWorldDepth,
    "bEpicSaveDepth16": bEpicSaveDepth16,
    "bEpicSeamlessModel": bEpicSeamlessModel,
    "bEpicSeamlessVAEDecode": bEpicSeamlessVAEDecode,
    "bEpicWrapPad": bEpicWrapPad,
    "bEpicLoopFrames": bEpicLoopFrames,
    "bEpicUnpad": bEpicUnpad,
    "bEpicWorldFromReference": bEpicWorldFromReference,
    "bEpicWorldEdit": bEpicWorldEdit,
    "bEpicWorldFeedback": bEpicWorldFeedback,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "bEpicWorldDepth": "bEpic World Depth (16-bit)",
    "bEpicSaveDepth16": "bEpic Save Depth (16-bit)",
    "bEpicSeamlessModel": "bEpic Seamless Model (tileable output)",
    "bEpicSeamlessVAEDecode": "bEpic Seamless VAE Decode",
    "bEpicWrapPad": "bEpic Wrap Pad (tile context)",
    "bEpicLoopFrames": "bEpic Loop Frames (seamless loop)",
    "bEpicUnpad": "bEpic Unpad (after Wrap Pad)",
    "bEpicWorldFromReference": "bEpic World From Reference",
    "bEpicWorldEdit": "bEpic World Edit",
    "bEpicWorldFeedback": "bEpic World Feedback",
}
