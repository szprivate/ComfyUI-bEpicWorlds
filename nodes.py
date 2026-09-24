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
        folder = os.path.join(folder_paths.get_output_directory(), "worlds_depth")
        os.makedirs(folder, exist_ok=True)
        stem = "".join(c for c in filename_prefix if c.isalnum() or c in "-_") or "depth"
        n = 1 + max([int(f[len(stem) + 1:-4]) for f in os.listdir(folder)
                     if f.startswith(stem + "_") and f.endswith(".png") and f[len(stem) + 1:-4].isdigit()] or [0])
        paths, previews = [], []
        try:
            for frame in image:
                rgb = np.clip(frame.cpu().numpy()[..., :3] * 255.0, 0, 255).astype(np.uint8)
                with torch.no_grad():
                    d = det.model.infer_image(np.ascontiguousarray(rgb[..., ::-1]), input_size=int(input_size))
                d = (d - d.min()) / max(1e-8, float(d.max() - d.min()))
                path = os.path.join(folder, f"{stem}_{n:05d}.png")
                n += 1
                Image.fromarray(np.round(d * 65535).astype(np.uint16)).save(path)
                paths.append(path)
                previews.append(np.repeat(d[..., None], 3, axis=2).astype(np.float32))
        finally:
            del det
        return {"ui": _ui_images(paths), "result": (torch.from_numpy(np.stack(previews)),)}


NODE_CLASS_MAPPINGS = {
    "bEpicWorldDepth": bEpicWorldDepth,
    "bEpicWorldFromReference": bEpicWorldFromReference,
    "bEpicWorldEdit": bEpicWorldEdit,
    "bEpicWorldFeedback": bEpicWorldFeedback,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "bEpicWorldDepth": "bEpic World Depth (16-bit)",
    "bEpicWorldFromReference": "bEpic World From Reference",
    "bEpicWorldEdit": "bEpic World Edit",
    "bEpicWorldFeedback": "bEpic World Feedback",
}
