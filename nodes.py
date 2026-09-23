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
        assets = os.path.join(summary["folder"], "assets")
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


NODE_CLASS_MAPPINGS = {
    "bEpicWorldFromReference": bEpicWorldFromReference,
    "bEpicWorldEdit": bEpicWorldEdit,
    "bEpicWorldFeedback": bEpicWorldFeedback,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "bEpicWorldFromReference": "bEpic World From Reference",
    "bEpicWorldEdit": "bEpic World Edit",
    "bEpicWorldFeedback": "bEpic World Feedback",
}
