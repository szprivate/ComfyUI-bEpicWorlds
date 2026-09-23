"""Reference image (+ spec, + optional depth / heightmap / panorama) → a world.

What comes out is a scene in the bEpic viewer's schema (see SCHEMA.md): an
environment matched to the picture, a terrain with ground textures cut from
it, vegetation and rocks scattered by biome, a reference camera standing where
the picture was taken — with the picture laid over its view for comparison —
and, when a depth map is given, the picture itself pushed out into 3D as the
hero view. The files it points at are written to `assets/`.

Item ids are fixed words ("env", "terrain", "refcam", "hero", "scatter_pine"…)
so an agent can name what it wants to change without looking anything up.
"""

import math
import os

import numpy as np

from . import reference as ref
from . import spec as specmod
from . import terrain as terr
from . import textures as tex

SCHEMA_VERSION = 1
HEIGHT_RES = 257


def _transform(position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    return {"position": [float(v) for v in position], "rotation": [float(v) for v in rotation],
            "scale": [float(v) for v in scale], "pivot": [0.0, 0.0, 0.0],
            "visible": True, "tracks": {}, "parent": None}


def src(path, name=None):
    """How a scene item points at a file: by absolute path, served by the
    viewer's own file route (hence `external`)."""
    path = os.path.abspath(path)
    return {"path": path, "name": name or os.path.basename(path), "external": True}


def _save(img_or_arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = img_or_arr if hasattr(img_or_arr, "save") else ref.to_pil(img_or_arr)
    im.save(path)
    return path


def _mix(a, b, t):
    return ref.hex_of(ref.rgb_of(a) * (1 - t) + ref.rgb_of(b) * t)


def _scale_hex(c, k):
    return ref.hex_of(np.clip(ref.rgb_of(c) * k, 0, 1))


def environment_item(analysis, recipe, panorama_src=None):
    sun = recipe["sun"]
    night = sun["elevation"] < 0
    item = {"id": "env", "kind": "environment", "name": "Environment", **_transform()}
    item["sky"] = {
        "mode": "panorama" if panorama_src else "gradient",
        "top": analysis["sky_top"],
        "horizon": analysis["sky_horizon"],
        "bottom": analysis["ground_horizon"],
        "src": panorama_src,
    }
    item["sun"] = {
        "azimuth": float(sun["azimuth"]),
        "elevation": float(sun["elevation"]),
        "color": sun["color"],
        "intensity": 0.35 if night else 2.6,
        "shadows": True,
    }
    item["fog"] = {"color": _mix(analysis["sky_horizon"], analysis["ground_horizon"], 0.3),
                   "density": float(recipe["fog"])}
    item["ambient"] = {"sky": analysis["sky_top"], "ground": analysis["ground"],
                       "intensity": 0.25 if night else 0.9}
    return item


def terrain_item(size, height, heightmap_src, layers, snow, seed, roughness):
    item = {"id": "terrain", "kind": "terrain", "name": "Terrain", **_transform()}
    item["terrain"] = {
        "heightmap": heightmap_src, "encoding": "rg16",
        "seed": int(seed), "roughness": float(roughness),
        "size": [float(size), float(size)], "height": float(height),
        "segments": 256,
        "layers": layers,
        "rules": {"rockSlope": [24, 38], "cliffSlope": [40, 56],
                  "peak": [0.68, 0.8] if snow else [0.9, 1.01]},
    }
    return item


def scatter_item(entry, index, clear, foliage, rock_color):
    kind = entry["type"]
    color = entry.get("color") or {
        "pine": _scale_hex(foliage, 0.8), "tree": foliage, "bush": _scale_hex(foliage, 1.1),
        "grass": _scale_hex(foliage, 1.35), "rock": rock_color, "model": "#ffffff"}[kind]
    item = {"id": f"scatter_{kind}" if index == 0 else f"scatter_{kind}_{index}",
            "kind": "scatter", "name": f"{kind.title()}s", **_transform()}
    item["scatter"] = {
        "target": "terrain",
        "source": {"type": kind, "src": entry.get("src")},
        "count": int(entry.get("count", 500)),
        "seed": int(entry.get("seed", index * 7 + 11)),
        "scale": [float(v) for v in entry.get("scale", [0.8, 1.4])],
        "layer": int(entry.get("layer", 0)),
        "maxSlope": float(entry.get("maxSlope", 32 if kind != "rock" else 60)),
        "color": color,
        "wind": float(entry.get("wind", 0.0 if kind == "rock" else (0.6 if kind == "grass" else 0.25))),
        "clear": clear,
    }
    return item


def build_world(reference, out_dir, name="world", spec=None, depth=None, heightmap=None,
                panorama=None, fov=50.0, world_size=240.0, eye_height=1.7, seed=None):
    """Build a world into `out_dir`. Returns (scene, meta).

    `reference`, `depth`, `heightmap` and `panorama` take anything
    reference.load_rgb reads: a path, a PIL image, an array or a ComfyUI
    tensor. Only `reference` is needed.
    """
    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    rgb = ref.load_rgb(reference)
    ref_path = _save(rgb, os.path.join(assets, "reference.png"))

    analysis = ref.analyze(rgb, fov=fov)
    recipe = specmod.resolve(spec, analysis)
    if seed is not None:
        recipe["terrain"]["seed"] = int(seed)
    t = recipe["terrain"]
    size, height = float(world_size), float(t.get("height", 20))

    # Where you stand: toward the near side, with most of the land ahead.
    spawn_uv = (0.5, 0.75)
    if heightmap is not None:
        h = terr.from_image(heightmap, HEIGHT_RES)
    else:
        h = terr.shape_for_view(
            terr.fbm(HEIGHT_RES, seed=int(t.get("seed", 0)), roughness=float(t.get("roughness", 0.5))),
            spawn=spawn_uv)
    hm_path = _save(terr.encode_rg16(h), os.path.join(assets, "heightmap_v1.png"))
    _save(h, os.path.join(assets, "heightmap_preview.png"))

    # Ground from the bottom of the picture; a rock layer from the same grain,
    # pulled toward the picture's own grey; cliffs darker; peaks snow or pale.
    ground_tex = tex.from_reference(rgb, (0.2, 0.74, 0.8, 1.0))
    g_path = _save(ground_tex, os.path.join(assets, "tex_ground.png"))
    grey = float(ref.rgb_of(analysis["ground"]).mean())
    rock_rgb = np.array([grey * 0.95, grey * 0.93, grey * 0.9]) * 0.9 + 0.08
    rock_color = ref.hex_of(rock_rgb)
    r_path = _save(tex.tinted(ground_tex, rock_rgb, 0.75), os.path.join(assets, "tex_rock.png"))
    layers = [
        {"name": "ground", "src": src(g_path), "color": "#ffffff", "tile": 9.0},
        {"name": "rock", "src": src(r_path), "color": "#ffffff", "tile": 14.0},
        {"name": "cliff", "src": None, "color": _scale_hex(rock_color, 0.62), "tile": 20.0},
        {"name": "peak", "src": None, "color": "#f2f4f7" if recipe["snow"] else _scale_hex(analysis["ground"], 1.25),
         "tile": 20.0},
    ]

    spawn_x = (spawn_uv[0] - 0.5) * size
    spawn_z = (spawn_uv[1] - 0.5) * size
    ground_y = terr.height_at(h, *spawn_uv) * height
    eye = [spawn_x, ground_y + eye_height, spawn_z]
    pitch = float(analysis["pitch"])

    pano_src = None
    if panorama is not None:
        pano_src = src(_save(ref.load_rgb(panorama), os.path.join(assets, "panorama.png")))

    items = [
        environment_item(analysis, recipe, pano_src),
        terrain_item(size, height, src(hm_path), layers, recipe["snow"],
                     t.get("seed", 0), t.get("roughness", 0.5)),
    ]
    # Nothing grows in the reference view's foreground: the picture shows
    # what is there, and a tree planted on the lens would hide all of it.
    clear = {"center": [spawn_x, spawn_z - 6.0], "radius": 12.0}
    counts = {}
    for entry in recipe["scatter"]:
        k = entry["type"]
        items.append(scatter_item(entry, counts.get(k, 0), clear, analysis["foliage"], rock_color))
        counts[k] = counts.get(k, 0) + 1

    w, hgt = analysis["size"]
    cam = {"id": "refcam", "kind": "camera", "name": "Reference",
           **_transform(eye, (pitch, 0, 0)),
           "fov": float(fov), "resolution": [int(w), int(hgt)],
           "reference": {"src": src(ref_path), "opacity": 0.5, "wipe": 1.0}}
    items.append(cam)

    if depth is not None:
        d_path = _save(ref.load_gray(depth), os.path.join(assets, "depth.png"))
        items.append({"id": "hero", "kind": "depthmesh", "name": "Reference view",
                      **_transform(eye, (pitch, 0, 0)),
                      "depthmesh": {"src": src(ref_path), "depth": src(d_path), "fov": float(fov),
                                    "near": 2.0, "far": size * 0.45, "cut": 0.12,
                                    "invert": False, "segments": 256}})

    scene = {
        "version": 1, "fps": 24, "length": 240, "lengthSet": False,
        "activeCamera": None, "items": items,
        "walk": {"spawn": eye, "yaw": 0.0, "eyeHeight": float(eye_height), "speed": 5.0,
                 "bounds": {"center": [0.0, 0.0], "radius": size * 0.46}},
        "world": {"name": name, "version": 1, "schema": SCHEMA_VERSION},
    }
    meta = {"name": name, "analysis": analysis,
            "recipe": {k: v for k, v in recipe.items() if k != "spec"},
            "spec": recipe["spec"], "fov": fov, "world_size": size, "eye_height": eye_height}
    return scene, meta


def regenerate_heightmap(scene, out_dir, version, seed=None, roughness=None, height=None,
                         heightmap=None):
    """A new terrain shape for an existing world, written under a new name so
    a viewer holding the old one fetches the new one."""
    item = next(i for i in scene["items"] if i.get("kind") == "terrain")
    t = item["terrain"]
    if seed is not None:
        t["seed"] = int(seed)
    if roughness is not None:
        t["roughness"] = float(roughness)
    if height is not None:
        t["height"] = float(height)
    if heightmap is not None:
        h = terr.from_image(heightmap, HEIGHT_RES)
    else:
        h = terr.shape_for_view(terr.fbm(HEIGHT_RES, seed=int(t.get("seed", 0)),
                                         roughness=float(t.get("roughness", 0.5))))
    path = _save(terr.encode_rg16(h), os.path.join(out_dir, "assets", f"heightmap_v{version}.png"))
    _save(h, os.path.join(out_dir, "assets", "heightmap_preview.png"))
    t["heightmap"] = src(path)
    t["encoding"] = "rg16"
    # Keep the walk and the reference camera standing on the new ground.
    size = t["size"][0]
    walk = scene.get("walk") or {}
    sp = walk.get("spawn")
    if sp:
        u, v = sp[0] / size + 0.5, sp[2] / size + 0.5
        ground = terr.height_at(h, u, v) * t["height"]
        dy = ground + walk.get("eyeHeight", 1.7) - sp[1]
        sp[1] += dy
        for it in scene["items"]:
            if it.get("id") in ("refcam", "hero"):
                it["position"][1] += dy
    return scene
