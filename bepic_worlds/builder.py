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
FAR_MIN = 30.0          # metres: the least depth a depth mesh's far end is given


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
                       "intensity": 0.4 if night else 1.6}
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
        "grass": _scale_hex(foliage, 1.35), "rock": rock_color, "column": "#e6e4df",
        "model": "#ffffff"}[kind]
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
        "wind": float(entry.get("wind", 0.0 if kind in ("rock", "column") else (0.6 if kind == "grass" else 0.25))),
        "clear": clear,
    }
    # A regular layout (columns in a hall) instead of a random one, and a
    # stretch of the height alone (a column as tall as the room).
    if entry.get("grid"):
        item["scatter"]["grid"] = [float(v) for v in entry["grid"]][:2]
    if entry.get("aspect"):
        item["scatter"]["aspect"] = float(entry["aspect"])
    return item


def depth_range(dmap, fov, pitch, eye_height, world_size):
    """How a relative depth map becomes metres, so the picture's ground lands
    on the terrain's. Returns (near, far, curve).

    A depth model gives inverse depth up to an unknown scale AND offset:
    d = a/z + b. The ground in the lower middle of the frame fixes both: for a
    camera `eye_height` above flat ground, tilted by `pitch`, geometry says how
    far the ground is along every row, and the map's values on those rows fit
    against 1/z. That is the viewer's near/far form, 1/z = d/near + (1-d)/far.

    The fit only covers the depths the ground rows span — a few metres. When
    it puts the far end closer than FAR_MIN (a map that isn't quite affine at
    the back does that), `curve` keeps the fit where it was measured and
    stretches only what lies beyond it out to FAR_MIN: a list of [d, z] the
    viewer interpolates in inverse depth. Otherwise curve is None.
    When no ground fits a line, one point — the bottom middle — sets `near`.
    """
    h, w = dmap.shape[:2]
    tan_half = math.tan(math.radians(fov) / 2.0)
    p = math.radians(pitch)
    far_cap = max(FAR_MIN, world_size * 0.45)
    inv_z, ds = [], []
    for r in range(int(h * 0.62), int(h * 0.99), max(1, h // 120)):
        v = (r + 0.5) / h
        off = math.atan((v - 0.5) * 2 * tan_half)          # below the optical axis
        below = off - p                                     # below the horizon
        if below < math.radians(4):
            continue
        z = eye_height / math.sin(below) * math.cos(off)    # along the camera axis
        row = dmap[r, int(w * 0.38): int(w * 0.62)]
        if row.size:
            inv_z.append(1.0 / z)
            ds.append(float(np.median(row)))
    if len(ds) >= 6:
        A = np.vstack([np.asarray(inv_z), np.ones(len(inv_z))]).T
        (a, b), *_ = np.linalg.lstsq(A, np.asarray(ds), rcond=None)
        resid = np.asarray(ds) - A @ np.array([a, b])
        if a > 1e-4 and float(np.abs(resid).mean()) < 0.06:
            # d = a/z + b  =>  1/z = d/a - b/a
            inv_far = max(-b / a, 1.0 / 5000.0)
            near, far = 1.0 / (1.0 / a + inv_far), min(1.0 / inv_far, 5000.0)
            if far >= FAR_MIN:
                return round(float(max(0.05, near)), 3), round(float(far), 2), None
            d_lo = max(min(ds), 0.02)                       # the farthest ground measured
            z_of = lambda d: 1.0 / max((d - b) / a, 1e-6)
            curve = [[0.0, float(far_cap)], [round(float(d_lo), 4), round(float(z_of(d_lo)), 3)]]
            for d in (0.25, 0.5, 0.75, 1.0):
                if d > d_lo + 0.02:
                    curve.append([d, round(float(z_of(d)), 3)])
            return round(float(max(0.05, near)), 3), round(float(far_cap), 2), curve
    patch = dmap[int(h * 0.94):, int(w * 0.4):int(w * 0.6)]
    d_bottom = float(np.clip(np.median(patch) if patch.size else 1.0, 0.05, 1.0))
    below = math.radians(fov / 2.0 - pitch)
    if below <= math.radians(2):
        return 2.0, far_cap, None
    z_bottom = eye_height / math.sin(below) * math.cos(math.radians(fov / 2.0))
    inv_near = (1.0 / z_bottom - (1.0 - d_bottom) / far_cap) / d_bottom
    near = 1.0 / inv_near if inv_near > 0 else z_bottom
    return round(float(max(0.05, near)), 3), round(float(far_cap), 2), None


def depth_to_z(d, near, far, curve=None):
    """Metres for a depth-map value — the viewer's own conversion."""
    if curve:
        pts = sorted(curve)
        if d <= pts[0][0]:
            return pts[0][1]
        for (d0, z0), (d1, z1) in zip(pts, pts[1:]):
            if d <= d1:
                t = (d - d0) / max(d1 - d0, 1e-9)
                return 1.0 / ((1 - t) / z0 + t / z1)
        return pts[-1][1]
    return 1.0 / (d / near + (1.0 - d) / far)


def has_ceiling(dmap, horizon):
    """True when the depth map shows something near overhead — a ceiling —
    rather than sky, which a depth model puts at (or near) infinity."""
    h = dmap.shape[0]
    top = dmap[: max(1, int(h * 0.15))]
    band = dmap[max(0, int(h * (horizon - 0.04))): max(1, int(h * (horizon + 0.04)))]
    t = float(np.median(top))
    b = float(np.median(band)) if band.size else 0.0
    return t > 0.25 and t > 2.0 * max(b, 0.02)


def ceiling_height(dmap, fov, pitch, eye_height, near, far, curve=None):
    """How high the ceiling is, from the depth up ahead in the map."""
    h, w = dmap.shape[:2]
    patch = dmap[int(h * 0.03): int(h * 0.12), int(w * 0.4): int(w * 0.6)]
    d = float(np.median(patch)) if patch.size else 0.5
    z = depth_to_z(d, near, far, curve)
    v = 0.075                                           # the patch's middle row
    y_cam = (0.5 - v) * 2 * math.tan(math.radians(fov) / 2) * z
    p = math.radians(pitch)
    above_eye = y_cam * math.cos(p) + z * math.sin(p)
    return float(np.clip(eye_height + above_eye, 2.1, 15.0))


def build_world(reference, out_dir, name="world", spec=None, depth=None, heightmap=None,
                panorama=None, fov=50.0, world_size=240.0, eye_height=1.7, seed=None,
                assets_dir=None):
    """Build a world into `out_dir`. Returns (scene, meta).

    `reference`, `depth`, `heightmap` and `panorama` take anything
    reference.load_rgb reads: a path, a PIL image, an array or a ComfyUI
    tensor. Only `reference` is needed. `assets_dir` is where its files go
    (default `<out_dir>/assets`) — a rebuild gets a folder of its own, so the
    versions before it keep the files they point at.
    """
    assets = assets_dir or os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    rgb = ref.load_rgb(reference)
    ref_path = _save(rgb, os.path.join(assets, "reference.png"))

    analysis = ref.analyze(rgb, fov=fov)
    dmap = ref.load_gray(depth) if depth is not None else None
    # Indoors can't be told from colours alone (an overcast sky and a
    # concrete ceiling are both grey), but a depth map says it plainly.
    analysis["indoor"] = bool(dmap is not None and has_ceiling(dmap, analysis["horizon"]))
    recipe = specmod.resolve(spec, analysis)
    if recipe["biome"] == "interior":
        return build_interior(rgb, analysis, recipe, out_dir, assets, ref_path, dmap, name=name,
                              fov=fov, world_size=world_size, eye_height=eye_height)
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

    if dmap is not None:
        d_path = _save(terr.encode_rg16(dmap), os.path.join(assets, "depth.png"))
        near, far, curve = depth_range(dmap, fov, pitch, eye_height, size)
        hero = {"src": src(ref_path), "depth": src(d_path), "encoding": "rg16", "fov": float(fov),
                "near": near, "far": far, "cut": 0.12, "invert": False, "segments": 256}
        if curve:
            hero["curve"] = curve
        items.append({"id": "hero", "kind": "depthmesh", "name": "Reference view",
                      **_transform(eye, (pitch, 0, 0)), "depthmesh": hero})

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


def build_interior(rgb, analysis, recipe, out_dir, assets, ref_path, dmap, name="world",
                   fov=50.0, world_size=240.0, eye_height=1.7):
    """An interior: floor and ceiling textured from the picture, columns on a
    grid, flat indoor light, and the picture itself as the hero view.

    The room's height comes from the depth map (the ceiling up ahead);
    without one, 3.2 m. The floor is a terrain with no relief; the ceiling is
    a second one turned upside down and marked not walkable.
    """
    size = float(world_size)
    pitch = float(analysis["pitch"])
    eye = [0.0, float(eye_height), size * 0.25]
    near, far, curve = depth_range(dmap, fov, pitch, eye_height, size) if dmap is not None else (1.0, size * 0.45, None)
    ceiling = ceiling_height(dmap, fov, pitch, eye_height, near, far, curve) if dmap is not None else 3.2

    flat = np.zeros((HEIGHT_RES, HEIGHT_RES), dtype=np.float32)
    hm_path = _save(terr.encode_rg16(flat), os.path.join(assets, "heightmap_v1.png"))
    _save(flat, os.path.join(assets, "heightmap_preview.png"))
    floor_path = _save(tex.from_reference(rgb, (0.25, 0.8, 0.75, 1.0)), os.path.join(assets, "tex_floor.png"))
    # The ceiling's texture from its brightest open patch — the middle top of
    # a picture is as likely to be a duct or a beam as the ceiling itself.
    top = min(0.35, max(0.12, analysis["horizon"] - 0.12))
    boxes = [(x, y, x + 0.24, y + 0.14) for x in (0.05, 0.25, 0.45, 0.7)
             for y in (0.02, max(0.02, top - 0.14))]
    lum = lambda b: float(tex.crop(rgb, b).mean())
    ceil_path = _save(tex.from_reference(rgb, max(boxes, key=lum)), os.path.join(assets, "tex_ceiling.png"))

    def plain_layers(first):
        return [first] + [{"name": n, "src": None, "color": analysis["ground"], "tile": 10.0}
                          for n in ("rock", "cliff", "peak")]

    floor = terrain_item(size, 0.0, src(hm_path),
                         plain_layers({"name": "floor", "src": src(floor_path), "color": "#ffffff", "tile": 6.0}),
                         False, 0, 0.3)
    floor["name"] = "Floor"
    floor["terrain"]["segments"] = 32
    ceil = terrain_item(size, 0.0, src(hm_path),
                        plain_layers({"name": "ceiling", "src": src(ceil_path), "color": "#ffffff", "tile": 8.0}),
                        False, 0, 0.3)
    ceil.update(id="ceiling", name="Ceiling", position=[0.0, round(ceiling, 3), 0.0], rotation=[180.0, 0.0, 0.0])
    ceil["terrain"]["segments"] = 16
    ceil["terrain"]["walkable"] = False

    env = {"id": "env", "kind": "environment", "name": "Environment", **_transform()}
    # Indoors the distance is the room fading out, in the colour the picture's
    # far end has — and the sky is that same colour, so where floor and
    # ceiling end there is nothing to see but the fade.
    far_col = _scale_hex(_mix(analysis["sky_horizon"], analysis["ground_horizon"], 0.5), 0.85)
    env["sky"] = {"mode": "gradient", "top": far_col, "horizon": far_col, "bottom": far_col, "src": None}
    # Light from overhead fittings, not a sun: soft, from above, no shadows.
    env["sun"] = {"azimuth": 0.0, "elevation": 75.0, "color": "#fff8ee", "intensity": 0.6, "shadows": False}
    env["fog"] = {"color": far_col, "density": 0.02}
    # Near-white fill, strong enough (three's lights are physical: a surface
    # shows albedo x light x intensity / pi) that the textures read at about
    # their own brightness — the picture's colours are in them already.
    env["ambient"] = {"sky": "#ebe7df", "ground": "#dcd6ca", "intensity": 4.0}

    items = [env, floor, ceil]
    # Columns on a grid, as tall as the room — kept out of the part of the
    # room the picture already shows (it has its own), and off the lens.
    light = max(analysis["palette"], key=lambda p: p["lum"])["hex"]
    # Kept clear: exactly the wedge the picture shows (it has its own columns)
    # — everything beside and behind it gets the world's — and the spot you
    # stand on.
    w_px, h_px = analysis["size"]
    half = math.degrees(math.atan(math.tan(math.radians(fov) / 2) * w_px / h_px)) + 3.0
    clear = [{"wedge": {"apex": [eye[0], eye[2]], "yaw": 0.0, "half": round(half, 2), "range": far if far < 1000 else 120.0}},
             {"center": [eye[0], eye[2]], "radius": 3.0}]
    for i, entry in enumerate(recipe["scatter"]):
        entry = dict(entry)
        if entry["type"] == "column":
            entry.setdefault("aspect", ceiling)
            entry.setdefault("color", light)
            gx, gz = entry.get("grid") or (8.5, 8.5)
            entry["count"] = max(int(entry.get("count", 0)), int((size / gx + 1) * (size / gz + 1)))
        items.append(scatter_item(entry, i, clear, analysis["foliage"], analysis["ground"]))

    w, hgt = analysis["size"]
    items.append({"id": "refcam", "kind": "camera", "name": "Reference", **_transform(eye, (pitch, 0, 0)),
                  "fov": float(fov), "resolution": [int(w), int(hgt)],
                  "reference": {"src": src(ref_path), "opacity": 0.5, "wipe": 1.0}})
    if dmap is not None:
        d_path = _save(terr.encode_rg16(dmap), os.path.join(assets, "depth.png"))
        hero = {"src": src(ref_path), "depth": src(d_path), "encoding": "rg16", "fov": float(fov),
                "near": near, "far": far, "cut": 0.12, "invert": False, "segments": 320}
        if curve:
            hero["curve"] = curve
        items.append({"id": "hero", "kind": "depthmesh", "name": "Reference view",
                      **_transform(eye, (pitch, 0, 0)), "depthmesh": hero})

    scene = {
        "version": 1, "fps": 24, "length": 240, "lengthSet": False, "activeCamera": None, "items": items,
        "walk": {"spawn": eye, "yaw": 0.0, "eyeHeight": float(eye_height), "speed": 4.0,
                 "bounds": {"center": [0.0, 0.0], "radius": size * 0.46}},
        "world": {"name": name, "version": 1, "schema": SCHEMA_VERSION},
    }
    meta = {"name": name, "analysis": analysis,
            "interior": {"ceiling": ceiling, "near": near, "far": far, "curve": curve},
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
