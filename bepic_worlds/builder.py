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

from . import lights as lightsmod
from . import reference as ref
from . import spec as specmod
from . import surface
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


def pbr_layer(name, tex_rgb, assets, crop=None, roughness=None, tile=9.0, color="#ffffff", normal_scale=1.0,
              delight=0.85, baked=0.0):
    """A terrain layer with PBR maps: the texture without its highlights, a
    normal map and a roughness map beside it, and the roughness it was read as.
    `delight` is how much of the highlights come out (a ceiling's brightness is
    mostly the lamps on it, and it keeps more of it)."""
    p = surface.pbr_set(tex_rgb, source_region=crop, roughness=roughness, delight_strength=delight)
    a_path = _save(p["albedo"], os.path.join(assets, f"tex_{name}.png"))
    n_path = _save(p["normal"], os.path.join(assets, f"tex_{name}_normal.png"))
    r_path = os.path.join(assets, f"tex_{name}_rough.png")
    surface.save_gray(p["rough_map"], r_path)
    return {"name": name, "src": src(a_path), "normal": src(n_path), "rough": src(r_path),
            "roughness": p["roughness"], "normalScale": float(normal_scale), "color": color, "tile": float(tile),
            "baked": float(baked)}


def light_sources(rgb, analysis):
    """Where the picture's lights are, across it (u, 0..1): lamps, and the sun
    when it is in frame. What a floor's reflections are looked for under."""
    us = [l["u"] for l in lightsmod.find_lamps(rgb, analysis["horizon"])]
    sun = analysis.get("sun") or {}
    if sun.get("in_frame"):
        h, w = rgb.shape[:2]
        hf = 2 * math.atan(math.tan(math.radians(analysis["fov"]) / 2) * w / h)
        us.append(0.5 + math.tan(math.radians(sun["azimuth"])) / (2 * math.tan(hf / 2)))
    return us


def render_settings(interior):
    """How a world is drawn: a tone curve that keeps colours as they are
    (Khronos PBR Neutral), bloom so lamps and the sun glow, and reflections
    captured from the world itself at the reference view. `fill` is how much
    of the hemisphere fill stays once reflections carry light too — more
    indoors, where lamps light the ceiling sideways and nothing captures it."""
    return {"tone": "neutral", "exposure": 1.0, "bloom": 0.35 if interior else 0.12,
            "reflections": "capture", "fill": 0.8 if interior else 0.35}


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
    item["render"] = render_settings(False)
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
        "lamp": "#fff6e8", "model": "#ffffff"}[kind]
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
    # Hung this far above the ground (a lamp under a ceiling), and glowing.
    for key in ("lift", "emissive", "roughness"):
        if entry.get(key) is not None:
            item["scatter"][key] = float(entry[key])
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
                assets_dir=None, pitch=None):
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
    if pitch is not None:
        # A tilt measured from objects of known size (assets.fit_pitch) beats
        # the horizon guessed from colours; the horizon row follows from it.
        analysis["pitch"] = float(pitch)
        analysis["horizon"] = round(0.5 + math.tan(math.radians(float(pitch))) / (2 * math.tan(math.radians(fov) / 2)), 4)
        analysis["pitch_source"] = "objects"
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

    # Ground from the bottom of the picture, as glossy as the picture's
    # reflections say; a rock layer from the same grain, pulled toward the
    # picture's own grey; cliffs darker; peaks snow or pale.
    ground_box = (0.2, 0.74, 0.8, 1.0)
    ground_tex = tex.from_reference(rgb, ground_box)
    floor_rough, _ = surface.floor_gloss(rgb, analysis["horizon"], light_sources(rgb, analysis))
    grey = float(ref.rgb_of(analysis["ground"]).mean())
    rock_rgb = np.array([grey * 0.95, grey * 0.93, grey * 0.9]) * 0.9 + 0.08
    rock_color = ref.hex_of(rock_rgb)
    peak = {"name": "peak", "src": None, "color": "#f2f4f7" if recipe["snow"] else _scale_hex(analysis["ground"], 1.25),
            "tile": 20.0, "roughness": 0.55 if recipe["snow"] else 0.9}
    layers = [
        pbr_layer("ground", ground_tex, assets, roughness=floor_rough, tile=9.0,
                  normal_scale=round(0.1 + 0.25 * floor_rough, 2)),
        pbr_layer("rock", tex.tinted(ground_tex, rock_rgb, 0.75), assets, roughness=0.85, tile=14.0, normal_scale=0.8),
        {"name": "cliff", "src": None, "color": _scale_hex(rock_color, 0.62), "tile": 20.0, "roughness": 0.9},
        peak,
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
    if dmap is not None:
        # With a depth map the picture itself stands in 3D across its whole
        # view: kept clear is the wedge it covers, out to where it ends —
        # beside and behind it the world's own trees grow.
        near, far, curve = depth_range(dmap, fov, pitch, eye_height, size)
        w_px, h_px = analysis["size"]
        # Wider than the view by a canopy's breadth: a tree just outside the
        # edge still reaches into the frame.
        half = math.degrees(math.atan(math.tan(math.radians(fov) / 2) * w_px / h_px)) + 12.0
        clear = [{"wedge": {"apex": [eye[0], eye[2]], "yaw": 0.0, "half": round(half, 2),
                            "range": round(min(far, 400.0), 1)}},
                 {"center": [eye[0], eye[2]], "radius": 10.0}]
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
    # The ceiling's texture from its brightest open patch — the middle top of
    # a picture is as likely to be a duct or a beam as the ceiling itself.
    top = min(0.35, max(0.12, analysis["horizon"] - 0.12))
    boxes = [(x, y, x + 0.24, y + 0.14) for x in (0.05, 0.25, 0.45, 0.7)
             for y in (0.02, max(0.02, top - 0.14))]
    lum = lambda b: float(tex.crop(rgb, b).mean())
    ceil_box = max(boxes, key=lum)
    floor_box = (0.25, 0.8, 0.75, 1.0)
    floor_rough, _ = surface.floor_gloss(rgb, analysis["horizon"], light_sources(rgb, analysis))
    # Relief is gentler the glossier the floor: on a near-mirror, small bumps
    # read as ripples on water.
    floor_layer = pbr_layer("floor", tex.from_reference(rgb, floor_box), assets, roughness=floor_rough, tile=6.0,
                            normal_scale=round(0.15 + 0.5 * floor_rough, 2), baked=0.25)
    ceil_layer = pbr_layer("ceiling", tex.from_reference(rgb, ceil_box), assets, crop=tex.crop(rgb, ceil_box),
                           tile=8.0, normal_scale=0.5, delight=0.2, baked=0.75)

    def plain_layers(first):
        return [first] + [{"name": n, "src": None, "color": analysis["ground"], "tile": 10.0, "roughness": 0.85}
                          for n in ("rock", "cliff", "peak")]

    floor = terrain_item(size, 0.0, src(hm_path), plain_layers(floor_layer), False, 0, 0.3)
    floor["name"] = "Floor"
    floor["terrain"]["segments"] = 32
    ceil = terrain_item(size, 0.0, src(hm_path), plain_layers(ceil_layer), False, 0, 0.3)
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
    env["render"] = render_settings(True)

    items = [env, floor, ceil]

    # The picture's own lamps, where the depth map says they hang.
    w_px, h_px = analysis["size"]
    lamp_color = "#fff6e8"
    if dmap is not None:
        found = lightsmod.find_lamps(rgb, analysis["horizon"], max_lamps=8)
        placed = lightsmod.place(found, lambda d: depth_to_z(d, near, far, curve), dmap, fov, w_px / h_px, pitch, eye,
                                 ceiling_y=ceiling - 0.05)
        for i, lamp in enumerate(placed):
            lamp_color = placed[0]["color"]
            items.append({"id": f"light_{i + 1}", "kind": "light", "name": f"Lamp {i + 1}",
                          **_transform([lamp["position"][0], min(lamp["position"][1], ceiling - 0.06),
                                        lamp["position"][2]], (0, lamp["yaw"], 0)),
                          "light": {"type": "point", "color": lamp["color"], "intensity": 18.0,
                                    "distance": 14.0, "decay": 2.0, "shadows": False,
                                    "fixture": {"shape": "tube", "size": [lamp["length"], lamp["width"], 0.05],
                                                "emissive": 6.0}}})
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
    # Strip lights on the same grid under the rest of the ceiling: they glow
    # (and bloom); the light itself is the fill and the captured reflections.
    items.append(scatter_item({"type": "lamp", "count": 2000, "grid": [8.5, 4.25], "scale": [1.0, 1.0],
                               "layer": -1, "lift": round(ceiling - 0.04, 3), "color": lamp_color,
                               "emissive": 6.0}, 0, clear, analysis["foliage"], analysis["ground"]))

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
