"""Real objects from the reference: crops for image-to-3D, placement, texture.

The pipeline this serves (the agent runs the ComfyUI parts):

    reference ──SAM3 "car"──► masks ──object_crops──► one clean crop per object
    best crop ──Hunyuan3D──► untextured mesh (GLB)
    mesh + crop ──project_texture──► textured mesh: the photo's own pixels on its front
    mesh + mask box ──locate──► where it stands in the world, and how big it is

and, for surfaces:

    reference region ──material_crop──► a tileable patch ──upscale + Chord──► PBR maps

Geometry conventions follow the rest of the package: metres, Y up, the
reference camera looking down -Z from `refcam.position`, tilted by its pitch.
"""

import math
import os

import numpy as np
from PIL import Image

from .reference import load_rgb, to_pil
from . import textures as tex


# ── masks and crops ──────────────────────────────────────────────────────────

def mask_array(src):
    """A mask image (any mode) as a boolean array."""
    with Image.open(src) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return a > 0.5


def largest_component(m):
    """The biggest connected piece of a mask — an object seen past a pillar
    comes as two pieces, and only one of them is the object's outline."""
    try:
        from scipy import ndimage
    except Exception:
        return m
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def instances(mask_paths, min_area=0.0015):
    """Objects in a set of SAM3 masks (one image per detected instance):
    [{path, bbox: [x0, y0, x1, y1] in 0..1, area, solidity, pieces, clipped,
    score}], best first.

    "Best" is the one an image-to-3D model can rebuild whole: in one piece
    (nothing standing in front of it), filling its box (not cut by the
    frame or by something else), and big enough to have detail. Specks are
    dropped."""
    out = []
    for p in mask_paths:
        m = mask_array(p)
        h, w = m.shape
        if m.sum() / float(h * w) < min_area:
            continue
        main = largest_component(m)
        pieces_share = float(main.sum()) / max(1.0, float(m.sum()))
        ys, xs = np.nonzero(main)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        area = len(xs) / float(h * w)
        solidity = len(xs) / float((x1 - x0) * (y1 - y0))
        clipped = x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1
        score = area ** 0.5 * solidity * pieces_share * (0.3 if clipped else 1.0)
        out.append({"path": p, "bbox": [x0 / w, y0 / h, x1 / w, y1 / h], "area": round(area, 5),
                    "solidity": round(solidity, 3), "whole": round(pieces_share, 3), "clipped": bool(clipped),
                    "score": round(score, 5)})
    out.sort(key=lambda o: -o["score"])
    return out


def object_crop(reference, mask_path, size=1024, pad=0.12):
    """The object on a plain white ground, centred in a square — what an
    image-to-3D model wants — plus an RGBA version (alpha = mask) that
    becomes the mesh's texture. Returns (rgb_image, rgba_image, crop_box)
    where crop_box is the square's [x0, y0, x1, y1] in 0..1 of the picture."""
    rgb = load_rgb(reference)
    H, W = rgb.shape[:2]
    m = mask_array(mask_path)
    if m.shape != (H, W):
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    m = largest_component(m)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0, sy0 = cx - side // 2, cy - side // 2
    canvas = np.ones((side, side, 3), dtype=np.float32)
    alpha = np.zeros((side, side), dtype=np.float32)
    ax0, ay0 = max(0, sx0), max(0, sy0)
    ax1, ay1 = min(W, sx0 + side), min(H, sy0 + side)
    patch = rgb[ay0:ay1, ax0:ax1]
    pm = m[ay0:ay1, ax0:ax1].astype(np.float32)
    ox, oy = ax0 - sx0, ay0 - sy0
    canvas[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch * pm[..., None] + (1 - pm[..., None])
    alpha[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = pm
    rgb_im = to_pil(canvas).resize((size, size), Image.LANCZOS)
    rgba = np.concatenate([canvas, alpha[..., None]], axis=-1)
    rgba_im = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").resize((size, size), Image.LANCZOS)
    return rgb_im, rgba_im, [sx0 / W, sy0 / H, (sx0 + side) / W, (sy0 + side) / H]


def material_crop(reference, box, size=768):
    """A tileable patch of a surface in the picture (box in 0..1)."""
    return to_pil(tex.from_reference(load_rgb(reference), box, size=size))


# ── where an object stands ───────────────────────────────────────────────────

def _camera(scene):
    cam = next((i for i in scene["items"] if i.get("id") == "refcam"), None)
    if cam is None:
        raise ValueError("the world has no reference camera ('refcam') to place things from")
    w, h = cam.get("resolution", [1920, 1080])
    return cam, math.radians(cam.get("fov", 50.0)), w / max(h, 1)


_HM_CACHE = {}


def _heights(path, encoding):
    """A heightmap as 0..1 heights, read once per file."""
    key = (path, os.path.getmtime(path))
    if key not in _HM_CACHE:
        from .terrain import decode_rg16
        with Image.open(path) as im:
            _HM_CACHE[key] = decode_rg16(im) if encoding == "rg16" else np.asarray(im.convert("L"), np.float32) / 255
    return _HM_CACHE[key]


def _ground_y(scene, x, z):
    """Height of the walkable ground at x, z (a flat floor at the terrain's
    height when there is no heightmap to read here)."""
    t = next((i for i in scene["items"] if i.get("kind") == "terrain"
              and (i.get("terrain") or {}).get("walkable", True) is not False), None)
    if t is None:
        return 0.0
    base = t.get("position", [0, 0, 0])[1]
    tt = t.get("terrain") or {}
    hm = (tt.get("heightmap") or {}).get("path")
    height = float(tt.get("height", 0) or 0)
    if not hm or height <= 0 or not os.path.isfile(hm):
        return base
    from .terrain import height_at
    h = _heights(hm, tt.get("encoding"))
    size = tt.get("size", [200, 200])
    u, v = x / size[0] + 0.5, z / size[1] + 0.5
    return base + height_at(h, u, v) * height


def locate(scene, bbox):
    """Where an object whose picture box is `bbox` stands, and its size.

    The bottom middle of the box is where it touches the ground: the ray
    through that pixel from the reference camera is walked out until it
    meets the ground. The box's height at that distance is the object's
    height in metres. Returns {position: [x, y, z], height, width, distance,
    yaw} — yaw turning the object's front toward the camera."""
    cam, vfov, aspect = _camera(scene)
    eye = cam["position"]
    pitch = math.radians(cam.get("rotation", [0, 0, 0])[0])
    tan_v = math.tan(vfov / 2)
    tan_h = tan_v * aspect
    x0, y0, x1, y1 = bbox
    u, v = (x0 + x1) / 2, y1

    def ray(u, v):
        dx, dy = (u - 0.5) * 2 * tan_h, (0.5 - v) * 2 * tan_v
        # camera -> world, pitch about X; the camera looks down -Z
        return np.array([dx, dy * math.cos(p_) + math.sin(p_), dy * math.sin(p_) - math.cos(p_)])

    p_ = pitch
    d = ray(u, v)
    d /= np.linalg.norm(d)
    # March out to the ground (it may be sloped terrain), then refine.
    t_prev, hit = 0.0, None
    for t in np.concatenate([np.linspace(0.3, 30, 300), np.linspace(30.5, 400, 200)]):
        p = np.array(eye) + d * t
        if p[1] <= _ground_y(scene, p[0], p[2]):
            lo, hi = t_prev, t
            for _ in range(24):
                mid = (lo + hi) / 2
                q = np.array(eye) + d * mid
                if q[1] <= _ground_y(scene, q[0], q[2]):
                    hi = mid
                else:
                    lo = mid
            hit = np.array(eye) + d * hi
            break
        t_prev = t
    if hit is None:
        raise ValueError("that object's foot doesn't meet the ground in the reference view")
    # Depth along the camera axis at the contact point sets the metres per picture height.
    fwd = np.array([0.0, math.sin(pitch), -math.cos(pitch)])
    z_axis = float(np.dot(hit - np.array(eye), fwd))
    height = (y1 - y0) * 2 * tan_v * z_axis
    width = (x1 - x0) * 2 * tan_h * z_axis
    to_cam = np.array(eye) - hit
    yaw = math.degrees(math.atan2(to_cam[0], to_cam[2]))
    return {"position": [round(float(v), 3) for v in hit], "height": round(float(height), 3),
            "width": round(float(width), 3), "distance": round(float(np.linalg.norm(to_cam)), 2),
            "yaw": round(yaw, 1)}


def fit_pitch(objects, fov, eye_height=1.7, aspect=1.5):
    """The camera tilt that makes objects of known size come out that size.

    For an object standing on the ground, its foot and its top are seen at
    two angles below the horizon, and tan(top) / tan(foot) = (eye - H) / eye
    whatever its distance. The horizon — the camera's tilt — is the one
    unknown, and a picture's horizon is exactly what an analysis of colours
    gets wrong indoors (a beam, a duct). `objects`: [{bbox: [x0,y0,x1,y1],
    height: metres}]. Returns (pitch in degrees, per-object pitches); the
    median, so one misjudged object doesn't tip it."""
    tan_v = math.tan(math.radians(fov) / 2)
    found = []
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        H = float(o["height"])
        a_top = math.atan((y0 - 0.5) * 2 * tan_v)          # below the optical axis (+ down)
        a_foot = math.atan((y1 - 0.5) * 2 * tan_v)
        ratio = (eye_height - H) / eye_height

        def f(p):                                        # p: pitch, + up
            b_foot, b_top = a_foot - p, a_top - p           # below the horizon (tilting up lowers it in frame)
            if b_foot <= 1e-4:
                return None
            return math.tan(b_top) / math.tan(b_foot) - ratio
        lo, hi = math.radians(-45), math.radians(45)
        pts = [lo + (hi - lo) * i / 400 for i in range(401)]
        vals = [(p, f(p)) for p in pts]
        vals = [(p, v) for p, v in vals if v is not None]
        root = None
        for (p0, v0), (p1, v1) in zip(vals, vals[1:]):
            if v0 == 0 or (v0 < 0) != (v1 < 0):
                for _ in range(40):
                    pm = (p0 + p1) / 2
                    vm = f(pm)
                    if vm is None:
                        break
                    if (vm < 0) == (v0 < 0):
                        p0, v0 = pm, vm
                    else:
                        p1 = pm
                root = (p0 + p1) / 2
                break
        if root is not None:
            found.append(math.degrees(root))
    if not found:
        raise ValueError("no object gave a usable horizon (are their feet in view?)")
    found.sort()
    return round(found[len(found) // 2], 2), [round(v, 2) for v in found]


# ── the photo's pixels on the mesh ───────────────────────────────────────────

def project_texture(glb_in, rgba_crop, glb_out, front=(0.0, 0.0, 1.0)):
    """Texture an untextured mesh with the crop it was made from.

    An image-to-3D model builds the object as seen in its input image, so the
    crop is a projection of the object's front: each vertex takes its colour
    from where it falls in the crop, looking along `front`. Faces turned away
    get the same pixels, which reads as the object's own colours rather than
    a hole. The mesh is also recentred to stand on y=0, one unit tall.
    Returns the mesh's extent after normalising (width, height, depth).
    """
    import trimesh
    scene = trimesh.load(glb_in, force="scene")
    mesh = trimesh.util.concatenate([g for g in scene.dump() if isinstance(g, trimesh.Trimesh)])
    v = mesh.vertices.copy()
    lo, hi = v.min(0), v.max(0)
    size = hi - lo
    v -= np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    v /= max(size[1], 1e-6)                            # one unit tall, standing on y = 0
    mesh.vertices = v
    lo, hi = v.min(0), v.max(0)

    rgba = rgba_crop if isinstance(rgba_crop, Image.Image) else Image.open(rgba_crop)
    a = np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a[..., 3] > 0.5)
    W, H = rgba.size
    bx0, bx1, by0, by1 = xs.min() / W, (xs.max() + 1) / W, ys.min() / H, (ys.max() + 1) / H
    f = np.asarray(front, dtype=np.float64)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, f)
    right /= np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else 1
    pu = v @ right
    pv = v @ up
    uu = bx0 + (pu - pu.min()) / max(np.ptp(pu), 1e-6) * (bx1 - bx0)
    vv = by0 + (pv.max() - pv) / max(np.ptp(pv), 1e-6) * (by1 - by0)
    uv = np.stack([uu, 1.0 - vv], axis=-1)

    # Fill the texture's empty (masked-out) ground with the object's own
    # colours, so edge texels don't bleed white.
    colour = a[..., :3]
    m = a[..., 3] > 0.5
    fill = colour[m].mean(0) if m.any() else np.array([0.5, 0.5, 0.5])
    filled = np.where(m[..., None], colour, fill)
    img = Image.fromarray((filled * 255).astype(np.uint8), "RGB")
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=img, roughnessFactor=0.6, metallicFactor=0.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    os.makedirs(os.path.dirname(glb_out), exist_ok=True)
    mesh.export(glb_out)
    return [round(float(x), 4) for x in (hi - lo)]
