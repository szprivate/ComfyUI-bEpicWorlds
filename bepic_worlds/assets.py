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


def surface_box(mask_paths, prefer_near=True, reference=None):
    """Where to cut a material patch from a surface SAM3 found: a square lying
    wholly inside the mask (its largest piece), big, near the bottom of the
    picture (the near ground, where the texture is sharpest) and — when the
    `reference` picture is given — plain: of the spots nearly as good, the one
    with the least contrast, so a painted line or a drain doesn't become the
    material. Returns [x0, y0, x1, y1] in 0..1."""
    from scipy import ndimage
    m = None
    for p in mask_paths:
        a = mask_array(p)
        m = a if m is None else (m | a)
    if m is None or not m.any():
        raise ValueError("the mask is empty — nothing of that surface was found")
    m = largest_component(m)
    h, w = m.shape
    dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
    score = dist * ((0.5 + np.arange(h, dtype=np.float32)[:, None] / h) if prefer_near else 1.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    if reference is not None:
        lum = load_rgb(reference).mean(axis=2)
        if lum.shape != m.shape:
            lum = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                             dtype=np.float32) / 255
        good = score >= 0.7 * float(score[y, x])
        ys, xs = np.nonzero(good)
        step = max(1, len(ys) // 400)
        best = None
        for cy, cx in zip(ys[::step], xs[::step]):
            r = max(4, int(dist[cy, cx] * 0.7))
            p = lum[max(0, cy - r):cy + r:2, max(0, cx - r):cx + r:2]
            flat = float(p.std()) / (0.5 + score[cy, cx] / score[y, x])   # plain, and still big and near
            if best is None or flat < best[0]:
                best = (flat, cy, cx)
        _, y, x = best
    half = max(4.0, float(dist[y, x]) * 0.7)        # a square inside the clear disc
    box = [(x - half) / w, (y - half) / h, (x + half) / w, (y + half) / h]
    return [round(float(min(1.0, max(0.0, v))), 4) for v in box]


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

def find_horizon(img):
    """The row (0..1 from the top) where a panorama's sky meets the ground:
    where brightness falls most steeply going down (the sky is brighter than
    the land under it), looked for between 30 % and 95 % of the height; the
    steepest change either way when nothing falls clearly."""
    a = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    rows = np.convolve(a.mean(axis=1), np.ones(5) / 5, mode="same")
    drop = rows[:-1] - rows[1:]
    lo, hi = int(0.3 * 255), int(0.95 * 255)
    seg = drop[lo:hi] if drop[lo:hi].max() > 2.0 else np.abs(drop[lo:hi])
    return (lo + int(np.argmax(seg)) + 0.5) / 256


def level_horizon(img, horizon=None, tolerance=0.03):
    """A panorama whose horizon isn't on its middle row — what an image model
    makes without a 360 LoRA — remapped so it is, as an equirectangular sky
    must be: the sky above is stretched over the upper half, the ground below
    over the lower. Returns (image, horizon row it had)."""
    h = find_horizon(img) if horizon is None else float(horizon)
    if abs(h - 0.5) <= tolerance:
        return img, h
    W, H = img.size
    y = np.arange(H, dtype=np.float32) / H
    src = np.where(y < 0.5, y / 0.5 * h, h + (y - 0.5) / 0.5 * (1 - h))
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    idx = np.clip(src * H, 0, H - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, H - 1)
    f = (idx - i0)[:, None, None]
    out = a[i0] * (1 - f) + a[i1] * f
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), h


def normalize_glb(glb_in, glb_out):
    """A mesh that brings its own materials (Meshy, …) or none: only moved to
    stand on y = 0, centred, one unit tall. Its textures stay untouched."""
    import trimesh
    scene = trimesh.load(glb_in, force="scene")
    lo, hi = scene.bounds
    size = hi - lo
    s = 1.0 / max(float(size[1]), 1e-6)
    t = np.eye(4)
    t[:3, :3] *= s
    t[:3, 3] = -np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) * s
    scene.apply_transform(t)
    os.makedirs(os.path.dirname(os.path.abspath(glb_out)), exist_ok=True)
    scene.export(glb_out)
    return [round(float(x) * s, 4) for x in size]


def decimate(vertices, faces, max_faces=40000):
    """Fewer triangles by vertex clustering: vertices sharing a grid cell
    merge into their mean, and faces that collapse go. Crude for sharp CAD
    edges, but an image-to-3D mesh (surface nets: even, dense, soft) takes it
    well — 200k triangles per copy is more than a walkable world can afford.
    The cell size is searched for the largest count at or under `max_faces`."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= max_faces:
        return v, f
    ext = float(np.ptp(v, axis=0).max()) or 1.0

    def cluster(step):
        q = np.floor((v - v.min(0)) / (ext * step)).astype(np.int64)
        _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        nv = np.zeros((len(counts), 3))
        np.add.at(nv, inv, v)
        nv /= counts[:, None]
        nf = inv[f]
        keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[keep]
        _, first = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)
        return nv, nf[np.sort(first)]

    lo, hi = 1 / 2048, 1 / 24                          # cell size as a share of the object's extent
    best = cluster(hi)
    for _ in range(12):
        mid = (lo * hi) ** 0.5
        nv, nf = cluster(mid)
        if len(nf) > max_faces:
            lo = mid
        else:
            hi, best = mid, (nv, nf)
    used = np.unique(best[1])
    remap = np.full(len(best[0]), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return best[0][used], remap[best[1]]


def project_texture(glb_in, rgba_crop, glb_out, front=(0.0, 0.0, 1.0), max_faces=40000):
    """Texture an untextured mesh with the crop it was made from.

    An image-to-3D model builds the object as seen in its input image, so the
    crop is a projection of the object's front: faces turned towards `front`
    take their colour from where they fall in the crop. Faces the picture never
    saw (the sides, the top, the far end) take the same place in a heavily
    blurred copy — the object's own colours, without a tail light smeared along
    the whole flank. The mesh is decimated to `max_faces`, recentred to stand on
    y = 0 and scaled one unit tall.
    Returns the mesh's extent after normalising (width, height, depth).
    """
    import trimesh
    from PIL import ImageFilter
    scene = trimesh.load(glb_in, force="scene")
    mesh = trimesh.util.concatenate([g for g in scene.dump() if isinstance(g, trimesh.Trimesh)])
    v, faces = decimate(mesh.vertices, mesh.faces, max_faces)
    lo, hi = v.min(0), v.max(0)
    size = hi - lo
    v = v - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    v /= max(size[1], 1e-6)                            # one unit tall, standing on y = 0
    lo, hi = v.min(0), v.max(0)
    smooth = trimesh.Trimesh(v, faces, process=False)
    normals = smooth.vertex_normals.copy()
    fnorm = smooth.face_normals

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
    pu, pv = v @ right, v @ up
    uu = bx0 + (pu - pu.min()) / max(np.ptp(pu), 1e-6) * (bx1 - bx0)
    vv = by0 + (pv.max() - pv) / max(np.ptp(pv), 1e-6) * (by1 - by0)

    # The atlas: the crop on the left half, a blurred copy on the right.
    colour = a[..., :3]
    m = a[..., 3] > 0.5
    fill = colour[m].mean(0) if m.any() else np.array([0.5, 0.5, 0.5])
    filled = Image.fromarray((np.where(m[..., None], colour, fill) * 255).astype(np.uint8), "RGB")
    blurred = filled.filter(ImageFilter.GaussianBlur(max(4, int(0.06 * max(W, H)))))
    atlas = Image.new("RGB", (W * 2, H))
    atlas.paste(filled, (0, 0))
    atlas.paste(blurred, (W, 0))

    # Each face is seen or unseen as a whole, so its corners are split off
    # (a face straddling both halves would smear the atlas between them).
    seen = fnorm @ f > 0.45                          # squarely towards the camera; slanted corners blur
    corners = faces.reshape(-1)
    fv = v[corners]
    fn = normals[corners]
    half = np.repeat(np.where(seen, 0.0, 0.5), 3)
    u = uu[corners] * 0.5 + half
    uv = np.stack([u, 1.0 - vv[corners]], axis=-1)
    out = trimesh.Trimesh(fv, np.arange(len(corners)).reshape(-1, 3), vertex_normals=fn, process=False)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas, roughnessFactor=0.6, metallicFactor=0.0)
    out.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    os.makedirs(os.path.dirname(os.path.abspath(glb_out)), exist_ok=True)
    out.export(glb_out)
    return [round(float(x), 4) for x in (hi - lo)]
