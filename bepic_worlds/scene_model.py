"""A 3D model of the whole picture as the heart of a world: where it stands,
how big it is, and the ground that meets it.

Image-to-3D engines give two kinds of model:

- **camera models** (SHARP, MoGe): built in the reference camera's own space,
  in metres. They go exactly where the camera says — nothing to guess. The
  terrain is then brought to *their* ground.
- **object models** (Meshy, Trellis, Tripo, Hunyuan, Rodin): centred on
  themselves, of no particular size, turned any way. Their layout is the
  engine's reading of the picture, not a measurement of it (Meshy made an
  L-shaped block from a street between two rows of houses), so they are
  fitted by what survives that:
    * **size** from height: the picture's buildings, measured in metres by its
      depth map, against the model's, above their own ground;
    * **ground** from the model's rim: where its outer edge meets the ground,
      the terrain must be;
    * **turn and place** by a trimmed fit of what the reference camera would
      see of the model to the picture's own 3D points, from twelve starting
      turns, keeping the best.

Either way the terrain is flattened under the model's footprint to the
model's ground level, blending back into the land around it, so walking off
the model's street onto the world's ground has no step.

Conventions are the package's: metres, Y up, the reference camera looking
down -Z from `refcam.position`, tilted by its pitch about X. A model item's
transform is T(position)·R(0, yaw, 0)·S(scale); its GLB is left as the engine
wrote it (the transform does the placing), unless it has to be textured.
"""

import math
import os

import numpy as np
from PIL import Image

SKY_DISP = 0.0015            # a depth-map value below this is sky (the viewer's rule)
GRID = (160, 90)             # the z-buffer that decides what the camera sees of a model


# ── rotations, as three.js composes them (Euler XYZ, degrees) ────────────────

def _rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_matrix(rotation):
    x, y, z = (list(rotation) + [0, 0, 0])[:3]
    return _rx(x) @ _ry(y) @ _rz(z)


# ── the model ────────────────────────────────────────────────────────────────

def load_mesh(path):
    """One triangle mesh from a GLB/OBJ/PLY/…, in the file's own coordinates."""
    import trimesh
    scene = trimesh.load(path, force="scene")
    parts = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not parts:
        raise ValueError("the model has no triangles")
    return trimesh.util.concatenate(parts)


def surface_samples(mesh, n=60000, seed=0):
    """Points spread evenly over the surface, and their normals."""
    import trimesh
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, np.float64), np.asarray(mesh.face_normals[fi], np.float64)


def _occupancy(xz, n=96):
    lo = xz.min(0)
    ext = max(float(np.ptp(xz, axis=0).max()), 1e-6)
    ij = np.clip(((xz - lo) / ext * (n - 1)).astype(int), 0, n - 1)
    grid = np.zeros((n, n), bool)
    grid[ij[:, 1], ij[:, 0]] = True
    return grid, ij


def ground_level(mesh):
    """The height where the model meets the ground, in its own units: the
    lowest substantial level among the upward-facing faces along its outer rim
    (a diorama's slab, the ends of its street). Roofs face up too, but they
    are higher; the model's very bottom, when nothing faces up at the rim."""
    from scipy import ndimage
    c = mesh.triangles_center
    up = mesh.face_normals[:, 1] > 0.85
    area = mesh.area_faces
    grid, ij = _occupancy(c[:, [0, 2]])
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, iterations=2))
    inside = ndimage.distance_transform_edt(grid)
    rim = inside[ij[:, 1], ij[:, 0]] <= 3.0
    pick = up & rim
    if pick.sum() < 20:
        return float(mesh.bounds[0][1])
    ys, w = c[pick, 1], area[pick]
    lo, hi = float(ys.min()), float(ys.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(ys, bins=24, range=(lo, hi), weights=w)
    need = 0.15 * float(hist.max())
    k = int(np.argmax(hist >= need))
    sel = (ys >= edges[k]) & (ys <= edges[k + 1])
    return float(np.average(ys[sel], weights=w[sel]))


# ── the picture, in 3D ───────────────────────────────────────────────────────

def _depth_to_z(d, near, far, curve=None):
    d = np.asarray(d, np.float64)
    if curve:
        pts = sorted(curve)
        ds = np.array([p[0] for p in pts])
        inv = 1.0 / np.array([p[1] for p in pts])
        return 1.0 / np.interp(d, ds, inv)
    return 1.0 / (d / near + (1.0 - d) / far)


def _item(scene, iid):
    return next((i for i in scene.get("items", []) if i.get("id") == iid), None)


def ground_heights(scene, x, z):
    """Walkable ground height at many x, z at once (assets._ground_y, vectorised)."""
    from .assets import _heights
    x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
    t = next((i for i in scene["items"] if i.get("kind") == "terrain"
              and (i.get("terrain") or {}).get("walkable", True) is not False), None)
    if t is None:
        return np.zeros_like(x)
    base = float(t.get("position", [0, 0, 0])[1])
    tt = t.get("terrain") or {}
    hm = (tt.get("heightmap") or {}).get("path")
    height = float(tt.get("height", 0) or 0)
    if not hm or height <= 0 or not os.path.isfile(hm):
        return np.full_like(x, base)
    h = _heights(hm, tt.get("encoding"))
    size = tt.get("size", [200, 200])
    n = h.shape[0] - 1
    fx = np.clip(x / size[0] + 0.5, 0, 1) * n
    fy = np.clip(z / size[1] + 0.5, 0, 1) * n
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, n), np.minimum(y0 + 1, n)
    ax, ay = fx - x0, fy - y0
    top = h[y0, x0] * (1 - ax) + h[y0, x1] * ax
    bot = h[y1, x0] * (1 - ax) + h[y1, x1] * ax
    return base + (top * (1 - ay) + bot * ay) * height


def settle_camera(scene):
    """Stand the reference camera, the picture's depth mesh and the walk start
    eye-height above the ground again — an edit to the terrain alone (a
    flattened height, a moved base) can leave them floating or buried, and
    everything fitted to the picture would float with them. Returns the metres
    moved (0 when they already stood right)."""
    walk = scene.get("walk") or {}
    eye_h = float(walk.get("eyeHeight", 1.7))
    cam = _item(scene, "refcam")
    anchor = cam["position"] if cam else walk.get("spawn")
    if not anchor:
        return 0.0
    ground = float(ground_heights(scene, [anchor[0]], [anchor[2]])[0])
    dy = ground + eye_h - float(anchor[1])
    if abs(dy) < 0.05:
        return 0.0
    for iid in ("refcam", "hero"):
        it = _item(scene, iid)
        if it:
            it["position"][1] = round(float(it["position"][1]) + dy, 4)
    if walk.get("spawn"):
        sp = walk["spawn"]
        g = float(ground_heights(scene, [sp[0]], [sp[2]])[0])
        sp[1] = round(g + eye_h, 4)
    return round(dy, 3)


def picture_points(scene, rows=160):
    """The reference picture as 3D points in the world, where the world's depth
    mesh (`hero`) puts them: {points, ground (bool: lies on the terrain), uv}.
    Sky is left out."""
    from .terrain import decode_rg16
    hero = _item(scene, "hero")
    if hero is None or hero.get("kind") != "depthmesh":
        raise ValueError("the world has no depth mesh of its picture ('hero') to fit a model to")
    d = hero["depthmesh"]
    with Image.open(d["src"]["path"]) as im:
        aspect = im.width / max(1, im.height)
    with Image.open(d["depth"]["path"]) as im:
        if d.get("encoding") == "rg16":
            disp = decode_rg16(im)
        else:
            disp = np.asarray(im.convert("L"), np.float32) / 255.0
    cols = max(8, int(round(rows * aspect)))
    disp = np.asarray(Image.fromarray(disp.astype(np.float32), "F").resize((cols, rows), Image.BILINEAR))
    if d.get("invert"):
        disp = 1.0 - disp
    v, u = np.mgrid[0:rows, 0:cols].astype(np.float64)
    u, v = (u + 0.5) / cols, (v + 0.5) / rows
    keep = disp >= SKY_DISP
    z = _depth_to_z(disp[keep], float(d.get("near", 1.0)), float(d.get("far", 100.0)), d.get("curve"))
    tan_v = math.tan(math.radians(float(d.get("fov", 50.0))) / 2)
    tan_h = tan_v * aspect
    cam = np.stack([(u[keep] - 0.5) * 2 * tan_h * z, (0.5 - v[keep]) * 2 * tan_v * z, -z], axis=-1)
    world = cam @ euler_matrix(hero.get("rotation", [0, 0, 0])).T + np.asarray(hero.get("position", [0, 0, 0]))
    gy = ground_heights(scene, world[:, 0], world[:, 2])
    return {"points": world, "ground": np.abs(world[:, 1] - gy) < 0.35, "height": world[:, 1] - gy,
            "uv": np.stack([u[keep], v[keep]], -1)}


# ── what the reference camera sees ───────────────────────────────────────────

class RefCamera:
    def __init__(self, scene):
        cam = _item(scene, "refcam")
        if cam is None:
            raise ValueError("the world has no reference camera ('refcam')")
        self.eye = np.asarray(cam["position"], np.float64)
        self.rot = euler_matrix(cam.get("rotation", [0, 0, 0]))
        w, h = cam.get("resolution", [1920, 1080])
        self.aspect = w / max(1, h)
        self.tan_v = math.tan(math.radians(float(cam.get("fov", 50.0))) / 2)
        self.tan_h = self.tan_v * self.aspect

    def project(self, pts):
        """(u, v, depth along the axis) for world points; depth <= 0 is behind."""
        c = (np.asarray(pts) - self.eye) @ self.rot           # world -> camera (rot is orthonormal)
        z = -c[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        return 0.5 + c[:, 0] / (2 * self.tan_h * zs), 0.5 - c[:, 1] / (2 * self.tan_v * zs), z

    def visible(self, pts, normals=None):
        """Which points the camera sees: in frame, in front, facing it, and
        nearest in their cell of a coarse depth buffer."""
        u, v, z = self.project(pts)
        ok = (z > 0.2) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if normals is not None:
            to_eye = self.eye - pts
            ok &= np.einsum("ij,ij->i", normals, to_eye) > 0
        idx = np.nonzero(ok)[0]
        if not len(idx):
            return ok
        cell = (v[idx] * GRID[1]).astype(int) * GRID[0] + (u[idx] * GRID[0]).astype(int)
        zmin = np.full(GRID[0] * GRID[1], np.inf)
        np.minimum.at(zmin, cell, z[idx])
        near = z[idx] <= zmin[cell] * 1.04 + 0.1
        out = np.zeros(len(pts), bool)
        out[idx[near]] = True
        return out


# ── fitting an object model ──────────────────────────────────────────────────

def _ry_mat(deg):
    return _ry(deg)


def fit_object_model(scene, mesh, height_m=None, yaw=None, starts=12, iters=18, seed=0):
    """Size, turn and place an object model on the picture. Returns
    {position, yaw, scale, ground_level (model units), error_m, coverage,
    size_m: [w, h, d], scale_from}.

    `height_m` fixes the height of the model's tallest parts (its 95th
    percentile above its ground) instead of reading it from the picture;
    `yaw` fixes the turn."""
    from scipy.spatial import cKDTree
    pic = picture_points(scene)
    P_all = pic["points"]
    struct = ~pic["ground"] & (pic["height"] > 0.3)
    P = P_all[struct] if struct.sum() > 200 else P_all
    tree = cKDTree(P)
    cam = RefCamera(scene)

    g = ground_level(mesh)
    M, N = surface_samples(mesh, 24000, seed=seed)
    above = M[:, 1] - g
    h_model = float(np.percentile(above[above > 0], 95)) if (above > 0).any() else float(np.ptp(M[:, 1]))
    if height_m:
        h_pic, scale_from = float(height_m), "given height"
    else:
        h_pic = float(np.percentile(pic["height"][struct], 95)) if struct.sum() > 200 else h_model
        scale_from = "the picture's buildings"
    s0 = h_pic / max(h_model, 1e-6)

    fp = M[:, [0, 2]]
    centre = np.array([(fp[:, 0].min() + fp[:, 0].max()) / 2, g, (fp[:, 1].min() + fp[:, 1].max()) / 2])
    local = M - centre                                      # model: footprint centre on the origin, ground at 0
    pxz = P[:, [0, 2]]
    start_xz = (np.percentile(pxz, 10, axis=0) + np.percentile(pxz, 90, axis=0)) / 2

    def place(yaw_deg, s, txz):
        R = _ry_mat(yaw_deg)
        y0 = float(ground_heights(scene, [txz[0]], [txz[1]])[0])
        return local @ R.T * s + np.array([txz[0], y0, txz[1]]), N @ R.T, y0

    def score(W, Nw):
        vis = cam.visible(W, Nw)
        if vis.sum() < 100:
            return np.inf, vis
        d1, _ = tree.query(W[vis])
        d2, _ = cKDTree(W[vis]).query(P[:: max(1, len(P) // 8000)])
        t1 = np.sort(d1)[: int(len(d1) * 0.7)].mean()
        t2 = np.sort(d2)[: int(len(d2) * 0.7)].mean()
        return float(t1 + t2) / 2, vis

    best = None
    yaws = [float(yaw)] if yaw is not None else [i * 360.0 / starts for i in range(starts)]
    for y0 in yaws:
        cur_yaw, s, txz = y0, s0, start_xz.copy()
        for _ in range(iters):
            W, Nw, _gy = place(cur_yaw, s, txz)
            vis = cam.visible(W, Nw)
            if vis.sum() < 100:
                break
            src = W[vis]
            d, j = tree.query(src)
            keep = d <= np.percentile(d, 70)
            a, b = src[keep][:, [0, 2]], P[j[keep]][:, [0, 2]]
            ca, cb = a.mean(0), b.mean(0)
            A, B = a - ca, b - cb
            if yaw is None:
                # the turn (about Y) that best maps A onto B, in the ground plane
                num = float((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum())
                den = float((A * B).sum())
                d_yaw = -math.degrees(math.atan2(num, den))
                d_yaw = max(-15.0, min(15.0, d_yaw))
                cur_yaw += d_yaw
                Rd = _ry_mat(d_yaw)[[0, 2]][:, [0, 2]]
                A = A @ Rd.T
            # size stays within a third of what the heights said
            k = float((A * B).sum() / max((A * A).sum(), 1e-9))
            s = float(np.clip(s * (1 + 0.5 * (k - 1)), s0 / 1.33, s0 * 1.33)) if not height_m else s
            txz = txz + (cb - ca) * 0.8
        W, Nw, gy = place(cur_yaw, s, txz)
        err, vis = score(W, Nw)
        if best is None or err < best["err"]:
            cov_d, _ = cKDTree(W).query(P[:: max(1, len(P) // 8000)])
            best = {"err": err, "yaw": cur_yaw % 360.0, "scale": s, "txz": txz, "gy": gy,
                    "coverage": float((cov_d < max(1.0, 0.04 * h_pic)).mean()), "W": W}
    if best is None or not np.isfinite(best["err"]):
        raise ValueError("the model couldn't be placed: the reference camera sees none of it")
    R = _ry_mat(best["yaw"])
    s = best["scale"]
    target = np.array([best["txz"][0], best["gy"], best["txz"][1]])
    position = target - (R @ centre) * s                   # so that T + R·s·centre lands on target
    ext = np.ptp(M, axis=0) * s
    return {"position": [round(float(v), 4) for v in position], "yaw": round(float(best["yaw"]), 2),
            "scale": round(float(s), 5), "ground_level": round(float(g), 5), "error_m": round(best["err"], 3),
            "coverage": round(best["coverage"], 3), "size_m": [round(float(v), 2) for v in ext],
            "scale_from": scale_from, "footprint_world": best["W"][:, [0, 2]]}


# ── a model built in the camera's own space ─────────────────────────────────

# How a camera model's axes map onto the world camera's (which looks down -Z,
# Y up): OpenCV-style engines (SHARP) have Y down and Z forward.
FRAMES = {"camera_cv": np.diag([1.0, -1.0, -1.0]), "camera_gl": np.eye(3)}


def fit_camera_model(scene, mesh, frame="camera_cv", scale=1.0):
    """Place a model made in the reference camera's space: the camera's pose
    carries it into the world. Returns the item transform and the model's
    ground level in world metres (read near the camera, where its ground is)."""
    cam = _item(scene, "refcam")
    if cam is None:
        raise ValueError("the world has no reference camera ('refcam')")
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {sorted(FRAMES)}")
    R = euler_matrix(cam.get("rotation", [0, 0, 0])) @ FRAMES[frame]
    eye = np.asarray(cam["position"], np.float64)
    M, N = surface_samples(mesh, 50000)
    W = (M * scale) @ R.T + eye
    Nw = N @ R.T
    up = Nw[:, 1] > 0.85
    near = np.hypot(W[:, 0] - eye[0], W[:, 2] - eye[2]) < 25.0
    pick = up & near
    ys = W[pick, 1] if pick.sum() > 50 else W[:, 1]
    level = float(np.percentile(ys, 20))
    # the model's rotation as Euler XYZ for the item: camera pitch, flipped axes
    # become a scale sign (Y and Z flip = a turn of 180° about X).
    rx = float(cam.get("rotation", [0, 0, 0])[0])
    if frame == "camera_cv":
        rotation, sc = [rx + 180.0, 0.0, 0.0], [scale, scale, scale]
    else:
        rotation, sc = [rx, 0.0, 0.0], [scale, scale, scale]
    ext = np.ptp(W, axis=0)
    return {"position": [round(float(v), 4) for v in eye], "rotation": rotation, "scale": sc,
            "ground_world": round(level, 4), "size_m": [round(float(v), 2) for v in ext],
            "footprint_world": W[:, [0, 2]]}


# ── the terrain meets the model ─────────────────────────────────────────────

def flatten_under(scene, footprint_xz, level, out_path, margin_m=8.0):
    """Flatten the walkable terrain to `level` (world metres) under a
    footprint (world x, z points), blending back to the land over `margin_m`.
    The terrain is lifted or lowered as a whole first when `level` lies
    outside its range. Writes the heightmap to `out_path`; returns the ids
    changed."""
    from scipy import ndimage
    from .terrain import decode_rg16, encode_rg16
    t = next((i for i in scene["items"] if i.get("kind") == "terrain"
              and (i.get("terrain") or {}).get("walkable", True) is not False), None)
    if t is None:
        return []
    tt = t["terrain"]
    size = float(tt.get("size", [240, 240])[0])
    height = float(tt.get("height", 0) or 0)
    hm_path = (tt.get("heightmap") or {}).get("path")
    if hm_path and os.path.isfile(hm_path):
        with Image.open(hm_path) as im:
            h = decode_rg16(im) if tt.get("encoding") == "rg16" else np.asarray(im.convert("L"), np.float32) / 255
    else:
        h = np.zeros((257, 257), np.float32)
    n = h.shape[0]
    base = float(t.get("position", [0, 0, 0])[1])
    if height <= 0:
        height = 1.0
        tt["height"] = height
        h = np.zeros_like(h)
    lvl = (level - base) / height
    if not 0.0 <= lvl <= 1.0:
        # move the whole terrain so the level sits inside its range, at the
        # height it already has where the model stands
        fx, fz = np.median(footprint_xz, axis=0)
        from .terrain import height_at
        here = height_at(h, fx / size + 0.5, fz / size + 0.5)
        base = level - here * height
        t["position"][1] = round(base, 4)
        lvl = here
    uv = np.clip((np.asarray(footprint_xz) / size + 0.5) * (n - 1), 0, n - 1).astype(int)
    mask = np.zeros((n, n), bool)
    mask[uv[:, 1], uv[:, 0]] = True
    cell = size / (n - 1)
    mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=max(1, int(round(1.5 / cell)))))
    dist = ndimage.distance_transform_edt(~mask) * cell
    tb = np.clip(dist / max(margin_m, 1e-6), 0, 1)
    w = tb * tb * (3 - 2 * tb)                              # 0 under the model, 1 past the margin
    out = (lvl * (1 - w) + h * w).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    encode_rg16(np.clip(out, 0, 1)).save(out_path)
    tt["heightmap"] = {"path": os.path.abspath(out_path), "name": os.path.basename(out_path), "external": True}
    tt["encoding"] = "rg16"
    return [t["id"]]


def footprint_clear(footprint_xz, pad=2.0):
    """A circle of scatter-free ground over a footprint."""
    lo, hi = footprint_xz.min(0), footprint_xz.max(0)
    c = (lo + hi) / 2
    return {"center": [round(float(c[0]), 3), round(float(c[1]), 3)],
            "radius": round(float(np.linalg.norm(hi - lo) / 2 + pad), 3)}


# ── the picture on a model that came without colours ─────────────────────────

def camera_texture(mesh, to_world, scene, picture, out_path, max_faces=300000):
    """Texture an untextured model with the reference picture, projected from
    the reference camera: faces the camera sees take the picture's pixels where
    they fall in it; the rest take the same place in a blurred copy (the scene's
    own colours, without a window smeared across a roof). `to_world` is the 4x4
    that places the model. Writes a GLB in the model's own coordinates."""
    import trimesh
    from PIL import ImageFilter
    from .assets import decimate
    v, f = decimate(mesh.vertices, mesh.faces, max_faces)
    vw = v @ to_world[:3, :3].T + to_world[:3, 3]
    cam = RefCamera(scene)
    u, vv, z = cam.project(vw)
    tri = trimesh.Trimesh(vw, f, process=False)
    centres = tri.triangles_center
    seen = cam.visible(centres, tri.face_normals)
    # a finer buffer than the fit's, so a wall behind a wall stays unseen
    cu, cv, cz = cam.project(centres)
    ok = seen & (cz > 0.2)
    idx = np.nonzero(ok)[0]
    if len(idx):
        gw, gh = 480, max(1, int(480 / cam.aspect))
        cell = (np.clip(cv[idx], 0, 0.999) * gh).astype(int) * gw + (np.clip(cu[idx], 0, 0.999) * gw).astype(int)
        zmin = np.full(gw * gh, np.inf)
        np.minimum.at(zmin, cell, cz[idx])
        seen = np.zeros(len(f), bool)
        seen[idx[cz[idx] <= zmin[cell] * 1.02 + 0.05]] = True
    with Image.open(picture) as im:
        pic = im.convert("RGB")
    W, H = pic.size
    blurred = pic.filter(ImageFilter.GaussianBlur(max(4, int(0.03 * max(W, H)))))
    atlas = Image.new("RGB", (W * 2, H))
    atlas.paste(pic, (0, 0))
    atlas.paste(blurred, (W, 0))
    corners = f.reshape(-1)
    half = np.repeat(np.where(seen, 0.0, 0.5), 3)
    uu = np.clip(u[corners], 0.0005, 0.9995) * 0.5 + half
    uv = np.stack([uu, 1.0 - np.clip(vv[corners], 0.0005, 0.9995)], axis=-1)
    out = trimesh.Trimesh(v[corners], np.arange(len(corners)).reshape(-1, 3), process=False)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas, roughnessFactor=0.85, metallicFactor=0.0)
    out.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.export(out_path)
    return {"faces": int(len(f)), "seen_share": round(float(seen.mean()), 3)}


def has_colour(mesh):
    """True when a mesh brings its own colours (a texture or vertex colours)."""
    kind = getattr(mesh.visual, "kind", None)
    if kind == "texture":
        return getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None) is not None \
            or getattr(getattr(mesh.visual, "material", None), "image", None) is not None
    if kind == "vertex":
        c = np.asarray(mesh.visual.vertex_colors)
        return c.size > 0 and float(np.ptp(c[:, :3])) > 0
    return False


def item_matrix(position, rotation, scale):
    m = np.eye(4)
    m[:3, :3] = euler_matrix(rotation) @ np.diag(scale)
    m[:3, 3] = position
    return m
