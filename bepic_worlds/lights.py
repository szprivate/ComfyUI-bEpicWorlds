"""Lamps in the picture, found and placed in the world.

A lamp is a bright, nearly colourless blob — a strip light, a panel, a
bulb — among ordinary surfaces. Each one found becomes a light in the world,
where the depth map says it hangs, with a fixture of its shape and a light of
its colour. Only blobs above the horizon count: below it, a bright blob is
usually the lamp's reflection in the floor.
"""

import math

import numpy as np
from PIL import Image

from .reference import load_rgb, hex_of

WORK = 320            # pictures are searched at this width


def _label(mask):
    """Connected components (4-neighbour) of a boolean mask: (labels, count)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            labels[y, x] = n
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        stack.append((ny, nx))
    return labels, n


def find_lamps(src, horizon, max_lamps=10):
    """[{u, v, w, h, angle, color, area}] for lamp-like blobs above the horizon,
    brightest-and-biggest first. u, v are the centre in 0..1 of the picture;
    angle is the long axis in the image, radians."""
    rgb = load_rgb(src)
    H, W = rgb.shape[:2]
    s = WORK / W
    small = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (WORK, max(1, int(H * s))), Image.BILINEAR), dtype=np.float32) / 255.0
    h, w = small.shape[:2]
    lum = small.mean(-1)
    sat = small.max(-1) - small.min(-1)
    rows = max(1, int(h * min(0.95, horizon + 0.02)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:rows] = (lum[:rows] > max(0.9, float(np.percentile(lum, 99.3)))) & (sat[:rows] < 0.2)
    labels, n = _label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(xs) < 3 or len(xs) > 0.02 * h * w:
            continue
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.vstack([xs - cx, ys - cy])) if len(xs) > 2 else np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, 1]
        length = 4 * math.sqrt(max(vals[1], 0.25))
        width = 4 * math.sqrt(max(vals[0], 0.25))
        col = small[ys, xs].mean(0)
        out.append({"u": float(cx / w), "v": float(cy / h), "w": float(length / w), "h": float(width / h),
                    "angle": float(math.atan2(major[1], major[0])), "area": int(len(xs)),
                    "color": hex_of(col / max(1e-6, float(col.max())))})
    out.sort(key=lambda b: -b["area"])
    return out[:max_lamps]


def place(lamps, depth_to_z, dmap, fov, aspect, pitch, eye, ceiling_y=None):
    """Lamps as world positions under the camera at `eye`, tilted by `pitch`.
    `depth_to_z(d)` turns a depth-map value into metres along the camera axis.

    With `ceiling_y` (an interior), a lamp is where the ray through it meets
    the ceiling: a strip light is thin and bright, and the depth map is least
    sure exactly there, while the ceiling's height is known. Lamps whose ray
    doesn't rise to it are left out.
    Returns [{position, length, width, yaw, color}]."""
    tan_v = math.tan(math.radians(fov) / 2)
    tan_h = tan_v * aspect
    p = math.radians(pitch)
    H, W = dmap.shape[:2]
    placed = []
    for lamp in lamps:
        dx = (lamp["u"] - 0.5) * 2 * tan_h            # ray through the lamp, camera space, per unit depth
        dy = (0.5 - lamp["v"]) * 2 * tan_v
        if ceiling_y is not None:
            rise = dy * math.cos(p) + math.sin(p)       # world-up per unit of camera depth
            if rise < 0.01:
                continue
            z = (ceiling_y - eye[1]) / rise
            if not 0.3 < z < 400:
                continue
        else:
            x0, y0 = int(lamp["u"] * (W - 1)), int(lamp["v"] * (H - 1))
            win = dmap[max(0, y0 - 3): y0 + 4, max(0, x0 - 3): x0 + 4]
            d = float(win.max()) if win.size else float(dmap[y0, x0])
            z = depth_to_z(d)
        cx, cy = dx * z, dy * z
        # Camera -> world: pitch about X, then the camera's position.
        wy = cy * math.cos(p) + (-z) * -math.sin(p)
        wz = cy * math.sin(p) - z * math.cos(p)
        pos = [eye[0] + cx, eye[1] + wy - 0.05, eye[2] + wz]
        # Size from the blob's extent at that distance; yaw from its long axis,
        # assuming it hangs level (strip lights mostly do).
        length = max(0.2, min(4.0, lamp["w"] * 2 * tan_h * z))
        width = max(0.05, min(1.2, lamp["h"] * 2 * tan_v * z))
        yaw = -math.degrees(lamp["angle"]) if length > width * 2 else 0.0
        placed.append({"position": [round(v, 3) for v in pos], "length": round(length, 3),
                       "width": round(width, 3), "yaw": round(yaw, 1), "color": lamp["color"]})
    return placed
