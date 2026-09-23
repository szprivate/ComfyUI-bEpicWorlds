"""Reading a reference image: what a world built from it has to agree with.

Everything here is measured, not generated: the colours of the sky, the
horizon and the ground; where the horizon sits, which says how the camera
that took the picture was tilted; where the sun is, when it is in frame; and a
small palette that tints everything else. A world built to these numbers
looks, from the reference camera, like it belongs to the picture.

Only numpy and Pillow, so an agent can call it without ComfyUI loaded.
"""

import colorsys
import math

import numpy as np
from PIL import Image

STAT_SIZE = 256          # images are measured at this size — plenty for colours


def load_rgb(src):
    """An RGB image as float32 HxWx3 in [0, 1], from a path, a PIL image, an
    array (HxWx3, uint8 or float) or a ComfyUI IMAGE tensor (BxHxWxC)."""
    if src is None:
        return None
    if isinstance(src, str):
        with Image.open(src) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGB"), dtype=np.float32) / 255.0
    arr = src
    if hasattr(arr, "detach"):                        # a torch tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_gray(src):
    """A single-channel float image in [0, 1] (depth maps, heightmaps).

    Keeps whatever precision the source has: a 16-bit PNG stays 16-bit (a
    depth map needs it — inverse depth puts everything past a few metres in
    the bottom few percent of the range), and a float tensor stays float."""
    if src is None:
        return None
    im = None
    if isinstance(src, str):
        im = Image.open(src)
    elif isinstance(src, Image.Image):
        im = src
    if im is not None and im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        hi = 65535.0 if im.mode.startswith("I;16") or a.max() > 255 else 255.0
        return np.clip(a / hi, 0.0, 1.0)
    rgb = load_rgb(im if im is not None else src)
    return rgb.mean(axis=-1)


def to_pil(arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def _small(rgb):
    h, w = rgb.shape[:2]
    s = STAT_SIZE / max(h, w)
    if s >= 1:
        return rgb
    im = to_pil(rgb).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def hex_of(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0, 1)
    return "#%02x%02x%02x" % tuple(int(v * 255 + 0.5) for v in c[:3])


def rgb_of(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _smooth(v, k=5):
    k = max(1, int(k))
    pad = np.pad(v, (k, k), mode="edge")
    ker = np.ones(2 * k + 1, dtype=np.float32) / (2 * k + 1)
    return np.convolve(pad, ker, mode="same")[k:-k]


def find_horizon(rgb):
    """The row where sky turns into land, as a fraction of the height, and how
    sure that is (0..1).

    A horizon is a horizontal edge that runs across the frame. So each pixel
    counts when the colour changes more going down than going across, and the
    row where the most of the width does that wins — gaps are fine, which is
    what lets a pier, a person or a lamp post stand in front of it. Rows in
    the outer 12 % / 10 % are not considered: there, a horizon can't be told
    from the frame edge.
    """
    h, w = rgb.shape[:2]
    s = max(1, h // 80)
    dy = np.zeros((h, w), dtype=np.float32)
    dy[s:-s] = np.linalg.norm(rgb[2 * s:] - rgb[:-2 * s], axis=-1)
    dx = np.zeros((h, w), dtype=np.float32)
    dx[:, s:-s] = np.linalg.norm(rgb[:, 2 * s:] - rgb[:, :-2 * s], axis=-1)
    across = _smooth(((dy > 0.06) & (dy > dx * 1.5)).mean(axis=1).astype(np.float32),
                     max(1, h // 100))
    lo, hi = int(h * 0.12), int(h * 0.9)
    if hi <= lo:
        return 0.5, 0.0
    i = int(np.argmax(across[lo:hi]))
    return (lo + i) / h, float(np.clip(across[lo + i] * 1.6, 0, 1))


def palette(rgb, k=6, iters=10, seed=0):
    """The image's main colours, most common first: [{hex, share, hue, sat, lum}]."""
    px = rgb.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if len(px) > 6000:
        px = px[rng.choice(len(px), 6000, replace=False)]
    centres = px[rng.choice(len(px), k, replace=False)].copy()
    for _ in range(iters):
        d = ((px[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = px[lab == j]
            if len(m):
                centres[j] = m.mean(0)
    counts = np.bincount(lab, minlength=k).astype(np.float32)
    order = np.argsort(-counts)
    out = []
    for j in order:
        if counts[j] == 0:
            continue
        h, l, s = colorsys.rgb_to_hls(*[float(v) for v in centres[j]])
        out.append({"hex": hex_of(centres[j]), "share": round(float(counts[j] / counts.sum()), 3),
                    "hue": round(h * 360, 1), "sat": round(s, 3), "lum": round(l, 3)})
    return out


def _band(rgb, a, b):
    h = rgb.shape[0]
    a, b = max(0, int(a * h)), min(h, max(int(a * h) + 1, int(b * h)))
    return rgb[a:b].reshape(-1, 3).mean(0)


def analyze(src, fov=50.0):
    """Everything a world needs to agree with the picture.

    `fov` is the reference camera's vertical field of view in degrees — it is
    not in the pixels, so it is a guess unless told (50 is a normal lens).
    Angles follow the world's convention: the reference camera looks down -Z;
    a sun azimuth of 0 is straight ahead, +90 to the right; elevation is up
    from the horizon.
    """
    full = load_rgb(src)
    rgb = _small(full)
    h, w = rgb.shape[:2]
    lum = _lum(rgb)
    vfov = math.radians(fov)
    hfov = 2 * math.atan(math.tan(vfov / 2) * w / h)

    horizon, confident = find_horizon(rgb)
    # A horizon above the middle of the frame means the camera looks down.
    pitch = -math.degrees(math.atan((0.5 - horizon) * 2 * math.tan(vfov / 2)))

    sky_top = _band(rgb, 0.0, min(0.1, horizon * 0.5))
    sky_horizon = _band(rgb, max(0.0, horizon - 0.08), horizon)
    ground_horizon = _band(rgb, horizon, min(1.0, horizon + 0.08))
    ground = _band(rgb, 0.72, 1.0)

    # The sun, when it is in the picture: the brightest few pixels above the
    # horizon, if they are properly bright.
    sun = None
    sky_rows = max(1, int(horizon * h))
    sky_lum = lum[:sky_rows]
    if sky_lum.size:
        # Near white, small, and well above the rest of the sky — a bright
        # cloud is none of the last two.
        thresh = max(0.95, float(np.percentile(sky_lum, 99.8)))
        ys, xs = np.nonzero(sky_lum >= thresh)
        compact = len(xs) <= 0.02 * sky_lum.size and (
            len(xs) < 3 or (np.ptp(xs) < w * 0.15 and np.ptp(ys) < h * 0.15))
        if len(xs) >= 3 and compact and thresh - float(np.median(sky_lum)) > 0.25:
            cx, cy = xs.mean() / w, ys.mean() / h
            az = math.degrees(math.atan((cx - 0.5) * 2 * math.tan(hfov / 2)))
            el = pitch + math.degrees(math.atan((0.5 - cy) * 2 * math.tan(vfov / 2)))
            col = rgb[:sky_rows][sky_lum >= thresh].mean(0)
            sun = {"azimuth": round(az, 1), "elevation": round(max(2.0, el), 1),
                   "color": hex_of(col / max(1e-6, col.max())), "in_frame": True}

    brightness = float(lum.mean())
    warmth = float((rgb[..., 0] - rgb[..., 2]).mean())
    if sun is None:
        # Out of frame: light from behind and to one side of the camera, as a
        # photographer would have it, higher for a brighter picture and lower
        # (and warmer) for a warm one.
        el = float(np.clip(15 + brightness * 60 - max(0.0, warmth) * 80, 5, 70))
        tint = np.array([1.0, 0.95 - max(0.0, warmth) * 0.6, 0.88 - max(0.0, warmth) * 1.0])
        sun = {"azimuth": 150.0, "elevation": round(el, 1),
               "color": hex_of(np.clip(tint, 0.3, 1.0)), "in_frame": False}

    # Distant things fade toward the sky: the flatter the colour near the
    # horizon, the thicker the air.
    sat_h = colorsys.rgb_to_hls(*[float(v) for v in ground_horizon])[2]
    fog_density = float(np.clip(0.012 - sat_h * 0.02, 0.0015, 0.012))

    pal = palette(rgb)
    greens = [p for p in pal if 60 <= p["hue"] <= 170 and p["sat"] > 0.12 and p["lum"] < 0.75]
    foliage = min(greens, key=lambda p: p["lum"])["hex"] if greens else hex_of(ground * 0.7)

    return {
        "size": [int(full.shape[1]), int(full.shape[0])],
        "fov": fov,
        "horizon": round(horizon, 4),
        "horizon_confidence": round(confident, 3),
        "pitch": round(pitch, 2),
        "sky_top": hex_of(sky_top),
        "sky_horizon": hex_of(sky_horizon),
        "ground_horizon": hex_of(ground_horizon),
        "ground": hex_of(ground),
        "sun": sun,
        "brightness": round(brightness, 3),
        "warmth": round(warmth, 3),
        "fog_density": round(fog_density, 5),
        "palette": pal,
        "foliage": foliage,
    }
