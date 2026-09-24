"""What a surface in the picture is made of, as PBR maps.

A photographed patch of floor carries three things at once: its colour, its
relief (the grain, the joints, the cracks) and the way it reflects light — a
wet-looking car-park floor shows streaks of the lamps in it, grass shows none.
These are separated here into what a PBR material wants:

    albedo     the colour with the reflections taken out, so the renderer's
               own lights and reflections aren't painted on twice;
    normal     relief from the patch's fine brightness detail (tangent space,
               +Y up, as three.js reads it);
    roughness  one number for the surface, from how much of it is highlight,
               and a map that varies it with the detail.

All tileable in, tileable out (every filter wraps around), numpy + Pillow only.
"""

import numpy as np

from .reference import to_pil


def _lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _blur(a, sigma):
    """Gaussian blur, periodic — done in the frequency domain, so it wraps at
    the edges exactly as a tiling texture does."""
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    g = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    if a.ndim == 2:
        return np.real(np.fft.ifft2(np.fft.fft2(a) * g)).astype(np.float32)
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(a[..., c]) * g)) for c in range(a.shape[2])],
                    axis=-1).astype(np.float32)


def gloss(rgb):
    """A cautious roughness for a surface seen on its own (0.45 … 0.95).

    Bright colourless patches standing above their surroundings may be
    reflections — or paint, or a lamp's own light falling on it, which look
    the same in one patch. So this only moves a surface from matte toward
    satin; a floor's real gloss comes from floor_gloss, which can tell.
    """
    h, w = rgb.shape[:2]
    if w > 256:
        rgb = np.asarray(to_pil(rgb).resize((256, max(8, int(h * 256 / w)))), dtype=np.float32) / 255.0
    lum = _lum(rgb)
    lift = lum - _blur(lum, rgb.shape[1] / 16.0)
    sat = rgb.max(-1) - rgb.min(-1)
    share = float(((lift > 0.1) & (sat < 0.15)).mean())
    peak = float(np.percentile(lift, 99))
    rough = 0.92 - 1.2 * share - 0.6 * max(0.0, peak - 0.1)
    return float(np.clip(rough, 0.45, 0.95)), {"highlight_share": round(share, 4), "peak": round(peak, 3)}


def floor_gloss(rgb, horizon, sources):
    """How mirror-like the ground is, as a roughness (0.12 … 0.95).

    A glossy floor shows each light source reflected straight below it: the
    ground under a lamp (or under the sun) is brighter than the ground a
    little to either side. `sources` are the lights' horizontal positions
    (u, 0..1 across the picture). Without any, the cautious gloss() is used.
    """
    h, w = rgb.shape[:2]
    top = int(h * min(0.97, horizon + 0.04))
    ground = _lum(rgb[top:])
    if not sources or ground.shape[0] < 4:
        return gloss(rgb[top:])[0], {"method": "patch"}
    cols = ground.mean(axis=0)
    found = []
    for u in sources:
        c = int(u * (w - 1))
        half = max(2, int(w * 0.025))
        under = cols[max(0, c - half): c + half + 1].mean()
        side = np.concatenate([cols[max(0, c - 5 * half): max(0, c - 3 * half)],
                               cols[c + 3 * half: c + 5 * half]])
        if side.size == 0:
            continue
        found.append(float(under / max(side.mean(), 1e-3)) - 1.0)
    # The mean of the two strongest: a single lamp above a white lane line
    # mustn't make a floor a mirror, and cars or pillars under the others
    # mustn't hide a real reflection.
    top = sorted(found, reverse=True)[:2]
    best = max(0.0, float(np.mean(top))) if top else 0.0
    rough = 0.9 - 1.6 * best
    return float(np.clip(rough, 0.18, 0.95)), {"method": "reflection", "brightening": round(best, 3)}


def delight(rgb, strength=0.85):
    """The colour without the reflections: highlights pulled back to the
    surface around them."""
    lum = _lum(rgb)
    base = _blur(rgb, 6)
    base_l = _lum(base)
    excess = np.clip((lum - base_l * 1.15) / np.maximum(base_l, 0.05), 0, 1)[..., None] * strength
    return np.clip(rgb * (1 - excess) + base * excess, 0, 1)


def normal_map(rgb, strength=1.0):
    """Tangent-space normals from fine brightness detail (bright = raised)."""
    lum = _lum(rgb)
    height = lum - _blur(lum, 8)                    # detail only: no slope from the lighting
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    gy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    n = np.stack([-gx * strength * 10, gy * strength * 10, np.ones_like(lum)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def roughness_map(rgb, base):
    """`base` roughness, rougher where the patch is dark and busy (joints,
    dirt, grain) and smoother where it is plain."""
    lum = _lum(rgb)
    detail = np.abs(lum - _blur(lum, 4))
    detail = detail / max(float(np.percentile(detail, 98)), 1e-4)
    dark = 1.0 - lum / max(float(lum.mean()) * 2, 1e-4)
    r = base + 0.18 * (np.clip(detail, 0, 1) - 0.4) + 0.08 * (dark - 0.5)
    return np.clip(r, 0.04, 1.0)


def pbr_set(rgb, source_region=None, roughness=None, delight_strength=0.85):
    """albedo, normal, roughness map and roughness value for one texture.
    `roughness` when it is known (floor_gloss); else it is measured on
    `source_region`, the untiled crop (tiling blends highlights away)."""
    if roughness is not None:
        rough, info = float(roughness), {"method": "given"}
    else:
        rough, info = gloss(source_region if source_region is not None else rgb)
    return {
        "albedo": delight(rgb, delight_strength),
        "normal": normal_map(rgb),
        "rough_map": roughness_map(rgb, rough),
        "roughness": round(rough, 3),
        "info": info,
    }


def save_gray(a, path):
    to_pil(np.stack([a] * 3, axis=-1)).convert("L").save(path)
