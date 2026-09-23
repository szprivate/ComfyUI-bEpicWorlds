"""Heightmaps: made up, or taken from an image, and stored so the browser can
read them at full precision.

Layout convention (shared with the viewer): a heightmap covers the terrain
square; column 0 is -X, row 0 is -Z — the far side, the direction the
reference camera looks. Values are 0..1, times the terrain's `height`.

A browser reads an image into 8-bit channels, so a 16-bit greyscale PNG would
come back terraced. Heights are therefore written as "rg16": the high byte in
red, the low byte in green, height = (R·256 + G) / 65535.
"""

import numpy as np
from PIL import Image

from .reference import load_gray


def _upsample(grid, n):
    im = Image.fromarray(grid.astype(np.float32), mode="F")
    return np.asarray(im.resize((n, n), Image.BICUBIC), dtype=np.float32)


def fbm(n=257, seed=0, octaves=7, roughness=0.5, base=3):
    """Fractal value noise, n x n, in 0..1: big shapes first, each octave
    finer and `roughness` times as strong as the last."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        g = base * (2 ** o) + 1
        if g > n:
            break
        out += _upsample(rng.random((g, g), dtype=np.float32), n) * amp
        total += amp
        amp *= roughness
    out /= max(total, 1e-6)
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / max(hi - lo, 1e-6)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def shape_for_view(h, spawn=(0.5, 0.75), flat_radius=0.07, rim=0.55, far_rise=0.35):
    """Make raw noise into a place to stand in.

    - flat where you start, so the reference view begins on level ground;
    - rising toward the far side, which is where the hills of a landscape
      photograph usually are;
    - rising at the rim, so the world ends in ridges rather than a cliff edge
      — the walk is bounded by the land itself.
    `spawn` is (u, v) in the map, u along +X and v along +Z.
    """
    n = h.shape[0]
    v, u = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1)
    su, sv = spawn
    d = np.hypot(u - su, v - sv)
    edge = np.maximum(np.abs(u - 0.5), np.abs(v - 0.5))

    shaped = h * 0.55
    shaped = shaped + _smoothstep(0.0, 0.7, np.clip(sv - v, 0, 1)) * far_rise * (0.6 + 0.8 * h)
    shaped = shaped + _smoothstep(0.34, 0.5, edge) * rim * (0.7 + 0.6 * h)
    near = _smoothstep(flat_radius, flat_radius * 2.5, d)
    base_level = float(shaped[d < flat_radius].mean()) if (d < flat_radius).any() else 0.0
    shaped = base_level * (1 - near) + shaped * near
    lo, hi = float(shaped.min()), float(shaped.max())
    return (shaped - lo) / max(hi - lo, 1e-6)


def from_image(src, n=257):
    """A heightmap someone made (or a depth model did): grey, resized, 0..1."""
    g = load_gray(src)
    im = Image.fromarray(g.astype(np.float32), mode="F").resize((n, n), Image.BICUBIC)
    h = np.asarray(im, dtype=np.float32)
    lo, hi = float(h.min()), float(h.max())
    return (h - lo) / max(hi - lo, 1e-6)


def encode_rg16(h):
    """0..1 heights as an RGB image: R the high byte, G the low byte."""
    q = np.clip(np.round(np.asarray(h) * 65535), 0, 65535).astype(np.uint32)
    rgb = np.zeros(q.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (q >> 8).astype(np.uint8)
    rgb[..., 1] = (q & 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def decode_rg16(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return (a[..., 0] * 256 + a[..., 1]) / 65535.0


def height_at(h, u, v):
    """Bilinear height (0..1) at map coordinates u, v in 0..1."""
    n = h.shape[0] - 1
    x, y = np.clip(u, 0, 1) * n, np.clip(v, 0, 1) * n
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, n), min(y0 + 1, n)
    fx, fy = x - x0, y - y0
    top = h[y0, x0] * (1 - fx) + h[y0, x1] * fx
    bot = h[y1, x0] * (1 - fx) + h[y1, x1] * fx
    return float(top * (1 - fy) + bot * fy)
