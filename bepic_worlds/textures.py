"""Ground textures cut out of the reference itself.

The ground in the lower part of a landscape photograph is the best texture
there is for that world: its colour, grain and light are the picture's own.
A crop of it is made to tile — blended with a copy of itself shifted by half,
so the seam falls where the copy's middle is — and a terrain repeats it.
"""

import numpy as np
from PIL import Image

from .reference import load_rgb, to_pil


def crop(rgb, box):
    """box = (left, top, right, bottom) as fractions of the image."""
    h, w = rgb.shape[:2]
    l, t, r, b = box
    x0, x1 = int(l * w), max(int(l * w) + 2, int(r * w))
    y0, y1 = int(t * h), max(int(t * h) + 2, int(b * h))
    return rgb[y0:y1, x0:x1]


def _square(rgb, size):
    """The largest centred square, at `size` pixels."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    im = to_pil(rgb[y0:y0 + s, x0:x0 + s]).resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def tileable(rgb, size=512):
    """`rgb` made to repeat without a visible seam.

    One axis at a time: blended with a copy shifted half a tile along that
    axis, taking the copy only near that axis's borders — where its own seam
    (in its middle) can't show. Doing both axes at once in a square mask left
    the copy's seam visible as a cross through the tile.
    """
    a = _square(rgb, size)
    t = np.linspace(0.0, 1.0, size, dtype=np.float32)
    edge = np.abs(t - 0.5) * 2                                  # 0 middle .. 1 border
    m = np.clip((edge - 0.5) / 0.5, 0, 1)
    m = m * m * (3 - 2 * m)
    a = a * (1 - m[None, :, None]) + np.roll(a, size // 2, axis=1) * m[None, :, None]
    a = a * (1 - m[:, None, None]) + np.roll(a, size // 2, axis=0) * m[:, None, None]
    return a


def flatten_light(rgb, strength=0.8):
    """Take out the large-scale light falloff of a photographed patch, so the
    tiles don't repeat a bright corner. Keeps the grain, evens the mean."""
    im = to_pil(rgb)
    small = im.resize((4, 4), Image.BILINEAR).resize(im.size, Image.BICUBIC)
    low = np.asarray(small, dtype=np.float32) / 255.0
    mean = rgb.reshape(-1, 3).mean(0)
    return np.clip(rgb - (low - mean) * strength, 0, 1)


def from_reference(src, box, size=512):
    """A tiling texture from part of a reference image."""
    return tileable(flatten_light(crop(load_rgb(src), box)), size)


def tinted(rgb, color, amount=0.6):
    """Pull a texture toward a colour while keeping its detail — a rock layer
    from a ground crop, say."""
    c = np.asarray(color, dtype=np.float32)
    lum = rgb.mean(-1, keepdims=True)
    detail = lum / max(float(lum.mean()), 1e-6)
    return np.clip(rgb * (1 - amount) + (c * detail) * amount, 0, 1)
