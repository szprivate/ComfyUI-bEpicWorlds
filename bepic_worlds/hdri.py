"""HDR environments from the web: found on Poly Haven, fitted to the world.

A real photographed HDRI lights a world better than a generated sky can:
its sun is where the light really comes from, its sky has the range a
photograph of the sun needs, and every reflection in the world shows it.
So a world's environment is looked for first among Poly Haven's HDRIs (all
CC0, free to use for anything) and only made up when none fits.

- `search` scores Poly Haven's HDRIs against words and the attributes Poly
  Haven gives each one (time of day, weather, indoor/outdoor, urban/nature,
  how open the sky is) and returns the best, with a thumbnail each to look at.
- `fetch` downloads one (only ever from Poly Haven's own hosts).
- `install` reads it, finds its sun, turns it so that sun stands where the
  world's does, evens its brightness, and writes it where the world keeps it.

The panorama's layout is three.js's: the middle column's left quarter (u =
0.25) looks down -Z, the reference camera's direction; u = 0.5 is +X. So an
azimuth (0 = -Z, +90 = +X) is at u = 0.25 + azimuth / 360.
"""

import json
import math
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request

import numpy as np

API = "https://api.polyhaven.com"
HOSTS = ("api.polyhaven.com", "dl.polyhaven.org", "cdn.polyhaven.com")
USER_AGENT = "bEpicWorlds/1.0 (ComfyUI-bEpicWorlds; world environments)"
RESOLUTIONS = ("1k", "2k", "4k", "8k")

_LIST_CACHE = {"at": 0.0, "data": None}


def _get(url, timeout=60, binary=False):
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in HOSTS:
        raise ValueError(f"not a Poly Haven address: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data if binary else json.loads(data.decode("utf-8"))


def catalogue():
    """Every Poly Haven HDRI with its metadata, kept for an hour."""
    if _LIST_CACHE["data"] is None or time.time() - _LIST_CACHE["at"] > 3600:
        _LIST_CACHE["data"] = _get(f"{API}/assets?t=hdris")
        _LIST_CACHE["at"] = time.time()
    return _LIST_CACHE["data"]


# Words people use, and the attribute values Poly Haven uses for them.
_WORDS = {
    "time_of_day": {"sunrise": ("sunrise", "dawn", "morning"), "sunset": ("sunset", "dusk", "golden", "evening"),
                    "midday": ("midday", "noon", "day", "afternoon", "daylight"),
                    "night": ("night", "moon", "dark"), "morning-afternoon": ("morning", "afternoon")},
    "weather": {"clear": ("clear", "sunny", "blue"), "partly_cloudy": ("partly", "scattered", "cumulus", "cloudy"),
                "overcast": ("overcast", "grey", "gray", "dull", "cloudy", "diffuse", "rain", "rainy"),
                "foggy": ("fog", "foggy", "mist", "misty", "haze", "hazy")},
    "environment": {"outdoor": ("outdoor", "outside", "street", "town", "city", "forest", "field"),
                    "indoor": ("indoor", "inside", "interior", "room", "hall", "garage", "studio")},
}


def _flat(asset):
    a = asset.get("attributes") or {}
    return " ".join([asset.get("name", ""), asset.get("description", ""), asset.get("category", "")]
                    + list(asset.get("categories") or []) + list(asset.get("tags") or [])
                    + [f"{k} {v}" for k, v in a.items()]).lower().replace("_", " ")


def search(query="", time_of_day=None, weather=None, environment=None, urban=None, open_sky=False, limit=8):
    """The HDRIs that fit best: [{id, name, score, attributes, categories, tags,
    thumbnail, page}]. `query` is free words ("overcast old town street");
    the other arguments, when given, are attributes an HDRI must not
    contradict (Poly Haven's own values: time_of_day sunrise | sunset |
    midday | night …, weather clear | partly_cloudy | overcast | foggy,
    environment outdoor | indoor; `urban` true/false). `open_sky` prefers
    HDRIs that are mostly sky over a low, distant horizon (Poly Haven's
    "skies" / "pure skies") — for a world that brings its own buildings and
    wants only sky and distance from the panorama, not someone else's
    courtyard walls towering over it."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(query).lower()) if len(w) > 2]
    wanted = {"time_of_day": time_of_day, "weather": weather, "environment": environment}
    for key, table in _WORDS.items():
        if wanted.get(key):
            continue
        for value, cues in table.items():
            if any(w in cues for w in words):
                wanted[key] = value
                break
    out = []
    for aid, asset in catalogue().items():
        a = asset.get("attributes") or {}
        cats = [c.lower() for c in asset.get("categories") or []]
        score = 0.0
        bad = False
        for key, value in wanted.items():
            if not value:
                continue
            have = str(a.get(key) or "").lower()
            if have and have == str(value).lower():
                score += 4.0
            elif have and key != "time_of_day":
                bad = True
            elif have:
                score -= 1.5
        if bad:
            continue
        if urban is not None:
            is_urban = "urban" in cats or "urban" in _flat(asset)
            if bool(urban) != is_urban:
                score -= 3.0
            else:
                score += 1.5
        hay = _flat(asset)
        score += sum(1.0 for w in words if w in hay)
        if a.get("sky_view") == "open" and wanted.get("environment") != "indoor":
            score += 0.3
        if open_sky:
            score += 4.0 if "pure skies" in cats else 3.0 if "skies" in cats else 0.0
            if a.get("sky_view") == "obstructed":
                score -= 3.0
        out.append((score, aid, asset))
    out.sort(key=lambda t: (-t[0], t[1]))
    hits = []
    for score, aid, asset in out[:max(1, int(limit))]:
        hits.append({"id": aid, "name": asset.get("name"), "score": round(score, 2),
                     "attributes": asset.get("attributes") or {}, "categories": asset.get("categories") or [],
                     "tags": (asset.get("tags") or [])[:12],
                     "thumbnail": f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=512&height=256",
                     "page": f"https://polyhaven.com/a/{aid}"})
    return hits


def fetch(asset_id, resolution="4k", folder=None):
    """Download one HDRI's .hdr (from Poly Haven only). Returns its path."""
    aid = re.sub(r"[^a-z0-9_]+", "", str(asset_id).lower())
    if not aid:
        raise ValueError("an HDRI id is letters, digits and _")
    res = resolution if resolution in RESOLUTIONS else "4k"
    files = _get(f"{API}/files/{aid}")
    entry = ((files.get("hdri") or {}).get(res) or {}).get("hdr")
    if not entry:
        have = sorted((files.get("hdri") or {}).keys())
        raise ValueError(f"'{aid}' has no {res} .hdr (it has: {', '.join(have)})")
    folder = folder or os.path.join(tempfile.gettempdir(), "bepic_hdri")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{aid}_{res}.hdr")
    if not (os.path.isfile(path) and os.path.getsize(path) == entry.get("size")):
        data = _get(entry["url"], timeout=600, binary=True)
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
    return path


# ── reading one ──────────────────────────────────────────────────────────────

def read(path):
    """An HDR panorama as float32 RGB (H x W x 3), linear."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"can't read {os.path.basename(path)} as an HDR image (.hdr, or .exr where OpenCV reads it)")
    return img[..., ::-1].astype(np.float32)


def write(path, rgb):
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, np.ascontiguousarray(rgb[..., ::-1].astype(np.float32))):
        raise ValueError(f"couldn't write {path}")


def _lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _hex(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return "#%02x%02x%02x" % tuple(int(v * 255 + 0.5) for v in c[:3])


def _small(rgb, w=512):
    import cv2
    h = max(1, w // 2)
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def find_sun(rgb):
    """Where the sun is in a panorama, if it has one: {azimuth, elevation,
    color, strength (how many times brighter than the sky's median)}, or None
    for an overcast or indoor one."""
    s = _small(rgb)
    lum = _lum(s)
    h, w = lum.shape
    sky = lum[: h // 2]
    med = float(np.median(sky)) + 1e-6
    peak = float(sky.max())
    # A sun outshines its sky by thousands (66 000x on a clear day); a lit
    # window or a lamp in an overcast street by tens.
    if peak < 300.0 * med:
        return None
    thr = max(peak * 0.25, 300.0 * med)
    ys, xs = np.nonzero(sky >= thr)
    if not len(xs):
        return None
    # the brightest blob, not every bright cloud: around the peak
    py, px = np.unravel_index(int(np.argmax(sky)), sky.shape)
    dx = np.minimum(np.abs(xs - px), w - np.abs(xs - px))
    near = (dx < w * 0.05) & (np.abs(ys - py) < h * 0.08)
    xs, ys = xs[near], ys[near]
    wts = sky[ys, xs]
    # a circular mean across the seam
    ang = (xs + 0.5) / w * 2 * math.pi
    u = (math.atan2(float((np.sin(ang) * wts).sum()), float((np.cos(ang) * wts).sum())) / (2 * math.pi)) % 1.0
    v = float(((ys + 0.5) * wts).sum() / wts.sum()) / h
    az = ((u - 0.25) * 360.0 + 180.0) % 360.0 - 180.0
    el = (0.5 - v) * 180.0
    col = s[: h // 2][ys, xs].mean(0)
    return {"azimuth": round(az, 1), "elevation": round(max(1.0, el), 1),
            "color": _hex(col / max(float(col.max()), 1e-6)), "strength": round(peak / med, 1), "u": u}


def colours(rgb):
    """The panorama's colours for the world's fill light, fog and the sky
    colours a gradient falls back to — as they'd look on screen."""
    s = _small(rgb, 256)
    h = s.shape[0]
    top = s[: h // 6].reshape(-1, 3)
    band = s[int(h * 0.4): int(h * 0.5)].reshape(-1, 3)
    # the sky seen along the horizon, not the walls of a street in front of it
    horizon = band[_lum(band) >= np.percentile(_lum(band), 70)]
    below = s[int(h * 0.5): int(h * 0.56)].reshape(-1, 3)
    ground = s[int(h * 0.6):].reshape(-1, 3)
    ref = float(np.percentile(_lum(s[: h // 2]), 75)) + 1e-6

    def disp(px):
        c = np.median(px, axis=0) / ref
        return _hex((c / (1 + c)) * 1.6)                   # a gentle tone curve
    return {"top": disp(top), "horizon": disp(horizon), "bottom": disp(below), "ground": disp(ground)}


def install(src, dst, match_azimuth=None, target_median=1.0, clamp=400.0):
    """Put an HDRI in a world: turned so its sun stands at `match_azimuth`
    (degrees; None keeps it where it is), scaled so the sky's median brightness
    is `target_median` (so every HDRI lights at about the same level before
    the world's exposure is matched), its hottest pixels held at `clamp` times
    that (an unclipped sun makes speckles in filtered reflections). Written as
    Radiance .hdr. Returns {sun (after turning), colours, turned_by, size}."""
    rgb = read(src)
    h, w = rgb.shape[:2]
    sun = find_sun(rgb)
    turned = 0.0
    if sun is not None and match_azimuth is not None:
        turned = ((float(match_azimuth) - sun["azimuth"] + 180.0) % 360.0) - 180.0
        shift = int(round(turned / 360.0 * w))
        rgb = np.roll(rgb, shift, axis=1)
        sun = find_sun(rgb)
    sky = _lum(_small(rgb))[: 128]
    k = float(target_median) / (float(np.median(sky)) + 1e-6)
    rgb = np.minimum(rgb * k, float(target_median) * float(clamp))
    write(dst, rgb)
    out = {"sun": sun, "colours": colours(rgb), "turned_by": round(turned, 1), "size": [int(w), int(h)]}
    if sun:
        sun.pop("u", None)
    return out
