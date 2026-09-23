"""A world spec: what kind of world to make, in a form a person or an agent
can write.

A spec is either JSON or plain text. Plain text is read for keywords — a
biome ("forest", "desert", "alpine"…), a time of day ("sunset", "noon"…) and
words like "misty" or "dense". JSON can say all of that exactly, and more:

    {
      "biome": "forest",            # meadow | forest | alpine | desert | hills | snow
      "time": "late afternoon",     # or "sun_elevation": 25
      "sun_azimuth": 140,
      "fog": 0.006,                 # exponential fog density
      "terrain": {"height": 30, "roughness": 0.55, "seed": 3},
      "scatter": [{"type": "pine", "count": 1800, "scale": [0.8, 1.6]}],
      "snow": false,
      "notes": "anything else — kept with the world, for whoever edits it next"
    }

Whatever the spec leaves out is taken from the reference image (see
reference.analyze), and whatever the image can't say comes from the biome.
"""

import json
import re

BIOMES = {
    "meadow": {
        "terrain": {"height": 16, "roughness": 0.45},
        "scatter": [{"type": "grass", "count": 9000, "scale": [0.6, 1.3], "layer": 0},
                    {"type": "tree", "count": 220, "scale": [0.8, 1.5], "layer": 0},
                    {"type": "bush", "count": 450, "scale": [0.6, 1.2], "layer": 0}],
        "snow": False,
    },
    "forest": {
        "terrain": {"height": 26, "roughness": 0.5},
        "scatter": [{"type": "pine", "count": 2400, "scale": [0.8, 1.7], "layer": 0},
                    {"type": "bush", "count": 700, "scale": [0.6, 1.2], "layer": 0},
                    {"type": "grass", "count": 4000, "scale": [0.6, 1.1], "layer": 0}],
        "snow": False,
    },
    "hills": {
        "terrain": {"height": 32, "roughness": 0.5},
        "scatter": [{"type": "tree", "count": 500, "scale": [0.8, 1.5], "layer": 0},
                    {"type": "grass", "count": 6000, "scale": [0.6, 1.2], "layer": 0},
                    {"type": "rock", "count": 250, "scale": [0.4, 1.6], "layer": 1}],
        "snow": False,
    },
    "alpine": {
        "terrain": {"height": 70, "roughness": 0.6},
        "scatter": [{"type": "pine", "count": 1500, "scale": [0.8, 1.6], "layer": 0, "maxSlope": 30},
                    {"type": "rock", "count": 500, "scale": [0.5, 2.2], "layer": 1}],
        "snow": True,
    },
    "desert": {
        "terrain": {"height": 14, "roughness": 0.35},
        "scatter": [{"type": "rock", "count": 350, "scale": [0.4, 1.8], "layer": -1},
                    {"type": "bush", "count": 120, "scale": [0.4, 0.9], "layer": 0}],
        "snow": False,
    },
    # Indoors: a flat floor, a ceiling at the height the depth map measures,
    # columns on a grid — a car park, a hall, a warehouse. See builder.build_interior.
    "interior": {
        "terrain": {"height": 0.0, "roughness": 0.3},
        "scatter": [{"type": "column", "count": 400, "grid": [8.5, 8.5], "scale": [1.0, 1.0], "layer": -1}],
        "snow": False,
    },
    "snow": {
        "terrain": {"height": 30, "roughness": 0.45},
        "scatter": [{"type": "pine", "count": 900, "scale": [0.8, 1.6], "layer": 0},
                    {"type": "rock", "count": 200, "scale": [0.5, 1.8], "layer": 1}],
        "snow": True,
    },
}

BIOME_WORDS = {
    "forest": ["forest", "woods", "woodland", "pine", "jungle", "trees"],
    "meadow": ["meadow", "field", "grassland", "pasture", "prairie", "lawn"],
    "alpine": ["alpine", "mountain", "mountains", "peaks", "highland", "cliffs"],
    "desert": ["desert", "dunes", "sand", "canyon", "arid", "badlands"],
    "snow": ["snow", "winter", "tundra", "ice", "arctic", "frozen"],
    "hills": ["hills", "hilly", "valley", "countryside", "rolling"],
    "interior": ["interior", "indoor", "indoors", "inside", "garage", "parking", "hall", "room",
                 "warehouse", "corridor", "basement", "tunnel", "underground", "factory"],
}

TIMES = {"dawn": 5, "sunrise": 5, "morning": 25, "noon": 65, "midday": 65,
         "afternoon": 45, "late afternoon": 25, "golden hour": 10, "evening": 12,
         "sunset": 4, "dusk": 2, "twilight": 1, "night": -8}

SCATTER_TYPES = ("pine", "tree", "bush", "grass", "rock", "column", "model")


def parse(spec):
    """A spec dict from JSON text, plain text, a dict or nothing."""
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return dict(spec)
    text = str(spec).strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    return from_text(text)


def from_text(text):
    """Keywords out of a sentence: biome, time of day, fog, density, snow."""
    t = " " + text.lower() + " "
    out = {"notes": text.strip()}
    scores = {b: sum(len(re.findall(r"\b%s\b" % w, t)) for w in words)
              for b, words in BIOME_WORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best]:
        out["biome"] = best
    for phrase in sorted(TIMES, key=len, reverse=True):     # "late afternoon" before "afternoon"
        if phrase in t:
            out["time"] = phrase
            break
    if re.search(r"\b(misty|mist|foggy|fog|hazy|haze)\b", t):
        out["fog_scale"] = 2.5
    elif re.search(r"\b(clear|crisp)\b", t):
        out["fog_scale"] = 0.4
    if re.search(r"\b(dense|thick|lush|overgrown)\b", t):
        out["density"] = 1.8
    elif re.search(r"\b(sparse|barren|empty|few)\b", t):
        out["density"] = 0.4
    if re.search(r"\b(flat|plain|plains)\b", t):
        out.setdefault("terrain", {})["height"] = 6
    elif re.search(r"\b(steep|dramatic|rugged)\b", t):
        out.setdefault("terrain", {})["roughness"] = 0.62
    if re.search(r"\b(snowy|snow-capped|snowcapped)\b", t):
        out["snow"] = True
    return out


def guess_biome(analysis):
    """A biome from the picture alone, when nothing was said."""
    if analysis.get("indoor"):
        return "interior"
    ground = analysis.get("ground", "#777777").lstrip("#")
    r, g, b = (int(ground[i:i + 2], 16) / 255 for i in (0, 2, 4))
    if min(r, g, b) > 0.75:
        return "snow"
    greens = [p for p in analysis.get("palette", []) if 60 <= p["hue"] <= 170 and p["sat"] > 0.12]
    green_share = sum(p["share"] for p in greens)
    if green_share > 0.35 and max(p["lum"] for p in greens) < 0.35:
        return "forest"
    if green_share > 0.2:
        return "meadow"
    if r > g > b and r - b > 0.12:
        return "desert"
    return "hills"


def resolve(spec, analysis):
    """The full recipe: the spec, filled in from the picture, then the biome."""
    s = parse(spec)
    biome = s.get("biome") if s.get("biome") in BIOMES else guess_biome(analysis)
    preset = BIOMES[biome]

    terrain = dict(preset["terrain"])
    terrain.update({k: v for k, v in (s.get("terrain") or {}).items() if v is not None})
    terrain.setdefault("seed", int(s.get("seed", 0) or 0))

    density = float(s.get("density", 1.0) or 1.0)
    scatter = s.get("scatter")
    if not isinstance(scatter, list):
        scatter = [dict(x, count=int(x["count"] * density)) for x in preset["scatter"]]
    scatter = [x for x in scatter if isinstance(x, dict) and x.get("type") in SCATTER_TYPES]

    sun = dict(analysis["sun"])
    if "time" in s and str(s["time"]).lower() in TIMES:
        sun["elevation"] = TIMES[str(s["time"]).lower()]
    if s.get("sun_elevation") is not None:
        sun["elevation"] = float(s["sun_elevation"])
    if s.get("sun_azimuth") is not None:
        sun["azimuth"] = float(s["sun_azimuth"])

    fog = s.get("fog")
    if fog is None:
        fog = analysis["fog_density"] * float(s.get("fog_scale", 1.0) or 1.0)

    return {
        "biome": biome,
        "terrain": terrain,
        "scatter": scatter,
        "sun": sun,
        "fog": float(fog),
        "snow": bool(s.get("snow", preset["snow"])),
        "notes": s.get("notes", ""),
        "spec": s,
    }
