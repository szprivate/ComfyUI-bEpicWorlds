"""Worlds on disk: one folder each, every version kept, feedback alongside.

    <root>/<name>/
        world.json          the current scene — the viewer can load this file as is
        meta.json           what it was made from, and the history of edits
        versions/v001.json  every version, so nothing an agent does is final
        assets/             heightmaps, textures, the reference, depth, panorama
        feedback.jsonl      notes left in the viewer, one JSON object per line
        feedback/           the snapshot taken with each note

`<root>` is ComfyUI's output folder + "/worlds" when running inside ComfyUI,
else $BEPIC_WORLDS_ROOT, else ../../output/worlds from this package when it
sits in custom_nodes. Every path is built from a sanitised name, so nothing
here reads or writes outside the root.

Edits are small operations (see `OPS`), applied to a copy, validated, and
saved as a new version. They are what an agent sends — each one names an item
by its fixed id and says what to change.
"""

import base64
import copy
import json
import math
import os
import re
import secrets
import time

from . import builder
from . import spec as specmod

KINDS = {"environment", "terrain", "scatter", "depthmesh", "light", "camera", "model",
         "primitive", "imageplane", "group"}
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def default_root():
    env = os.environ.get("BEPIC_WORLDS_ROOT")
    if env:
        return os.path.abspath(env)
    try:
        import folder_paths
        return os.path.join(folder_paths.get_output_directory(), "worlds")
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.basename(here) == "custom_nodes":
        return os.path.join(os.path.dirname(here), "output", "worlds")
    return os.path.abspath("worlds")


def safe_name(name):
    n = _NAME_RE.sub("_", str(name or "").strip()).strip("_")[:64]
    if not n:
        raise ValueError("a world needs a name made of letters, digits, - or _")
    return n


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _read(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _dig(obj, path, create=False):
    """(parent, key) for a dotted path into nested dicts / lists."""
    parts = [p for p in str(path).split(".") if p != ""]
    if not parts:
        raise ValueError("an empty path")
    cur = obj
    for p in parts[:-1]:
        key = int(p) if isinstance(cur, list) else p
        if isinstance(cur, list):
            cur = cur[key]
        elif key not in cur or cur[key] is None:
            if not create:
                raise ValueError(f"nothing at '{path}' ('{p}' is missing)")
            cur[key] = {}
            cur = cur[key]
        else:
            cur = cur[key]
    last = parts[-1]
    return cur, (int(last) if isinstance(cur, list) else last)


class WorldStore:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or default_root())

    # ── places ───────────────────────────────────────────────────────────
    def dir(self, name):
        path = os.path.join(self.root, safe_name(name))
        if os.path.commonpath([self.root, os.path.abspath(path)]) != self.root:
            raise ValueError("that name leaves the worlds folder")
        return path

    def exists(self, name):
        return os.path.isfile(os.path.join(self.dir(name), "world.json"))

    def names(self):
        try:
            entries = sorted(os.listdir(self.root))
        except FileNotFoundError:
            return []
        return [n for n in entries if os.path.isfile(os.path.join(self.root, n, "world.json"))]

    def _unique(self, name):
        base = safe_name(name)
        if not self.exists(base):
            return base
        i = 2
        while self.exists(f"{base}_{i}"):
            i += 1
        return f"{base}_{i}"

    # ── making and reading ───────────────────────────────────────────────
    def create(self, reference, name="world", overwrite=False, note="created", **kwargs):
        """Build a world. With `overwrite`, a world of that name is rebuilt as
        its next version — the older versions and the feedback stay."""
        name = safe_name(name) if overwrite else self._unique(name)
        d = self.dir(name)
        previous = self.meta(name) if self.exists(name) else None
        version = self.version(name) + 1 if previous is not None else 1
        assets = os.path.join(d, "assets") if version == 1 else os.path.join(d, "assets", f"v{version:03d}")
        scene, meta = builder.build_world(reference, d, name=name, assets_dir=assets, **kwargs)
        # What it was made from, so `rebuild` can make it again with one thing changed.
        meta["inputs"] = {k: os.path.abspath(v) for k, v in
                          dict(kwargs, reference=reference).items()
                          if k in ("reference", "depth", "heightmap", "panorama") and isinstance(v, str) and os.path.isfile(v)}
        meta["seed"] = kwargs.get("seed")
        if previous is not None:
            meta["created"] = previous.get("created", _now())
            meta["history"] = previous.get("history", []) + [
                {"version": version, "time": _now(), "note": note if note != "created" else "rebuilt"}]
        else:
            meta["created"] = _now()
            meta["history"] = [{"version": 1, "time": _now(), "note": note}]
        _write(os.path.join(d, "meta.json"), meta)
        self._save_scene(name, scene, version)
        return self.summary(name)

    def rebuild(self, name, note="rebuilt", **changes):
        """Make a world again from the inputs it was made from, with `changes`
        (pitch, fov, spec, world_size, eye_height, seed, depth, …) — its next
        version. Edits made since (add_asset, set_material, …) are not carried
        over: rebuild first, then add."""
        meta = self.meta(name)
        inputs = dict(meta.get("inputs") or {})
        if not inputs.get("reference") or not os.path.isfile(inputs["reference"]):
            raise ValueError(f"'{name}' doesn't record the files it was made from; create it again instead")
        kwargs = {"spec": meta.get("spec"), "fov": meta.get("fov"), "world_size": meta.get("world_size"),
                  "eye_height": meta.get("eye_height"), "seed": meta.get("seed")}
        pitch = (meta.get("analysis") or {}).get("pitch")
        if (meta.get("analysis") or {}).get("pitch_source") == "objects" and pitch is not None:
            kwargs["pitch"] = pitch                      # a measured tilt survives a rebuild
        kwargs.update({k: v for k, v in inputs.items() if k != "reference"})
        kwargs.update({k: v for k, v in changes.items() if v is not None})
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.create(inputs["reference"], name=name, overwrite=True, note=note, **kwargs)

    def load(self, name, version=None):
        d = self.dir(name)
        path = (os.path.join(d, "versions", f"v{int(version):03d}.json") if version
                else os.path.join(d, "world.json"))
        scene = _read(path)
        if scene is None:
            raise FileNotFoundError(f"no world '{name}'" + (f" version {version}" if version else ""))
        return scene

    def meta(self, name):
        return _read(os.path.join(self.dir(name), "meta.json"), {})

    def _save_scene(self, name, scene, version):
        d = self.dir(name)
        scene.setdefault("world", {})["name"] = name
        scene["world"]["version"] = version
        scene["world"].pop("feedback", None)          # pins are added when served, not stored
        _write(os.path.join(d, "versions", f"v{version:03d}.json"), scene)
        _write(os.path.join(d, "world.json"), scene)

    def version(self, name):
        return int(((self.load(name).get("world") or {}).get("version")) or 1)

    def summary(self, name):
        """What an agent needs to know to talk about a world, and nothing more."""
        scene = self.load(name)
        meta = self.meta(name)
        items = []
        for it in scene.get("items", []):
            one = {"id": it.get("id"), "kind": it.get("kind"), "name": it.get("name")}
            if it.get("kind") == "environment":
                one.update(sun=it.get("sun"), fog=it.get("fog"), sky_mode=(it.get("sky") or {}).get("mode"))
            elif it.get("kind") == "terrain":
                t = it.get("terrain") or {}
                one.update({k: t.get(k) for k in ("size", "height", "seed", "roughness")},
                           layers=[l.get("name") for l in t.get("layers", [])])
            elif it.get("kind") == "scatter":
                s = it.get("scatter") or {}
                one.update(type=(s.get("source") or {}).get("type"), count=s.get("count"),
                           scale=s.get("scale"), layer=s.get("layer"), wind=s.get("wind"),
                           clear=s.get("clear"))
            elif it.get("kind") == "camera":
                one.update(position=it.get("position"), rotation=it.get("rotation"), fov=it.get("fov"))
            items.append(one)
        fb = self.feedback(name, "open")
        return {
            "name": name,
            "version": self.version(name),
            "folder": self.dir(name),
            "world_json": os.path.join(self.dir(name), "world.json"),
            "biome": (meta.get("recipe") or {}).get("biome"),
            "notes": (meta.get("recipe") or {}).get("notes", ""),
            "walk": scene.get("walk"),
            "items": items,
            "open_feedback": len(fb),
            "history": meta.get("history", [])[-10:],
        }

    def scene_for_viewer(self, name, version=None):
        """The scene with its open feedback attached as pins."""
        scene = self.load(name, version)
        scene.setdefault("world", {})["feedback"] = [
            {"id": f["id"], "n": f.get("n"), "text": f["text"], "point": f.get("point")}
            for f in self.feedback(name, "open") if f.get("point")]
        return scene

    # ── editing ──────────────────────────────────────────────────────────
    def edit(self, name, ops, note=""):
        """Apply `ops` (a list of operations, see OPS) as one new version."""
        if isinstance(ops, str):
            ops = json.loads(ops)
        if isinstance(ops, dict):
            ops = [ops]
        if not isinstance(ops, list) or not ops:
            raise ValueError("ops must be a non-empty list of operations")
        scene = copy.deepcopy(self.load(name))
        version = self.version(name) + 1
        changed = []
        for i, op in enumerate(ops):
            try:
                changed += self._apply(name, scene, op, version)
            except Exception as e:
                raise ValueError(f"op {i} ({op.get('op') if isinstance(op, dict) else op!r}): {e}") from None
        validate(scene)
        self._save_scene(name, scene, version)
        meta = self.meta(name)
        meta.setdefault("history", []).append({"version": version, "time": _now(),
                                               "note": note or "", "ops": ops})
        _write(os.path.join(self.dir(name), "meta.json"), meta)
        out = self.summary(name)
        out["changed"] = sorted(set(changed))
        return out

    def revert(self, name, version, note=""):
        old = self.load(name, version)
        new_version = self.version(name) + 1
        self._save_scene(name, copy.deepcopy(old), new_version)
        meta = self.meta(name)
        meta.setdefault("history", []).append({"version": new_version, "time": _now(),
                                               "note": note or f"back to version {version}"})
        _write(os.path.join(self.dir(name), "meta.json"), meta)
        return self.summary(name)

    def _item(self, scene, item_id):
        for it in scene.get("items", []):
            if it.get("id") == item_id:
                return it
        raise ValueError(f"no item '{item_id}' — ids here: "
                         + ", ".join(i.get("id", "?") for i in scene.get("items", [])))

    def _apply(self, name, scene, op, version):
        if not isinstance(op, dict) or "op" not in op:
            raise ValueError("each operation is an object with an 'op'")
        kind = op["op"]
        if kind == "set":
            item = self._item(scene, op["id"])
            if str(op["path"]).split(".")[0] in ("id", "kind"):
                raise ValueError("an item's id and kind can't be changed")
            parent, key = _dig(item, op["path"], create=True)
            parent[key] = op["value"]
            return [item["id"]]
        if kind == "set_scene":
            if str(op["path"]).split(".")[0] in ("items", "world"):
                raise ValueError("use the item operations for items")
            parent, key = _dig(scene, op["path"], create=True)
            parent[key] = op["value"]
            return ["scene"]
        if kind == "remove":
            before = len(scene["items"])
            scene["items"] = [i for i in scene["items"] if i.get("id") != op["id"]]
            if len(scene["items"]) == before:
                raise ValueError(f"no item '{op['id']}'")
            return [op["id"]]
        if kind == "add_item":
            item = copy.deepcopy(op["item"])
            if item.get("kind") not in KINDS:
                raise ValueError(f"kind must be one of {sorted(KINDS)}")
            item.setdefault("id", f"{item['kind']}_{secrets.token_hex(3)}")
            if any(i.get("id") == item["id"] for i in scene["items"]):
                raise ValueError(f"id '{item['id']}' is taken")
            for k, v in builder._transform().items():
                item.setdefault(k, v)
            item.setdefault("name", item["id"])
            scene["items"].append(item)
            return [item["id"]]
        if kind == "add_scatter":
            meta = self.meta(name)
            analysis = meta.get("analysis") or {}
            existing = [i for i in scene["items"] if i.get("kind") == "scatter"
                        and (i.get("scatter") or {}).get("source", {}).get("type") == op.get("type")]
            if op.get("type") not in specmod.SCATTER_TYPES:
                raise ValueError(f"type must be one of {specmod.SCATTER_TYPES}")
            refcam = next((i for i in scene["items"] if i.get("id") == "refcam"), None)
            clear = None
            if refcam:
                clear = {"center": [refcam["position"][0], refcam["position"][2] - 6.0], "radius": 12.0}
            item = builder.scatter_item(op, len(existing), clear, analysis.get("foliage", "#3d5a2a"),
                                        analysis.get("ground", "#777777"))
            while any(i.get("id") == item["id"] for i in scene["items"]):
                item["id"] += "_x"
            scene["items"].append(item)
            return [item["id"]]
        if kind == "scale_scatter":
            ids = [op["id"]] if op.get("id") else [i["id"] for i in scene["items"] if i.get("kind") == "scatter"]
            factor = float(op["factor"])
            for one in ids:
                s = self._item(scene, one)["scatter"]
                s["count"] = max(0, int(round(s["count"] * factor)))
            return ids
        if kind == "clear_area":
            ids = [op["id"]] if op.get("id") else [i["id"] for i in scene["items"] if i.get("kind") == "scatter"]
            area = {"center": [float(v) for v in op["center"]][:2], "radius": float(op["radius"])}   # a circle
            for one in ids:
                s = self._item(scene, one)["scatter"]
                cur = s.get("clear")
                cur = [] if not cur else ([cur] if isinstance(cur, dict) else list(cur))
                s["clear"] = cur + [area]
            return ids
        if kind == "regenerate_terrain":
            builder.regenerate_heightmap(scene, self.dir(name), version, seed=op.get("seed"),
                                         roughness=op.get("roughness"), height=op.get("height"))
            return ["terrain", "refcam", "hero", "scene"]
        if kind == "add_asset":
            return self._add_asset(name, scene, op, version)
        if kind == "set_material":
            return self._set_material(name, scene, op, version)
        if kind == "set_sky":
            return self._set_sky(name, scene, op, version)
        if kind == "set_time":
            t = str(op["time"]).lower()
            if t not in specmod.TIMES:
                raise ValueError(f"time must be one of {sorted(specmod.TIMES)}")
            env = self._item(scene, "env")
            env["sun"]["elevation"] = float(specmod.TIMES[t])
            if env["sun"]["elevation"] < 12:
                env["sun"]["color"] = "#ffb070" if env["sun"]["elevation"] >= 0 else "#8fa6d9"
            return ["env"]
        raise ValueError(f"unknown op '{kind}' — known: {', '.join(OPS)}")

    # ── assets from the picture ──────────────────────────────────────────
    def _add_asset(self, name, scene, op, version):
        """A generated model (image-to-3D) placed where the picture shows it.

        op: {glb, texture?, textured?, label, and where:
          bboxes: [[x0, y0, x1, y1], …] (0..1 of the reference: standing where
            the picture shows it, as tall as it looks, facing the camera), or
          positions: [[x, z] or [x, y, z], …] (on the ground when y is left out), or
          scatter: {count, center: [x, z], radius, seed?, spacing?} (on the ground,
            no closer than `spacing` metres),
        height? (metres; needed for positions and scatter), yaw? (degrees; else
        facing the camera for boxes, random for positions and scatter)}.
        A mesh that comes untextured is textured from `texture` (the RGBA
        picture it was made from); one that comes `textured` (Meshy, …) keeps
        its materials. Either way it is normalised to stand on y = 0 one unit
        tall and copied into the world's assets; one model item per spot."""
        import random
        from . import assets as assetsmod
        glb = op.get("glb")
        if not glb or not os.path.isfile(glb):
            raise ValueError("add_asset needs `glb`: the generated mesh file")
        label = safe_name(op.get("label") or "asset").lower()
        out = os.path.join(self.dir(name), "assets", "models", f"{label}_v{version}.glb")
        texture = op.get("texture")
        if op.get("textured"):
            assetsmod.normalize_glb(glb, out)
        elif texture and os.path.isfile(texture):
            assetsmod.project_texture(glb, texture, out)
        else:
            assetsmod.normalize_glb(glb, out)
        rng = random.Random(op.get("seed", 0))
        height = float(op.get("height") or 1.0)
        spots = []
        for box in op.get("bboxes") or []:
            loc = assetsmod.locate(scene, [float(v) for v in box])
            spots.append((loc["position"], loc["height"], loc["yaw"], box))
        for pos in op.get("positions") or []:
            p = [float(v) for v in pos]
            p = [p[0], assetsmod._ground_y(scene, p[0], p[1]), p[1]] if len(p) == 2 else p[:3]
            spots.append((p, height, rng.uniform(0, 360), None))
        sc = op.get("scatter")
        if sc:
            cx, cz = [float(v) for v in sc.get("center", [0, 0])]
            radius, spacing = float(sc.get("radius", 10)), float(sc.get("spacing", height))
            placed = []
            for _ in range(int(sc.get("count", 5)) * 30):
                if len(placed) >= int(sc.get("count", 5)):
                    break
                a, r = rng.uniform(0, 6.2832), radius * rng.random() ** 0.5
                x, z = cx + r * math.cos(a), cz + r * math.sin(a)
                if all((x - px) ** 2 + (z - pz) ** 2 >= spacing ** 2 for px, pz in placed):
                    placed.append((x, z))
                    spots.append(([x, assetsmod._ground_y(scene, x, z), z], height, rng.uniform(0, 360), None))
        if not spots:
            raise ValueError("add_asset needs `bboxes` (where the picture shows it), `positions` or `scatter`")
        taken = {i.get("id") for i in scene["items"]}
        ids = []
        for n, (pos, h, yaw, box) in enumerate(spots):
            h = float(op["height"]) if op.get("height") and box is not None else h
            if op.get("yaw") is not None:
                yaw = float(op["yaw"])
            iid = f"{label}_{n + 1}"
            while iid in taken:
                iid += "_x"
            taken.add(iid)
            item = {"id": iid, "kind": "model", "name": f"{label.title()} {n + 1}",
                    **builder._transform(pos, (0.0, yaw, 0.0), (h, h, h)),
                    "src": builder.src(out, f"{label}.glb")}
            item["src"]["format"] = "glb"
            if box is not None:
                item["from_picture"] = [round(float(v), 4) for v in box]
            scene["items"].append(item)
            ids.append(iid)
        return ids

    def _set_sky(self, name, scene, op, version):
        """A panorama for the sky: op {panorama (2:1 image), horizon? (0..1:
        the row its horizon is on; found when left out), level? (default
        true: move the horizon to the middle row, as equirectangular needs)}."""
        from PIL import Image
        from . import assets as assetsmod
        pano = op.get("panorama")
        if not pano or not os.path.isfile(pano):
            raise ValueError("set_sky needs `panorama`: a 2:1 equirectangular image")
        env = self._item(scene, "env")
        dst = os.path.join(self.dir(name), "assets", "sky", f"sky_v{version}.png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(pano) as im:
            im = im.convert("RGB")
            if op.get("level", True):
                im, _h = assetsmod.level_horizon(im, op.get("horizon"))
            im.save(dst)
        sky = env.setdefault("sky", {})
        sky["mode"] = "panorama"
        sky["src"] = builder.src(dst)
        return ["env"]

    def _set_material(self, name, scene, op, version):
        """PBR maps from a material model (Chord) onto a terrain layer.

        op: {id (terrain id), layer (0..3), albedo, normal?, roughness?, tile?,
        normalScale?, baked?}. The files are copied into the world's assets."""
        import shutil
        item = self._item(scene, op.get("id") or "terrain")
        if item.get("kind") != "terrain":
            raise ValueError(f"'{item.get('id')}' is not a terrain")
        layer = int(op.get("layer", 0))
        layers = (item.get("terrain") or {}).get("layers") or []
        if not 0 <= layer < len(layers):
            raise ValueError("layer must be 0..3")
        L = layers[layer]
        folder = os.path.join(self.dir(name), "assets", "materials")
        os.makedirs(folder, exist_ok=True)
        for key, field in (("albedo", "src"), ("normal", "normal"), ("roughness", "rough")):
            path = op.get(key)
            if not path:
                continue
            if not os.path.isfile(path):
                raise ValueError(f"no such file for {key}: {path}")
            dst = os.path.join(folder, f"{item['id']}_{layer}_{key}_v{version}{os.path.splitext(path)[1] or '.png'}")
            shutil.copyfile(path, dst)
            L[field] = builder.src(dst)
        if op.get("albedo"):
            L["color"] = "#ffffff"
        if op.get("description"):
            L["description"] = str(op["description"])[:400]
        for key in ("tile", "normalScale", "baked", "roughness_value"):
            if op.get(key) is not None:
                L["roughness" if key == "roughness_value" else key] = float(op[key])
        return [item["id"]]

    # ── feedback ─────────────────────────────────────────────────────────
    def _fb_path(self, name):
        return os.path.join(self.dir(name), "feedback.jsonl")

    def feedback(self, name, status=None):
        out = []
        try:
            with open(self._fb_path(name), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            pass
        except FileNotFoundError:
            pass
        return [f for f in out if not status or f.get("status") == status]

    def _fb_write(self, name, entries):
        path = self._fb_path(name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for f in entries:
                fh.write(json.dumps(f) + "\n")
        os.replace(tmp, path)

    def add_feedback(self, name, text, point=None, camera=None, snapshot=None, version=None,
                     author="user"):
        if not self.exists(name):
            raise FileNotFoundError(f"no world '{name}'")
        text = str(text or "").strip()
        if not text:
            raise ValueError("feedback needs some text")
        entries = self.feedback(name)
        fid = "fb_" + secrets.token_hex(4)
        entry = {"id": fid, "n": len(entries) + 1, "created": _now(),
                 "version": int(version or self.version(name)), "author": author,
                 "text": text[:4000], "status": "open",
                 "point": [float(v) for v in point][:3] if point else None,
                 "camera": camera if isinstance(camera, dict) else None,
                 "snapshot": None}
        if snapshot:
            entry["snapshot"] = self._save_snapshot(name, fid, snapshot)
        entries.append(entry)
        self._fb_write(name, entries)
        return entry

    def _save_snapshot(self, name, fid, data_url):
        m = re.match(r"data:image/(png|jpeg|jpg|webp);base64,(.+)$", str(data_url), re.S)
        if not m:
            return None
        raw = base64.b64decode(m.group(2))
        if len(raw) > 6 * 1024 * 1024:
            return None
        ext = "jpg" if m.group(1) in ("jpeg", "jpg") else m.group(1)
        path = os.path.join(self.dir(name), "feedback", f"{fid}.{ext}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
        return path

    def resolve_feedback(self, name, ids, reply="", version=None):
        ids = set([ids] if isinstance(ids, str) else ids or [])
        entries = self.feedback(name)
        done = []
        for f in entries:
            if f["id"] in ids and f.get("status") != "resolved":
                f["status"] = "resolved"
                f["resolved"] = _now()
                f["resolved_in"] = int(version or self.version(name))
                if reply:
                    f["reply"] = str(reply)[:4000]
                done.append(f["id"])
        self._fb_write(name, entries)
        missing = sorted(ids - set(done) - {f["id"] for f in entries if f.get("status") == "resolved"})
        return {"resolved": done, "unknown": missing}


# Operations, as documented for agents (also served by the MCP `schema` tool).
OPS = {
    "set": "{op, id, path, value} — set a field of an item by dotted path, e.g. "
           "id 'env', path 'sun.elevation', value 20; id 'terrain', path 'terrain.height'.",
    "set_scene": "{op, path, value} — a scene field, e.g. 'walk.speed', 'walk.eyeHeight'.",
    "add_item": "{op, item} — a whole item (kind, id optional); see SCHEMA.md.",
    "remove": "{op, id} — remove an item.",
    "add_scatter": "{op, type, count, scale?, layer?, maxSlope?, wind?, color?} — "
                   "type: pine | tree | bush | grass | rock | model (model needs src).",
    "scale_scatter": "{op, factor, id?} — multiply instance counts (all scatters, or one).",
    "clear_area": "{op, center:[x,z], radius, id?} — keep scatter out of a circle.",
    "regenerate_terrain": "{op, seed?, roughness?, height?} — a new terrain shape.",
    "set_time": "{op, time} — dawn, morning, noon, afternoon, late afternoon, golden hour, "
                "evening, sunset, dusk, twilight, night.",
    "add_asset": "{op, glb, texture?, textured?, label, bboxes: [[x0,y0,x1,y1],…] (0..1 of the reference; "
                 "placed where its foot meets the ground, as tall as the box says, facing the camera) | "
                 "positions: [[x,z] or [x,y,z],…] | scatter: {count, center:[x,z], radius, spacing?, seed?}, "
                 "height? (metres, for positions/scatter), yaw?} — a generated model (image-to-3D), textured "
                 "from its picture unless it comes textured. Files may be ComfyUI refs {filename, subfolder, type}.",
    "set_material": "{op, id?: 'terrain', layer: 0..3, albedo, normal?, roughness?, tile?, normalScale?, "
                    "baked?, description?} — PBR maps (e.g. from Chord) onto a terrain layer.",
    "set_sky": "{op, panorama, horizon?, level?} — a 2:1 sky panorama (the sky slot makes one); its "
               "horizon is moved to the middle row unless level is false.",
}


def validate(scene):
    """Enough checking that the viewer can read what an edit left behind."""
    items = scene.get("items")
    if not isinstance(items, list):
        raise ValueError("the scene has no item list")
    seen = set()
    for it in items:
        if it.get("kind") not in KINDS:
            raise ValueError(f"item '{it.get('id')}' has unknown kind '{it.get('kind')}'")
        if it.get("id") in seen:
            raise ValueError(f"two items share the id '{it.get('id')}'")
        seen.add(it.get("id"))
        for vec in ("position", "rotation", "scale"):
            v = it.get(vec)
            if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)):
                raise ValueError(f"item '{it.get('id')}': {vec} must be three numbers")
        if it["kind"] == "scatter":
            s = it.get("scatter") or {}
            if not isinstance(s.get("count"), int) or s["count"] < 0 or s["count"] > 200000:
                raise ValueError(f"item '{it['id']}': scatter.count must be 0..200000")
    return True
