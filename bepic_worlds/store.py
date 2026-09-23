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
import os
import re
import secrets
import time

from . import builder
from . import spec as specmod

KINDS = {"environment", "terrain", "scatter", "depthmesh", "camera", "model",
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
            area = {"center": [float(v) for v in op["center"]][:2], "radius": float(op["radius"])}
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
