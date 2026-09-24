"""One interface to the worlds, two ways of reaching them.

- **HttpBackend** talks to a running ComfyUI through the /bepic_worlds/*
  routes. This is the normal case for an agent: the world is built inside
  ComfyUI, lands in ComfyUI's own output folder (wherever a launcher put it),
  and opens in the viewer the user is looking at. A reference image on the
  agent's side is uploaded first, through ComfyUI's /upload/image.
- **LocalBackend** uses the library directly on a folder. No ComfyUI needed —
  but no viewer to open either.

`backend()` picks HTTP when ComfyUI answers, local otherwise. Both return plain
dicts, so the MCP server and the CLI don't care which they have.
"""

import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.parse
import urllib.request

from .store import WorldStore, OPS

DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


class BackendError(RuntimeError):
    pass


class HttpBackend:
    kind = "comfyui"

    def __init__(self, url=DEFAULT_URL, timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _req(self, method, path, body=None, raw=False, headers=None):
        data = None
        hdrs = dict(headers or {})
        if body is not None and not isinstance(body, bytes):
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        elif isinstance(body, bytes):
            data = body
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error")
            except Exception:
                msg = None
            raise BackendError(msg or f"ComfyUI answered {e.code} for {path}") from None
        except urllib.error.URLError as e:
            raise BackendError(f"ComfyUI is not reachable at {self.url} ({e.reason})") from None
        return payload if raw else json.loads(payload.decode("utf-8"))

    def alive(self):
        try:
            self._req("GET", "/bepic_worlds/info")
            return True
        except BackendError:
            return False

    def _image(self, ref):
        """A path on this machine is uploaded; anything else is passed on as
        ComfyUI names it ({filename, subfolder, type}, or a path it can see)."""
        if not ref:
            return None
        if isinstance(ref, str) and os.path.isfile(ref):
            boundary = uuid.uuid4().hex
            name = os.path.basename(ref)
            with open(ref, "rb") as fh:
                content = fh.read()
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                    f"Content-Type: {ctype}\r\n\r\n").encode() + content + \
                   (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n"
                    f"worlds_refs\r\n--{boundary}--\r\n").encode()
            up = self._req("POST", "/upload/image", body,
                           headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            return {"filename": up["name"], "subfolder": up.get("subfolder", ""), "type": up.get("type", "input")}
        return ref

    def create(self, reference, name="world", open_in_viewer=True, **kw):
        body = {"reference": self._image(reference), "name": name, "open_in_viewer": open_in_viewer}
        for k in ("depth", "heightmap", "panorama"):
            if kw.get(k):
                body[k] = self._image(kw.pop(k))
        body.update({k: v for k, v in kw.items() if v is not None})
        return self._req("POST", "/bepic_worlds/create", body)

    def list(self):
        return self._req("GET", "/bepic_worlds/list")["worlds"]

    def describe(self, name):
        return self._req("GET", "/bepic_worlds/world?" + urllib.parse.urlencode({"name": name, "summary": 1}))

    def world_json(self, name, version=None):
        q = {"name": name}
        if version:
            q["version"] = version
        return self._req("GET", "/bepic_worlds/world?" + urllib.parse.urlencode(q))

    def edit(self, name, ops, note="", open_in_viewer=True):
        return self._req("POST", "/bepic_worlds/edit",
                         {"name": name, "ops": ops, "note": note, "open_in_viewer": open_in_viewer})

    def revert(self, name, version, note="", open_in_viewer=True):
        return self._req("POST", "/bepic_worlds/revert",
                         {"name": name, "version": version, "note": note, "open_in_viewer": open_in_viewer})

    def feedback(self, name, status="open"):
        return self._req("GET", "/bepic_worlds/feedback?" +
                         urllib.parse.urlencode({"name": name, "status": status or "all"}))["feedback"]

    def snapshot_bytes(self, entry):
        ref = entry.get("snapshot_view")
        if not ref:
            return None
        try:
            return self._req("GET", "/view?" + urllib.parse.urlencode(ref), raw=True)
        except BackendError:
            return None

    def add_feedback(self, name, text, point=None, camera=None, author="agent"):
        return self._req("POST", "/bepic_worlds/feedback",
                         {"name": name, "text": text, "point": point, "camera": camera, "author": author})

    def resolve(self, name, ids, reply=""):
        return self._req("POST", "/bepic_worlds/feedback/resolve", {"name": name, "ids": ids, "reply": reply})

    def open(self, name, version=None):
        return self._req("POST", "/bepic_worlds/open", {"name": name, "version": version})

    def calibrate(self, name, wait=60.0):
        """Have the viewer match the world to its reference; wait for the new
        version (up to `wait` seconds) and return its history entry."""
        import time
        req = self._req("POST", "/bepic_worlds/calibrate", {"name": name})
        deadline = time.time() + max(0.0, wait)
        while time.time() < deadline:
            time.sleep(1.5)
            d = self.describe(name)
            if d["version"] > req["version_before"]:
                return {"matched": True, "version": d["version"], "note": (d.get("history") or [{}])[-1].get("note")}
        return {"matched": False, "note": "no new version yet — is the world open in a viewer? "
                                          "(the match runs in the browser)", **req}


class LocalBackend:
    kind = "local"

    def __init__(self, root=None, url=DEFAULT_URL):
        self.store = WorldStore(root)
        self.http = HttpBackend(url, timeout=10)

    def create(self, reference, name="world", open_in_viewer=True, **kw):
        out = self.store.create(reference, name=name, **{k: v for k, v in kw.items() if v is not None})
        if open_in_viewer:
            out["viewer"] = self._try_open(out["name"])
        return out

    def list(self):
        return [{"name": n, "version": self.store.version(n),
                 "open_feedback": len(self.store.feedback(n, "open"))} for n in self.store.names()]

    def describe(self, name):
        return self.store.summary(name)

    def world_json(self, name, version=None):
        return self.store.load(name, version)

    def edit(self, name, ops, note="", open_in_viewer=True):
        out = self.store.edit(name, ops, note=note)
        if open_in_viewer:
            out["viewer"] = self._try_open(name)
        return out

    def revert(self, name, version, note="", open_in_viewer=True):
        out = self.store.revert(name, version, note=note)
        if open_in_viewer:
            out["viewer"] = self._try_open(name)
        return out

    def feedback(self, name, status="open"):
        return self.store.feedback(name, None if status in (None, "all") else status)

    def snapshot_bytes(self, entry):
        p = entry.get("snapshot")
        if p and os.path.isfile(p):
            with open(p, "rb") as fh:
                return fh.read()
        return None

    def add_feedback(self, name, text, point=None, camera=None, author="agent"):
        return self.store.add_feedback(name, text, point=point, camera=camera, author=author)

    def resolve(self, name, ids, reply=""):
        return self.store.resolve_feedback(name, ids, reply=reply)

    def open(self, name, version=None):
        return self.http.open(name, version)

    def calibrate(self, name, wait=60.0):
        return self.http.calibrate(name, wait)

    def _try_open(self, name):
        try:
            return self.http.open(name)
        except BackendError as e:
            return {"opened": None, "note": f"not opened in the viewer: {e}"}


def backend(url=None, root=None, mode=None):
    """HTTP when ComfyUI answers (or when asked for), local otherwise."""
    mode = mode or os.environ.get("BEPIC_WORLDS_MODE", "auto")
    url = url or DEFAULT_URL
    if mode == "local" or (root and mode == "auto"):
        return LocalBackend(root, url)
    http = HttpBackend(url)
    if mode == "http" or http.alive():
        return http
    return LocalBackend(root, url)


SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SCHEMA.md")


def schema_text():
    try:
        with open(SCHEMA_PATH, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        text = "(SCHEMA.md not found)"
    ops = "\n".join(f"- `{k}`: {v}" for k, v in OPS.items())
    return f"{text}\n\n## Edit operations\n{ops}\n"
