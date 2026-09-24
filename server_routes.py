"""HTTP routes: what an agent (through the MCP server or the CLI) and the
viewer use to reach the worlds while ComfyUI is running.

All under /bepic_worlds/. Reads are GET; anything that makes, changes or
opens something is POST, so a link or an <img> on some page can't. Every world
lives under <output>/worlds, reached only through sanitised names; a reference
image given by path must sit in ComfyUI's input, output or temp folder. No
route starts a process.
"""

import json
import os

import folder_paths
from aiohttp import web
from server import PromptServer

from .bepic_worlds import store as store_mod
from .bepic_worlds.store import WorldStore

_store = None


def store():
    global _store
    root = os.path.join(folder_paths.get_output_directory(), "worlds")
    if _store is None or _store.root != os.path.abspath(root):
        _store = WorldStore(root)
    return _store


def _comfy_dirs():
    out = []
    for get in (folder_paths.get_input_directory, folder_paths.get_output_directory,
                folder_paths.get_temp_directory):
        try:
            out.append(os.path.realpath(get()))
        except Exception:
            pass
    return out


def resolve_image(ref):
    """An image an API caller named: {filename, subfolder, type} as ComfyUI's
    own /view takes it, or a path — either way, inside input/output/temp."""
    if not ref:
        return None
    if isinstance(ref, dict):
        base = {"input": folder_paths.get_input_directory,
                "output": folder_paths.get_output_directory,
                "temp": folder_paths.get_temp_directory}.get(ref.get("type") or "input")
        if base is None:
            raise ValueError("type must be input, output or temp")
        path = os.path.join(base(), ref.get("subfolder") or "", ref.get("filename") or "")
    else:
        path = str(ref)
        if not os.path.isabs(path):
            path = os.path.join(folder_paths.get_input_directory(), path)
    real = os.path.realpath(path)
    for d in _comfy_dirs():
        try:
            if os.path.commonpath([real, d]) == d:
                if not os.path.isfile(real):
                    raise ValueError(f"no such image: {ref}")
                return real
        except ValueError as e:
            if "no such image" in str(e):
                raise
            continue
    raise ValueError("images must be in ComfyUI's input, output or temp folder "
                     "(upload one with /upload/image first)")


def view_ref(path):
    """{filename, subfolder, type} for a file under output — how a remote
    caller fetches it through ComfyUI's /view."""
    if not path:
        return None
    out = os.path.realpath(folder_paths.get_output_directory())
    real = os.path.realpath(path)
    try:
        if os.path.commonpath([real, out]) != out:
            return None
    except ValueError:
        return None
    sub = os.path.relpath(os.path.dirname(real), out)
    return {"filename": os.path.basename(real), "subfolder": "" if sub == "." else sub.replace(os.sep, "/"),
            "type": "output"}


def open_in_viewer(name, version=None, focus=True):
    """Show a world in the bEpic viewer: its own tab, replaced if already open.
    `focus` brings the tab forward; a refresh (pins resolved) leaves the user
    where they are."""
    scene = store().scene_for_viewer(name, version)
    PromptServer.instance.send_sync("bepic.viewer.update", {
        "tabs": {f"world:{name}": []},
        "unique_id": None,
        "scene_data": json.dumps(scene),
        "scene_replace": True,
        "tab_label": f"World: {name}",
        "focus_tab": bool(focus),
    })
    return {"opened": name, "version": (scene.get("world") or {}).get("version")}


def _feedback_out(entries):
    out = []
    for f in entries:
        f = dict(f)
        f["snapshot_view"] = view_ref(f.get("snapshot"))
        out.append(f)
    return out


def _err(e, status=400):
    return web.json_response({"error": str(e)}, status=status)


async def _json(request):
    try:
        data = await request.json()
    except Exception:
        data = None
    if not isinstance(data, dict):
        raise ValueError("send a JSON object")
    return data


def register():
    routes = PromptServer.instance.routes

    @routes.get("/bepic_worlds/info")
    async def info(_request):
        from .bepic_worlds import __version__
        return web.json_response({"root": store().root, "version": __version__,
                                  "ops": store_mod.OPS})

    @routes.get("/bepic_worlds/list")
    async def list_worlds(_request):
        s = store()
        return web.json_response({"worlds": [
            {"name": n, "version": s.version(n), "open_feedback": len(s.feedback(n, "open"))}
            for n in s.names()]})

    @routes.get("/bepic_worlds/world")
    async def get_world(request):
        try:
            name = request.query.get("name", "")
            if request.query.get("summary"):
                return web.json_response(store().summary(name))
            v = request.query.get("version")
            return web.json_response(store().load(name, int(v) if v else None))
        except FileNotFoundError as e:
            return _err(e, 404)
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/create")
    async def create(request):
        try:
            d = await _json(request)
            kwargs = {k: d[k] for k in ("spec", "fov", "world_size", "eye_height", "seed") if d.get(k) is not None}
            for k in ("depth", "heightmap", "panorama"):
                if d.get(k):
                    kwargs[k] = resolve_image(d[k])
            out = store().create(resolve_image(d.get("reference")), name=d.get("name") or "world",
                                 overwrite=bool(d.get("overwrite")), note=d.get("note") or "created", **kwargs)
            if d.get("open_in_viewer", True):
                out["viewer"] = open_in_viewer(out["name"])
            return web.json_response(out)
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/edit")
    async def edit(request):
        try:
            d = await _json(request)
            out = store().edit(d.get("name", ""), d.get("ops"), note=d.get("note", ""))
            if d.get("open_in_viewer", True):
                out["viewer"] = open_in_viewer(out["name"])
            return web.json_response(out)
        except FileNotFoundError as e:
            return _err(e, 404)
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/revert")
    async def revert(request):
        try:
            d = await _json(request)
            out = store().revert(d.get("name", ""), int(d.get("version")), note=d.get("note", ""))
            if d.get("open_in_viewer", True):
                out["viewer"] = open_in_viewer(out["name"])
            return web.json_response(out)
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/open")
    async def open_route(request):
        try:
            d = await _json(request)
            return web.json_response(open_in_viewer(d.get("name", ""), d.get("version")))
        except FileNotFoundError as e:
            return _err(e, 404)
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/calibrate")
    async def calibrate(request):
        """Ask the open viewer to match a world to its reference picture. The
        measuring is done there (it renders); the result comes back as a new
        version with the error before and after in its note."""
        try:
            d = await _json(request)
            name = d.get("name", "")
            before = store().version(name)
            open_in_viewer(name)
            PromptServer.instance.send_sync("bepic.world.calibrate", {"name": name})
            return web.json_response({"requested": name, "version_before": before,
                                      "note": "the viewer saves the match as the next version"})
        except FileNotFoundError as e:
            return _err(e, 404)
        except Exception as e:
            return _err(e)

    @routes.get("/bepic_worlds/feedback")
    async def feedback_list(request):
        try:
            status = request.query.get("status") or None
            return web.json_response({"feedback": _feedback_out(
                store().feedback(request.query.get("name", ""), None if status == "all" else status))})
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/feedback")
    async def feedback_add(request):
        try:
            d = await _json(request)
            entry = store().add_feedback(d.get("name", ""), d.get("text", ""), point=d.get("point"),
                                         camera=d.get("camera"), snapshot=d.get("snapshot"),
                                         version=d.get("version"), author=d.get("author") or "user")
            return web.json_response(_feedback_out([entry])[0])
        except FileNotFoundError as e:
            return _err(e, 404)
        except Exception as e:
            return _err(e)

    @routes.post("/bepic_worlds/feedback/resolve")
    async def feedback_resolve(request):
        try:
            d = await _json(request)
            out = store().resolve_feedback(d.get("name", ""), d.get("ids") or [],
                                           reply=d.get("reply", ""), version=d.get("version"))
            if out["resolved"]:
                # The pins go from the viewer now, not on the next edit.
                open_in_viewer(d.get("name", ""), focus=False)
            return web.json_response(out)
        except Exception as e:
            return _err(e)
