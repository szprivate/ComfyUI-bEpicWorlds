"""MCP server: the world builder as tools for an agent.

Run it over stdio:
    python -m bepic_worlds.mcp_server              # ComfyUI at $COMFYUI_URL, else local
    python -m bepic_worlds.mcp_server --root D:/worlds --mode local

The loop the tools are shaped for:
    create_world  →  (the user walks it in the bEpic viewer and leaves notes)
    get_feedback  →  edit_world  →  resolve_feedback  →  (the user looks again)

Needs the `mcp` package (pip install mcp). Everything else is in this package.
"""

import argparse
import json

from mcp.server.fastmcp import FastMCP, Image

from . import client

mcp = FastMCP("bepic-worlds", instructions=(
    "Build walkable 3D worlds from reference images and improve them from the user's "
    "feedback. Worlds open in the bEpic Image Viewer inside ComfyUI, where the user walks "
    "them (walk mode) and pins notes to places. Call world_schema once to learn the item "
    "ids, fields and edit operations. Typical loop: create_world → wait for the user → "
    "get_feedback → edit_world (one version per round, with a note) → resolve_feedback "
    "with a short reply saying what changed."))

_backend = None


def be():
    global _backend
    if _backend is None:
        _backend = client.backend()
    return _backend


def _dump(obj):
    return json.dumps(obj, indent=1, default=str)


@mcp.tool()
def world_schema() -> str:
    """The world format: item kinds and ids (env, terrain, scatter_*, refcam, hero), their
    fields and units, and every edit operation with its arguments. Read before editing."""
    return client.schema_text()


@mcp.tool()
def create_world(reference: str, name: str = "world", spec: str = "", depth: str = "",
                 heightmap: str = "", panorama: str = "", fov: float = 50.0,
                 world_size: float = 240.0, seed: int = 0, overwrite: bool = False,
                 open_in_viewer: bool = True) -> str:
    """Build a world from a reference image and (by default) open it in the viewer.

    reference: path to an image (uploaded to ComfyUI if it is on this machine), or a file
        name in ComfyUI's input folder.
    spec: what to make, in words ("misty pine forest at sunset, sparse") or JSON
        ({"biome": "alpine", "time": "morning", "terrain": {"height": 60}, "scatter": [...]}).
        Biomes: meadow, forest, hills, alpine, desert, snow, interior (car parks, halls — also chosen when the depth map shows a ceiling). Anything left out comes from
        the picture.
    depth / heightmap / panorama: optional images — a depth map of the reference (adds the
        picture as a 3D hero view), a terrain heightmap, a 360° sky.
    fov: vertical field of view of the reference, degrees. world_size: metres across.
    Returns a summary: name (may get a suffix if taken), version, items and their ids.
    """
    out = be().create(reference, name=name, spec=spec or None, depth=depth or None,
                      heightmap=heightmap or None, panorama=panorama or None, fov=fov,
                      world_size=world_size, seed=seed, overwrite=overwrite,
                      open_in_viewer=open_in_viewer)
    return _dump(out)


@mcp.tool()
def list_worlds() -> str:
    """Every world, with its current version and how many open feedback notes it has."""
    return _dump(be().list())


@mcp.tool()
def describe_world(name: str) -> str:
    """A compact description of a world: biome, sun, fog, terrain, each scatter (type, count,
    scale), cameras, walk settings, open feedback count and recent history."""
    return _dump(be().describe(name))


@mcp.tool()
def get_world_json(name: str, version: int = 0) -> str:
    """The full scene JSON of a world (a version number for an older one). Large; prefer
    describe_world unless you need an exact field."""
    return _dump(be().world_json(name, version or None))


@mcp.tool()
def edit_world(name: str, ops: str, note: str = "", open_in_viewer: bool = True) -> str:
    """Apply edit operations as ONE new version, and show it in the viewer.

    ops: a JSON list, e.g.
      [{"op": "scale_scatter", "id": "scatter_pine", "factor": 0.5},
       {"op": "set", "id": "env", "path": "fog.density", "value": 0.004},
       {"op": "clear_area", "center": [12, -30], "radius": 10},
       {"op": "set_time", "time": "sunset"},
       {"op": "regenerate_terrain", "seed": 4, "height": 45}]
    note: what this version changes and why (shown in the history).
    Fails without changing anything if any operation is invalid; the error says which.
    Feedback points are [x, y, z] in metres — clear_area takes [x, z] from them.
    """
    parsed = json.loads(ops) if isinstance(ops, str) else ops
    return _dump(be().edit(name, parsed, note=note, open_in_viewer=open_in_viewer))


@mcp.tool()
def revert_world(name: str, version: int, note: str = "") -> str:
    """Make an older version current again (as a new version — nothing is lost)."""
    return _dump(be().revert(name, version, note=note))


@mcp.tool()
def get_feedback(name: str, status: str = "open", include_snapshots: bool = True):
    """The user's notes on a world. Each has an id, a number, the text, the 3D point it was
    pinned to ([x, y, z] metres, or null), the camera it was seen from, and the version it
    was about. With include_snapshots, the picture of what the user saw comes along (up to 6).
    status: open | resolved | all."""
    entries = be().feedback(name, status)
    content = [_dump([{k: v for k, v in f.items() if k not in ("snapshot", "snapshot_view")}
                      for f in entries]) if entries else "no feedback"]
    if include_snapshots:
        for f in [f for f in entries if f.get("snapshot") or f.get("snapshot_view")][:6]:
            data = be().snapshot_bytes(f)
            if data:
                fmt = "png" if data[:4] == b"\x89PNG" else "jpeg"
                content.append(f"snapshot for {f['id']} (#{f.get('n')}):")
                content.append(Image(data=data, format=fmt))
    return content


@mcp.tool()
def resolve_feedback(name: str, ids: list[str], reply: str = "") -> str:
    """Mark notes as dealt with (their pins disappear from the viewer). reply: one line the
    user will read, e.g. "halved the pines on the left slope (v4)"."""
    return _dump(be().resolve(name, ids, reply=reply))


@mcp.tool()
def calibrate_world(name: str, wait_seconds: float = 60.0) -> str:
    """Match the world's look to its reference picture, by measurement: the viewer renders the
    world from the reference camera, compares it region by region with the picture
    (brightness, colour, contrast) and sets exposure, fill light, sun, fog and each surface's
    baked light for the least error. Saved as a new version whose note gives the error before
    and after and the brightness of each band against the picture. Needs the world open in a
    viewer (it is opened for you); waits up to wait_seconds for the result."""
    return _dump(be().calibrate(name, wait_seconds))


@mcp.tool()
def open_in_viewer(name: str, version: int = 0) -> str:
    """Open (or refresh) a world's tab in the bEpic viewer. Needs ComfyUI running."""
    return _dump(be().open(name, version or None))


def main(argv=None):
    global _backend
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=None, help="ComfyUI URL (default $COMFYUI_URL or http://127.0.0.1:8188)")
    ap.add_argument("--root", default=None, help="worlds folder, for local mode")
    ap.add_argument("--mode", choices=["auto", "http", "local"], default=None)
    args = ap.parse_args(argv)
    _backend = client.backend(url=args.url, root=args.root, mode=args.mode)
    mcp.run()


if __name__ == "__main__":
    main()
