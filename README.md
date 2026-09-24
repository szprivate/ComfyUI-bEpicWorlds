# ComfyUI-bEpicWorlds

Walkable, game-like worlds (no game actions) built from reference images — by
a ComfyUI node, or by an agent on request — and improved from the feedback you
pin to them while walking them in the **bEpic Image Viewer**.

```
reference image ─► analysis ─► world (sky, sun, fog, terrain, textures, vegetation, reference camera)
      + spec           │                          │
 ("misty pine          ▼                          ▼
  forest, sunset")  output/worlds/<name>/  ──►  bEpic viewer: walk it, pin notes
                                                          │
                    agent: get_feedback ◄─────────────────┘
                           edit_world ──► version n+1 ──► viewer refreshes
```

Three ways in, one library underneath (`bepic_worlds`, numpy + Pillow):

| | for | how |
|---|---|---|
| **MCP server** | agents (agentY, Claude Code, any MCP client) | `python -m bepic_worlds.mcp_server` |
| **CLI** | agents that run shell commands, scripts | `python -m bepic_worlds …` |
| **ComfyUI nodes** | workflows | *bEpic World From Reference*, *bEpic World Edit*, *bEpic World Feedback*, *bEpic World Depth (16-bit)* |

Worlds are **shown** by the bEpic Image Viewer (`ComfyUI-ImageViewer`,
master): walk mode, the reference overlay, Match, and feedback pins live there.

## Install

In `ComfyUI/custom_nodes/` (restart ComfyUI once):

```bash
git clone <this repo> ComfyUI-bEpicWorlds
```

For the MCP server, the `mcp` package (already in most ComfyUI environments):
`pip install mcp`.

## Using it from an agent

Register the MCP server with ComfyUI's Python, so it has numpy and Pillow:

```json
{
  "mcpServers": {
    "bepic-worlds": {
      "command": "D:/ai/comfyui/.venv/Scripts/python.exe",
      "args": ["-s", "-m", "bepic_worlds.mcp_server"],
      "cwd": "D:/AI/comfyui/custom_nodes/ComfyUI-bEpicWorlds",
      "env": {"COMFYUI_URL": "http://127.0.0.1:8188"}
    }
  }
}
```

With ComfyUI running, the server works **through ComfyUI** (the `/bepic_worlds/*`
routes): worlds land in ComfyUI's own output folder — wherever a launcher put it
— and open in the viewer you are looking at. A reference image on the agent's
disk is uploaded to ComfyUI first. Without ComfyUI it falls back to building
worlds in a local folder (`--root`, or `$BEPIC_WORLDS_ROOT`).

Tools: `world_schema`, `create_world`, `list_worlds`, `describe_world`,
`get_world_json`, `edit_world`, `revert_world`, `calibrate_world`, `get_feedback`
(returns your snapshots as images), `resolve_feedback`, `open_in_viewer`.

`calibrate_world` matches a world's look to its reference by measurement: the
viewer renders the world from the reference camera (without the picture's own
depth mesh), compares it with the picture on a 3 x 3 grid — median brightness,
colour balance, spread — and sets exposure, fill, sun, fog and each surface's
baked light for the least error. The result is saved as a new version whose
note gives the error before and after. The same runs from the viewer's
**Match** button. It needs the world open in a browser (it is opened for you).

### The loop
1. You ask the agent for a world ("a misty pine valley like this photo").
2. It calls `create_world`; the viewer opens a **World: name** tab.
3. You walk it (**Walk** in the previz toolbar), and press **Note** to pin
   notes to places: "fewer trees here", "sky too saturated". Each note keeps the
   spot, your view, and a snapshot.
4. You tell the agent to look; it calls `get_feedback`, sees what you saw,
   `edit_world`s a new version with a note, and `resolve_feedback`s with a reply.
5. The viewer refreshes; resolved pins disappear; every version is kept
   (`revert_world` goes back).

## CLI

```bash
python -m bepic_worlds create ref.jpg --name valley --spec "misty pine forest at sunset"
python -m bepic_worlds describe valley
python -m bepic_worlds edit valley '[{"op": "scale_scatter", "factor": 0.5}]' --note "half the trees"
python -m bepic_worlds feedback valley
python -m bepic_worlds resolve valley fb_1a2b3c4d --reply "thinned them"
python -m bepic_worlds calibrate valley          # match the look to the reference (needs the viewer open)
python -m bepic_worlds schema
```

## What's in a world

See [SCHEMA.md](SCHEMA.md) — item kinds, ids, fields, units, and every edit
operation. In short: `env` (sky, sun, fog), `terrain` (heightmap, four texture
layers blended by slope and height), `scatter_*` (pines, trees, bushes, grass,
rocks, or a model; wind), `refcam` (the reference camera, with the picture
overlaid), `hero` (the reference pushed into 3D by a depth map, when given).

What comes from the **picture**: sky and horizon colours, the horizon line →
camera tilt, the sun (when it is in frame, else a photographer's guess), fog
density, ground textures cut from the picture itself, foliage tint. What comes
from the **spec**: biome, time of day, density, terrain height, what grows.

## Routes (ComfyUI)

`GET  /bepic_worlds/info · /list · /world?name=[&version=][&summary=1] · /feedback?name=&status=`
`POST /bepic_worlds/create · /rebuild · /edit · /revert · /open · /calibrate · /feedback · /feedback/resolve`
`POST /bepic_worlds/stage_reference · /object_crops · /fit_camera · /material_crop` (real assets from the picture — see SCHEMA.md)

Everything that changes something is POST; worlds live under `output/worlds`
and are reached by sanitised name only; reference images must be in ComfyUI's
input, output or temp folder; no route starts a process.

## agentY

agentY has a **World Builder** specialist (`run_world_builder`) that drives all of
this — including the real-asset pipelines (SAM3 → Hunyuan3D 2.1 → `add_asset`,
SAM3 → Chord → `set_material`) — from its `src/tools/worlds.py`.

## Status

First pass. Planned: multiple references (one per region or camera), specs
written by a VLM node, props from image-to-3D, Gaussian-splat vistas (via
ComfyUI-Sharp), video textures for ambient motion, streaming for large worlds.
See `docs/worlds/PLAN.md` in the viewer repo.
