# World schema (v1)

A world is a scene in the bEpic viewer's previz format, plus four item kinds
and three scene fields. The viewer (`ComfyUI-ImageViewer`, `js/bEpicViewer_scene3d.js`,
`parseScene`) is the reference reader; this file is the contract.

## Units and axes
Metres. Y is up. The reference camera looks down **−Z**. Rotations are degrees,
XYZ order. A sun **azimuth** of 0 is straight ahead (−Z), +90 is to the right
(+X), 180 behind; **elevation** is degrees above the horizon.
Colours are `"#rrggbb"`. A file is `{"path": "<absolute>", "name": "...", "external": true}`.

## Every item
```json
{"id": "terrain", "kind": "terrain", "name": "Terrain",
 "position": [0,0,0], "rotation": [0,0,0], "scale": [1,1,1], "pivot": [0,0,0],
 "visible": true, "tracks": {}, "parent": null}
```
Ids made by the builder are fixed: `env`, `terrain`, `scatter_<type>` (then
`scatter_<type>_1`…), `refcam`, `hero`.

## `environment` (id `env`, one per world)
| field | meaning |
|---|---|
| `sky.mode` | `gradient` or `panorama` |
| `sky.top`, `sky.horizon`, `sky.bottom` | gradient colours |
| `sky.src` | equirectangular panorama file (mode `panorama`) |
| `sun.azimuth`, `sun.elevation` | degrees (see axes) |
| `sun.color`, `sun.intensity` | light colour; intensity ~2.5 day, ~0.3 night |
| `sun.shadows` | true/false |
| `fog.color`, `fog.density` | exponential fog; 0.002 clear … 0.02 thick |
| `ambient.sky`, `ambient.ground`, `ambient.intensity` | hemisphere fill light |

## `terrain` (id `terrain`)
| field | meaning |
|---|---|
| `terrain.heightmap` | file; `encoding` `rg16` (height = (R·256+G)/65535) or `gray` |
| `terrain.seed`, `terrain.roughness` | the procedural shape, when there is no heightmap |
| `terrain.size` | `[x, z]` metres, centred on the item |
| `terrain.height` | metres at heightmap value 1 |
| `terrain.segments` | mesh resolution per side (≤ 512) |
| `terrain.layers[0..3]` | `{name, src?, color, tile}` — ground, rock, cliff, peak; `tile` = metres per texture repeat |
| `terrain.rules` | `rockSlope: [from, to]`, `cliffSlope: [from, to]` (degrees), `peak: [from, to]` (0..1 of height) |
| `terrain.splat` | optional RGBA image: layer weights, overrides `rules` |

## `scatter` (ids `scatter_*`)
| field | meaning |
|---|---|
| `scatter.target` | terrain id to grow on (`terrain`) |
| `scatter.source.type` | `pine`, `tree`, `bush`, `grass`, `rock`, or `model` (+ `source.src`) |
| `scatter.count` | instances, 0 … 200000 |
| `scatter.seed` | placement seed |
| `scatter.scale` | `[min, max]` size multiplier |
| `scatter.layer` | terrain layer it grows on (0 ground, 1 rock…); −1 anywhere |
| `scatter.maxSlope` | degrees |
| `scatter.color` | tint |
| `scatter.wind` | sway strength, 0 = still |
| `scatter.clear` | `{center: [x, z], radius}` or a list of them: kept empty |

## `depthmesh` (id `hero`)
The reference picture pushed out by its depth map, drawn unlit, standing where
the reference camera stands.
`depthmesh.src` (picture), `depthmesh.depth` (map: bright = near unless `invert`),
`fov` (vertical, degrees), `near`, `far` (metres the map's ends mean),
`cut` (tear threshold at depth edges, 0.05–0.3), `segments`.

## Cameras
Standard previz camera (`fov` vertical degrees, `resolution` `[w, h]`) plus
`reference: {src, opacity, wipe}` — the picture this view must match, laid over
the camera's gate when looking through it (`wipe` 0..1 is how much of the width
it covers).

## Scene fields
```json
"walk":  {"spawn": [x, y, z], "yaw": 0, "eyeHeight": 1.7, "speed": 5,
          "bounds": {"center": [x, z], "radius": 110}},
"world": {"name": "valley", "version": 3, "schema": 1,
          "feedback": [{"id": "fb_…", "n": 1, "text": "…", "point": [x, y, z]}]}
```
`world.feedback` is added when a world is served to the viewer (open notes, as
pins); it is not stored in `world.json`.

## Feedback entries (`feedback.jsonl`)
```json
{"id": "fb_1a2b3c4d", "n": 1, "created": "…", "version": 2, "author": "user",
 "text": "too many trees on the left", "status": "open",
 "point": [3.0, 1.2, -10.0],
 "camera": {"position": [x, y, z], "yaw": 12.0, "pitch": -4.0, "fov": 50},
 "snapshot": "<world>/feedback/fb_1a2b3c4d.jpg",
 "resolved_in": 3, "reply": "thinned the pines on that slope"}
```
