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
`scatter_<type>_1`…), `refcam`, `hero`, and for an interior also `ceiling`.

**Interiors** (biome `interior` — chosen by words like garage / hall / indoor, or
when the depth map shows a ceiling): `terrain` is a flat floor, `ceiling` a
second terrain turned over at the measured height and not walkable, a `column`
scatter on a grid, and overhead light without shadows in place of a sun.

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
| `render.tone` | tone curve: `neutral` (keeps colours; default for worlds), `aces`, `agx`, `linear` |
| `render.exposure` | multiplies the viewer's own exposure |
| `render.bloom` | glow strength around bright things (lamps, the sun), 0 = off |
| `render.reflections` | `capture`: reflections and image light taken from the world itself at the walk spawn; `off` |
| `render.fill` | share of the hemisphere fill kept once reflections carry light too (0.35 outdoors, 0.8 indoors) |

## `terrain` (id `terrain`)
| field | meaning |
|---|---|
| `terrain.heightmap` | file; `encoding` `rg16` (height = (R·256+G)/65535) or `gray` |
| `terrain.seed`, `terrain.roughness` | the procedural shape, when there is no heightmap |
| `terrain.size` | `[x, z]` metres, centred on the item |
| `terrain.height` | metres at heightmap value 1 |
| `terrain.segments` | mesh resolution per side (≤ 512) |
| `terrain.layers[0..3]` | `{name, src?, color, tile, normal?, rough?, roughness, normalScale, baked}` — ground, rock, cliff, peak; `tile` = metres per texture repeat; PBR: a normal map, a roughness map, the measured roughness (0.02 mirror … 1 matte), relief strength, and `baked` 0..1: how much of the texture's photographed light it keeps as glow |
| `terrain.rules` | `rockSlope: [from, to]`, `cliffSlope: [from, to]` (degrees), `peak: [from, to]` (0..1 of height) |
| `terrain.splat` | optional RGBA image: layer weights, overrides `rules` |
| `terrain.walkable` | `false` for a surface you don't stand on (an interior's ceiling: a terrain turned over, rotation `[180, 0, 0]`) |

## `scatter` (ids `scatter_*`)
| field | meaning |
|---|---|
| `scatter.target` | terrain id to grow on (`terrain`) |
| `scatter.source.type` | `pine`, `tree`, `bush`, `grass`, `rock`, `column`, or `model` (+ `source.src`) |
| `scatter.count` | instances, 0 … 200000 |
| `scatter.seed` | placement seed |
| `scatter.scale` | `[min, max]` size multiplier |
| `scatter.layer` | terrain layer it grows on (0 ground, 1 rock…); −1 anywhere |
| `scatter.maxSlope` | degrees |
| `scatter.color` | tint |
| `scatter.wind` | sway strength, 0 = still |
| `scatter.clear` | kept empty: circles `{center: [x, z], radius}` and wedges `{wedge: {apex: [x, z], yaw, half, range}}` (a view a picture already covers; yaw 0 = −Z, `half` degrees each side), one or a list |
| `scatter.grid` | `[dx, dz]` metres: a regular layout instead of random (columns in a hall) |
| `scatter.aspect` | stretches the height alone (a unit column × the room height) |
| `scatter.lift` | metres above the ground (lamps hung under a ceiling) |
| `scatter.emissive` | glow strength (a `lamp`) |
| `scatter.roughness` | surface roughness, or unset for the type's own |

## `light` (ids `light_*`)
A point light and its fixture. `light.color`, `light.intensity` (candela, ~18 for a strip light),
`light.distance` (reach, m), `light.decay` (2 = physical), `light.shadows`,
`light.fixture` `{shape: tube|panel|none, size: [length, width, thickness], emissive}`.
Interiors get one per lamp found in the picture, hung where the ray through it meets the ceiling,
plus a `scatter_lamp` grid of glowing fixtures under the rest of the ceiling.

## `depthmesh` (id `hero`)
The reference picture pushed out by its depth map, drawn unlit, standing where
the reference camera stands.
`depthmesh.src` (picture), `depthmesh.depth` (map: bright = near unless `invert`),
`fov` (vertical, degrees), `near`, `far` (metres the map's ends mean),
`cut` (tear threshold at depth edges, 0.05–0.3), `segments`, `encoding` (`rg16` 16-bit, or `gray`),
`curve` (optional `[[d, metres], …]`, interpolated in inverse depth, where the map isn't one straight line).

## `model` (ids `<label>_<n>`, from `add_asset`)
A generated mesh (image-to-3D) standing where the picture shows the object.
`src` (a GLB in the world's `assets/models/`, normalised to one unit tall on
y = 0 and textured from the object's crop), `scale` = its height in metres,
`rotation[1]` = yaw (facing the reference camera unless given),
`from_picture` = the box `[x0, y0, x1, y1]` (0..1) it was placed from.

## Real assets from the picture
The routes an agent strings together (all POST, all JSON):
1. `stage_reference {name}` → the reference as a ComfyUI input image, for SAM3.
2. SAM3 on it with the prompt `"<label>:N"` (up to N instances), one mask per instance.
3. `object_crops {name, label, masks, limit?, crops?}` → the instances, best first
   (whole, unoccluded, big), each with `bbox`, `score`, `placement` and, for the
   first `crops`, a `crop` (on white, for image-to-3D) and a `texture` (RGBA).
4. `fit_camera {name, objects: [{bbox, height}]}` → the camera tilt that makes
   objects of known height (cars 1.5 m) come out that tall; `rebuild {name, pitch}`
   when it differs from the world's (a rebuild remakes the world from the files it
   was made from — objects and materials added since are not carried over).
5. Image-to-3D on the best `crop`, then `edit` with `add_asset {glb, texture, label, bboxes}`.
6. `material_crop {name, masks | box, label?}` → a patch of a surface for a material
   model (Chord): from SAM3's masks of the surface the clearest near square is cut;
   `edit` with `set_material {layer, albedo, normal, roughness}`.

Depth for `create` comes best from the **bEpic World Depth (16-bit)** node: 8-bit
depth leaves the far end of a picture only a few steps and the hero view terraces.
Generated meshes are decimated to ~40k triangles; the faces the picture never saw
take a blurred copy of the crop.

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

## Matching the reference
`calibrate_world` / the viewer's **Match** writes these fields and nothing else:
`env` `render.exposure`, `ambient.intensity`, `sun.intensity`, `fog.density`, and
each terrain's `terrain.layers.0.baked` — as one `edit` whose note reads
`matched to the reference: error A → B (N% less); brightness vs picture: top …, middle …, bottom …`.

## Feedback entries (`feedback.jsonl`)
```json
{"id": "fb_1a2b3c4d", "n": 1, "created": "…", "version": 2, "author": "user",
 "text": "too many trees on the left", "status": "open",
 "point": [3.0, 1.2, -10.0],
 "camera": {"position": [x, y, z], "yaw": 12.0, "pitch": -4.0, "fov": 50},
 "snapshot": "<world>/feedback/fb_1a2b3c4d.jpg",
 "resolved_in": 3, "reply": "thinned the pines on that slope"}
```
