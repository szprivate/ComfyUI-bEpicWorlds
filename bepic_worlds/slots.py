"""The generative steps of world building, and which workflow does each.

A *slot* is one step — depth, segmentation, image → 3D, texture refinement,
texture generation, PBR material, sky, object picture. Each is done by a
ComfyUI workflow in API format whose nodes are titled for the slot:

- ``IN:<name>`` on a LoadImage: an input image;
- ``IN:<name>.<field>`` on any node: sets that node's input ``field`` (a
  prompt's text, a seed, a denoise …). Several nodes may carry the same name,
  and one node several, separated by ``|``: ``IN:denoise.denoise|seed.seed``;
- ``OUT:<name>`` on a save node (SaveImage, SaveGLB …): an output.

The pack ships one or more workflows per slot (``slots/*.json``, listed in
``slots/slots.json`` with a default). A choice made per slot — one of those,
or the name of any template in an agent's library, titled the same way — is
kept in ``<worlds root>/slots.json``. Running the workflows is the caller's
business (an agent, through ComfyUI's /prompt); the pack only says what they
are.
"""

import json
import os

SLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slots")


def _read(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def manifest():
    data = _read(os.path.join(SLOTS_DIR, "slots.json"), {}) or {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def template(name):
    name = os.path.basename(str(name or ""))
    wf = _read(os.path.join(SLOTS_DIR, name + ".json")) if name and name != "slots" else None
    if wf is None:
        raise FileNotFoundError(f"no slot workflow '{name}' in this pack")
    return wf


def choices(root):
    """The workflow chosen for each slot (the default where none was)."""
    chosen = _read(os.path.join(root, "slots.json"), {}) or {}
    return {slot: chosen.get(slot) or spec.get("default") for slot, spec in manifest().items()}


def choose(root, slot, template_name):
    slots = manifest()
    if slot not in slots:
        raise ValueError(f"no slot '{slot}' — slots are: {', '.join(slots)}")
    path = os.path.join(root, "slots.json")
    chosen = _read(path, {}) or {}
    if template_name:
        chosen[slot] = str(template_name)
    else:
        chosen.pop(slot, None)
    os.makedirs(root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chosen, fh, indent=1)
    return choices(root)
