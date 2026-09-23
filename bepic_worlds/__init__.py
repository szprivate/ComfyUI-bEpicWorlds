"""bEpic Worlds: walkable worlds from reference images.

    from bepic_worlds import WorldStore
    store = WorldStore()                       # <ComfyUI output>/worlds
    store.create("ref.jpg", name="valley", spec="misty pine forest at sunset")
    store.edit("valley", [{"op": "scale_scatter", "factor": 0.5}], note="fewer trees")
    store.feedback("valley", "open")

Plain Python (numpy + Pillow). The ComfyUI node pack around it, the MCP
server (bepic_worlds.mcp_server) and the CLI (python -m bepic_worlds) are thin
layers over the same WorldStore.
"""

from .store import WorldStore, OPS, default_root, safe_name
from .builder import build_world
from .reference import analyze

__version__ = "0.1.0"
__all__ = ["WorldStore", "OPS", "default_root", "safe_name", "build_world", "analyze"]
