"""bEpic Worlds — walkable worlds from reference images, for ComfyUI and agents.

The library is `bepic_worlds` (plain Python); this file makes it a ComfyUI
node pack: three nodes, and the /bepic_worlds/* routes the viewer and the
agent tools use. The bEpic Image Viewer (ComfyUI-ImageViewer, `worlds`
branch) is what shows and walks the worlds.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import server_routes

try:
    server_routes.register()
except Exception as e:                      # no server (tests, a CLI import)
    print(f"[bEpicWorlds] routes not registered: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
