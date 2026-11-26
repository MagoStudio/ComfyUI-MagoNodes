import logging

from .wan_tiled_sampler import (
    NODE_CLASS_MAPPINGS as WAN_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as WAN_DISPLAY_MAPPINGS,
)
from .ltx_tiled_sampler import (
    NODE_CLASS_MAPPINGS as LTX_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LTX_DISPLAY_MAPPINGS,
)
from .conditioning_cache import (
    NODE_CLASS_MAPPINGS as COND_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COND_DISPLAY_MAPPINGS,
)
from .wan_kijai_tiled_patch import (
    NODE_CLASS_MAPPINGS as WAN_KJ_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as WAN_KJ_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **WAN_CLASS_MAPPINGS, **LTX_CLASS_MAPPINGS, **COND_CLASS_MAPPINGS, **WAN_KJ_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **WAN_DISPLAY_MAPPINGS, **LTX_DISPLAY_MAPPINGS, **COND_DISPLAY_MAPPINGS, **WAN_KJ_DISPLAY_MAPPINGS,
}

# Older ComfyUI versions can still load the pack's other nodes.
try:
    from .sol3_attention import MagoSol3Attention
except ImportError as exc:
    if exc.name not in {"comfy_api", "comfy_api.latest", "comfy.patcher_extension"}:
        raise
    logging.getLogger(__name__).warning("Update ComfyUI to enable Sol³ Sparse Attention (%s).", exc)
else:
    NODE_CLASS_MAPPINGS["MagoSol3Attention"] = MagoSol3Attention
    NODE_DISPLAY_NAME_MAPPINGS["MagoSol3Attention"] = "Sol³ Sparse Attention (Mago)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
