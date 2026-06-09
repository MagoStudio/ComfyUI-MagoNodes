import os
import torch
import folder_paths


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning cache — save / load CONDITIONING to disk
#
# For workflows with a FIXED prompt (e.g. LTX HDR conversion), bake the text
# conditioning once with SaveConditioning, then in the production graph reuse it
# with LoadConditioning and delete the text-encoder nodes entirely.  That frees
# the (large) text encoder from VRAM — only the diffusion model + VAE need to
# load.  Cache the conditioning BEFORE any per-input guide node
# (e.g. LTXAddVideoICLoRAGuide), since the guide changes every run.
# ─────────────────────────────────────────────────────────────────────────────

_SUBDIR = "conditioning"
_PLACEHOLDER = "<no cached conditioning — run Save first>"


def _cache_dir():
    d = os.path.join(folder_paths.get_output_directory(), _SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _to_cpu(obj):
    """Recursively move tensors to CPU so the saved file is portable (no GPU /
    device coupling).  Handles the CONDITIONING structure: list[[tensor, dict]]."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _resolve_name(filename):
    name = os.path.basename((filename or "").strip()) or "conditioning"
    if not name.endswith(".pth"):
        name += ".pth"
    return name


class MagoSaveConditioning:
    """
    Save a CONDITIONING to <output>/conditioning/<filename>.pth and pass it
    through unchanged.  Use it inline or as a terminal node to bake a fixed
    prompt once.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename": ("STRING", {
                    "default": "prompt",
                    "tooltip": "Base name for the cache file (saved under "
                               "output/conditioning/<name>.pth). Use distinct names "
                               "for positive and negative, e.g. hdr_pos / hdr_neg.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Mago Nodes/Conditioning"

    def save(self, conditioning, filename):
        path = os.path.join(_cache_dir(), _resolve_name(filename))
        torch.save(_to_cpu(conditioning), path)
        print(f"[Mago] Saved CONDITIONING → {path}")
        return (conditioning,)


class MagoLoadConditioning:
    """
    Load a CONDITIONING previously saved with MagoSaveConditioning.  Lets a
    production graph skip the text encoder entirely.
    """

    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(_cache_dir()) if f.endswith(".pth")]
        return {
            "required": {
                "filename": (sorted(files) if files else [_PLACEHOLDER],),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "Mago Nodes/Conditioning"

    @classmethod
    def IS_CHANGED(s, filename):
        path = os.path.join(_cache_dir(), filename)
        return os.path.getmtime(path) if os.path.exists(path) else float("nan")

    def load(self, filename):
        if filename == _PLACEHOLDER:
            raise FileNotFoundError(
                "No cached conditioning found. Run MagoSaveConditioning first, "
                "then refresh the node list.")
        path = os.path.join(_cache_dir(), os.path.basename(filename))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cached conditioning not found: {path}")
        # weights_only=False: CONDITIONING dicts hold non-tensor metadata; these
        # are your own locally-generated files.
        cond = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[Mago] Loaded CONDITIONING ← {path}")
        return (cond,)


NODE_CLASS_MAPPINGS = {
    "MagoSaveConditioning": MagoSaveConditioning,
    "MagoLoadConditioning": MagoLoadConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagoSaveConditioning": "Save Conditioning (Mago)",
    "MagoLoadConditioning": "Load Conditioning (Mago)",
}
