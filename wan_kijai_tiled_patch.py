"""
WAN Tiled Patch for kijai's WanVideoWrapper (WANVIDEOMODEL → WANVIDEOMODEL).

Companion to ``wan_tiled_sampler.py``.  That module tiles native ComfyUI WAN
(``MODEL``) by installing ``model_options["model_function_wrapper"]`` — a hook
kijai's ``WanVideoSampler`` never fires (it runs its own sampling loop over its
own ``WANVIDEOMODEL`` type; there is no ``MODEL``↔``WANVIDEOMODEL`` bridge).

This node brings the same per-step MultiDiffusion spatial tiling to the kijai
stack by wrapping the transformer's ``forward`` directly.  kijai's sampler grabs
``transformer = patcher.model.diffusion_model`` once and calls it for every model
evaluation; we wrap that ``forward`` so each call is split into overlapping
spatial tiles, evaluated at a training-distribution token count, and blended back
with trapezoidal windows — exactly the algorithm in ``wan_tiled_sampler``'s
``_run_tiled``, adapted to kijai's forward contract.

Why this is enough for tiling to fire (and correct):
  * RoPE self-corrects.  ``grid_sizes`` is computed *inside* forward from the
    (sliced) latent's shape, and ``freqs`` is a fixed 1024-length bank reused
    unchanged across tiles (spatial axes are not RIFLEX-scaled), so a smaller
    tile simply indexes fewer positions.
  * ``seq_len`` is recomputed per tile — fewer tokens is the whole point.
  * ``vace_data`` context is sliced spatially and circular-padded to the patch
    size, the same token-count fix ``wan_tiled_sampler`` applies to VACE.
    ``forward_vace`` then force-aligns vace tokens to the tile's latent token
    count, so slicing + padding is sufficient.

Tiling is done in patch units (latent // 2) and scaled back by 2, so every tile
boundary is even: the transformer's stride-2 patch conv floors, and
``unpatchify`` reconstructs ``grid_sizes * patch_size`` — an odd tile extent
would return a latent one pixel short and break the blend.

v1 scope (VACE inpainting, the wan22fun use case):
  * Slices ``x`` (latent), ``y`` (image cond) and ``vace_data`` spatially.
  * Teacache must be OFF — per-tile forwards each carry a distinct token layout,
    so a shared ``pred_id`` cache is invalid; ``pred_id`` is forced to ``None``.
  * If any other spatially-shaped conditioning (``attn_cond``, ``fun_ref``,
    ``add_cond``, controlnet, camera…) is present and non-None, the wrapper falls
    back to a single full-frame forward and logs once, rather than silently
    corrupting it.  Extend the slicing table below to support those paths.
  * Not combined with kijai temporal context-windows in the same pass.
"""

import math
import torch

from .wan_tiled_sampler import _compute_tile_starts, _make_window_1d


_WAN_PATCH_SIZE_SPATIAL = 2  # WAN spatial patch size (temporal is 1)

# Conditioning kwargs that are spatially shaped and NOT handled by the tile
# slicer yet.  If any is present and non-None we skip tiling for that call.
_UNSUPPORTED_SPATIAL_CONDS = (
    "attn_cond", "fun_ref", "fun_camera", "add_cond", "controlnet",
    "camera_embed", "uni3c_data", "s2v_pose", "wananim_pose_latents",
)


def _build_even_tile_specs(H, W, tiles_h, tiles_w, overlap_h, overlap_w, debug=False):
    """Per-tile latent-space bounds + trapezoidal blend windows, with every
    boundary forced even by tiling the patch grid (H//2, W//2) and scaling ×2.

    Windows are linear-trapezoidal and sum to 1.0 everywhere (same construction
    as ``wan_tiled_sampler``), so blended predictions are an exact partition."""
    Hp, Wp = H // 2, W // 2

    def _pct_to_patches(total_p, n_tiles, pct):
        if n_tiles <= 1 or pct == 0:
            return 0
        approx = math.ceil(total_p / n_tiles)
        return max(0, round(approx * pct / 100))

    oh_p = _pct_to_patches(Hp, tiles_h, overlap_h)
    ow_p = _pct_to_patches(Wp, tiles_w, overlap_w)
    h_starts_p, tile_hp = _compute_tile_starts(Hp, tiles_h, oh_p)
    w_starts_p, tile_wp = _compute_tile_starts(Wp, tiles_w, ow_p)

    if debug:
        print(f"[WanKJTiled] latent {H}x{W} -> patch grid {Hp}x{Wp} "
              f"tiles_h={tiles_h} tiles_w={tiles_w} "
              f"h_starts_p={h_starts_p} tile_hp={tile_hp} "
              f"w_starts_p={w_starts_p} tile_wp={tile_wp}")

    specs = []
    for hi, hs_p in enumerate(h_starts_p):
        he_p = min(hs_p + tile_hp, Hp)
        fade_l_h = max(0, h_starts_p[hi - 1] + tile_hp - hs_p) if hi > 0 else 0
        fade_r_h = max(0, hs_p + tile_hp - h_starts_p[hi + 1]) if hi < len(h_starts_p) - 1 else 0
        for wi, ws_p in enumerate(w_starts_p):
            we_p = min(ws_p + tile_wp, Wp)
            fade_l_w = max(0, w_starts_p[wi - 1] + tile_wp - ws_p) if wi > 0 else 0
            fade_r_w = max(0, ws_p + tile_wp - w_starts_p[wi + 1]) if wi < len(w_starts_p) - 1 else 0

            # latent coords (even by construction)
            hs, he = hs_p * 2, he_p * 2
            ws, we = ws_p * 2, we_p * 2
            # fades are in patch units → ×2 for the latent-space window
            win_h = _make_window_1d(he - hs, fade_l_h * 2, fade_r_h * 2, torch.float32, "cpu")
            win_w = _make_window_1d(we - ws, fade_l_w * 2, fade_r_w * 2, torch.float32, "cpu")
            window = (win_h[:, None] * win_w[None, :]).reshape(1, 1, he - hs, we - ws)
            specs.append({"h_start": hs, "h_end": he, "w_start": ws, "w_end": we,
                          "window": window})

    if debug:
        wt = torch.zeros(1, 1, H, W, dtype=torch.float32)
        for s in specs:
            wt[:, :, s["h_start"]:s["h_end"], s["w_start"]:s["w_end"]] += s["window"]
        mn, mx = wt.min().item(), wt.max().item()
        ok = abs(mn - 1.0) < 0.02 and abs(mx - 1.0) < 0.02
        print(f"[WanKJTiled] blend weight sanity: min={mn:.4f} max={mx:.4f} "
              f"({'OK' if ok else 'WARN'})")
    return specs


def _slice_vace_tile(vace_data, hs, he, ws, we, tile_seq_len, debug=False):
    """Slice each vace entry's context (list of [Cv, F, H, W]) to the tile and
    circular-pad H/W up to the spatial patch size, mirroring wan_tiled_sampler's
    VACE handling.  Sets the entry's ``seq_len`` to the tile's token count; keeps
    ``scale``/``start``/``end``.  Returns a new list (originals untouched)."""
    out = []
    for entry in vace_data:
        ctx_list = entry["context"]
        new_ctx = []
        for ctx in ctx_list:
            if isinstance(ctx, torch.Tensor) and ctx.dim() == 4 \
                    and ctx.shape[-2] >= he and ctx.shape[-1] >= we:
                tile = ctx[:, :, hs:he, ws:we]
                ph = (_WAN_PATCH_SIZE_SPATIAL - tile.shape[-2] % _WAN_PATCH_SIZE_SPATIAL) % _WAN_PATCH_SIZE_SPATIAL
                pw = (_WAN_PATCH_SIZE_SPATIAL - tile.shape[-1] % _WAN_PATCH_SIZE_SPATIAL) % _WAN_PATCH_SIZE_SPATIAL
                if ph or pw:
                    tile = torch.nn.functional.pad(tile, (0, pw, 0, ph), mode="circular")
                new_ctx.append(tile.contiguous())
            else:
                # shape doesn't match the latent grid → pass through unchanged
                new_ctx.append(ctx)
                if debug:
                    shp = tuple(ctx.shape) if isinstance(ctx, torch.Tensor) else type(ctx).__name__
                    print(f"[WanKJTiled] vace ctx not sliced (shape {shp})")
        out.append({**entry, "context": new_ctx, "seq_len": tile_seq_len})
    return out


def _make_tiled_forward(transformer):
    """Return a ``forward`` replacement bound to ``transformer`` that performs
    per-call spatial MultiDiffusion, reading its live config from
    ``transformer._mago_tile_cfg`` and delegating to the original forward stored
    at ``transformer._mago_orig_forward``."""
    warned = {"unsupported": False, "teacache": False}

    def tiled_forward(*args, **kwargs):
        orig = transformer._mago_orig_forward
        cfg = getattr(transformer, "_mago_tile_cfg", None)

        # positional-arg safety: kijai always calls forward with keywords, but
        # if anything calls positionally we can't safely tile → passthrough.
        if cfg is None or args or "x" not in kwargs:
            return orig(*args, **kwargs)

        tiles_h, tiles_w = cfg["tiles_h"], cfg["tiles_w"]
        debug = cfg["debug"]
        if cfg["bypass_tiling"] or (tiles_h <= 1 and tiles_w <= 1):
            return orig(*args, **kwargs)

        x = kwargs["x"]
        if not isinstance(x, (list, tuple)) or len(x) != 1 \
                or not isinstance(x[0], torch.Tensor) or x[0].dim() != 4:
            return orig(*args, **kwargs)  # unexpected layout → don't risk it

        # unsupported spatial conditioning present → single full-frame forward
        present = [k for k in _UNSUPPORTED_SPATIAL_CONDS if kwargs.get(k) is not None]
        if present:
            if debug or not warned["unsupported"]:
                warned["unsupported"] = True
                print(f"[WanKJTiled] unsupported spatial cond {present} present "
                      f"→ tiling skipped for this call (running full frame).")
            return orig(*args, **kwargs)

        C, F, H, W = x[0].shape
        specs = _build_even_tile_specs(H, W, tiles_h, tiles_w,
                                       cfg["overlap_h"], cfg["overlap_w"], debug=debug)
        if len(specs) <= 1:
            return orig(*args, **kwargs)

        # teacache uses pred_id to reuse a cached residual keyed on the full-frame
        # token layout — invalid per tile.  Force it off for the tiled calls.
        if kwargs.get("pred_id") is not None and not warned["teacache"]:
            warned["teacache"] = True
            print("[WanKJTiled] pred_id/teacache active → disabled for tiled forward "
                  "(per-tile token layouts are cache-incompatible).")

        y = kwargs.get("y")
        vace_data = kwargs.get("vace_data")
        vace_is_dict_list = (isinstance(vace_data, list) and len(vace_data) > 0
                             and isinstance(vace_data[0], dict))

        accum = torch.zeros(C, F, H, W, dtype=torch.float32, device=x[0].device)
        w_acc = torch.zeros(1, 1, H, W, dtype=torch.float32, device=x[0].device)

        for spec in specs:
            hs, he, ws, we = spec["h_start"], spec["h_end"], spec["w_start"], spec["w_end"]
            th, tw = he - hs, we - ws
            tile_seq_len = int(math.ceil((th * tw) / 4 * F))

            tkw = dict(kwargs)
            tkw["x"] = [x[0][:, :, hs:he, ws:we].contiguous()]
            tkw["seq_len"] = tile_seq_len
            tkw["pred_id"] = None
            if isinstance(y, (list, tuple)) and len(y) == 1 \
                    and isinstance(y[0], torch.Tensor) and y[0].dim() == 4 \
                    and y[0].shape[-2] == H and y[0].shape[-1] == W:
                tkw["y"] = [y[0][:, :, hs:he, ws:we].contiguous()]
            if vace_is_dict_list:
                tkw["vace_data"] = _slice_vace_tile(vace_data, hs, he, ws, we,
                                                    tile_seq_len, debug=debug)

            pred, _ = orig(**tkw)
            tile_pred = pred[0]
            if tile_pred.shape[-2:] != (th, tw):
                # transformer returned an unexpected spatial size → abort tiling
                # cleanly with a full-frame pass rather than a corrupt blend.
                print(f"[WanKJTiled] tile pred shape {tuple(tile_pred.shape)} != "
                      f"({th},{tw}); falling back to full forward.")
                return orig(*args, **kwargs)

            window = spec["window"].to(device=tile_pred.device, dtype=torch.float32)
            accum[:, :, hs:he, ws:we] += tile_pred.float() * window
            w_acc[:, :, hs:he, ws:we] += window

        blended = (accum / w_acc.clamp(min=1e-8)).to(x[0].dtype)
        return [blended], None

    return tiled_forward


def _install_tiled_forward(transformer, cfg):
    """Idempotently wrap ``transformer.forward`` and set the live tile config.

    kijai already mutates the transformer instance every run (block-swap flags,
    etc.), so wrapping its forward in the same spirit is safe.  Re-applying just
    updates ``_mago_tile_cfg``; ``bypass_tiling``/``tiles<=1`` make the wrapper a
    pure passthrough, so a persisted wrap never changes non-tiled behaviour."""
    if not getattr(transformer, "_mago_tiled_wrapped", False):
        transformer._mago_orig_forward = transformer.forward
        transformer.forward = _make_tiled_forward(transformer)
        transformer._mago_tiled_wrapped = True
    transformer._mago_tile_cfg = cfg


class WanVideoTiledPatch:
    """kijai WanVideoWrapper patch node: adds per-step MultiDiffusion spatial
    tiling to a ``WANVIDEOMODEL`` so it can inpaint/generate above the model's
    single-pass resolution (e.g. 1920+, native-4k) while keeping large temporal
    chunks.  Plug the patched model into ``WanVideoSampler``."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "tiles_h": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                            "tooltip": "Spatial tiles along height. 1 = no height tiling."}),
                "tiles_w": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                            "tooltip": "Spatial tiles along width. 1 = no width tiling."}),
                "overlap_h": ("INT", {"default": 25, "min": 0, "max": 50, "step": 1,
                              "tooltip": "Height overlap, % of tile height (0-50)."}),
                "overlap_w": ("INT", {"default": 25, "min": 0, "max": 50, "step": 1,
                              "tooltip": "Width overlap, % of tile width (0-50)."}),
            },
            "optional": {
                "bypass_tiling": ("BOOLEAN", {"default": False,
                                  "tooltip": "Pass through unchanged (plain WanVideoSampler)."}),
                "debug": ("BOOLEAN", {"default": False,
                          "tooltip": "Print tile layout + blend-weight sanity check."}),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "patch"
    CATEGORY = "Mago Nodes/Sampling"

    def patch(self, model, tiles_h, tiles_w, overlap_h, overlap_w,
              bypass_tiling=False, debug=False):
        cfg = {"tiles_h": tiles_h, "tiles_w": tiles_w,
               "overlap_h": overlap_h, "overlap_w": overlap_w,
               "bypass_tiling": bypass_tiling, "debug": debug}
        m = model.clone()
        _install_tiled_forward(m.model.diffusion_model, cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {"WanVideoTiledPatch": WanVideoTiledPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVideoTiledPatch": "WAN Video Tiled Patch (kijai)"}
