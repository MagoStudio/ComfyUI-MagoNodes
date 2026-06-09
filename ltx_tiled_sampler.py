import math
import torch


# ─────────────────────────────────────────────────────────────────────────────
# LTX Tiled Sampler — model patch (MODEL → MODEL)
#
# Per-step MultiDiffusion spatial/temporal tiling for LTX-Video, installed as a
# model_function_wrapper so it works with the vanilla SamplerCustomAdvanced (plug
# the patched MODEL into your guider).  The per-tile guide-conditioning adaptation
# (keyframe_idxs / guide_attention_entries cropping) is based on the 10S Nodes LTX
# tiled sampler, but with the coordinate/grid-mask handling reworked here to fix
# correctness bugs in that implementation.  The trapezoidal blend window, the
# Stage-2 tiling geometry/defaults, and the causal-VAE coordinate conventions
# (temporal scale 8, spatial scale 32) follow the official Lightricks LTX-Video
# implementation (ltx_core/tiling.py).
#
# Unlike the all-in-one sampler, tile geometry is resolved lazily on the first
# model call (the patch node never sees the latent) and cached per resolution.
# ─────────────────────────────────────────────────────────────────────────────


def _compute_tile_starts(total_size, n_tiles, overlap):
    """Return (starts_list, tile_size) for a single dimension."""
    if n_tiles <= 1 or total_size <= 1:
        return [0], total_size
    tile_size = math.ceil((total_size + (n_tiles - 1) * overlap) / n_tiles)
    tile_size = min(tile_size, total_size)
    if n_tiles == 2:
        return [0, max(0, total_size - tile_size)], tile_size
    starts = []
    stride = (total_size - tile_size) / (n_tiles - 1)
    for i in range(n_tiles):
        starts.append(int(round(i * stride)))
    return starts, tile_size


def _make_window_1d(size, fade_left, fade_right, dtype, device):
    """
    Linear trapezoidal blend window matching the official LTX tiling implementation
    (compute_trapezoidal_mask_1d in ltx_core/tiling.py).  Ramps exclude exact
    0.0/1.0 endpoints so weights never reach zero inside a tile.
    """
    win = torch.ones(size, dtype=dtype, device=device)
    if fade_left > 0:
        fl = min(fade_left, size)
        win[:fl] = torch.linspace(0.0, 1.0, fl + 2, dtype=dtype, device=device)[1:-1]
    if fade_right > 0:
        fr = min(fade_right, size)
        win[size - fr:] = torch.linspace(1.0, 0.0, fr + 2, dtype=dtype, device=device)[1:-1]
    return win


def _adapt_guide_for_tile(c, t_start, t_end, h_start, h_end, w_start, w_end,
                          H_full, W_full, debug=False):
    """
    Adapt LTX guide conditioning (keyframe_idxs, guide_attention_entries) for one
    spatio-temporal tile.  Returns a new dict; does not modify c in-place.  On any
    error returns c unchanged.

    Root cause this fixes — LTX's _process_input does
        kf_grid_mask = grid_mask[-keyframe_idxs.shape[2]:]
    where grid_mask covers only the tile's tokens.  When the full-video
    keyframe_idxs token count exceeds the tile token count, the slice returns the
    whole (shorter) grid_mask → 'guide pre_filter_counts (N) != keyframe grid mask
    length (M)'.  We instead filter keyframe_idxs by the pixel coords stored in the
    tensor (axis 0=t, 1=y, 2=x) and shift survivors to tile-relative space so they
    line up with the tile's 0-indexed positional encoding.

    Coordinate conventions (LTX causal VAE): temporal scale 8, spatial scale 32.
      t_pixel_start = max(0, t_latent*8 - 7);  t_pixel_end = t_latent*8 + 1
      spatial pixel = latent_index * 32
    """
    _VAE_T = 8
    _VAE_S = 32
    _y_lo = h_start * _VAE_S
    _y_hi = h_end   * _VAE_S
    _x_lo = w_start * _VAE_S
    _x_hi = w_end   * _VAE_S
    _t_tile_size      = t_end - t_start
    _t_tile_px_offset = max(0, t_start * _VAE_T - (_VAE_T - 1))

    kf_idxs = c.get('keyframe_idxs', None)
    if kf_idxs is None:
        return c

    try:
        if kf_idxs.dim() not in (3, 4) or kf_idxs.shape[1] < 3:
            raise ValueError(f"unexpected keyframe_idxs shape {kf_idxs.shape}")

        N = kf_idxs.shape[2]

        if kf_idxs.dim() == 4:
            t_px = kf_idxs[0, 0, :, 0]
            y_px = kf_idxs[0, 1, :, 0]
            x_px = kf_idxs[0, 2, :, 0]
        else:
            t_px = kf_idxs[0, 0, :]
            y_px = kf_idxs[0, 1, :]
            x_px = kf_idxs[0, 2, :]

        t_lat_abs = (t_px.long() + _VAE_T - 1) // _VAE_T
        t_lat_rel = t_lat_abs - t_start

        if debug:
            print(f"    [ltx_guide] tile t:[{t_start},{t_end})"
                  f" h:[{h_start},{h_end}) w:[{w_start},{w_end})")

        # Apply the temporal lower bound only when the guide is LONGER than the
        # tile.  When the whole guide fits one tile's temporal capacity, dropping
        # pre-tile frames would discard conditioning the model needs (e.g. a
        # 25-frame guide on a 50-frame target).
        _guide_t_frames   = int(t_lat_abs.max().item()) + 1
        _needs_lower_bound = (_guide_t_frames > _t_tile_size)

        coord_mask = (
            (y_px >= _y_lo) & (y_px < _y_hi) &
            (x_px >= _x_lo) & (x_px < _x_hi) &
            (t_lat_rel < _t_tile_size)
        )
        if _needs_lower_bound:
            coord_mask = coord_mask & (t_lat_rel >= 0)

        if debug:
            print(f"    [ltx_guide] kf: {N} → {int(coord_mask.sum().item())} tokens selected")

        new_c = dict(c)
        if coord_mask.any():
            sel_t_abs    = t_lat_abs[coord_mask]
            sel_t_rel    = t_lat_rel[coord_mask]
            in_tile      = (sel_t_rel >= 0)
            # In-tile frames: causal tile-local coords matching the patchifier
            # ([0,1],[1,9],[9,17]…).  Pre-tile frames: absolute_pixel - tile_offset
            # (proper negative interval, the LTX has_negative_time convention).
            causal_start = (sel_t_rel * _VAE_T - (_VAE_T - 1)).clamp(min=0)
            causal_end   = sel_t_rel * _VAE_T + 1
            abs_start    = (sel_t_abs * _VAE_T - (_VAE_T - 1)).clamp(min=0) - _t_tile_px_offset
            abs_end      = sel_t_abs * _VAE_T + 1 - _t_tile_px_offset
            tile_t_start = torch.where(in_tile, causal_start, abs_start)
            tile_t_end   = torch.where(in_tile, causal_end,   abs_end)

            if kf_idxs.dim() == 4:
                new_kf = kf_idxs[:, :, coord_mask, :].clone()
                new_kf[:, 1, :, :] -= _y_lo
                new_kf[:, 2, :, :] -= _x_lo
                new_kf[:, 0, :, 0] = tile_t_start.to(new_kf.dtype)
                new_kf[:, 0, :, 1] = tile_t_end.to(new_kf.dtype)
                new_c['keyframe_idxs'] = new_kf
            else:
                new_kf = kf_idxs[:, :, coord_mask].clone()
                new_kf[:, 1, :] -= _y_lo
                new_kf[:, 2, :] -= _x_lo
                new_kf[:, 0, :] = tile_t_start.to(new_kf.dtype)
                new_c['keyframe_idxs'] = new_kf

            # Recount pre_filter_count per guide_attention_entry, dropping entries
            # whose tokens fall entirely outside this tile.
            for src_key in ('transformer_options', None):
                if src_key == 'transformer_options':
                    to = c.get('transformer_options', {})
                    guide_entries = to.get('guide_attention_entries', None)
                else:
                    guide_entries = c.get('guide_attention_entries', None)
                    to = c

                if guide_entries:
                    new_entries = []
                    offset = 0
                    n_zero = 0
                    for entry in guide_entries:
                        pfc        = entry['pre_filter_count']
                        entry_mask = coord_mask[offset:offset + pfc]
                        new_pfc    = int(entry_mask.sum().item())
                        if new_pfc > 0:
                            new_entries.append({**entry, 'pre_filter_count': new_pfc})
                        else:
                            n_zero += 1
                        offset += pfc

                    if debug and n_zero:
                        print(f"    [ltx_guide] dropped {n_zero}/{len(guide_entries)}"
                              f" zero-token entries")

                    if src_key == 'transformer_options':
                        new_to = dict(to)
                        new_to['guide_attention_entries'] = new_entries
                        new_c['transformer_options'] = new_to
                    else:
                        new_c['guide_attention_entries'] = new_entries
                    break
        else:
            # No guide tokens fall in this tile — disable guide conditioning.
            new_c.pop('keyframe_idxs', None)
            to = new_c.get('transformer_options', {})
            if 'guide_attention_entries' in to:
                new_to = dict(to)
                new_to.pop('guide_attention_entries', None)
                new_c['transformer_options'] = new_to
            new_c.pop('guide_attention_entries', None)
            if debug:
                print(f"    [ltx_guide] no guide tokens in tile — guide disabled")

        return new_c

    except Exception:
        if debug:
            import traceback
            print(f"    [ltx_guide] error — passthrough\n{traceback.format_exc()}")
        return c


def _build_ltx_tile_specs(F, H, W, tiles_t, tiles_h, tiles_w,
                          overlap_t, overlap_h, overlap_w, debug=False):
    """Build 3D (T, H, W) tile bounds + trapezoidal blend windows for a latent of
    size (F, H, W).  Pure geometry; windows are built on CPU and moved to the
    latent's device at blend time.  Cached per resolution by the wrapper."""
    t_starts, t_tile_sz = _compute_tile_starts(F, tiles_t, overlap_t)
    h_starts, h_tile_sz = _compute_tile_starts(H, tiles_h, overlap_h)
    w_starts, w_tile_sz = _compute_tile_starts(W, tiles_w, overlap_w)

    if debug:
        print(f"[LTXTiled] latent F×H×W={F}×{H}×{W}  "
              f"T:n={len(t_starts)} sz={t_tile_sz} starts={t_starts}  "
              f"H:n={len(h_starts)} sz={h_tile_sz} starts={h_starts}  "
              f"W:n={len(w_starts)} sz={w_tile_sz} starts={w_starts}")

    specs = []
    for ti, t_start in enumerate(t_starts):
        for hi, h_start in enumerate(h_starts):
            for wi, w_start in enumerate(w_starts):
                t_end = min(t_start + t_tile_sz, F)
                h_end = min(h_start + h_tile_sz, H)
                w_end = min(w_start + w_tile_sz, W)

                t_fl = (max(0, min(t_starts[ti-1] + t_tile_sz, F) - t_start) if ti > 0 else 0)
                t_fr = (max(0, t_end - t_starts[ti+1]) if ti < len(t_starts) - 1 else 0)
                h_fl = (max(0, min(h_starts[hi-1] + h_tile_sz, H) - h_start) if hi > 0 else 0)
                h_fr = (max(0, h_end - h_starts[hi+1]) if hi < len(h_starts) - 1 else 0)
                w_fl = (max(0, min(w_starts[wi-1] + w_tile_sz, W) - w_start) if wi > 0 else 0)
                w_fr = (max(0, w_end - w_starts[wi+1]) if wi < len(w_starts) - 1 else 0)

                t_win = _make_window_1d(t_end - t_start, t_fl, t_fr, torch.float32, 'cpu')
                h_win = _make_window_1d(h_end - h_start, h_fl, h_fr, torch.float32, 'cpu')
                w_win = _make_window_1d(w_end - w_start, w_fl, w_fr, torch.float32, 'cpu')
                window_3d = (t_win[:, None, None] * h_win[None, :, None] * w_win[None, None, :]) \
                    .view(1, 1, t_end - t_start, h_end - h_start, w_end - w_start)

                specs.append(dict(
                    t_start=t_start, t_end=t_end,
                    h_start=h_start, h_end=h_end,
                    w_start=w_start, w_end=w_end,
                    window_3d=window_3d, H_full=H, W_full=W,
                ))
                if debug:
                    print(f"  · tile T[{t_start}:{t_end}] H[{h_start}:{h_end}] W[{w_start}:{w_end}] "
                          f"fades T({t_fl},{t_fr}) H({h_fl},{h_fr}) W({w_fl},{w_fr})")

    if debug:
        w_check = torch.zeros(1, 1, F, H, W, dtype=torch.float32)
        for spec in specs:
            ts, te = spec['t_start'], spec['t_end']
            hs, he = spec['h_start'], spec['h_end']
            ws, we = spec['w_start'], spec['w_end']
            w_check[:, :, ts:te, hs:he, ws:we] += spec['window_3d']
        print(f"  · blend weight sanity: min={w_check.min().item():.4f} "
              f"max={w_check.max().item():.4f}")

    return specs


def _make_ltx_tile_wrapper(tiling_cfg, existing_wrapper, debug=False):
    """
    Returns a model_function_wrapper implementing per-step MultiDiffusion tiling
    for LTX.  At each model call (sigma step) it tiles the full noisy latent
    (B, C, F, H, W), adapts guide conditioning + denoise_mask per tile, runs the
    model per tile, and blends predictions back with trapezoidal windows.  Because
    blending happens every step, all tiles share one trajectory — no temporal
    ghosting from independent per-tile schedules.

    Tile geometry is built lazily from the latent size and cached per resolution.
    """
    specs_cache = {}
    logged = [False]  # one-time "wrapper invoked" confirmation

    def _get_specs(F, H, W):
        key = (F, H, W)
        if key not in specs_cache:
            specs_cache[key] = _build_ltx_tile_specs(
                F, H, W, tiling_cfg["tiles_t"], tiling_cfg["tiles_h"], tiling_cfg["tiles_w"],
                tiling_cfg["overlap_t"], tiling_cfg["overlap_h"], tiling_cfg["overlap_w"],
                debug=debug)
        return specs_cache[key]

    def _call(model_fn, d):
        if existing_wrapper is not None:
            return existing_wrapper(model_fn, d)
        return model_fn(d['input'], d['timestep'], **d.get('c', {}))

    def wrapper(model_fn, input_dict):
        x_full = input_dict['input']

        # One-time confirmation that the patch is actually being invoked by the
        # sampler.  A model patch can silently fail to propagate (no error, just
        # untiled output); with debug=True this line proves the wrapper is live —
        # and tells you if it's passing through instead of tiling.
        if debug and not logged[0]:
            logged[0] = True
            shape = tuple(x_full.shape) if isinstance(x_full, torch.Tensor) else type(x_full).__name__
            chained = " (chaining an existing wrapper)" if existing_wrapper is not None else ""
            print(f"[LTXTiled] wrapper INVOKED — first model call, input={shape}{chained}")

        if not isinstance(x_full, torch.Tensor) or x_full.dim() != 5:
            if debug:
                print(f"[LTXTiled] passthrough — input is not a 5D video latent; not tiling")
            return _call(model_fn, input_dict)  # not a 5D video latent — passthrough

        B, C, F, H, W = x_full.shape
        tile_specs = _get_specs(F, H, W)
        if len(tile_specs) <= 1:
            if debug:
                print(f"[LTXTiled] passthrough — latent {F}×{H}×{W} yields a single tile; not tiling")
            return _call(model_fn, input_dict)

        dtype  = x_full.dtype
        device = x_full.device
        accum = torch.zeros(B, C, F, H, W, dtype=torch.float32, device=device)
        w_acc = torch.zeros(1, 1, F, H, W, dtype=torch.float32, device=device)
        base_c = input_dict.get('c', {})

        for spec in tile_specs:
            ts, te = spec['t_start'], spec['t_end']
            hs, he = spec['h_start'], spec['h_end']
            ws, we = spec['w_start'], spec['w_end']
            window_3d = spec['window_3d'].to(device=device, dtype=torch.float32)

            tile_x = x_full[:, :, ts:te, hs:he, ws:we].contiguous()
            tile_c = _adapt_guide_for_tile(
                base_c, ts, te, hs, he, ws, we,
                spec['H_full'], spec['W_full'], debug=debug)

            # Slice denoise_mask to tile dims so LTX patchifies it to the tile's
            # token count (not the full-video count).
            dm = tile_c.get('denoise_mask', None)
            if dm is not None and isinstance(dm, torch.Tensor):
                try:
                    if dm.dim() == 5 and dm.shape[-3:] == (F, H, W):
                        tile_c = dict(tile_c)
                        tile_c['denoise_mask'] = dm[:, :, ts:te, hs:he, ws:we]
                    elif dm.dim() == 4 and dm.shape[-3:] == (F, H, W):
                        tile_c = dict(tile_c)
                        tile_c['denoise_mask'] = dm[:, ts:te, hs:he, ws:we]
                except Exception:
                    pass

            tile_pred = _call(model_fn, {**input_dict, 'input': tile_x, 'c': tile_c})

            if isinstance(tile_pred, torch.Tensor) and tile_pred.shape == tile_x.shape:
                accum[:, :, ts:te, hs:he, ws:we] += tile_pred.float() * window_3d
                w_acc[:, :, ts:te, hs:he, ws:we] += window_3d
            elif debug:
                info = (tuple(tile_pred.shape) if isinstance(tile_pred, torch.Tensor)
                        else type(tile_pred).__name__)
                print(f"    [ltx_tile] tile [{ts}:{te},{hs}:{he},{ws}:{we}] "
                      f"pred shape mismatch: {info} vs {tuple(tile_x.shape)}")

        return (accum / w_acc.clamp(min=1e-8)).to(dtype)

    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LtxTiledSamplerPatch:
    """
    Model patch (MODEL → MODEL) adding per-step MultiDiffusion spatial/temporal
    tiling to an LTX-Video model.  Works with the vanilla SamplerCustomAdvanced:
    insert it on the model line (after your LoRA/ICLoRA loaders, before the
    guider).  Each model evaluation is split into overlapping T×H×W tiles kept at
    training-distribution token counts and blended back with trapezoidal windows,
    sharing one trajectory across tiles.  Guide conditioning (keyframe_idxs /
    guide_attention_entries) and denoise_mask are cropped per tile.

    Apply this patch LAST in a chain of model patches — it needs the
    model_function_wrapper slot (an existing wrapper is chained, but a later patch
    that overwrites the slot without chaining would drop this one).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tiles_h": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Tiles along height (H). Official LTX Stage-2 default: 2. "
                               "Set 1 to disable.",
                }),
                "tiles_w": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Tiles along width (W). Official LTX Stage-2 default: 2. "
                               "Set 1 to disable.",
                }),
                "overlap_h": ("INT", {
                    "default": 6, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Overlap in latent units between H tiles. "
                               "Official LTX default: 6 (≈48 px at 8× VAE).",
                }),
                "overlap_w": ("INT", {
                    "default": 6, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Overlap in latent units between W tiles. "
                               "Official LTX default: 6 (≈48 px at 8× VAE).",
                }),
            },
            "optional": {
                "tiles_t": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Tiles along the temporal (frames) axis. 1 disables temporal "
                               "tiling (recommended for most clips). Official Stage-2 uses 2.",
                }),
                "overlap_t": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Overlap in latent frames between temporal tiles. "
                               "Official LTX default: 8.",
                }),
                "bypass_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return the model unpatched — single-pass sampling.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print tile layout, blend-weight sanity check, and per-tile "
                               "guide-conditioning info to the console.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Mago Nodes/Sampling"

    def patch(self, model, tiles_h, tiles_w, overlap_h, overlap_w,
              tiles_t=1, overlap_t=8, bypass_tiling=False, debug=False):
        tiling_possible = tiles_t > 1 or tiles_h > 1 or tiles_w > 1
        if bypass_tiling or not tiling_possible:
            return (model,)

        tiling_cfg = {
            "tiles_t": tiles_t, "tiles_h": tiles_h, "tiles_w": tiles_w,
            "overlap_t": overlap_t, "overlap_h": overlap_h, "overlap_w": overlap_w,
        }
        m = model.clone()
        existing = m.model_options.get("model_function_wrapper", None)
        m.model_options["model_function_wrapper"] = _make_ltx_tile_wrapper(
            tiling_cfg, existing, debug=debug)
        return (m,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LtxTiledSamplerPatch": LtxTiledSamplerPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LtxTiledSamplerPatch": "LTX Tiled Sampler (Model Patch)",
}
