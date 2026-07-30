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

import contextlib
import math
import os

import torch

from .wan_tiled_sampler import (
    _compute_tile_starts, _make_window_1d,
    _parse_schedule, _schedule_lookup, _round_even, _resize_spatial,
    _resize_cond_planes,
)


_WAN_PATCH_SIZE_SPATIAL = 2  # WAN spatial patch size (temporal is 1)

# Warnings that describe a persistent condition (a build that can't be scheduled,
# a control signal left at full res) rather than a per-call event: print once per
# process so a 40-step run doesn't emit 40 copies, but never gate them behind
# `debug` — each one means the node is silently doing less than asked.
_WARNED = set()


def _warn_once(key, msg):
    """Print ``msg`` the first time ``key`` is seen in this process."""
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg)


try:  # present on any build that can run compile_args backend="inductor"
    from torch._inductor import config as _inductor_config
except Exception:  # minimal / non-inductor torch builds
    _inductor_config = None


@contextlib.contextmanager
def _frugal_compile():
    """Disable inductor's shape-padding pass for forwards run inside this block.

    Every tile — and every downscaled pass — carries a token count the compiled
    transformer has not seen before, so on a ``torch.compile``'d model with
    ``dynamic=False`` (the WanVideoTorchCompileSettings default) the first such
    forward triggers a fresh inductor compile *inside this wrapper*, at the worst
    possible moment: both the high and low models resident and the sampler holding
    its latents.  Inductor's ``pad_mm`` pass then benchmarks padded vs unpadded
    matmuls for the tile's shapes (``should_pad_common`` → ``should_pad_bench`` →
    ``do_bench`` → ``aten.addmm``), allocating GiB-scale scratch and OOMing the run
    *during compilation* — the very failure the tiling was meant to avoid, and one
    that surfaces as a ``BackendCompilerFailed`` pointing at this module rather than
    at a sampling step.

    ``shape_padding`` gates a compile-time graph pass, so switching it off around
    the call only affects graphs compiled within the block; tile shapes already in
    the compile cache run untouched.  The cost is therefore not a slower run but a
    possibly-unpadded kernel for the graphs compiled here — the trade this node
    exists to make.  Set ``MAGO_TILED_SHAPE_PADDING=1`` to keep torch's default.
    """
    if (_inductor_config is None
            or not hasattr(_inductor_config, "shape_padding")
            or os.environ.get("MAGO_TILED_SHAPE_PADDING") == "1"):
        yield
        return
    prev = _inductor_config.shape_padding
    _inductor_config.shape_padding = False
    try:
        yield
    finally:
        _inductor_config.shape_padding = prev

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


def _vace_grid_mismatch(vace_data, H, W):
    """Describe the first vace context that is NOT a ``[Cv, F, H, W]`` tensor on the
    latent grid, or ``None`` when every context in every entry is one — i.e. when
    ``_slice_vace_tile`` / ``_resize_vace_downscale`` can handle the whole list.

    ``forward_vace`` sizes its hint tokens to the entry's ``seq_len``, which must
    match the forward's own token count, so there is no per-entry escape: a context
    left full-res is either silently truncated or a mid-run shape error.  Callers
    bail to a full-frame forward instead."""
    for entry in vace_data:
        ctx_list = entry.get("context") if isinstance(entry, dict) else None
        if not isinstance(ctx_list, (list, tuple)) or len(ctx_list) == 0:
            return f"entry {type(entry).__name__} without a context list"
        for ctx in ctx_list:
            if not (isinstance(ctx, torch.Tensor) and ctx.dim() == 4
                    and ctx.shape[-2] == H and ctx.shape[-1] == W):
                return (str(tuple(ctx.shape)) if isinstance(ctx, torch.Tensor)
                        else type(ctx).__name__)
    return None


def _slice_vace_tile(vace_data, H, W, hs, he, ws, we, tile_seq_len):
    """Slice each vace entry's context (list of [Cv, F, H, W]) to the tile and
    circular-pad H/W up to the spatial patch size, mirroring wan_tiled_sampler's
    VACE handling.  Sets the entry's ``seq_len`` to the tile's token count; keeps
    ``scale``/``start``/``end``.  Returns a new list (originals untouched).

    ``seq_len`` must equal the token count of the forward this feeds, so callers
    clear ``_vace_grid_mismatch`` first; the else branch is a safety net."""
    out = []
    for entry in vace_data:
        ctx_list = entry["context"]
        new_ctx = []
        for ctx in ctx_list:
            if isinstance(ctx, torch.Tensor) and ctx.dim() == 4 \
                    and ctx.shape[-2] == H and ctx.shape[-1] == W:
                tile = ctx[:, :, hs:he, ws:we]
                ph = (_WAN_PATCH_SIZE_SPATIAL - tile.shape[-2] % _WAN_PATCH_SIZE_SPATIAL) % _WAN_PATCH_SIZE_SPATIAL
                pw = (_WAN_PATCH_SIZE_SPATIAL - tile.shape[-1] % _WAN_PATCH_SIZE_SPATIAL) % _WAN_PATCH_SIZE_SPATIAL
                if ph or pw:
                    tile = torch.nn.functional.pad(tile, (0, pw, 0, ph), mode="circular")
                new_ctx.append(tile.contiguous())
            else:
                new_ctx.append(ctx)
                shp = tuple(ctx.shape) if isinstance(ctx, torch.Tensor) else type(ctx).__name__
                _warn_once(f"vace_slice_{shp}",
                           f"[WanKJTiled] vace ctx {shp} reached the tile slicer without "
                           f"matching the latent grid ({H}x{W}); it stays full-res while "
                           f"seq_len follows the tile, so forward_vace will truncate this "
                           f"entry's control signal. Please report this shape.")
        out.append({**entry, "context": new_ctx, "seq_len": tile_seq_len})
    return out


def _resize_vace_downscale(vace_data, H, W, oh, ow, new_seq_len):
    """Spatially downscale each vace entry's context (list of [Cv, F, H, W]) to
    (oh, ow) and set the entry's ``seq_len`` to the downscaled token count.  Returns
    a new list (originals untouched).

    A VACE context is ``cat([inactive, reactive, mask_planes])``, so the mode is
    picked per plane (``_resize_cond_planes``) — the latents want ``area``, the
    trailing pixel-unshuffled mask must stay binary.  Same ``seq_len`` invariant as
    ``_slice_vace_tile``."""
    out = []
    for entry in vace_data:
        new_ctx = []
        for ctx in entry["context"]:
            if isinstance(ctx, torch.Tensor) and ctx.dim() == 4 \
                    and ctx.shape[-2] == H and ctx.shape[-1] == W:
                new_ctx.append(_resize_cond_planes(ctx, oh, ow, 0).contiguous())
            else:
                new_ctx.append(ctx)
                shp = tuple(ctx.shape) if isinstance(ctx, torch.Tensor) else type(ctx).__name__
                _warn_once(f"vace_downscale_{shp}",
                           f"[WanKJTiled] vace ctx {shp} reached the downscaler without "
                           f"matching the latent grid ({H}x{W}); it stays full-res while "
                           f"seq_len follows the downscaled latent, so forward_vace will "
                           f"truncate this entry's control signal. Please report this shape.")
        out.append({**entry, "context": new_ctx, "seq_len": new_seq_len})
    return out


def _run_downscaled_kj(orig, kwargs, scale, y, vace_data, vace_is_dict_list):
    """Multiscale (coarse-to-fine) whole-frame pass for the kijai forward.

    Downscales the latent (and its spatially-shaped conditioning) to ``scale`` %,
    runs a SINGLE full-frame forward at the reduced token count, and upscales the
    predicted latent back to the original size.  WAN derives its RoPE grid from the
    latent's h/w, so a smaller latent brings token count + positions back in
    training distribution — this keeps global motion coherent in the early (high
    sigma) steps, where independent per-tile motion would otherwise drift.

    Mirrors ``wan_tiled_sampler._run_downscaled`` but at kijai's transformer.forward
    contract (x=[z] 4D, seq_len recomputed, freqs bank reused unchanged).  Returns
    the forward's ``([pred], cache)`` tuple with the prediction at the original size,
    or ``None`` if the target size isn't actually smaller (caller runs full-res)."""
    x = kwargs["x"]
    C, F, H, W = x[0].shape
    oh, ow = _round_even(H * scale / 100.0), _round_even(W * scale / 100.0)
    if oh >= H and ow >= W:
        return None  # not actually a downscale → let caller run full-res

    # Subsample the noisy latent with nearest-exact (NOT averaging): at high sigma
    # x is noise-dominated and area/bilinear would collapse the noise variance, so
    # the model would see a latent whose level no longer matches `t`.  The clean
    # conditioning below is signal, so it uses antialiased `area`.
    x_small = _resize_spatial(x[0], oh, ow, "nearest-exact").contiguous()
    new_seq_len = int(math.ceil((oh * ow) / 4 * F))

    tkw = dict(kwargs)
    tkw["x"] = [x_small]
    tkw["seq_len"] = new_seq_len
    if kwargs.get("pred_id") is not None:
        _warn_once("teacache_downscale",
                   "[WanKJTiled] pred_id/teacache active → disabled for the downscaled "
                   "pass (its token layout differs from full-res).")
    tkw["pred_id"] = None  # token layout differs from full-res → teacache invalid

    if isinstance(y, (list, tuple)) and len(y) == 1 \
            and isinstance(y[0], torch.Tensor) and y[0].dim() == 4 \
            and y[0].shape[-2] == H and y[0].shape[-1] == W:
        # y is cat([mask, latent]) on I2V/FunInP but two stacked latents on the
        # Fun-Control paths, so the mode is picked per plane from its values rather
        # than from y's channel count — see _resize_cond_planes.  y[0] is 4D.
        tkw["y"] = [_resize_cond_planes(y[0], oh, ow, 0).contiguous()]
    if vace_is_dict_list:
        tkw["vace_data"] = _resize_vace_downscale(vace_data, H, W, oh, ow, new_seq_len)

    with _frugal_compile():  # new token count → may compile here; keep it cheap
        out = orig(**tkw)
    if not (isinstance(out, (list, tuple)) and len(out) == 2
            and isinstance(out[0], (list, tuple)) and len(out[0]) == 1
            and isinstance(out[0][0], torch.Tensor)):
        _warn_once("downscale_ret",
                   f"[WanKJTiled] downscaled forward returned {type(out).__name__}, not "
                   f"([tensor], cache) — cannot upscale it back to ({H},{W}). The sampler "
                   f"is receiving whatever the transformer returned, at reduced resolution.")
        return out
    pred, cache = out
    # Always hand the sampler a full-resolution prediction so its latent state
    # stays full-res across the whole run (and any high/low-noise split).  Resize
    # whenever the prediction isn't already full-res, not only from the expected
    # (oh, ow): on any unexpected size the alternative is leaking a downscaled
    # latent into the sampler's full-res update.  Mirrors wan_tiled_sampler's
    # _run_downscaled, which tests `!= (H, W)` for the same reason.
    if pred[0].shape[-2:] == (H, W):
        return pred, cache
    if pred[0].shape[-2:] != (oh, ow):
        _warn_once("downscale_shape",
                   f"[WanKJTiled] downscaled pred shape {tuple(pred[0].shape)} is neither "
                   f"the requested ({oh},{ow}) nor full-res ({H},{W}); resizing to full-res "
                   f"anyway. Check the WanVideoWrapper build's forward contract.")
    up = _resize_spatial(pred[0], H, W, "bilinear").to(x[0].dtype)
    return [up], cache


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
        sched_active = cfg.get("scale_sched") is not None
        # `bypass_tiling` means "no tiling", NOT "no multiscale" — same contract as
        # the native node (wan_tiled_sampler: `default_tile_on` at _patch_model, and
        # the early-out at WanTiledSampler.sample) and as documented in README.md's
        # bypass_tiling row and its tooltip.  So tiling is gated by `tile_on`, and
        # the wrapper only steps aside when there is nothing at all to do.
        tile_on = (tiles_h > 1 or tiles_w > 1) and not cfg["bypass_tiling"]
        if not sched_active and not tile_on:
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

        y = kwargs.get("y")
        vace_data = kwargs.get("vace_data")
        vace_is_dict_list = (isinstance(vace_data, list) and len(vace_data) > 0
                             and isinstance(vace_data[0], dict))

        # A vace context off the latent grid can be neither sliced nor downscaled
        # without corrupting seq_len (see _vace_grid_mismatch).  It's a property of
        # the shapes, not of this step → take the same exit as an unsupported
        # spatial cond and run the whole frame.
        if vace_is_dict_list:
            bad_vace = _vace_grid_mismatch(vace_data, H, W)
            if bad_vace is not None:
                _warn_once(f"vace_grid_{bad_vace}",
                           f"[WanKJTiled] vace context {bad_vace} is not on the latent "
                           f"grid ({H}x{W}), so it can be neither tiled nor downscaled "
                           f"without corrupting the VACE control signal → running every "
                           f"step full-frame at full resolution. Tiling and multiscale "
                           f"are INACTIVE for this run.")
                return orig(*args, **kwargs)

        # --- multiscale (coarse-to-fine) schedule ---------------------------------
        # kijai passes the 0-based step index straight into forward as
        # `current_step` (nodes_sampler base_params), so the schedule lookup needs
        # no sigma matching.  scale < 100 → downscale the whole frame (tiling off);
        # scale == 100 → tiled path below.  Natural pairing: low-res whole-frame
        # early → full-res tiled late.
        #
        # A missing `current_step` must NOT be read as step 0: on a build that
        # doesn't forward it, every step would resolve to the schedule's first
        # entry, pinning the whole run to (say) 50% with tiling never engaging —
        # uniformly soft output and no error.  Unknown step → no schedule, which
        # degrades to plain full-res tiling rather than to a silent downscale.
        # (The native node avoids this entirely by deriving the step from
        # transformer_options["sample_sigmas"], which kijai's loop doesn't pass.)
        step = None
        cs = kwargs.get("current_step")
        if cs is not None:
            try:
                step = int(cs)
            except (TypeError, ValueError):
                step = None

        scale = 100.0
        if sched_active:
            if step is None:
                _warn_once("no_current_step",
                           "[WanKJTiled] this WanVideoWrapper build does not pass "
                           "`current_step` into transformer.forward, so scale_schedule "
                           "cannot be resolved → running every step at full resolution "
                           "(tiling only). Multiscale is INACTIVE.")
            else:
                scale = max(1.0, min(100.0, _schedule_lookup(cfg["scale_sched"], step)))

        if debug and warned.get("last_step_logged") != step:
            warned["last_step_logged"] = step
            mode = (f"downscale {round(scale)}% (whole-frame)" if scale < 100
                    else ("tiled" if tile_on else "full"))
            print(f"[WanKJTiled] step {step if step is not None else '?'}: "
                  f"scale={round(scale)}% → {mode}")

        if scale < 100:
            res = _run_downscaled_kj(orig, kwargs, scale, y, vace_data,
                                     vace_is_dict_list)
            if res is not None:
                return res
            # target size wasn't actually smaller → fall through to tiling / full

        # At scale == 100 under bypass_tiling there is nothing left to do: run the
        # full frame, or bypass would lose its meaning on exactly those steps.
        if not tile_on:
            return orig(*args, **kwargs)

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
                tkw["vace_data"] = _slice_vace_tile(vace_data, H, W, hs, he, ws, we,
                                                    tile_seq_len)

            with _frugal_compile():  # new token count → may compile here; keep it cheap
                out = orig(**tkw)
            # The forward contract is ([pred], cache); anything else can't be
            # blended, so bail to a full frame instead of raising out of a wrapper
            # the sampler doesn't know is there.
            if not (isinstance(out, (list, tuple)) and len(out) == 2
                    and isinstance(out[0], (list, tuple)) and len(out[0]) == 1
                    and isinstance(out[0][0], torch.Tensor)):
                print(f"[WanKJTiled] unexpected forward return {type(out).__name__} "
                      f"for a tile; falling back to full forward.")
                return orig(*args, **kwargs)
            tile_pred = out[0][0]
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
    updates ``_mago_tile_cfg``.

    Note the wrap is installed on ``patcher.model.diffusion_model``, which
    ``model.clone()`` SHARES — so both the wrapper and its cfg outlive this node.
    While the node is in the graph that's fine (every run rewrites the cfg), but
    if the node is later removed from the graph ``patch()`` stops running and the
    last cfg keeps applying — a stale ``tiles_h/tiles_w`` keeps tiling, and a stale
    ``scale_sched`` keeps downscaling, on runs that no longer use this node.
    Reload the model (or re-add the node with ``bypass_tiling`` and an empty
    schedule) to clear it.  Closing this properly needs an epoch stamp on the cfg
    or an uninstall path; tracked separately."""
    if not getattr(transformer, "_mago_tiled_wrapped", False):
        transformer._mago_orig_forward = transformer.forward
        transformer.forward = _make_tiled_forward(transformer)
        transformer._mago_tiled_wrapped = True
    transformer._mago_tile_cfg = cfg


class WanVideoTiledPatch:
    """kijai WanVideoWrapper patch node: adds per-step MultiDiffusion spatial
    tiling AND multiscale (coarse-to-fine ``scale_schedule``) to a ``WANVIDEOMODEL``
    so it can inpaint/generate above the model's single-pass resolution (e.g. 1920+,
    native-4k) while keeping large temporal chunks and coherent global motion.  Plug
    the patched model into ``WanVideoSampler``.

    Multiscale (coarse-to-fine, matching the native ``WanTiledSamplerPatch``): early
    high-sigma steps run the whole frame at reduced resolution — smaller latent →
    RoPE/token count back in training distribution → global motion stays coherent;
    late steps run full-res tiled to recover detail.  Empty ``scale_schedule`` =
    tiling only (unchanged behaviour).  The coarse-to-fine approach is based on the
    LTX-Video tiled-sampler work — Lightricks' official implementation and the 10S
    Nodes investigation into long-context sampling degradation — reworked here for
    WAN rather than ported from it."""

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
                "scale_schedule": ("STRING", {"default": "", "multiline": False,
                                   "tooltip": "Multiscale (coarse-to-fine) schedule, "
                                   "step → resolution %, e.g. '{0:50, 8:100}'. While the "
                                   "scheduled scale < 100 the whole frame is run at that "
                                   "resolution (tiling off, keeps global motion coherent); "
                                   "at 100 it runs tiled. Empty = always full-res (tiling only)."}),
                "bypass_tiling": ("BOOLEAN", {"default": False,
                                  "tooltip": "Skip tiling (unless a scale_schedule is set — "
                                  "a schedule still runs its downscaled steps). With no "
                                  "schedule this is a plain WanVideoSampler passthrough."}),
                "debug": ("BOOLEAN", {"default": False,
                          "tooltip": "Print tile layout + blend-weight sanity check + per-step scale."}),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "patch"
    CATEGORY = "Mago Nodes/Sampling"

    def patch(self, model, tiles_h, tiles_w, overlap_h, overlap_w,
              scale_schedule="", bypass_tiling=False, debug=False):
        cfg = {"tiles_h": tiles_h, "tiles_w": tiles_w,
               "overlap_h": overlap_h, "overlap_w": overlap_w,
               "scale_sched": _parse_schedule(scale_schedule),
               "bypass_tiling": bypass_tiling, "debug": debug}
        m = model.clone()
        _install_tiled_forward(m.model.diffusion_model, cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {"WanVideoTiledPatch": WanVideoTiledPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVideoTiledPatch": "WAN Video Tiled Patch (kijai)"}
