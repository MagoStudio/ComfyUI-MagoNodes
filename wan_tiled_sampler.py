import math
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


# ─────────────────────────────────────────────────────────────────────────────
# Tile geometry helpers
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
    Linear trapezoidal blend window.  Ramps exclude exact 0.0/1.0 endpoints so
    weights never reach zero inside a tile.
    """
    win = torch.ones(size, dtype=dtype, device=device)
    if fade_left > 0:
        fl = min(fade_left, size)
        win[:fl] = torch.linspace(0.0, 1.0, fl + 2, dtype=dtype, device=device)[1:-1]
    if fade_right > 0:
        fr = min(fade_right, size)
        win[size - fr:] = torch.linspace(1.0, 0.0, fr + 2, dtype=dtype, device=device)[1:-1]
    return win


# ─────────────────────────────────────────────────────────────────────────────
# Multiscale scheduling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_schedule(text):
    """
    Parse a lenient ``{step: value, ...}`` schedule string into a sorted list of
    (step:int, value:float) pairs.  Accepts JSON-ish input without quoted keys,
    e.g. ``{0:25, 10:50, 20:100}`` (commas optional, whitespace ignored).
    Returns None for empty / unparseable input.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    s = s.strip("{}").strip()
    if not s:
        return None
    pairs = []
    # split on commas and/or whitespace, keep only "k:v" tokens
    tokens = s.replace(",", " ").split()
    for tok in tokens:
        if ":" not in tok:
            continue
        k, _, v = tok.partition(":")
        try:
            pairs.append((int(float(k.strip())), float(v.strip())))
        except ValueError:
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    return pairs


def _schedule_lookup(sched, step):
    """Value of the largest key <= step (held-until-next-key). Before the first
    key, the first key's value extends backward."""
    val = sched[0][1]
    for k, v in sched:
        if k <= step:
            val = v
        else:
            break
    return val


def _round_even(x):
    """Round to the nearest positive even integer (WAN spatial patch size is 2)."""
    return max(2, int(round(x / 2.0)) * 2)


def _resize_spatial(t, out_h, out_w, mode):
    """
    Resize only the last two (H, W) dims of a tensor, leaving every leading dim
    (batch, channels, frames, refs, …) untouched.  Interpolation is independent
    per channel/frame, so folding the leading dims into the batch axis is exact.
    """
    if t.shape[-2] == out_h and t.shape[-1] == out_w:
        return t
    lead = t.shape[:-2]
    H, W = t.shape[-2], t.shape[-1]
    x = t.reshape(-1, 1, H, W).float()
    kwargs = {}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode=mode, **kwargs)
    return x.reshape(*lead, out_h, out_w).to(t.dtype)


def _nearest_step(start_sigmas, sigma):
    """Map an incoming sigma to a 0-based step index by nearest match against the
    per-step starting sigmas.  Robust to samplers that evaluate the model more
    than once per step (intermediate sigmas snap to the closest scheduled step)."""
    best_i, best_d = 0, float("inf")
    for i, s in enumerate(start_sigmas):
        d = abs(s - sigma)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _build_tile_specs(H, W, tiles_h, tiles_w, overlap_h, overlap_w, debug=False):
    """Build the per-tile bounds + trapezoidal blend windows for a latent of
    spatial size (H, W).  Pure geometry — no model state — so it can be computed
    lazily on the first model call (the patch node doesn't know the latent size
    ahead of time) and cached per resolution."""
    # Convert overlap % → latent pixels based on tile size.  Tile size isn't known
    # yet, so approximate from total / n_tiles, then recompute with the real size.
    def _pct_to_px(total, n_tiles, pct):
        if n_tiles <= 1 or pct == 0:
            return 0
        approx_tile = math.ceil(total / n_tiles)
        return max(0, round(approx_tile * pct / 100))

    overlap_h_px = _pct_to_px(H, tiles_h, overlap_h)
    overlap_w_px = _pct_to_px(W, tiles_w, overlap_w)
    h_starts, tile_h = _compute_tile_starts(H, tiles_h, overlap_h_px)
    w_starts, tile_w = _compute_tile_starts(W, tiles_w, overlap_w_px)

    overlap_h_px = max(0, round(tile_h * overlap_h / 100))
    overlap_w_px = max(0, round(tile_w * overlap_w / 100))
    h_starts, tile_h = _compute_tile_starts(H, tiles_h, overlap_h_px)
    w_starts, tile_w = _compute_tile_starts(W, tiles_w, overlap_w_px)

    if debug:
        print(f"[WanTiled] latent H×W={H}×{W} tiles_h={tiles_h} tiles_w={tiles_w} "
              f"overlap_h={overlap_h}% ({overlap_h_px}px) overlap_w={overlap_w}% ({overlap_w_px}px)")
        print(f"  h_starts={h_starts} tile_h={tile_h}")
        print(f"  w_starts={w_starts} tile_w={tile_w}")

    tile_specs = []
    for hi, hs in enumerate(h_starts):
        he = min(hs + tile_h, H)
        fade_left_h  = max(0, h_starts[hi - 1] + tile_h - hs) if hi > 0 else 0
        fade_right_h = max(0, hs + tile_h - h_starts[hi + 1]) if hi < len(h_starts) - 1 else 0

        for wi, ws in enumerate(w_starts):
            we = min(ws + tile_w, W)
            fade_left_w  = max(0, w_starts[wi - 1] + tile_w - ws) if wi > 0 else 0
            fade_right_w = max(0, ws + tile_w - w_starts[wi + 1]) if wi < len(w_starts) - 1 else 0

            win_h = _make_window_1d(he - hs, fade_left_h, fade_right_h, torch.float32, 'cpu')
            win_w = _make_window_1d(we - ws, fade_left_w, fade_right_w, torch.float32, 'cpu')
            # shape (1, 1, 1, Ht, Wt) — broadcast over (B, C, F)
            window_2d = (win_h[:, None] * win_w[None, :]).reshape(1, 1, 1, he - hs, we - ws)

            tile_specs.append({'h_start': hs, 'h_end': he,
                               'w_start': ws, 'w_end': we,
                               'window_2d': window_2d})
            if debug:
                print(f"  tile h:[{hs}:{he}] w:[{ws}:{we}] "
                      f"fade_h=({fade_left_h},{fade_right_h}) fade_w=({fade_left_w},{fade_right_w})")

    if debug:
        w_test = torch.zeros(1, 1, 1, H, W, dtype=torch.float32)
        for spec in tile_specs:
            hs, he, ws, we = spec['h_start'], spec['h_end'], spec['w_start'], spec['w_end']
            w_test[:, :, :, hs:he, ws:we] += spec['window_2d']
        mn, mx = w_test.min().item(), w_test.max().item()
        print(f"  blend weight sanity: min={mn:.4f} max={mx:.4f} "
              f"({'OK' if abs(mn - 1.0) < 0.02 and abs(mx - 1.0) < 0.02 else 'WARN'})")

    return tile_specs


# ─────────────────────────────────────────────────────────────────────────────
# MultiDiffusion + multiscale wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _make_wan_tile_wrapper(tiling_cfg, existing_wrapper, debug=False,
                           reference_latent_full=False, default_tile_on=True,
                           scale_sched=None):
    """
    Returns a model_function_wrapper implementing, per denoising step:

      * Multiscale (scale_sched): when the scheduled scale is < 100%, the whole
        latent (and its spatially-shaped conditioning) is downscaled, evaluated
        in a single full-frame pass, and the prediction upscaled back.  WAN
        derives its RoPE positions from the latent's h/w, so a smaller latent
        brings token count and positional encodings back in-distribution — this
        keeps global motion coherent in I2V where independent per-tile motion
        would otherwise drift.

      * Per-step MultiDiffusion spatial tiling (at scale == 100%): the latent is
        split into overlapping spatial tiles, each evaluated at a
        training-distribution token count, and blended back with trapezoidal
        windows (weights sum to 1.0 everywhere).

    Tiling is skipped while scale < 100% (tiling a downscaled latent defeats the
    purpose); the natural pairing is low-res whole-frame early → full-res tiled
    late.  At scale == 100% tiling follows default_tile_on (i.e. bypass_tiling).
    """
    sched_active = scale_sched is not None
    last_logged = [-1]
    specs_cache = {}

    def _get_specs(H, W):
        key = (H, W)
        if key not in specs_cache:
            specs_cache[key] = _build_tile_specs(
                H, W, tiling_cfg["tiles_h"], tiling_cfg["tiles_w"],
                tiling_cfg["overlap_h"], tiling_cfg["overlap_w"], debug=debug)
        return specs_cache[key]

    def _call_model(model_fn, d):
        if existing_wrapper is not None:
            return existing_wrapper(model_fn, d)
        return model_fn(d["input"], d["timestep"], **d.get("c", {}))

    def _run_downscaled(model_fn, input_dict, scale):
        x_full = input_dict["input"]
        H, W = x_full.shape[-2], x_full.shape[-1]
        oh, ow = _round_even(H * scale / 100.0), _round_even(W * scale / 100.0)
        if oh >= H and ow >= W:
            return _call_model(model_fn, input_dict)

        # Downscale the noisy latent by SUBSAMPLING (nearest), not averaging.
        # At high sigma x is noise-dominated; averaging (area/bilinear) collapses
        # the noise variance, so the model would receive a latent whose noise
        # level no longer matches `timestep` and would emit garbage in the
        # generated frames.  nearest-exact keeps every k-th sample, so the noise
        # stays unit-variance and the level still matches sigma (works for both
        # flow- and eps-parameterised models).  The clean conditioning tensors
        # below are signal, not noise, so they use antialiased `area`.
        x_small = _resize_spatial(x_full, oh, ow, "nearest-exact").contiguous()
        c = dict(input_dict.get("c", {}))
        for key in ("c_concat", "reference_latent", "vace_context"):
            v = c.get(key, None)
            if isinstance(v, torch.Tensor) and v.shape[-2] == H and v.shape[-1] == W:
                c[key] = _resize_spatial(v, oh, ow, "area").contiguous()
        dm = c.get("denoise_mask", None)
        if isinstance(dm, torch.Tensor) and dm.shape[-2] == H and dm.shape[-1] == W:
            c["denoise_mask"] = _resize_spatial(dm, oh, ow, "nearest-exact").contiguous()

        d = {**input_dict, "input": x_small, "c": c}
        pred = _call_model(model_fn, d)
        # Always hand the sampler a full-resolution prediction: the sampler's
        # latent state stays full-res across the whole run (and across WAN 2.2
        # high/low-noise sampler passes), so the downscale never persists — the
        # next pass always receives a full-res latent regardless of its schedule.
        if isinstance(pred, torch.Tensor) and pred.shape[-2:] != (H, W):
            pred = _resize_spatial(pred, H, W, "bilinear")
        return pred.to(x_full.dtype) if isinstance(pred, torch.Tensor) else pred

    def _run_tiled(model_fn, input_dict, tile_specs):
        x_full = input_dict["input"]
        B, C, F, H, W = x_full.shape
        dtype = x_full.dtype
        device = x_full.device

        accum = torch.zeros(B, C, F, H, W, dtype=torch.float32, device=device)
        w_acc = torch.zeros(1, 1, F, H, W, dtype=torch.float32, device=device)
        base_c = input_dict.get("c", {})

        for spec in tile_specs:
            hs, he = spec["h_start"], spec["h_end"]
            ws, we = spec["w_start"], spec["w_end"]
            window_2d = spec["window_2d"].to(device=device, dtype=torch.float32)

            tile_x = x_full[:, :, :, hs:he, ws:we].contiguous()
            tile_c = dict(base_c)

            # --- c_concat: (B, extra_ch, F, H, W) → slice spatial dims ---
            cc = tile_c.get("c_concat", None)
            if cc is not None and isinstance(cc, torch.Tensor) and cc.dim() == 5:
                try:
                    if cc.shape[-2] == H and cc.shape[-1] == W:
                        tile_c["c_concat"] = cc[:, :, :, hs:he, ws:we].contiguous()
                    elif debug:
                        print(f"  [wan_tile] c_concat spatial mismatch: "
                              f"{tuple(cc.shape)} vs x {tuple(x_full.shape)}")
                except Exception:
                    pass

            # --- denoise_mask: 5D (B,C,F,H,W) or 4D (B,F,H,W) → slice spatial dims ---
            dm = tile_c.get("denoise_mask", None)
            if dm is not None and isinstance(dm, torch.Tensor):
                try:
                    if dm.dim() == 5 and dm.shape[-2] == H and dm.shape[-1] == W:
                        tile_c["denoise_mask"] = dm[:, :, :, hs:he, ws:we].contiguous()
                    elif dm.dim() == 4 and dm.shape[-2] == H and dm.shape[-1] == W:
                        tile_c["denoise_mask"] = dm[:, :, hs:he, ws:we].contiguous()
                except Exception:
                    pass

            # --- reference_latent: (B, C, H, W) → optionally slice spatial dims ---
            # When reference_latent_full=True the full reference frame is passed to every
            # tile unchanged.  ref_conv appends its tokens to the sequence (no size
            # constraint), so the full image provides global scene context to all tiles.
            if not reference_latent_full:
                rl = tile_c.get("reference_latent", None)
                if rl is not None and isinstance(rl, torch.Tensor) and rl.dim() == 4:
                    try:
                        if rl.shape[-2] == H and rl.shape[-1] == W:
                            tile_c["reference_latent"] = rl[:, :, hs:he, ws:we].contiguous()
                    except Exception:
                        pass

            # --- vace_context: (B, num_refs, channels, F, H, W) → slice + pad spatial dims ---
            # WAN's _forward applies pad_to_patch_size(x, patch_size=(1,2,2)) with circular
            # padding before patch_embedding, but vace_context bypasses this step and goes
            # directly into vace_patch_embedding.  For tile dims that aren't multiples of the
            # spatial patch size (e.g. H=49), pad_to_patch_size makes x produce 25 H-patches
            # while vace_patch_embedding on the unpadded tile produces only 24.
            # Fix: slice the vace tile then apply the same circular padding so both produce
            # the same token count.
            _WAN_PATCH_SIZE = (1, 2, 2)  # temporal × height × width
            vc = tile_c.get("vace_context", None)
            if vc is not None and isinstance(vc, torch.Tensor) and vc.dim() == 6:
                try:
                    if vc.shape[-2] >= he and vc.shape[-1] >= we:
                        tile_vc = vc[:, :, :, :, hs:he, ws:we]
                        # Compute padding equivalent to pad_to_patch_size (dims F,H,W)
                        pad = ()
                        for i, ps in enumerate(_WAN_PATCH_SIZE):
                            sz = tile_vc.shape[3 + i]  # dims 3=F 4=H 5=W of 6D tensor
                            pad = (0, (ps - sz % ps) % ps) + pad
                        if any(p > 0 for p in pad):
                            # torch circular pad only supports up to 5D — reshape to 5D, pad, restore
                            B_vc, nr, ch_vc = tile_vc.shape[:3]
                            t5 = tile_vc.reshape(B_vc * nr, ch_vc, *tile_vc.shape[3:])
                            t5 = torch.nn.functional.pad(t5, pad, mode="circular")
                            tile_vc = t5.reshape(B_vc, nr, ch_vc, *t5.shape[2:])
                        tile_c["vace_context"] = tile_vc.contiguous()
                        if debug:
                            print(f"  [wan_tile_debug] vace tile h:[{hs}:{he}] w:[{ws}:{we}] "
                                  f"pad={pad} → shape {tuple(tile_vc.shape)}")
                    elif debug:
                        print(f"  [wan_tile_debug] vace spatial too small: "
                              f"shape={tuple(vc.shape)} he={he} we={we}")
                except Exception as e:
                    if debug:
                        import traceback
                        print(f"  [wan_tile_debug] vace exception: {e}\n{traceback.format_exc()}")
            elif vc is not None and debug:
                print(f"  [wan_tile_debug] vace not sliced: type={type(vc).__name__} "
                      f"{'dim=' + str(vc.dim()) if isinstance(vc, torch.Tensor) else ''}")

            tile_input_dict = {**input_dict, "input": tile_x, "c": tile_c}
            tile_pred = _call_model(model_fn, tile_input_dict)

            if isinstance(tile_pred, torch.Tensor) and tile_pred.shape == tile_x.shape:
                accum[:, :, :, hs:he, ws:we] += tile_pred.float() * window_2d
                w_acc[:, :, :, hs:he, ws:we] += window_2d
            elif debug:
                pred_info = (tuple(tile_pred.shape)
                             if isinstance(tile_pred, torch.Tensor)
                             else type(tile_pred).__name__)
                print(f"  [wan_tile] tile h:[{hs}:{he}] w:[{ws}:{we}] "
                      f"pred shape mismatch: {pred_info} vs {tuple(tile_x.shape)}")

        return (accum / w_acc.clamp(min=1e-8)).to(dtype)

    def wrapper(model_fn, input_dict):
        # Resolve the current step (0-based, local to this sampler pass) from the
        # incoming sigma vs. the full per-pass sigma schedule, which ComfyUI hands
        # to every model call in transformer_options["sample_sigmas"].  This makes
        # the wrapper sampler-agnostic — it needs no knowledge of steps/scheduler —
        # and gives correct local indices across a WAN 2.2 two-sampler split.
        step = 0
        if sched_active:
            try:
                to = input_dict.get("c", {}).get("transformer_options", {})
                ss = to.get("sample_sigmas", None)
                sigma = float(input_dict["timestep"].flatten()[0])
                if ss is not None and len(ss) > 1:
                    step = _nearest_step([float(s) for s in ss[:-1]], sigma)
            except Exception:
                step = 0

        x_full = input_dict["input"]
        H, W = x_full.shape[-2], x_full.shape[-1]
        tile_specs = _get_specs(H, W)

        scale = 100.0 if scale_sched is None else max(1.0, min(100.0, _schedule_lookup(scale_sched, step)))
        do_tile = default_tile_on and len(tile_specs) > 1

        if debug and step != last_logged[0]:
            last_logged[0] = step
            mode = ("downscale %d%% (whole-frame)" % round(scale)) if scale < 100 \
                else ("tiled" if do_tile else "full")
            print(f"[WanTiled] step {step}: scale={round(scale)}% → {mode}")

        if scale < 100:
            return _run_downscaled(model_fn, input_dict, scale)
        if do_tile:
            return _run_tiled(model_fn, input_dict, tile_specs)
        return _call_model(model_fn, input_dict)

    return wrapper


def _patch_model(model, tiles_h, tiles_w, overlap_h, overlap_w,
                 scale_sched=None, bypass_tiling=False,
                 reference_latent_full=True, debug=False):
    """Clone `model` and install the multiscale/tiling model_function_wrapper,
    chaining any wrapper already present.  Shared by the sampler and patch nodes."""
    tiling_cfg = {"tiles_h": tiles_h, "tiles_w": tiles_w,
                  "overlap_h": overlap_h, "overlap_w": overlap_w}
    default_tile_on = (tiles_h > 1 or tiles_w > 1) and not bypass_tiling
    m = model.clone()
    existing = m.model_options.get("model_function_wrapper", None)
    m.model_options["model_function_wrapper"] = _make_wan_tile_wrapper(
        tiling_cfg, existing, debug=debug,
        reference_latent_full=reference_latent_full,
        default_tile_on=default_tile_on,
        scale_sched=scale_sched)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

class WanTiledSampler:
    """
    Drop-in replacement for KSamplerAdvanced with per-step MultiDiffusion spatial
    tiling and multiscale (coarse-to-fine) scheduling for WAN 2.1 / 2.2 models.

    Tiling keeps each spatial patch at training-distribution token counts, which
    prevents the hue shift / conditioning drift that appears when sampling very
    large latents in a single pass.  This works well for V2V, where the
    conditioning splits cleanly per tile.

    For I2V, independent per-tile motion drifts apart even when the first frame /
    reference is honoured.  The multiscale schedule fixes this: run the *whole*
    frame downscaled in the early (high-noise) steps so global motion is decided
    coherently, then raise the resolution and hand off to tiling for detail.

    Works with WAN Fun Control (Wan22FunControlToVideo) and VACE by slicing /
    rescaling c_concat, denoise_mask, reference_latent, and vace_context.
    Temporal frames are never split (spatial-only), so causal temporal attention
    is never broken.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], {"advanced": True}),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0,
                    "step": 0.1, "round": 0.01,
                    "tooltip": "Classifier-Free Guidance scale. "
                               "WAN distilled models typically use cfg=1.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "advanced": True,
                }),
                "end_at_step": ("INT", {
                    "default": 10000, "min": 0, "max": 10000, "advanced": True,
                }),
                "return_with_leftover_noise": (["disable", "enable"], {"advanced": True}),
                "tiles_h": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of spatial tiles along the height axis. "
                               "Set 1 to disable height tiling.",
                }),
                "tiles_w": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of spatial tiles along the width axis. "
                               "Set 1 to disable width tiling.",
                }),
                "overlap_h": ("INT", {
                    "default": 25, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Overlap between height tiles as a percentage of tile height "
                               "(0–50%). 25% = one quarter of each tile overlaps its neighbour. "
                               "Larger values reduce seam artifacts from cross-tile motion "
                               "at the cost of more compute per step.",
                }),
                "overlap_w": ("INT", {
                    "default": 25, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Overlap between width tiles as a percentage of tile width "
                               "(0–50%). 25% = one quarter of each tile overlaps its neighbour.",
                }),
            },
            "optional": {
                "scale_schedule": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Multiscale (coarse-to-fine) schedule mapping step → resolution %. "
                               "e.g. {0:25, 10:50, 20:100} runs the WHOLE frame downscaled to 25% "
                               "for steps 0–9, 50% for 10–19, then full-res from 20. Values are "
                               "held until the next key. While scale < 100% the frame is evaluated "
                               "in a single pass (tiling is skipped) so global motion stays coherent "
                               "— the key fix for I2V drift. Empty = always 100%.",
                }),
                "bypass_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip tiling entirely — equivalent to plain KSamplerAdvanced "
                               "(unless a scale_schedule is set).",
                }),
                "reference_latent_full": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, the full reference frame is passed to every "
                               "tile instead of cropping it to the tile region.  Useful for "
                               "I2V workflows: the reference tokens are appended to the "
                               "sequence so there is no size constraint, and the full image "
                               "gives each tile global scene context.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print tile layout, blend weight sanity check, per-step "
                               "scale/tile decisions, and conditioning slice info.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Mago Nodes/Sampling"

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        tiles_h,
        tiles_w,
        overlap_h,
        overlap_w,
        scale_schedule="",
        bypass_tiling=False,
        reference_latent_full=True,
        debug=False,
        denoise=1.0,
    ):
        force_full_denoise = return_with_leftover_noise != "enable"
        disable_noise      = add_noise == "disable"

        scale_sched = _parse_schedule(scale_schedule)
        tiling_possible = tiles_h > 1 or tiles_w > 1

        # Wrapper is needed if a scale_schedule is requested, or if plain tiling is
        # active.  A bare bypass with no schedule → plain KSamplerAdvanced.
        sched_active = scale_sched is not None
        if not sched_active and (bypass_tiling or not tiling_possible):
            return self._plain_sample(
                model, noise_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise, disable_noise=disable_noise,
                start_step=start_at_step, last_step=end_at_step,
                force_full_denoise=force_full_denoise,
            )

        # --- install the multiscale/tiling wrapper on a model clone ---
        # Tile geometry and per-step sigma mapping are resolved lazily inside the
        # wrapper (from the latent size and transformer_options["sample_sigmas"]).
        model_clone = _patch_model(
            model, tiles_h, tiles_w, overlap_h, overlap_w,
            scale_sched=scale_sched, bypass_tiling=bypass_tiling,
            reference_latent_full=reference_latent_full, debug=debug,
        )

        return self._plain_sample(
            model_clone, noise_seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise, disable_noise=disable_noise,
            start_step=start_at_step, last_step=end_at_step,
            force_full_denoise=force_full_denoise,
        )

    # ------------------------------------------------------------------

    def _plain_sample(
        self, model, seed, steps, cfg, sampler_name, scheduler,
        positive, negative, latent,
        denoise=1.0, disable_noise=False,
        start_step=None, last_step=None, force_full_denoise=True,
    ):
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model, latent_image, latent.get("downscale_ratio_spacial", None)
        )

        if disable_noise:
            noise = torch.zeros(
                latent_image.size(), dtype=latent_image.dtype,
                layout=latent_image.layout, device="cpu"
            )
        else:
            batch_inds = latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = latent.get("noise_mask", None)
        callback    = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise, disable_noise=disable_noise,
            start_step=start_step, last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed,
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return (out,)


class WanTiledSamplerPatch:
    """
    Model patch (MODEL → MODEL) that installs the same per-step MultiDiffusion
    tiling + multiscale scheduling as WanTiledSampler, but as a wrapper on the
    model — so it works with any vanilla sampler (KSampler, KSamplerAdvanced,
    SamplerCustom…).  Plug the patched MODEL into your sampler of choice.

    Unlike the all-in-one node, the schedule's step indices are resolved at
    sample time from transformer_options["sample_sigmas"], so no sampler settings
    need to be known here, and a WAN 2.2 two-sampler (high/low-noise) split gets
    correct per-pass local step indices automatically.

    Apply this patch LAST in any chain of model patches: tiling needs the
    model_function_wrapper slot.  An existing wrapper is chained, but a later
    patch that overwrites the slot without chaining would drop this one.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tiles_h": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of spatial tiles along the height axis. Set 1 to disable.",
                }),
                "tiles_w": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of spatial tiles along the width axis. Set 1 to disable.",
                }),
                "overlap_h": ("INT", {
                    "default": 25, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Overlap between height tiles, % of tile height (0–50%).",
                }),
                "overlap_w": ("INT", {
                    "default": 25, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Overlap between width tiles, % of tile width (0–50%).",
                }),
            },
            "optional": {
                "scale_schedule": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Multiscale (coarse-to-fine) schedule, step → resolution %. "
                               "e.g. {0:25, 10:50, 20:100} runs the WHOLE frame downscaled to "
                               "25% for steps 0–9, 50% for 10–19, then full-res from 20 (held "
                               "until next key). While scale < 100% the frame runs in one pass "
                               "(no tiling) so global motion stays coherent — the I2V fix. "
                               "Step indices are local to each sampler pass. Empty = always 100%.",
                }),
                "bypass_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip tiling (unless a scale_schedule is set).",
                }),
                "reference_latent_full": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pass the full reference frame to every tile (I2V global context) "
                               "instead of cropping it to the tile region. Keep on for I2V.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print tile layout, blend-weight sanity check, and per-step "
                               "scale/tile decisions to the console.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Mago Nodes/Sampling"

    def patch(self, model, tiles_h, tiles_w, overlap_h, overlap_w,
              scale_schedule="", bypass_tiling=False,
              reference_latent_full=True, debug=False):
        scale_sched = _parse_schedule(scale_schedule)
        m = _patch_model(
            model, tiles_h, tiles_w, overlap_h, overlap_w,
            scale_sched=scale_sched, bypass_tiling=bypass_tiling,
            reference_latent_full=reference_latent_full, debug=debug,
        )
        return (m,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WanTiledSampler": WanTiledSampler,
    "WanTiledSamplerPatch": WanTiledSamplerPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTiledSampler": "WAN Tiled Sampler",
    "WanTiledSamplerPatch": "WAN Tiled Sampler (Model Patch)",
}
