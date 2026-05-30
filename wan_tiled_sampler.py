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
# MultiDiffusion wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _make_wan_tile_wrapper(tile_specs, existing_wrapper, debug=False, reference_latent_full=False):
    """
    Returns a model_function_wrapper implementing per-step MultiDiffusion spatial
    tiling for WAN models.

    At each model call the wrapper:
      1. Tiles the full noisy latent (B, C, F, H, W) into spatial patches.
      2. Slices spatially-shaped conditioning tensors per tile:
           c_concat       — (B, extra_ch, F, H, W)  Fun Control reference+mask
           denoise_mask   — (B, C, F, H, W)          WAN2.2 inpainting mask
           reference_latent — (B, C, H, W)           reference frame (WAN2.1 I2V)
      3. Calls the model for each tile independently.
      4. Blends tile predictions into a full-size output with pre-computed 2-D
         trapezoidal windows (weights sum to 1.0 everywhere).

    tile_specs: list of dicts with keys
        h_start, h_end, w_start, w_end  — latent-space bounds
        window_2d                        — (1,1,1,Ht,Wt) fp32 blend weights
    existing_wrapper: outer model_function_wrapper to chain after tile call, or None.
    """
    def wrapper(model_fn, input_dict):
        x_full = input_dict['input']
        B, C, F, H, W = x_full.shape
        dtype  = x_full.dtype
        device = x_full.device

        accum = torch.zeros(B, C, F, H, W, dtype=torch.float32, device=device)
        w_acc = torch.zeros(1, 1, F, H, W, dtype=torch.float32, device=device)

        base_c = input_dict.get('c', {})

        for spec in tile_specs:
            hs, he = spec['h_start'], spec['h_end']
            ws, we = spec['w_start'], spec['w_end']
            window_2d = spec['window_2d'].to(device=device, dtype=torch.float32)

            tile_x = x_full[:, :, :, hs:he, ws:we].contiguous()
            tile_c = dict(base_c)

            # --- c_concat: (B, extra_ch, F, H, W) → slice spatial dims ---
            cc = tile_c.get('c_concat', None)
            if cc is not None and isinstance(cc, torch.Tensor) and cc.dim() == 5:
                try:
                    if cc.shape[-2] == H and cc.shape[-1] == W:
                        tile_c['c_concat'] = cc[:, :, :, hs:he, ws:we].contiguous()
                    elif debug:
                        print(f"  [wan_tile] c_concat spatial mismatch: "
                              f"{tuple(cc.shape)} vs x {tuple(x_full.shape)}")
                except Exception:
                    pass

            # --- denoise_mask: 5D (B,C,F,H,W) or 4D (B,F,H,W) → slice spatial dims ---
            dm = tile_c.get('denoise_mask', None)
            if dm is not None and isinstance(dm, torch.Tensor):
                try:
                    if dm.dim() == 5 and dm.shape[-2] == H and dm.shape[-1] == W:
                        tile_c['denoise_mask'] = dm[:, :, :, hs:he, ws:we].contiguous()
                    elif dm.dim() == 4 and dm.shape[-2] == H and dm.shape[-1] == W:
                        tile_c['denoise_mask'] = dm[:, :, hs:he, ws:we].contiguous()
                except Exception:
                    pass

            # --- reference_latent: (B, C, H, W) → optionally slice spatial dims ---
            # When reference_latent_full=True the full reference frame is passed to every
            # tile unchanged.  ref_conv appends its tokens to the sequence (no size
            # constraint), so the full image provides global scene context to all tiles.
            if not reference_latent_full:
                rl = tile_c.get('reference_latent', None)
                if rl is not None and isinstance(rl, torch.Tensor) and rl.dim() == 4:
                    try:
                        if rl.shape[-2] == H and rl.shape[-1] == W:
                            tile_c['reference_latent'] = rl[:, :, hs:he, ws:we].contiguous()
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
            vc = tile_c.get('vace_context', None)
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
                            t5 = torch.nn.functional.pad(t5, pad, mode='circular')
                            tile_vc = t5.reshape(B_vc, nr, ch_vc, *t5.shape[2:])
                        tile_c['vace_context'] = tile_vc.contiguous()
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

            tile_input_dict = {**input_dict, 'input': tile_x, 'c': tile_c}
            if existing_wrapper is not None:
                tile_pred = existing_wrapper(model_fn, tile_input_dict)
            else:
                tile_pred = model_fn(tile_x, input_dict['timestep'], **tile_c)

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

    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class WanTiledSampler:
    """
    Drop-in replacement for KSamplerAdvanced with per-step MultiDiffusion spatial
    tiling for WAN 2.1 / 2.2 models.

    Tiling keeps each spatial patch at training-distribution token counts, which
    prevents the hue shift / conditioning drift that appears when sampling very
    large latents in a single pass.

    Works with WAN Fun Control (Wan22FunControlToVideo) by slicing c_concat,
    denoise_mask, and reference_latent per tile.  Temporal frames are never split
    (spatial-only tiling), so causal temporal attention is never broken.
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
                "bypass_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip tiling entirely — equivalent to plain KSamplerAdvanced.",
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
                    "tooltip": "Print tile layout, blend weight sanity check, and "
                               "per-step conditioning slice info.",
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
        bypass_tiling=False,
        reference_latent_full=True,
        debug=False,
        denoise=1.0,
    ):
        force_full_denoise = return_with_leftover_noise != "enable"
        disable_noise      = add_noise == "disable"

        # --- bypass: plain KSamplerAdvanced behaviour ---
        if bypass_tiling or (tiles_h <= 1 and tiles_w <= 1):
            return self._plain_sample(
                model, noise_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise, disable_noise=disable_noise,
                start_step=start_at_step, last_step=end_at_step,
                force_full_denoise=force_full_denoise,
            )

        # --- build tile specs ---
        latent = latent_image["samples"]
        # latent may be 4D (image) or 5D (video)
        if latent.dim() == 4:
            B, C, H, W = latent.shape
            F = 1
        else:
            B, C, F, H, W = latent.shape

        # Convert overlap % → latent pixels based on tile size.
        # Tile size isn't known yet so approximate from total size / n_tiles,
        # then recompute after _compute_tile_starts with the real tile_h/tile_w.
        def _pct_to_px(total, n_tiles, pct):
            if n_tiles <= 1 or pct == 0:
                return 0
            approx_tile = math.ceil(total / n_tiles)
            return max(0, round(approx_tile * pct / 100))

        overlap_h_px = _pct_to_px(H, tiles_h, overlap_h)
        overlap_w_px = _pct_to_px(W, tiles_w, overlap_w)

        h_starts, tile_h = _compute_tile_starts(H, tiles_h, overlap_h_px)
        w_starts, tile_w = _compute_tile_starts(W, tiles_w, overlap_w_px)

        # Recompute with actual tile size for accuracy
        overlap_h_px = max(0, round(tile_h * overlap_h / 100))
        overlap_w_px = max(0, round(tile_w * overlap_w / 100))
        h_starts, tile_h = _compute_tile_starts(H, tiles_h, overlap_h_px)
        w_starts, tile_w = _compute_tile_starts(W, tiles_w, overlap_w_px)

        if debug:
            print(f"[WanTiledSampler] latent={tuple(latent.shape)} "
                  f"tiles_h={tiles_h} tiles_w={tiles_w} "
                  f"overlap_h={overlap_h}% ({overlap_h_px}px) "
                  f"overlap_w={overlap_w}% ({overlap_w_px}px)")
            print(f"  h_starts={h_starts} tile_h={tile_h}")
            print(f"  w_starts={w_starts} tile_w={tile_w} overlap_w_px={overlap_w_px}")

        tile_specs = []
        for hi, hs in enumerate(h_starts):
            he = min(hs + tile_h, H)
            fade_left_h  = max(0, h_starts[hi - 1] + tile_h - hs) if hi > 0 else 0
            fade_right_h = max(0, hs + tile_h - h_starts[hi + 1]) if hi < len(h_starts) - 1 else 0

            for wi, ws in enumerate(w_starts):
                we = min(ws + tile_w, W)
                fade_left_w  = max(0, w_starts[wi - 1] + tile_w - ws) if wi > 0 else 0
                fade_right_w = max(0, ws + tile_w - w_starts[wi + 1]) if wi < len(w_starts) - 1 else 0

                win_h = _make_window_1d(he - hs, fade_left_h, fade_right_h,
                                        torch.float32, 'cpu')
                win_w = _make_window_1d(we - ws, fade_left_w, fade_right_w,
                                        torch.float32, 'cpu')
                # shape (1, 1, 1, Ht, Wt) — broadcast over (B, C, F)
                window_2d = (win_h[:, None] * win_w[None, :]).reshape(1, 1, 1, he - hs, we - ws)

                tile_specs.append({
                    'h_start': hs, 'h_end': he,
                    'w_start': ws, 'w_end': we,
                    'window_2d': window_2d,
                })

                if debug:
                    print(f"  tile h:[{hs}:{he}] w:[{ws}:{we}] "
                          f"fade_h=({fade_left_h},{fade_right_h}) "
                          f"fade_w=({fade_left_w},{fade_right_w})")

        # blend weight sanity check
        if debug:
            w_test = torch.zeros(1, 1, 1, H, W, dtype=torch.float32)
            for spec in tile_specs:
                hs, he = spec['h_start'], spec['h_end']
                ws, we = spec['w_start'], spec['w_end']
                w_test[:, :, :, hs:he, ws:we] += spec['window_2d']
            mn = w_test.min().item()
            mx = w_test.max().item()
            print(f"  blend weight sanity: min={mn:.4f} max={mx:.4f} "
                  f"({'OK' if abs(mn - 1.0) < 0.02 and abs(mx - 1.0) < 0.02 else 'WARN'})")

        # --- install wrapper on model clone ---
        model_clone = model.clone()
        existing_wrapper = model_clone.model_options.get('model_function_wrapper', None)
        tile_wrapper = _make_wan_tile_wrapper(tile_specs, existing_wrapper, debug=debug,
                                              reference_latent_full=reference_latent_full)
        model_clone.model_options['model_function_wrapper'] = tile_wrapper

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


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WanTiledSampler": WanTiledSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanTiledSampler": "WAN Tiled Sampler",
}
