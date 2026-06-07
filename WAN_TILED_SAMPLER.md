# WAN Tiled Sampler — Design Notes

## Problem

WAN models are trained on videos at specific spatial resolutions.  When sampling
at higher resolutions (e.g. 1920×1080), the token count grows quadratically and
the model operates outside its training distribution.  Symptoms:

- Hue shift / colour drift across the output
- Conditioning strength diluted (fewer text-conditioning tokens per spatial unit)
- Positional encodings outside the trained range

The same root cause was identified in LTX-Video's upscale pass and documented
in the 10S Nodes.

## Approach: per-step MultiDiffusion

**MultiDiffusion** (Bar-Tal et al., 2023) proposes running a diffusion model on
overlapping spatial tiles at each denoising step and blending the per-tile
predictions back into a single full-resolution estimate before the sampler takes
its step.

Key property: because blending happens *at every sigma step*, all tiles share a
single denoising trajectory.  Content never diverges between tiles the way it
would if each tile ran an independent full schedule.

### Blend windows

Each tile's prediction is weighted by a 2-D trapezoidal window — linear fade
from 0→1 over the overlap region, flat 1 in the non-overlapping centre.
Windows are designed so their sum equals exactly 1.0 at every spatial position
(verified by a debug sanity check).  The formula matches the
`compute_trapezoidal_mask_1d` used in ltx_core/tiling.py.

### Implementation

The wrapper is installed on `model.model_options['model_function_wrapper']`
before calling `comfy.sample.sample`.  ComfyUI calls this wrapper for every
model evaluation inside the sampler (every sigma step, both cond and uncond
passes).  The wrapper:

1. Receives `{"input": x_full, "timestep": t, "c": c, "cond_or_uncond": ...}`
2. Slices `x_full` and all spatially-shaped conditioning tensors per tile
3. Calls `model_fn` (or an existing chained wrapper) for each tile
4. Blends predictions with trapezoidal windows and returns the full result

Any pre-existing `model_function_wrapper` on the model (e.g. from other nodes)
is chained — our wrapper calls it instead of `model_fn` directly for each tile.

## Conditioning tensors sliced per tile

| Key | Shape | Notes |
|-----|-------|-------|
| `c_concat` | `(B, extra_ch, F, H, W)` | Fun Control reference video + mask |
| `denoise_mask` | `(B, C, F, H, W)` | WAN 2.2 per-frame inpainting mask |
| `reference_latent` | `(B, C, H, W)` | I2V reference frame (optional full-frame mode) |
| `vace_context` | `(B, refs, ch, F, H, W)` | VACE control video |

### VACE padding subtlety

WAN's `_forward` calls `pad_to_patch_size(x, patch_size=(1,2,2))` with
**circular** padding before `patch_embedding`, bringing odd spatial dimensions
up to the next even number.  `vace_context` bypasses this step and goes
directly into `vace_patch_embedding`.  Without correction, a tile with H=49
would have `patch_embedding` produce 25 H-patches (from padded H=50) while
`vace_patch_embedding` produces 24 (from unpadded H=49) → token count
mismatch at the VACE attention block.

Fix: slice the vace tile, then apply the equivalent circular padding manually
before setting it in the conditioning dict.  Because `torch.nn.functional.pad`
only supports non-constant padding on tensors up to 5D, the 6-D vace tensor
`(B, refs, ch, F, H, W)` is reshaped to `(B*refs, ch, F, H, W)`, padded, then
reshaped back.

### Reference latent — full vs cropped

`reference_latent` is processed by `ref_conv` (a Conv2d) whose output tokens
are **appended** to the attention sequence.  There is no size constraint between
these tokens and the tile's generated tokens.  The `reference_latent_full`
option (default on) therefore passes the complete reference frame to every tile,
giving each tile global scene context rather than only its own spatial crop.

## Overlap as percentage

Overlap is expressed as a percentage of tile size (0–50%) rather than latent
pixels so the value is resolution-independent.  Internally it is converted to
latent pixels as `round(tile_size * pct / 100)`.

Guidance:
- **12%** ≈ 64px at 1080p — minimal, avoids obvious seams in static scenes
- **25%** ≈ 160px at 1080p — recommended, handles moderate cross-tile motion
- **50%** ≈ 320px at 1080p — tiled-VAE style, maximum continuity, ~2× compute

## Multiscale (coarse-to-fine) scheduling

Tiling solves the *spatial* token-count problem but not the *temporal coherence*
problem in I2V.  Each tile runs its own denoising trajectory for motion, and
while MultiDiffusion blending keeps the trajectories tied at the latent level,
the model still decides motion per tile from local context.  In V2V the control
video pins the motion, so tiles agree; in I2V there is no such pin and tiles
drift apart over the sequence even though the first frame / reference is honoured.

The fix is to decide global motion *before* tiling, at low resolution.  WAN's
`rope_encode(t_len, h, w, …)` derives positional encodings directly from the
latent's spatial dims, and `_forward` crops its output back to the input dims.
So feeding a **downscaled whole-frame latent** into a single model call:

- brings token count and positional encodings back inside the training range
  (the original high-res motivation), and
- gives the model the *entire* frame as context, so motion is decided once,
  coherently, instead of per tile.

### Two schedules

`scale_schedule` maps step → resolution %, held until the next key:
`{0:25, 10:50, 20:100}`.  While the scheduled scale is < 100% the wrapper
downscales the latent and every spatially-shaped conditioning tensor, runs one
whole-frame evaluation, and upscales the x0 prediction back (`bilinear`).
Targets are rounded to the nearest even number so the spatial patch size (2)
divides cleanly and the VACE `pad_to_patch_size` mismatch above cannot occur in
the downscale path.  Tiling is skipped while scale < 100%.

### Downscale method — preserving the noise level

Unlike Kohya Deep Shrink (`PatchModelAddDownscale`), which downscales a *deep
feature map* inside the UNet — where the noise has already been absorbed — this
node downscales the *raw input latent* (the only lever a DiT like WAN exposes:
its first op is patchify, so a smaller latent simply produces fewer tokens).

That distinction matters at high sigma, where the latent is noise-dominated.
Averaging downsamplers (`area`, `bilinear`) collapse the variance of i.i.d.
noise — pooling a k×k block divides its variance by k² — so the model would
receive a latent whose noise level no longer matches `timestep` and would emit
pure noise in the generated frames (the conditioned frame 0 still looks fine,
since its inpaint mask is applied at full resolution outside the wrapper).

The fix: the **noisy latent is downscaled by subsampling (`nearest-exact`)**.
Subsampling keeps every k-th sample, so the noise stays unit-variance and the
level still matches sigma, for both flow- and eps-parameterised models.  The
*conditioning* tensors (`c_concat`, `reference_latent`, `vace_context`) are clean
signal, not noise, so they use antialiased `area`; `denoise_mask` uses
`nearest-exact` to keep its values crisp.

`tile_schedule` maps step → bypass flag (`1` = whole-frame, `0` = tile) and
overrides the `bypass_tiling` toggle at full resolution.

### Mapping a model call to a step

ComfyUI calls the wrapper with the current **sigma** as `timestep`
(`sampling_function(..., timestep=sigma)` in `comfy/samplers.py`), not a step
index.  The full per-pass sigma schedule is also handed to every model call as
`transformer_options["sample_sigmas"]` (set in `CFGGuider.inner_sample`), so the
wrapper maps the incoming sigma to the nearest entry of that schedule to recover
a 0-based step index — local to the current sampler pass, and robust to samplers
that evaluate the model more than once per step (intermediate sigmas snap to the
closest scheduled step).

Because `sample_sigmas` is read at call time, the wrapper needs no knowledge of
the sampler's `steps` / `scheduler` / `start_step` / `denoise`.  That is what
lets the same wrapper power both the all-in-one **sampler** node and the
**model-patch** node (`MODEL → MODEL`, Kohya-Deep-Shrink style), and what makes a
WAN 2.2 high/low-noise two-sampler split get correct per-pass step indices for
free.  Tile geometry, which depends on the latent's H×W, is likewise resolved
lazily on the first model call and cached per resolution, since the patch node
doesn't see the latent ahead of time.

## References

- Bar-Tal et al. (2023). *MultiDiffusion: Fusing Diffusion Paths for Controlled
  Image Generation.* https://multidiffusion.github.io
- Lightricks. *LTX-Video official tiling implementation* (`ltx_core/tiling.py`,
  `ltx_core/modality_tiling.py`) — source of the trapezoidal blend window
  formula and the `pad_to_patch_size` / circular-padding pattern.
- 10S Nodes https://github.com/TenStrip/10S-Comfy-nodes - Investigation on long-context sampling degradation.
- ComfyUI `model_function_wrapper` interface (`comfy/samplers.py` line ~322) —
  the hook point used to intercept every model evaluation.
