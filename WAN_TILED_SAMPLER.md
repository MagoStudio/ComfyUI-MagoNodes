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

## References

- Bar-Tal et al. (2023). *MultiDiffusion: Fusing Diffusion Paths for Controlled
  Image Generation.* https://multidiffusion.github.io
- Lightricks. *LTX-Video official tiling implementation* (`ltx_core/tiling.py`,
  `ltx_core/modality_tiling.py`) — source of the trapezoidal blend window
  formula and the `pad_to_patch_size` / circular-padding pattern.
- 10S Nodes https://github.com/TenStrip/10S-Comfy-nodes - Investigation on long-context sampling degradation.
- ComfyUI `model_function_wrapper` interface (`comfy/samplers.py` line ~322) —
  the hook point used to intercept every model evaluation.
