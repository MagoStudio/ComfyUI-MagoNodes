# ComfyUI-MagoNodes

MagoStudio nodes for ComfyUI:

- **WAN Tiled Sampler** — per-step MultiDiffusion tiling + multiscale scheduling
  for WAN 2.1 / 2.2 (all-in-one sampler node and a model-patch node).
- **LTX Tiled Sampler (Model Patch)** — per-step MultiDiffusion T×H×W tiling for
  LTX-Video, as a model patch for the vanilla `SamplerCustomAdvanced`.

## WAN Tiled Sampler

A drop-in replacement for `KSamplerAdvanced` that adds per-step **MultiDiffusion**
spatial tiling **and** multiscale (coarse-to-fine) scheduling for WAN 2.1 / 2.2
video models.

### Why

WAN models are trained at specific spatial resolutions. Sampling at higher
resolutions (e.g. 1080p) pushes token counts outside the training distribution
and causes hue shift / colour drift, diluted conditioning, and out-of-range
positional encodings.

**Tiling.** The node splits each denoising step into overlapping spatial tiles,
runs the model on each tile at a training-distribution token count, and blends
the predictions back into a single full-resolution estimate *before* the sampler
takes its step. Because blending happens at every sigma step, all tiles share one
denoising trajectory, so content never diverges between tiles. Tiling is
**spatial only** — temporal frames are never split, so causal temporal attention
stays intact.

Tiling works great for **V2V**, where the conditioning (control video, masks)
splits cleanly per tile. For **I2V** it isn't enough on its own: each tile's
*motion* is decided independently, so even though the first frame and reference
are honoured, the tiles drift apart over time.

**Multiscale (the I2V fix).** The `scale_schedule` runs the *whole* frame
downscaled during the early, high-noise steps — when global motion and layout are
decided — then raises the resolution and hands off to tiling for detail. WAN
derives its positional encodings from the latent's spatial size, so a smaller
latent automatically brings token count and positions back in-distribution. One
shared low-res pass keeps motion globally coherent; the later full-res tiled
passes add the sharpness.

Conditioning tensors (`c_concat`, `denoise_mask`, `reference_latent`,
`vace_context`) are sliced per tile and rescaled per scale step, so WAN Fun
Control, VACE, and I2V workflows all work.

See [WAN_TILED_SAMPLER.md](WAN_TILED_SAMPLER.md) for the full design notes.

### Installation

Clone (or copy) this repo into your ComfyUI `custom_nodes` folder and restart
ComfyUI:

```
ComfyUI/custom_nodes/ComfyUI-MagoNodes/
```

There are no extra dependencies.

### Usage

The pack ships **two nodes**, both under **Mago Nodes / Sampling**, that do the
same thing in different shapes — pick one:

- **WAN Tiled Sampler** — an all-in-one sampler. Use it exactly like
  `KSamplerAdvanced` (same inputs: model, conditioning, latent, sampler,
  scheduler, steps, cfg, seed, start/end step) plus the tiling/scale controls.
- **WAN Tiled Sampler (Model Patch)** — a `MODEL → MODEL` patch (Kohya
  Deep-Shrink style). Drop it between your model and **any** vanilla sampler
  (`KSampler`, `KSamplerAdvanced`, `SamplerCustom`…). Same tiling/scale controls,
  no sampler settings. **Apply it last** in a chain of model patches — it needs
  the `model_function_wrapper` slot (an existing wrapper is chained, but a later
  patch that overwrites the slot would drop it). This is the better fit for the
  WAN 2.2 two-sampler split: patch each model and the per-pass step indices are
  resolved automatically.

The tiling/scale controls are identical for both nodes:

| Input | Default | Description |
|-------|---------|-------------|
| `tiles_h` | 2 | Number of tiles along the height axis. `1` disables height tiling. |
| `tiles_w` | 2 | Number of tiles along the width axis. `1` disables width tiling. |
| `overlap_h` | 25 | Overlap between height tiles, as a percentage of tile height (0–50%). |
| `overlap_w` | 25 | Overlap between width tiles, as a percentage of tile width (0–50%). |
| `scale_schedule` | `""` | Multiscale schedule, step → resolution % (see below). Empty = always full-res. |
| `bypass_tiling` | False | Skip tiling entirely — behaves like a plain `KSamplerAdvanced` (unless a `scale_schedule` is set). |
| `reference_latent_full` | True | Pass the full reference frame to every tile (I2V global context) instead of cropping it. Keep on for I2V. |
| `debug` | False | Print tile layout, blend-weight sanity check, per-step scale/tile decisions, and slice info to the console. |

Overlap is a percentage so it stays resolution-independent. Guidance:

- **12%** — minimal; avoids obvious seams in mostly static scenes.
- **25%** — recommended; handles moderate cross-tile motion (the default).
- **50%** — maximum continuity (tiled-VAE style) at roughly 2× the compute.

### Multiscale scheduling

`scale_schedule` is written as `{step: value, ...}` (commas optional, whitespace
ignored). Each value is **held until the next key**, step `0` is the first
sampled step, and the value is a percentage of the latent's native size:

```
{0:25, 10:50, 20:100}
```

> Steps 0–9 run the whole frame at 25%, steps 10–19 at 50%, then 20→end at full
> resolution. While scale < 100% the frame is evaluated in **one whole-frame
> pass** (tiling is skipped, so global motion stays coherent); at 100% the node
> tiles for detail. So the schedule alone drives the coarse-to-fine handoff —
> low-res whole-frame early, full-res tiled late.

**Recommended I2V recipe** — a gentle coarse-to-fine ramp:

```
scale_schedule = {0:50, 15:100}
```

### Tips

- Leave `scale_schedule` empty for the original behaviour: pure full-res tiling.
- Setting both `tiles_h` and `tiles_w` to `1` (or enabling `bypass_tiling`) with
  no schedule makes the node identical to `KSamplerAdvanced`.
- More tiles = lower per-tile token count and memory, but more model evaluations
  per step. Start with 2×2 and increase only if you still see drift or run out of
  VRAM.
- The scale schedule's step indices are relative to the sampled range, so they
  respect `start_at_step` / `end_at_step`.
- The schedule is per **sampler pass**: in a WAN 2.2 high/low-noise (two-sampler)
  setup each node has its own `scale_schedule`, and step indices restart at 0 for
  each. The latent handed between passes is always full resolution — the
  downscaling lives inside each model eval and never persists — so a second pass
  starting at `{0:100}` just runs full-res.
- WAN distilled models typically use `cfg = 1`.

## LTX Tiled Sampler (Model Patch)

A `MODEL → MODEL` patch that adds per-step MultiDiffusion tiling along the
temporal (T), height (H), and width (W) axes to an LTX-Video model. It works with
the vanilla **`SamplerCustomAdvanced`** — no custom sampler needed.

### Wiring

Insert it on the model line, **after** your LoRA / ICLoRA loaders and **before**
the guider:

```
… → LTX ICLoRA Loader → LTX Tiled Sampler (Model Patch) → CFGGuider → SamplerCustomAdvanced
```

The guide conditioning (`keyframe_idxs` / `guide_attention_entries`) and
`denoise_mask` are cropped to each tile automatically, so ICLoRA / guide
workflows keep working. Apply this patch **last** in any chain of model patches —
it claims the `model_function_wrapper` slot (an existing wrapper is chained, but a
later patch that overwrites the slot would drop it).

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `tiles_h` | 2 | Tiles along height. `1` disables. (Official LTX Stage-2: 2.) |
| `tiles_w` | 2 | Tiles along width. `1` disables. (Official LTX Stage-2: 2.) |
| `overlap_h` | 6 | Overlap in latent units between H tiles (≈48 px at 8× VAE). |
| `overlap_w` | 6 | Overlap in latent units between W tiles. |
| `tiles_t` | 1 | Tiles along frames. `1` disables (recommended for most clips). |
| `overlap_t` | 8 | Overlap in latent frames between temporal tiles. |
| `bypass_tiling` | False | Return the model unpatched (single-pass). |
| `debug` | False | Print tile layout, blend-weight check, and per-tile guide info. |

Tiling is **per-step MultiDiffusion** — predictions are blended at every sigma
step so all tiles stay on one trajectory (no temporal ghosting from independent
per-tile schedules).

**Credits.** The trapezoidal blend window, the Stage-2 tiling geometry (2×2
spatial, 1 temporal), and the default overlaps follow the official Lightricks
LTX-Video implementation (`ltx_core/tiling.py`), as do the causal-VAE coordinate
conventions (temporal scale 8, spatial scale 32). The per-tile guide handling
(`keyframe_idxs` / `guide_attention_entries`) is based on the
[10S Nodes](https://github.com/TenStrip/10S-Comfy-nodes) LTX tiled sampler, with
the coordinate / grid-mask logic reworked to fix correctness bugs in that
implementation.

## References

- Bar-Tal et al. (2023). *MultiDiffusion: Fusing Diffusion Paths for Controlled
  Image Generation.* <https://multidiffusion.github.io>
- Lightricks — *LTX-Video official tiling implementation* (source of the
  trapezoidal blend-window formula).
- [10S Nodes](https://github.com/TenStrip/10S-Comfy-nodes) — investigation on
  long-context sampling degradation.
