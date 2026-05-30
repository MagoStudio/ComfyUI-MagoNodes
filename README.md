# ComfyUI-MagoNodes

MagoStudio nodes for ComfyUI. Currently a single node: **WAN Tiled Sampler**.

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

Find the node under **Mago Nodes / Sampling → WAN Tiled Sampler**. Use it exactly
like `KSamplerAdvanced` — same inputs (model, conditioning, latent, sampler,
scheduler, steps, cfg, seed, start/end step) — plus the tiling controls below.

| Input | Default | Description |
|-------|---------|-------------|
| `tiles_h` | 2 | Number of tiles along the height axis. `1` disables height tiling. |
| `tiles_w` | 2 | Number of tiles along the width axis. `1` disables width tiling. |
| `overlap_h` | 25 | Overlap between height tiles, as a percentage of tile height (0–50%). |
| `overlap_w` | 25 | Overlap between width tiles, as a percentage of tile width (0–50%). |
| `scale_schedule` | `""` | Multiscale schedule, step → resolution % (see below). Empty = always full-res. |
| `tile_schedule` | `""` | Per-step tiling toggle, step → bypass flag (`1` = whole-frame, `0` = tile). Empty = use `bypass_tiling`. |
| `bypass_tiling` | False | Skip tiling entirely — behaves like a plain `KSamplerAdvanced` (unless a schedule is set). |
| `reference_latent_full` | True | Pass the full reference frame to every tile (I2V global context) instead of cropping it. Keep on for I2V. |
| `debug` | False | Print tile layout, blend-weight sanity check, per-step scale/tile decisions, and slice info to the console. |

Overlap is a percentage so it stays resolution-independent. Guidance:

- **12%** — minimal; avoids obvious seams in mostly static scenes.
- **25%** — recommended; handles moderate cross-tile motion (the default).
- **50%** — maximum continuity (tiled-VAE style) at roughly 2× the compute.

### Multiscale scheduling

Both schedules are written as `{step: value, ...}` (commas optional, whitespace
ignored). Each value is **held until the next key**, and step `0` is the first
sampled step.

`scale_schedule` sets the resolution per step as a percentage of the latent's
native size:

```
{0:25, 10:50, 20:100}
```

> Steps 0–9 run the whole frame at 25%, steps 10–19 at 50%, then 20→end at full
> resolution. While scale < 100% the frame is evaluated in **one whole-frame
> pass** (tiling is skipped, so global motion stays coherent); at 100% the node
> tiles for detail.

`tile_schedule` independently controls whether each full-res step tiles
(`0`) or runs whole-frame (`1`):

```
{0:1, 20:0}
```

> Whole-frame for steps 0–19, tiled from 20. (Redundant with `scale_schedule`
> while scale < 100%, since low-res steps never tile.)

**Recommended I2V recipe** — pair a coarse-to-fine scale ramp with tiling only at
the end:

```
scale_schedule = {0:50, 15:100}
tile_schedule  = {0:1, 15:0}
```

### Tips

- Leave both schedules empty for the original behaviour: pure full-res tiling.
- Setting both `tiles_h` and `tiles_w` to `1` (or enabling `bypass_tiling`) with
  no schedules makes the node identical to `KSamplerAdvanced`.
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

## References

- Bar-Tal et al. (2023). *MultiDiffusion: Fusing Diffusion Paths for Controlled
  Image Generation.* <https://multidiffusion.github.io>
- Lightricks — *LTX-Video official tiling implementation* (source of the
  trapezoidal blend-window formula).
- [10S Nodes](https://github.com/TenStrip/10S-Comfy-nodes) — investigation on
  long-context sampling degradation.
