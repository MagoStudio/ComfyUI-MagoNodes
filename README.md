# ComfyUI-MagoNodes

MagoStudio nodes for ComfyUI. Currently a single node: **WAN Tiled Sampler**.

## WAN Tiled Sampler

A drop-in replacement for `KSamplerAdvanced` that adds per-step **MultiDiffusion**
spatial tiling for WAN 2.1 / 2.2 video models.

**Works for video2video modes only**!

### Why

WAN models are trained at specific spatial resolutions. Sampling at higher
resolutions (e.g. 1080p) pushes token counts outside the training distribution
and causes hue shift / colour drift, diluted conditioning, and out-of-range
positional encodings.

This node splits each denoising step into overlapping spatial tiles, runs the
model on each tile at a training-distribution token count, and blends the
predictions back into a single full-resolution estimate *before* the sampler
takes its step. Because blending happens at every sigma step, all tiles share one
denoising trajectory, so content never diverges between tiles.

Tiling is **spatial only** — temporal frames are never split, so causal temporal
attention stays intact. It works with WAN Fun Control, VACE, and I2V workflows
(conditioning tensors like `c_concat`, `denoise_mask`, `reference_latent`, and
`vace_context` are sliced per tile).

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
| `bypass_tiling` | False | Skip tiling entirely — behaves like a plain `KSamplerAdvanced`. |
| `reference_latent_full` | True | Pass the full reference frame to every tile (I2V global context) instead of cropping it. Keep on for I2V. |
| `debug` | False | Print tile layout, blend-weight sanity check, and per-step slice info to the console. |

Overlap is a percentage so it stays resolution-independent. Guidance:

- **12%** — minimal; avoids obvious seams in mostly static scenes.
- **25%** — recommended; handles moderate cross-tile motion (the default).
- **50%** — maximum continuity (tiled-VAE style) at roughly 2× the compute.

### Tips

- Setting both `tiles_h` and `tiles_w` to `1` (or enabling `bypass_tiling`) makes
  the node identical to `KSamplerAdvanced`.
- More tiles = lower per-tile token count and memory, but more model evaluations
  per step. Start with 2×2 and increase only if you still see drift or run out of
  VRAM.
- WAN distilled models typically use `cfg = 1`.

## References

- Bar-Tal et al. (2023). *MultiDiffusion: Fusing Diffusion Paths for Controlled
  Image Generation.* <https://multidiffusion.github.io>
- Lightricks — *LTX-Video official tiling implementation* (source of the
  trapezoidal blend-window formula).
- [10S Nodes](https://github.com/TenStrip/10S-Comfy-nodes) — investigation on
  long-context sampling degradation.
