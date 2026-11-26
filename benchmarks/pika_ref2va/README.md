# Sol³ vs usual attention on MiniMax H3 Ref2VA

On one NVIDIA GeForce RTX 5090, Sol³ saved **10.92 seconds of sampling** on a 5-second clip and **199.70 seconds** on a 15-second clip. Both branches used the Pika Flex Turbo HD Ref2VA LoRA at strength 1, eight Euler steps and the checkpoint's original output heads, without a PDD adapter.

## Measured sampler time

| Clip | Native → exported frames | Usual attention | Sol³ | Time saved | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 s | 124 → 120 | 94.06 s | 83.14 s | 10.92 s (11.61%) | 1.131× |
| 10 s | 243 → 240 | 290.79 s | 204.45 s | 86.34 s (29.69%) | 1.422× |
| 15 s | 362 → 360 | 595.77 s | 396.07 s | 199.70 s (33.52%) | 1.504× |

Each row is **one measured pair after one warmup pair** at that clip length. Both branches shared a noise seed within each pair. All three lengths used seed `245381526188768` for warmup and `245381526188769` for measurement. The runner timed each `SamplerCustomAdvanced` node from its running to finished events on the ComfyUI WebSocket. VAE decoding and MP4 encoding happen after that interval. Speedup is usual sampler time divided by Sol³ sampler time. Use the paired MP4s to compare motion, detail and audio.

The test system ran ComfyUI 0.46.0 on Windows portable with Python 3.13.14, PyTorch 2.14.0+cu130 and triton-windows 3.8.0.post29. H3 generated frames at 1280×736. Both branches kept the first 120, 240 or 360 frames, cropped eight pixels from the top and bottom, and exported 1280×720 MP4 at 24 fps with generated AAC audio trimmed to the clip length.

## Attention settings

The usual-attention branch (sampler 13) used H3's native attention. The Sol³ branch (sampler 113) added **Sol³ Sparse Attention (Mago)** with the following settings.

| Control | Value | Effect on this run |
| --- | ---: | --- |
| `dense_steps` | 2 | All blocks use dense attention for the first two of eight steps. |
| `dense_first_blocks` | 2 | Blocks 0 and 1 stay dense on all eight steps. |
| `min_tokens` | 4096 | Sequences shorter than 4096 tokens stay dense. |

Each measured Sol³ branch recorded 288 sparse calls across the remaining six steps and 48 blocks. Both branches shared the input image, text conditioning, starting latent, noise and sigmas. They used the simple schedule with denoise 1, video/audio sigma shifts of 12/3, and chunked feed-forward with two chunks and a sequence threshold of 4096.

## Run the paired workflow

1. Install this node pack and its Sol³ dependency. The sample also uses `MiniMaxChunkFeedForward` from ComfyUI-KJNodes. Restart ComfyUI after installation. [Sol³ installation](../../README.md#sol3-installation) · [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
2. Open the [five-second sample workflow](../../example_workflows/pika_ref2va_8step_720p_5s_attention_benchmark.json) in ComfyUI. Select the model files below in their loaders.
3. Upload your own reference image in **Load Image** (node 7). The measured runs used a separately supplied cat-detective image. Adjust the prompt in node 8 to match your image and queue the workflow. Both branches write MP4s to `ComfyUI/output/benchmarks/pika_ref2va_720p5s/` with the prefixes `usual` and `sol3`.

| Loader | File used for the benchmark |
| --- | --- |
| Diffusion model (1) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| LoRA (2) | `pika_h3_flex_turbo_8step_hd_ref2va.safetensors` |
| MiniMax text encoder (4) | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` |
| Video VAE (5) | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE (6) | `minimax_h3_audio_vae_fp32.safetensors` |

## Time 5-, 10- or 15-second clips

Start ComfyUI on its default port (8188). From this node pack's directory in a Windows portable install, run the command below. Set `--seconds` to `5`, `10` or `15`, and pass the reference filename relative to `ComfyUI/input/`. The runner reads its own API export. If your model names or prompt differ, edit the corresponding inputs in that file. [API export](prompt.api.json) · [Timing script](measure.py)

```powershell
..\..\..\python_embeded\python.exe .\benchmarks\pika_ref2va\measure.py --seconds 5 --reference-image "reference.png" --seed 245381526188768
```

Add `--check` to validate the graph, nodes, model selections and reference image on the running server before queuing jobs. The timing command runs a warmup pair and a measured pair, prints sampler times, and saves both videos from each pair under `ComfyUI/output/benchmarks/pika_ref2va_720p{5,10,15}s/` for the selected duration. Omit `--seed` on subsequent trials to use fresh noise and avoid reusing cached sampler outputs.
