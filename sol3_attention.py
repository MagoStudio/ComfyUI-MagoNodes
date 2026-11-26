"""Sol³ attention for the native Wan, LTX and MiniMax H3 diffusion models."""

import logging
from functools import partial
from importlib import import_module
from weakref import WeakKeyDictionary

import torch
from comfy_api.latest import io
from comfy.patcher_extension import CallbacksMP, PatcherInjection


def _load_backend(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Sol³ requires an NVIDIA SM80 or SM120 CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((8, 0), (12, 0)):
        raise ValueError(f"Sol³ supports SM80 and SM120. This model runs on SM{capability[0]}{capability[1]}.")
    try:
        import_module("triton")
        from sol3_attn import sol3_attn, release_workspace
        if capability == (8, 0):
            import_module("flashinfer")
    except ImportError as exc:
        raise RuntimeError(
            f"Could not load the Sol³ backend ({exc}). See the Sol³ installation "
            "section in ComfyUI-MagoNodes/README.md and use ComfyUI's Python."
        ) from exc
    return sol3_attn, release_workspace


def _attention_layers(diffusion):
    from comfy.ldm.lightricks.model import LTXVModel
    from comfy.ldm.wan.model import WanModel, WanSelfAttention

    if isinstance(diffusion, WanModel):
        layers = [block.self_attn for block in diffusion.blocks]
        if any(type(layer) is not WanSelfAttention for layer in layers):
            raise ValueError("Sol³ requires standard bidirectional Wan self-attention for this model.")
        return layers, "head_dim", False
    if isinstance(diffusion, LTXVModel):
        return [block.attn1 for block in diffusion.transformer_blocks], "dim_head", False

    from comfy.ldm.minimax.model import MiniMaxH3Model
    if isinstance(diffusion, MiniMaxH3Model):
        return [block.attn for block in diffusion.blocks], "head_dim", True
    raise ValueError(f"Sol³ supports native Wan, LTX and MiniMax H3 models. Received {type(diffusion).__name__}.")


def _h3_prefix_tokens(options, tokens):
    # Native H3 packs text, keyframes, references and audio before the target video.
    layout = options.get("minimax_h3_layout")
    if layout is None or layout.seq_len != tokens or not layout.segments:
        return None
    start, end, kind = layout.segments[-1]
    return start if kind == "video" and end == tokens and 0 <= start < end else None


class _Sol3Patch:
    def __init__(self, operator, release, dense_steps, min_tokens, dense_first_blocks, h3):
        self.operator = operator
        self.release = release
        self.dense_steps = dense_steps
        self.min_tokens = min_tokens
        self.dense_first_blocks = dense_first_blocks
        self.h3 = h3
        self.calls = 0
        self.hooks = WeakKeyDictionary()

    def inject(self, patcher):
        layers, _, _ = _attention_layers(patcher.get_model_object("diffusion_model"))
        self.hooks[patcher] = [
            layer.register_forward_pre_hook(self.scope_attention, with_kwargs=True)
            for layer in layers[self.dense_first_blocks:]
        ]

    def eject(self, patcher):
        for hook in self.hooks.pop(patcher, []):
            hook.remove()

    def prepare(self, patcher, timestep, model_options):
        opts = model_options["transformer_options"]
        sigmas = opts.get("sample_sigmas")
        # Compare once per denoiser evaluation. CFG and individual layers share the result.
        opts["mago_sol3_active"] = self.dense_steps == 0 or (
            sigmas is not None and self.dense_steps < len(sigmas) - 1
            and bool(timestep.flatten()[0] <= sigmas[self.dense_steps])
        )

    def cleanup(self, patcher):
        self.release()
        logging.info("Mago Sol³ ran %d sparse attention calls during this sampler pass.", self.calls)
        self.calls = 0

    def scope_attention(self, module, args, kwargs):
        x = args[0] if args else kwargs["x"]
        opts = kwargs.get("transformer_options", {})
        tokens = x.shape[-2]
        if (not opts.get("mago_sol3_active") or tokens < self.min_tokens
                or kwargs.get("mask") is not None or kwargs.get("context") is not None):
            return
        prefix = _h3_prefix_tokens(opts, tokens) if self.h3 else 0
        if prefix is not None:
            override = partial(self.attention, prefix, opts, module.comfy_attention)
            return args, dict(kwargs, transformer_options=dict(opts, optimized_attention_override=override))

    def attention(self, prefix, options, preferred, _original_attention, q, k, v, heads, mask=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
        from comfy.ldm.modules.attention import optimized_attention
        from comfy.model_prefetch import pause_malloc_graph

        # A native hook can issue further attention calls after the module's
        # own QKV call. Restore its normal backend on this local options copy.
        scoped = kwargs["transformer_options"]
        scoped.pop("optimized_attention_override")
        if "optimized_attention_override" in options:
            scoped["optimized_attention_override"] = options["optimized_attention_override"]

        if (mask is not None or not q.numel() or q.shape != k.shape or q.shape != v.shape
                or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16)
                or q.device != k.device or q.device != v.device or not q.is_cuda
                or q.shape[-1] != (128 if skip_reshape else heads * 128)
                or kwargs.get("scale") is not None or kwargs.get("enable_gqa") or kwargs.get("is_causal")
                or kwargs.get("attn_precision") == torch.float32 or kwargs.get("low_precision_attention") is False):
            # Re-enter through the original options so a checkpoint's preferred
            # backend and an existing model-wide override keep their precedence.
            kwargs.pop("_inside_attn_wrapper", None)
            kwargs["transformer_options"] = options
            return optimized_attention(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                                       skip_output_reshape=skip_output_reshape, preferred_attention=preferred, **kwargs)
        if skip_reshape:  # ComfyUI BHSD -> Sol³ BSHD
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        else:  # ComfyUI [B, S, H*D]
            q, k, v = (t.unflatten(-1, (heads, 128)) for t in (q, k, v))

        # Upstream accepts one sample at a time and caches scratch across blocks.
        # Keep persistent allocations outside ComfyUI's block allocator.
        with pause_malloc_graph():
            outputs = []
            for inputs in zip(q.split(1), k.split(1), v.split(1)):
                inputs = [t.to(torch.bfloat16).contiguous() for t in inputs]
                outputs.append(self.operator(*inputs, prefix_tokens=prefix))
                self.calls += 1
            out = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        out = out.to(q.dtype)
        return out.transpose(1, 2) if skip_output_reshape else out.flatten(2)


class MagoSol3Attention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagoSol3Attention", display_name="Sol³ Sparse Attention (Mago)",
            category="Mago Nodes/Sampling", search_aliases=["sol3", "sol cubed"],
            description="Apply Sol³ to native Wan, LTX or MiniMax H3 attention with 128-dimensional heads.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("dense_steps", default=1, min=0, max=10000,
                             tooltip="Keep the original attention for this many initial steps of each sampler pass."),
                io.Int.Input("min_tokens", default=4096, min=0, max=1048576, step=256,
                             tooltip="Keep the original attention on sequences shorter than this length."),
                io.Int.Input("dense_first_blocks", optional=True, default=1, min=0, max=1024,
                             tooltip="Keep the original attention in this many leading transformer blocks on every step. "
                                     "Use 2 for eight-step HyperFlow H3 to protect blocks 0 and 1."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, dense_steps=1, min_tokens=4096, dense_first_blocks=1):
        diffusion = model.get_model_object("diffusion_model")
        layers, dim_attr, h3 = _attention_layers(diffusion)
        if not layers or any(getattr(layer, dim_attr) != 128 for layer in layers):
            raise ValueError("This model's attention head dimension is incompatible with Sol³ (128 required).")
        operator, release = _load_backend(model.load_device)
        m = model.clone()
        patch = _Sol3Patch(operator, release, dense_steps, min_tokens, dense_first_blocks, h3)
        m.model_options["transformer_options"]["mago_sol3_active"] = dense_steps == 0
        m.set_injections("mago_sol3", [PatcherInjection(patch.inject, patch.eject)])
        for event, callback in ((CallbacksMP.ON_PREPARE_STATE, patch.prepare), (CallbacksMP.ON_CLEANUP, patch.cleanup)):
            m.remove_callbacks_with_key(event, "mago_sol3")
            m.add_callback_with_key(event, "mago_sol3", callback)
        return io.NodeOutput(m)
