"""Exercise the ComfyUI attention call sites with small models and a recording SOL3 operator."""
import importlib
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK.parent.parent), str(PACK.parent)]
if not torch.cuda.is_available():
    from comfy.cli_args import args
    args.cpu = True

from comfy.ldm.flux.math import rope
from comfy.ldm.wan.model import WanModel
from comfy.model_patcher import ModelPatcher

sol3_node = importlib.import_module(f"{PACK.name}.sol3_attention")


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
class Sol3IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.operator = Mock(side_effect=self.record_attention)
        self.release = Mock()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(sol3_node, "_load_backend", return_value=(self.operator, self.release)))
        stack.enter_context(torch.inference_mode())
        self.dense = Mock(spec=(), side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
        self.settings = dict(dtype=torch.bfloat16, device="cuda", operations=nn)

    def record_attention(self, q, k, v, **kwargs):
        for tensor in (q, k, v):
            self.assertEqual(tensor.shape, (1, q.shape[1], 2, 128))
            self.assertTrue(tensor.is_contiguous())
            self.assertEqual(tensor.dtype, torch.bfloat16)
        return v.clone()

    def install(self, diffusion, dense_steps=0, dense_first_blocks=0):
        model = nn.Module()
        model.diffusion_model = diffusion
        original = ModelPatcher(model, torch.device("cuda"), torch.device("cpu"))
        original.model_options["transformer_options"]["optimized_attention_override"] = self.dense
        patched = sol3_node.MagoSol3Attention.execute(
            original, dense_steps=dense_steps, min_tokens=0, dense_first_blocks=dense_first_blocks,
        )[0]
        self.assertIsNot(patched, original)
        patched.patch_model(load_weights=False)
        self.addCleanup(patched.unpatch_model, unpatch_weights=False)
        patched.pre_run()
        self.addCleanup(patched.cleanup)
        return patched

    def test_wan_warmup_batched_cfg_and_equal_length_cross_attention(self):
        self.settings["dtype"] = torch.float16
        wan = WanModel(dim=256, num_heads=2, num_layers=1, ffn_dim=256, freq_dim=32,
                       text_dim=256, **self.settings)
        self.addCleanup(self.release.assert_called_once)
        patched = self.install(wan, dense_steps=1)
        opts = patched.model_options["transformer_options"]
        sigmas = opts["sample_sigmas"] = torch.tensor([0.9, 0.5, 0.0], device="cuda")
        x = torch.randn(2, 67, 256, device="cuda", dtype=torch.float16)
        freqs = rope(torch.arange(67, device="cuda")[None], 128, 10000).unsqueeze(2)
        layer = wan.blocks[0].self_attn

        patched.prepare_state(sigmas[:1], patched.model_options)
        layer(x, freqs, transformer_options=opts)
        self.dense.assert_called_once()
        self.operator.assert_not_called()

        patched.prepare_state(sigmas[1:2], patched.model_options)
        got = layer(x, freqs, transformer_options=opts)
        self.assertEqual(self.operator.call_count, 2)
        expected = layer.o(layer.v(x).to(torch.bfloat16).to(torch.float16))
        torch.testing.assert_close(got, expected)

        wan.blocks[0].cross_attn(x, x.flip(1), transformer_options=opts)
        self.assertEqual(self.dense.call_count, 2)
        patched.unpatch_model(unpatch_weights=False)
        layer(x, freqs, transformer_options=opts)
        self.assertEqual(self.dense.call_count, 3)
        self.assertEqual(self.operator.call_count, 2)

    def test_ltx_mask_reaches_preexisting_backend(self):
        from comfy.ldm.lightricks.model import LTXVModel

        ltx = LTXVModel(num_layers=1, attention_head_dim=128, num_attention_heads=2,
                        caption_channels=256, cross_attention_dim=256, **self.settings)
        patched = self.install(ltx)
        opts = patched.model_options["transformer_options"]
        layer = ltx.transformer_blocks[0].attn1
        x = torch.randn(1, 67, 256, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(1, 1, 67, 67, device="cuda", dtype=torch.bfloat16)
        mask[..., 12:] = -torch.inf
        layer(x, mask=mask, transformer_options=opts)
        self.assertIs(self.dense.call_args.kwargs["mask"], mask)
        self.operator.assert_not_called()
        layer(x, transformer_options=opts)
        self.operator.assert_called_once()

    def test_h3_protects_first_blocks_and_packed_prefix_after_warmup(self):
        from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout

        model = MiniMaxH3Model(hidden_size=256, num_layers=3, token_refiner_num_layers=0,
                               num_attention_heads=2, ffn_hidden_size=256, time_embed_dim=64,
                               time_embed_hidden_size=64, text_dim=256, **self.settings)
        patched = self.install(model, dense_steps=2, dense_first_blocks=2)
        opts = patched.model_options["transformer_options"]
        sigmas = opts["sample_sigmas"] = torch.linspace(1, 0, 9, device="cuda")
        layout = PackedLayout(text_len=5, latent_t=1, latent_h=16, latent_w=16, audio_t=3)
        x = torch.randn(layout.seq_len, 256, device="cuda", dtype=torch.bfloat16)
        patched.prepare_state(sigmas[2:3], patched.model_options)
        model.blocks[2].attn(x, transformer_options=opts)
        self.dense.assert_called_once()
        self.operator.assert_not_called()
        opts["minimax_h3_layout"] = layout
        for step in range(8):
            patched.prepare_state(sigmas[step:step + 1], patched.model_options)
            for index, block in enumerate(model.blocks):
                with self.subTest(step=step, block=index):
                    self.dense.reset_mock()
                    self.operator.reset_mock()
                    block.attn(x, transformer_options=opts)
                    if step < 2 or index < 2:
                        self.dense.assert_called_once()
                        self.operator.assert_not_called()
                    else:
                        self.dense.assert_not_called()
                        self.operator.assert_called_once()
                        self.assertEqual(self.operator.call_args.kwargs["prefix_tokens"], 11)


if __name__ == "__main__":
    unittest.main()
