"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _dsv4_tail_graph_warmup_input_ids,
    _get_vlm_warmup_image_base64,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDsv4TailGraphWarmup(CustomTestCase):
    def test_uses_the_full_prefill_chunk(self):
        input_ids = _dsv4_tail_graph_warmup_input_ids(
            SimpleNamespace(
                enable_decoder_swa_bounded_replay=True,
                chunked_prefill_size=2048,
            ),
            {
                "is_generation": True,
                "architectures": ["DeepseekV4ForCausalLM"],
            },
        )

        self.assertEqual(len(input_ids), 2048)
        self.assertEqual(input_ids[:4], [10, 11, 12, 13])

    def test_unchunked_prefill_uses_the_eager_warm_gate_size(self):
        input_ids = _dsv4_tail_graph_warmup_input_ids(
            SimpleNamespace(
                enable_decoder_swa_bounded_replay=True,
                chunked_prefill_size=-1,
            ),
            {
                "is_generation": True,
                "architectures": ["DeepseekV4ForCausalLM"],
            },
        )

        self.assertEqual(len(input_ids), 8192)

    def test_skips_models_or_settings_without_tail_graphs(self):
        server_args = SimpleNamespace(
            enable_decoder_swa_bounded_replay=True,
            chunked_prefill_size=2048,
        )
        dsv4 = {
            "is_generation": True,
            "architectures": ["DeepseekV4ForCausalLM"],
        }

        self.assertIsNone(
            _dsv4_tail_graph_warmup_input_ids(
                server_args,
                {"is_generation": True, "architectures": ["Qwen3ForCausalLM"]},
            )
        )
        self.assertIsNone(
            _dsv4_tail_graph_warmup_input_ids(
                SimpleNamespace(
                    enable_decoder_swa_bounded_replay=False,
                    chunked_prefill_size=2048,
                ),
                dsv4,
            )
        )
        self.assertIsNone(
            _dsv4_tail_graph_warmup_input_ids(
                server_args, {**dsv4, "is_generation": False}
            )
        )
        with patch.object(
            envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS, "get", return_value=0
        ):
            self.assertIsNone(_dsv4_tail_graph_warmup_input_ids(server_args, dsv4))


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_k2_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            {"architectures": ["KimiK25ForConditionalGeneration"]}
        )
        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)

        png = base64.b64decode(KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_kimi_k3_uses_native_patch_grid_image(self):
        for model_info in (
            {"architectures": ["KimiK3ForConditionalGeneration"]},
            {"architectures": None, "model_type": "kimi_k3"},
        ):
            with self.subTest(model_info=model_info):
                self.assertEqual(
                    _get_vlm_warmup_image_base64(model_info),
                    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
                )

        png = base64.b64decode(KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (448, 448))

    def test_other_vlms_keep_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                {"architectures": ["Qwen3VLForConditionalGeneration"]}
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64({"architectures": None}),
            MINIMUM_PNG_PICTURE_BASE64,
        )


if __name__ == "__main__":
    unittest.main()
