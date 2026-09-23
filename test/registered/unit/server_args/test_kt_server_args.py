"""KTransformers arguments on the modular ServerArgs pipeline."""

import argparse
import os
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.field_order import POSITIONAL_FIELD_ORDER
from sglang.srt.arg_groups.kt_hook import (
    enforce_kt_cuda_graph_compatibility,
    handle_kt_compatibility,
    validate_kt_args,
)
from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolution_result,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


KT_DOWNSTREAM_FIELDS = (
    "kt_numa_nodes",
    "kt_gpu_experts_ratio",
    "kt_num_gpu_layers",
    "kt_prefill_stream_top_n",
    "kt_gpu_prefill_token_threshold",
    "record_kt_gpu_expert_distribution",
    "kt_enable_dynamic_expert_update",
    "kt_expert_placement_strategy",
    "kt_lora_path",
    "kt_expert_lora_path",
)


def _enabled_graph_config() -> CudaGraphConfig:
    return CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL),
        prefill=PhaseConfig(backend=Backend.BREAKABLE),
    )


class TestKTFieldMigration(unittest.TestCase):
    def test_all_downstream_fields_keep_their_cli_surface(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        expected_flags = {
            "--" + name.replace("_", "-") for name in KT_DOWNSTREAM_FIELDS
        }
        self.assertTrue(expected_flags <= set(parser._option_string_actions))

        namespace = parser.parse_args(
            [
                "--model",
                "dummy",
                "--kt-numa-nodes",
                "0",
                "1",
                "--kt-gpu-experts-ratio",
                "0.5",
                "--kt-num-gpu-layers",
                "3",
                "--kt-prefill-stream-top-n",
                "2",
                "--kt-gpu-prefill-token-threshold",
                "128",
                "--record-kt-gpu-expert-distribution",
                "--kt-enable-dynamic-expert-update",
                "--kt-expert-placement-strategy",
                "front-loading",
                "--kt-lora-path",
                "/adapter",
                "--kt-expert-lora-path",
                "/adapter",
            ]
        )
        args = ServerArgs.from_cli_args(namespace)
        self.assertEqual(args.kt_numa_nodes, [0, 1])
        self.assertEqual(args.kt_gpu_experts_ratio, 0.5)
        self.assertEqual(args.kt_num_gpu_layers, 3)
        self.assertEqual(args.kt_prefill_stream_top_n, 2)
        self.assertEqual(args.kt_gpu_prefill_token_threshold, 128)
        self.assertTrue(args.record_kt_gpu_expert_distribution)
        self.assertTrue(args.kt_enable_dynamic_expert_update)
        self.assertEqual(args.kt_expert_placement_strategy, "front-loading")
        self.assertEqual(args.kt_lora_path, "/adapter")
        self.assertEqual(args.kt_expert_lora_path, "/adapter")

    def test_kt_fields_keep_the_downstream_positional_order(self):
        expected = (
            "kt_weight_path",
            "kt_method",
            "kt_cpuinfer",
            "kt_threadpool_count",
            "kt_numa_nodes",
            "kt_num_gpu_experts",
            "kt_gpu_experts_ratio",
            "kt_num_gpu_layers",
            "kt_prefill_stream_top_n",
            "kt_max_deferred_experts_per_token",
            "kt_gpu_prefill_token_threshold",
            "record_kt_gpu_expert_distribution",
            "kt_enable_dynamic_expert_update",
            "kt_expert_placement_strategy",
            "kt_lora_path",
            "kt_expert_lora_path",
        )
        start = POSITIONAL_FIELD_ORDER.index("kt_weight_path")
        self.assertEqual(
            POSITIONAL_FIELD_ORDER[start : start + len(expected)], expected
        )


class TestKTDeclarations(unittest.TestCase):
    def test_legacy_lora_alias_resolves_without_writing_the_input(self):
        args = ServerArgs(model_path="dummy", kt_lora_path="/adapter")
        args.resolve_once()

        self.assertIsNone(args.kt_expert_lora_path)
        self.assertFalse(args.disable_cuda_graph)
        self.assertEqual(
            resolution_result(args, "kt_expert_lora_path"),
            "/adapter",
        )
        self.assertTrue(resolution_result(args, "disable_cuda_graph"))

    def test_conflicting_lora_aliases_fail_during_resolution(self):
        args = ServerArgs(
            model_path="dummy",
            kt_lora_path="/legacy",
            kt_expert_lora_path="/explicit",
        )
        with self.assertRaisesRegex(ValueError, "cannot point to different adapters"):
            args.resolve_once()

    def test_k3_environment_aliases_declare_the_cli_fields(self):
        environment = {
            "SGLANG_K3_SHARED_EXPERTS_ATTN_TP": "1",
            "SGLANG_K3_DENSE_MLP_ATTN_TP": "true",
        }
        with patch.dict(os.environ, environment, clear=False):
            args = ServerArgs(model_path="dummy")
            args.resolve_once()

        self.assertFalse(args.enable_shared_experts_attn_tp)
        self.assertFalse(args.enable_dense_mlp_attn_tp)
        self.assertTrue(resolution_result(args, "enable_shared_experts_attn_tp"))
        self.assertTrue(resolution_result(args, "enable_dense_mlp_attn_tp"))

    def test_kt_weights_disable_only_prefill_graph_capture(self):
        args = ServerArgs(model_path="dummy", kt_weight_path="/weights")
        declare_resolution(
            args,
            "test_graph_config",
            cuda_graph_config=_enabled_graph_config(),
        )

        enforce_kt_cuda_graph_compatibility(args)

        config = resolution_result(args, "cuda_graph_config")
        self.assertEqual(config.decode.backend, Backend.FULL)
        self.assertEqual(config.prefill.backend, Backend.DISABLED)

    def test_kt_lora_disables_both_graph_phases(self):
        args = ServerArgs(model_path="dummy", kt_lora_path="/adapter")
        declare_resolution(
            args,
            "test_graph_config",
            cuda_graph_config=_enabled_graph_config(),
        )
        handle_kt_compatibility(args)

        enforce_kt_cuda_graph_compatibility(args)

        config = resolution_result(args, "cuda_graph_config")
        self.assertEqual(config.decode.backend, Backend.DISABLED)
        self.assertEqual(config.prefill.backend, Backend.DISABLED)
        self.assertTrue(resolution_result(args, "disable_cuda_graph"))


class TestKTValidation(unittest.TestCase):
    def assert_invalid(self, message: str, **fields):
        with self.assertRaisesRegex(ValueError, message):
            validate_kt_args(ServerArgs(model_path="dummy", **fields))

    def test_scalar_and_topology_constraints(self):
        cases = (
            ("threadpool-count must be positive", {"kt_threadpool_count": 0}),
            (
                "must contain exactly",
                {"kt_threadpool_count": 2, "kt_numa_nodes": [0]},
            ),
            (
                "values must be non-negative",
                {"kt_threadpool_count": 1, "kt_numa_nodes": [-1]},
            ),
            ("num-gpu-experts must be non-negative", {"kt_num_gpu_experts": -1}),
            ("ratio must be between 0 and 1", {"kt_gpu_experts_ratio": 1.1}),
            ("num-gpu-layers must be non-negative", {"kt_num_gpu_layers": -1}),
            (
                "deferred-experts-per-token must be non-negative",
                {"kt_max_deferred_experts_per_token": -1},
            ),
            (
                "prefill-token-threshold must be non-negative",
                {"kt_gpu_prefill_token_threshold": -1},
            ),
            (
                "placement-strategy must be one of",
                {"kt_expert_placement_strategy": "invalid"},
            ),
        )
        for message, fields in cases:
            with self.subTest(fields=fields):
                self.assert_invalid(message, **fields)

    def test_weight_runtime_constraints(self):
        self.assert_invalid("requires --kt-num-gpu-experts", kt_weight_path="/w")
        self.assert_invalid(
            "requires --moe-a2a-backend none",
            kt_weight_path="/w",
            kt_num_gpu_experts=0,
            kt_cpuinfer=2,
            moe_a2a_backend="deepep",
        )
        self.assert_invalid(
            "requires --kt-cpuinfer",
            kt_weight_path="/w",
            kt_num_gpu_experts=0,
            kt_cpuinfer=0,
        )
        self.assert_invalid(
            "at least --kt-threadpool-count",
            kt_weight_path="/w",
            kt_num_gpu_experts=0,
            kt_cpuinfer=1,
            kt_threadpool_count=2,
        )
        self.assert_invalid(
            "requires a positive",
            kt_enable_dynamic_expert_update=True,
        )

    def test_decayed_lfu_stream_width_constraints(self):
        validate_kt_args(
            ServerArgs(
                model_path="dummy",
                kt_num_gpu_experts=8,
                kt_expert_placement_strategy="decayed-lfu",
            )
        )
        validate_kt_args(
            ServerArgs(
                model_path="dummy",
                kt_num_gpu_experts=8,
                kt_prefill_stream_top_n=0,
                kt_expert_placement_strategy="decayed-lfu",
            )
        )
        self.assert_invalid(
            "only valid with",
            kt_num_gpu_experts=8,
            kt_prefill_stream_top_n=1,
        )
        self.assert_invalid(
            "must satisfy 0 <= N <=",
            kt_num_gpu_experts=8,
            kt_prefill_stream_top_n=9,
            kt_expert_placement_strategy="decayed-lfu",
        )
        self.assert_invalid(
            "requires a positive",
            kt_num_gpu_experts=0,
            kt_expert_placement_strategy="decayed-lfu",
        )
        self.assert_invalid(
            "cannot be combined with --kt-gpu-experts-ratio",
            kt_num_gpu_experts=8,
            kt_gpu_experts_ratio=0.5,
            kt_expert_placement_strategy="decayed-lfu",
        )
        self.assert_invalid(
            "legacy --kt-enable-dynamic-expert-update",
            kt_num_gpu_experts=8,
            kt_enable_dynamic_expert_update=True,
            kt_expert_placement_strategy="decayed-lfu",
        )
        self.assert_invalid(
            "--kt-max-deferred-experts-per-token",
            kt_num_gpu_experts=8,
            kt_max_deferred_experts_per_token=0,
            kt_expert_placement_strategy="decayed-lfu",
        )
        self.assert_invalid(
            "single-batch overlap",
            kt_num_gpu_experts=8,
            enable_single_batch_overlap=True,
            kt_expert_placement_strategy="decayed-lfu",
        )
        self.assert_invalid(
            "cannot yet be combined with --enable-dp-attention",
            kt_num_gpu_experts=8,
            enable_dp_attention=True,
            kt_expert_placement_strategy="decayed-lfu",
        )

    def test_decayed_lfu_resolves_default_stream_width(self):
        args = ServerArgs(
            model_path="dummy",
            kt_num_gpu_experts=32,
            kt_expert_placement_strategy="decayed-lfu",
        )
        args.resolve_once()
        self.assertIsNone(args.kt_prefill_stream_top_n)
        self.assertEqual(resolution_result(args, "kt_prefill_stream_top_n"), 4)

        small = ServerArgs(
            model_path="dummy",
            kt_num_gpu_experts=2,
            kt_expert_placement_strategy="decayed-lfu",
        )
        small.resolve_once()
        self.assertEqual(resolution_result(small, "kt_prefill_stream_top_n"), 2)

    def test_server_lifecycle_invokes_kt_validation(self):
        args = ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            chunked_prefill_size=-1,
            kt_weight_path="/weights",
            kt_num_gpu_experts=0,
            kt_cpuinfer=0,
        )
        args.resolve_once()
        with self.assertRaisesRegex(ValueError, "requires --kt-cpuinfer"):
            args.check_server_args()

    def test_lora_constraints(self):
        self.assert_invalid(
            "requires --kt-weight-path",
            kt_expert_lora_path="/adapter",
        )
        base = {
            "kt_expert_lora_path": "/adapter",
            "kt_weight_path": "/weights",
            "kt_num_gpu_experts": 0,
            "kt_cpuinfer": 2,
        }
        self.assert_invalid(
            "supports CPU experts only",
            **{**base, "kt_num_gpu_experts": 1},
        )
        self.assert_invalid(
            "cannot be combined with --kt-gpu-prefill-token-threshold",
            **{**base, "kt_gpu_prefill_token_threshold": 1},
        )
        self.assert_invalid(
            "supports only AMXBF16",
            **{**base, "kt_method": "LLAMAFILE"},
        )
        self.assert_invalid(
            "max-running-requests >= 2",
            **{**base, "max_running_requests": 1},
        )
        validate_kt_args(ServerArgs(model_path="dummy", **base))


if __name__ == "__main__":
    unittest.main()
