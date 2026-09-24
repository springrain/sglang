"""SM100 MXFP4 must preserve KTEP's explicit compact expert routing."""

import sys
from types import ModuleType
from unittest.mock import patch

import torch
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmGenMxfp4MoeQuantInfo,
    _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_non_situ_standard_topk_uses_routed_api_with_compact_geometry():
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as trtllm_mod

    tokens, hidden, top_k = 2, 16, 3
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, -1, 2], [1, 2, -1]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.6, 0.0, 0.4], [0.25, 0.75, 0.0]], dtype=torch.float32
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.empty(tokens, 0),
        ),
    )

    tensors = {
        name: torch.full((1,), value, dtype=torch.float32)
        for value, name in enumerate(
            (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_bias",
                "w2_weight_bias",
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
            ),
            start=1,
        )
    }
    quant_info = FlashInferTrtllmGenMxfp4MoeQuantInfo(
        **tensors,
        global_num_experts=8,
        local_expert_offset=0,
        local_num_experts=3,
        intermediate_size_per_partition=32,
        hidden_size=hidden,
        flashinfer_mxfp4_moe_precision="bf16",
    )

    expected = (
        torch.arange(tokens * hidden, dtype=torch.float32)
        .reshape(tokens, hidden)
        .to(torch.bfloat16)
    )
    output = torch.empty_like(x)
    captured = {}

    def fake_routed_moe(**kwargs):
        captured.update(kwargs)
        kwargs["output"].copy_(expected)
        return kwargs["output"].clone()

    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.trtllm_fp4_block_scale_routed_moe = fake_routed_moe

    with (
        patch.dict(
            sys.modules,
            {
                "flashinfer": flashinfer,
                "flashinfer.fused_moe": fused_moe,
            },
        ),
        patch.object(
            trtllm_mod, "get_activation_type", return_value=17
        ) as activation_type,
        patch.object(trtllm_mod, "trtllm_moe_enable_pdl", return_value=False),
        patch.object(zero_copy_context, "get_moe_output_spec", return_value=output),
    ):
        result = _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output,
            quant_info,
            MoeRunnerConfig(activation="silu", is_gated=True),
        )

    activation_type.assert_called_once_with("silu", is_gated=True)
    routed_ids, routed_weights = captured["topk_ids"]
    torch.testing.assert_close(routed_ids, topk_ids)
    torch.testing.assert_close(routed_weights, topk_weights)
    assert captured["gemm1_bias"] is tensors["w13_weight_bias"]
    assert captured["gemm2_bias"] is tensors["w2_weight_bias"]
    assert captured["gemm1_alpha"] is tensors["gemm1_alpha"]
    assert captured["gemm1_beta"] is tensors["gemm1_beta"]
    assert captured["gemm1_clamp_limit"] is tensors["gemm1_clamp_limit"]
    assert captured["num_experts"] == 8
    assert captured["local_expert_offset"] == 0
    assert captured["local_num_experts"] == 3
    assert captured["top_k"] == top_k
    assert captured["routing_method_type"] == 1
    assert captured["activation_type"] == 17
    assert captured["do_finalize"] is True
    assert captured["output"] is output
    assert result.hidden_states is output
    torch.testing.assert_close(result.hidden_states, expected)


def test_non_situ_bypassed_topk_keeps_logits_api():
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as trtllm_mod

    tokens, hidden, num_experts, top_k = 2, 16, 8, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    router_logits = torch.randn(tokens, num_experts)
    dispatch_output = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=BypassedTopKOutput(
            hidden_states=x,
            router_logits=router_logits,
            topk_config=TopKConfig(top_k=top_k, renormalize=True),
        ),
    )
    dummy = torch.ones(1)
    quant_info = FlashInferTrtllmGenMxfp4MoeQuantInfo(
        w13_weight=dummy,
        w2_weight=dummy,
        w13_weight_scale=dummy,
        w2_weight_scale=dummy,
        w13_weight_bias=dummy,
        w2_weight_bias=dummy,
        gemm1_alpha=dummy,
        gemm1_beta=dummy,
        gemm1_clamp_limit=dummy,
        global_num_experts=num_experts,
        local_expert_offset=0,
        local_num_experts=num_experts,
        intermediate_size_per_partition=32,
        hidden_size=hidden,
        flashinfer_mxfp4_moe_precision="bf16",
    )
    output = torch.empty_like(x)
    calls = []

    def fake_logits_moe(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["output"].zero_()
        return (kwargs["output"],)

    def fail_routed_moe(**kwargs):
        raise AssertionError("bypassed top-k must keep the logits API")

    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.trtllm_fp4_block_scale_moe = fake_logits_moe
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.trtllm_fp4_block_scale_routed_moe = fail_routed_moe

    with (
        patch.dict(
            sys.modules,
            {
                "flashinfer": flashinfer,
                "flashinfer.fused_moe": fused_moe,
            },
        ),
        patch.object(trtllm_mod, "trtllm_moe_enable_pdl", return_value=False),
        patch.object(zero_copy_context, "get_moe_output_spec", return_value=output),
    ):
        result = _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output,
            quant_info,
            MoeRunnerConfig(activation="silu", is_gated=True),
        )

    assert len(calls) == 1
    args, kwargs = calls[0]
    torch.testing.assert_close(args[0], router_logits.to(torch.bfloat16))
    assert kwargs["output"] is output
    assert result.hidden_states is output
