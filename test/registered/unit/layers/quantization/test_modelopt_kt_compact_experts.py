"""CPU regression tests for compact KTransformers ModelOpt MoE experts."""

import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant as modelopt_mod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
    ModelOptFp8MoEMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FusedMoeWeightScaleSupported(Enum):
    BLOCK = "block"
    TENSOR = "tensor"


@pytest.fixture(autouse=True)
def _stub_fused_moe_scale_enum(monkeypatch):
    """Keep shape tests independent from the full fused-MoE runtime import."""

    module = ModuleType("sglang.srt.layers.moe.fused_moe_triton")
    module.FusedMoeWeightScaleSupported = _FusedMoeWeightScaleSupported
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.fused_moe_triton", module)


class _MockLayer(torch.nn.Module):
    def __init__(self, num_experts: int = 8, num_local_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_size = num_experts // num_local_experts
        self.moe_ep_rank = 0
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            is_gated=True,
        )


def _build_nvfp4_method(*, compact: bool = True):
    method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
    method.quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
    )
    method.enable_flashinfer_trtllm_moe = False
    method._cache_permute_indices = {}
    if compact:
        method._kt_compact_expert_rows = True
    return method


def _build_fp8_method(*, compact: bool = True):
    method = ModelOptFp8MoEMethod.__new__(ModelOptFp8MoEMethod)
    method.quant_config = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
    if compact:
        method._kt_compact_expert_rows = True
    return method


@pytest.mark.parametrize("resident_experts", [0, 1, 3, 8])
def test_nvfp4_create_weights_uses_compact_expert_rows(monkeypatch, resident_experts):
    monkeypatch.setattr(
        modelopt_mod, "swizzle_blockscale", lambda scale: scale.detach()
    )
    layer = _MockLayer()
    method = _build_nvfp4_method()

    method.create_weights(
        layer=layer,
        num_experts=resident_experts,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )

    expected_shapes = {
        "w13_weight": (resident_experts, 64, 32),
        "w2_weight": (resident_experts, 64, 16),
        "w13_weight_scale": (resident_experts, 64, 4),
        "w2_weight_scale": (resident_experts, 64, 2),
        "w13_weight_scale_2": (resident_experts, 2),
        "w2_weight_scale_2": (resident_experts,),
        "w13_input_scale": (resident_experts, 2),
        "w2_input_scale": (resident_experts,),
    }
    for name, shape in expected_shapes.items():
        assert tuple(getattr(layer, name).shape) == shape
    assert method.num_experts == resident_experts
    assert not layer.w13_input_scale._sglang_require_global_experts
    assert not layer.w2_input_scale._sglang_require_global_experts


def test_nvfp4_non_kt_input_scales_keep_global_expert_shape(monkeypatch):
    monkeypatch.setattr(
        modelopt_mod, "swizzle_blockscale", lambda scale: scale.detach()
    )
    layer = _MockLayer(num_experts=8, num_local_experts=4)
    method = _build_nvfp4_method(compact=False)

    method.create_weights(
        layer=layer,
        num_experts=4,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )

    assert tuple(layer.w13_weight.shape) == (4, 64, 32)
    assert tuple(layer.w13_input_scale.shape) == (8, 2)
    assert tuple(layer.w2_input_scale.shape) == (8,)
    assert layer.w13_input_scale._sglang_require_global_experts
    assert layer.w2_input_scale._sglang_require_global_experts


@pytest.mark.parametrize("resident_experts", [0, 1, 3, 8])
def test_fp8_create_weights_uses_compact_expert_rows(resident_experts):
    layer = _MockLayer()
    method = _build_fp8_method()

    method.create_weights(
        layer=layer,
        num_experts=resident_experts,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )

    expected_shapes = {
        "w13_weight": (resident_experts, 64, 64),
        "w2_weight": (resident_experts, 64, 32),
        "w13_weight_scale": (resident_experts, 2),
        "w2_weight_scale": (resident_experts,),
        "w13_input_scale": (resident_experts,),
        "w2_input_scale": (resident_experts,),
    }
    for name, shape in expected_shapes.items():
        assert tuple(getattr(layer, name).shape) == shape
    assert method.num_experts == resident_experts


@pytest.mark.parametrize("method_factory", [_build_nvfp4_method, _build_fp8_method])
@pytest.mark.parametrize("resident_experts", [0, 1, 3, 8])
def test_modelopt_runtime_geometry_keeps_global_count(method_factory, resident_experts):
    layer = SimpleNamespace(num_experts=8, num_local_experts=4, moe_ep_rank=1)
    method = method_factory()
    method.num_experts = resident_experts

    assert method._runtime_expert_geometry(layer) == (
        8,
        0,
        resident_experts,
    )

    del method._kt_compact_expert_rows
    assert method._runtime_expert_geometry(layer) == (8, 4, 4)


@pytest.mark.parametrize("method_factory", [_build_nvfp4_method, _build_fp8_method])
def test_zero_resident_experts_skip_modelopt_postprocess(monkeypatch, method_factory):
    monkeypatch.setattr(
        modelopt_mod, "swizzle_blockscale", lambda scale: scale.detach()
    )
    layer = _MockLayer()
    method = method_factory()
    method.create_weights(
        layer=layer,
        num_experts=0,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )

    method.process_weights_after_loading(layer)


def test_modelopt_methods_advertise_compact_kt_support():
    assert ModelOptNvFp4FusedMoEMethod.supports_kt_compact_expert_rows
    assert ModelOptFp8MoEMethod.supports_kt_compact_expert_rows


def test_fp8_compact_trtllm_keeps_canonical_triton_layout(monkeypatch):
    method = _build_fp8_method()
    method.num_experts = 1
    layer = _MockLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(1, 64, 64, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(1, 64, 32, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.w13_input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.w2_input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    monkeypatch.setattr(
        modelopt_mod,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_TRTLLM,
    )
    flashinfer_module = ModuleType("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm")

    def _unexpected_align(*args, **kwargs):
        raise AssertionError("compact ModelOpt FP8 must not use TRT-LLM layout")

    flashinfer_module.align_fp8_moe_weights_for_flashinfer_trtllm = _unexpected_align
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        flashinfer_module,
    )

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight, original_w13)
    torch.testing.assert_close(layer.w2_weight, original_w2)
    assert not hasattr(layer, "output1_scales_scalar")


def test_fp8_compact_standard_topk_uses_triton_runner(monkeypatch):
    method = _build_fp8_method()
    method.num_experts = 3
    method.moe_runner_config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=3,
        activation="silu",
        is_gated=True,
    )
    layer = SimpleNamespace(
        w13_weight=torch.empty(3, 64, 64),
        w2_weight=torch.empty(3, 64, 32),
        w13_weight_scale=torch.ones(3),
        w2_weight_scale=torch.ones(3),
        w13_input_scale=torch.tensor(1.0),
        w2_input_scale=torch.tensor(1.0),
    )
    topk_output = object()
    dispatch_output = SimpleNamespace(
        hidden_states=torch.empty(2, 64), topk_output=topk_output
    )
    expected = object()
    calls = []

    class _Runner:
        def run(self, dispatch, quant_info):
            calls.append((dispatch, quant_info))
            return expected

    method.runner = _Runner()
    monkeypatch.setattr(
        modelopt_mod,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_TRTLLM,
    )
    topk_module = ModuleType("sglang.srt.layers.moe.topk")
    topk_module.TopKOutputChecker = SimpleNamespace(
        format_is_bypassed=lambda output: False
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk_module)

    result = method.apply(layer, dispatch_output)

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] is dispatch_output
    assert calls[0][1].w13_weight is layer.w13_weight


def test_fp8_compact_rejects_unmaterialized_bypassed_topk(monkeypatch):
    method = _build_fp8_method()
    method.num_experts = 1
    method.moe_runner_config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=1,
        activation="silu",
        is_gated=True,
    )
    topk_module = ModuleType("sglang.srt.layers.moe.topk")
    topk_module.TopKOutputChecker = SimpleNamespace(
        format_is_bypassed=lambda output: True
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk_module)
    dispatch_output = SimpleNamespace(
        hidden_states=torch.empty(2, 64), topk_output=object()
    )

    with pytest.raises(RuntimeError, match="require explicit top-k ids"):
        method.apply(SimpleNamespace(), dispatch_output)


@pytest.mark.parametrize("resident_experts", [1, 3, 8])
def test_nvfp4_trtllm_quant_info_uses_compact_geometry(monkeypatch, resident_experts):
    method = _build_nvfp4_method()
    method.num_experts = resident_experts
    method.enable_flashinfer_trtllm_moe = True
    method._moe_runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM
    method.moe_runner_config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=resident_experts,
        activation="silu",
        is_gated=True,
    )
    layer = SimpleNamespace(
        num_experts=8,
        num_local_experts=8,
        moe_ep_rank=0,
        intermediate_size_per_partition=32,
        w13_weight=torch.nn.Parameter(torch.empty(resident_experts, 64, 32)),
        w2_weight=torch.nn.Parameter(torch.empty(resident_experts, 64, 16)),
        w13_weight_scale=torch.nn.Parameter(torch.empty(resident_experts, 64, 4)),
        w2_weight_scale=torch.nn.Parameter(torch.empty(resident_experts, 64, 2)),
        g1_scale_c=torch.nn.Parameter(torch.ones(resident_experts)),
        g1_alphas=torch.nn.Parameter(torch.ones(resident_experts)),
        g2_alphas=torch.nn.Parameter(torch.ones(resident_experts)),
        w13_input_scale_quant=torch.tensor(1.0),
    )
    captured = []

    class _QuantInfo(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class _Runner:
        def run(self, dispatch, quant_info):
            captured.append(quant_info)
            return quant_info

    method.runner = _Runner()
    flashinfer_module = ModuleType("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm")
    flashinfer_module.FlashInferTrtllmFp4MoeQuantInfo = _QuantInfo
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        flashinfer_module,
    )

    result = method.apply(
        layer,
        SimpleNamespace(hidden_states=torch.empty(2, 64), topk_output=object()),
    )

    assert result is captured[0]
    assert captured[0].global_num_experts == 8
    assert captured[0].local_expert_offset == 0
    assert captured[0].local_num_experts == resident_experts
