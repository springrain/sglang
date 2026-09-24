"""CPU-only regression coverage for compact KTransformers BF16/FP8 experts."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import fp8 as fp8_mod
from sglang.srt.layers.quantization import unquant as unquant_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_methods_advertise_compact_expert_rows():
    assert unquant_mod.UnquantizedFusedMoEMethod.supports_kt_compact_expert_rows
    assert fp8_mod.Fp8MoEMethod.supports_kt_compact_expert_rows


class _NoSpecialMoeBackend:
    def is_cutlass(self):
        return False

    def is_flashinfer_trtllm(self):
        return False

    def is_flashinfer_trtllm_routed(self):
        return False

    def is_hpc_ops(self):
        return False


class _MockLayer(torch.nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.hidden_size = 4
        self.intermediate_size_per_partition = 2
        self.moe_ep_rank = 0
        self.moe_runner_config = SimpleNamespace(is_gated=True)


def _install_fake_flashinfer_core(monkeypatch):
    flashinfer = ModuleType("flashinfer")
    fused_moe = ModuleType("flashinfer.fused_moe")
    core = ModuleType("flashinfer.fused_moe.core")

    def _identity_indices(cache, weight, *args, **kwargs):
        del cache, args, kwargs
        return torch.arange(weight.shape[0], device=weight.device)

    core._maybe_get_cached_w3_w1_permute_indices = _identity_indices
    core.get_w2_permute_indices_with_cache = _identity_indices
    core.convert_to_block_layout = lambda weight, block_k: weight
    flashinfer.fused_moe = fused_moe
    fused_moe.core = core
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core)


@pytest.mark.parametrize("resident_experts", [0, 1, 3])
def test_unquantized_postprocess_uses_physical_expert_rows(
    monkeypatch, resident_experts
):
    monkeypatch.setattr(unquant_mod, "_use_xpu_moe_ld_padding", lambda _: False)
    monkeypatch.setattr(unquant_mod, "_use_aiter", False)
    monkeypatch.setattr(unquant_mod, "_is_cpu", False)
    monkeypatch.setattr(unquant_mod, "_is_npu", False)
    if resident_experts:
        _install_fake_flashinfer_core(monkeypatch)

    layer = _MockLayer()
    method = unquant_mod.UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
    method.create_weights(
        layer=layer,
        num_experts=resident_experts,
        hidden_size=layer.hidden_size,
        intermediate_size_per_partition=layer.intermediate_size_per_partition,
        params_dtype=torch.bfloat16,
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape[0] == resident_experts
    assert layer.w2_weight.shape[0] == resident_experts


def _make_fp8_method(monkeypatch, *, serialized: bool):
    monkeypatch.setattr(
        fp8_mod, "get_moe_runner_backend", lambda: _NoSpecialMoeBackend()
    )
    method = fp8_mod.Fp8MoEMethod(
        fp8_mod.Fp8Config(is_checkpoint_fp8_serialized=serialized)
    )
    method._owns_moe_runner = False
    return method


def _make_fp8_layer(resident_experts: int) -> _MockLayer:
    layer = _MockLayer()
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.ones(resident_experts, 4, 4, dtype=torch.bfloat16),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.ones(resident_experts, 4, 2, dtype=torch.bfloat16),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.ones(resident_experts, 2, dtype=torch.float32),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.ones(resident_experts, dtype=torch.float32),
            requires_grad=False,
        ),
    )
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    return layer


def _fake_scaled_fp8_quant(tensor, scale=None):
    del scale
    return (
        torch.zeros_like(tensor, dtype=fp8_mod.fp8_dtype),
        torch.tensor(1.0, dtype=torch.float32, device=tensor.device),
    )


@pytest.mark.parametrize("resident_experts", [0, 1, 3])
def test_fp8_online_quantization_uses_physical_expert_rows(
    monkeypatch, resident_experts
):
    method = _make_fp8_method(monkeypatch, serialized=False)
    layer = _make_fp8_layer(resident_experts)
    monkeypatch.setattr(fp8_mod, "_is_hip", False)
    monkeypatch.setattr(fp8_mod, "_use_hip_int4", False)
    monkeypatch.setattr(fp8_mod, "scaled_fp8_quant", _fake_scaled_fp8_quant)

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape[0] == resident_experts
    assert layer.w2_weight.shape[0] == resident_experts
    assert layer.w13_weight_scale.shape == (resident_experts,)


@pytest.mark.parametrize("resident_experts", [0, 1, 3])
def test_fp8_serialized_requantization_uses_physical_expert_rows(
    monkeypatch, resident_experts
):
    method = _make_fp8_method(monkeypatch, serialized=True)
    layer = _make_fp8_layer(resident_experts)
    monkeypatch.setattr(fp8_mod, "_is_hip", False)
    monkeypatch.setattr(fp8_mod, "_use_hip_int4", False)
    monkeypatch.setattr(fp8_mod, "_is_fp8_fnuz", False)
    monkeypatch.setattr(
        fp8_mod, "per_tensor_dequantize", lambda weight, scale: weight.float()
    )
    monkeypatch.setattr(fp8_mod, "scaled_fp8_quant", _fake_scaled_fp8_quant)

    method.process_weights_after_loading(layer)

    assert layer.w13_weight_scale.shape == (resident_experts,)


@pytest.mark.parametrize(
    "method_factory",
    [
        lambda: unquant_mod.UnquantizedFusedMoEMethod(),
        lambda: fp8_mod.Fp8MoEMethod.__new__(fp8_mod.Fp8MoEMethod),
    ],
)
def test_runtime_geometry_keeps_global_count_and_compacts_local_rows(
    method_factory,
):
    method = method_factory()
    layer = SimpleNamespace(
        num_experts=8,
        num_local_experts=4,
        moe_ep_rank=1,
        w13_weight=torch.empty(2, 4, 4),
    )

    assert method._runtime_expert_geometry(layer) == (8, 4, 4)

    method._kt_compact_expert_rows = True
    assert method._runtime_expert_geometry(layer) == (8, 0, 2)


@pytest.mark.parametrize("resident_experts", [0, 1, 3])
def test_fp8_trtllm_activation_params_use_physical_rows(resident_experts):
    method = fp8_mod.Fp8MoEMethod.__new__(fp8_mod.Fp8MoEMethod)
    method.moe_runner_config = SimpleNamespace(
        gemm1_alpha=1.5,
        gemm1_beta=0.25,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
    )
    layer = SimpleNamespace(w13_weight=torch.empty(resident_experts, 4, 4))

    method._prepare_flashinfer_trtllm_activation_params(layer)

    assert layer._flashinfer_trtllm_gemm1_alpha.shape == (resident_experts,)
    assert layer._flashinfer_trtllm_gemm1_beta.shape == (resident_experts,)
    assert layer._flashinfer_trtllm_gemm1_clamp_limit is None


def test_unquantized_compact_rows_route_non_routed_trtllm(monkeypatch):
    captured = []

    class _Runner:
        def __init__(self, backend, config):
            captured.append((backend, config))

    monkeypatch.setattr(
        unquant_mod,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_TRTLLM,
    )
    monkeypatch.setattr(unquant_mod, "MoeRunner", _Runner)
    monkeypatch.setattr(unquant_mod, "_use_aiter", False)
    method = unquant_mod.UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
    method._kt_compact_expert_rows = True

    method.create_moe_runner(SimpleNamespace(), MoeRunnerConfig())

    assert captured[0][0] is MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED


def test_fp8_compact_rows_route_non_routed_trtllm(monkeypatch):
    captured = []

    class _Runner:
        def __init__(self, backend, config):
            captured.append((backend, config))

    monkeypatch.setattr(
        fp8_mod,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_TRTLLM,
    )
    monkeypatch.setattr(fp8_mod, "MoeRunner", _Runner)
    method = fp8_mod.Fp8MoEMethod.__new__(fp8_mod.Fp8MoEMethod)
    method.is_fp4_expert = False
    method._kt_compact_expert_rows = True

    method.create_moe_runner(SimpleNamespace(), MoeRunnerConfig())

    assert captured[0][0] is MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED


@pytest.mark.parametrize(
    "backend, error_pattern",
    [
        (MoeRunnerBackend.FLASHINFER_MEGAMOE, "global expert fleet"),
        (MoeRunnerBackend.HPC_OPS, "global/ragged expert metadata"),
        (MoeRunnerBackend.TRITON_KERNELS, "global/ragged expert metadata"),
    ],
)
def test_fp8_compact_rows_reject_incompatible_backends(
    monkeypatch, backend, error_pattern
):
    monkeypatch.setattr(fp8_mod, "get_moe_runner_backend", lambda: backend)
    method = fp8_mod.Fp8MoEMethod.__new__(fp8_mod.Fp8MoEMethod)
    method.is_fp4_expert = False
    method._kt_compact_expert_rows = True

    with pytest.raises(ValueError, match=error_pattern):
        method.create_moe_runner(SimpleNamespace(), MoeRunnerConfig())
