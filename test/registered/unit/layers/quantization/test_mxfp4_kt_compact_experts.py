"""CPU-only regression coverage for compact KTransformers MXFP4 experts."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    resolve_kt_cpu_activation,
    validate_kt_v4_mxfp4_activation,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization import mxfp4 as mxfp4_mod
from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
    Mxfp4FlashinferCutlassMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4_humming_moe import Mxfp4HummingMoEMethod
from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CPUPlatform:
    is_sm90 = False
    is_sm100 = False
    is_sm120 = False


class _MockLayer(torch.nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.hidden_size = 64
        self.intermediate_size_per_partition = 32
        self.moe_ep_rank = 0


def _build_method():
    method = mxfp4_mod.Mxfp4MoEMethod.__new__(mxfp4_mod.Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_mega_moe = False
    method.use_deep_gemm = False
    method.use_flashinfer = False
    method._fi_kernel = None
    return method


@pytest.mark.parametrize("resident_experts", [0, 1, 3, 8])
def test_create_weights_uses_resident_expert_rows(monkeypatch, resident_experts):
    monkeypatch.setattr(mxfp4_mod, "get_platform", lambda: _CPUPlatform())
    monkeypatch.setattr(mxfp4_mod, "_use_aiter", False)
    monkeypatch.setattr(mxfp4_mod, "_is_xpu", False)
    monkeypatch.setattr(mxfp4_mod, "_is_hip", False)
    monkeypatch.setattr(mxfp4_mod, "has_triton_kernels", False)

    layer = _MockLayer()
    method = _build_method()
    method.create_weights(
        layer=layer,
        num_experts=resident_experts,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
        with_bias=True,
    )

    expected_shapes = {
        "w13_weight": (resident_experts, 64, 32),
        "w13_weight_scale": (resident_experts, 64, 2),
        "w13_weight_bias": (resident_experts, 64),
        "w2_weight": (resident_experts, 64, 16),
        "w2_weight_scale": (resident_experts, 64, 1),
        "w2_weight_bias": (resident_experts, 64),
    }
    for name, shape in expected_shapes.items():
        assert tuple(getattr(layer, name).shape) == shape


def test_zero_resident_experts_skip_mxfp4_postprocess():
    layer = SimpleNamespace()
    method = _build_method()
    method.num_experts = 0

    method.process_weights_after_loading(layer)

    assert layer._mxfp4_backend == "kt_cpu_only"


def test_runtime_geometry_uses_compact_kt_ids():
    layer = SimpleNamespace(num_experts=8, num_local_experts=4, moe_ep_rank=1)
    method = _build_method()
    method.num_experts = 2

    assert method._runtime_expert_geometry(layer) == (8, 4, 4)

    method._kt_compact_expert_rows = True
    assert method._runtime_expert_geometry(layer) == (8, 0, 2)


def test_specialized_mxfp4_backends_declare_compact_kt_support():
    for method_cls in (
        mxfp4_mod.Mxfp4DynamicQuantMoEMethod,
        Mxfp4FlashinferCutlassMoEMethod,
        Mxfp4FlashinferTrtllmMoEMethod,
        Mxfp4HummingMoEMethod,
        Mxfp4MarlinMoEMethod,
    ):
        assert method_cls.supports_kt_compact_expert_rows is True


def test_flashinfer_trtllm_runtime_geometry_uses_weight_rows():
    method = Mxfp4FlashinferTrtllmMoEMethod.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    layer = SimpleNamespace(
        num_experts=8,
        num_local_experts=8,
        moe_ep_rank=0,
        w13_weight=torch.empty(3, 16, 8),
    )

    assert method._runtime_expert_geometry(layer) == (8, 0, 8)

    method._kt_compact_expert_rows = True
    assert method._runtime_expert_geometry(layer) == (8, 0, 3)


@pytest.mark.parametrize("resident_experts, expected_calls", [(0, 0), (1, 1)])
def test_kt_postprocess_skips_only_empty_gpu_image(
    monkeypatch, resident_experts, expected_calls
):
    calls = []

    class _GpuMethod:
        def process_weights_after_loading(self, layer):
            calls.append(layer)

    wrapper = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    wrapper.gpu_method = _GpuMethod()
    wrapper.num_gpu_experts = resident_experts
    wrapper.kt_config = SimpleNamespace(
        method="MXFP4", kt_enable_dynamic_expert_update=False
    )
    wrapper.tp_rank = 1
    wrapper.wrapper = None
    wrapper.kt_expert_lora_enabled = False
    layer = SimpleNamespace()
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper._mxfp4_pipeline_backend_supported",
        lambda method, target_layer: False,
    )

    wrapper.process_weights_after_loading(layer)

    assert len(calls) == expected_calls


@pytest.mark.parametrize("resident_experts", [0, 1, 3])
def test_kt_runner_uses_compact_mxfp4_geometry(resident_experts):
    captured = []

    class _GpuMethod:
        supports_kt_compact_expert_rows = True
        runner = object()

        def create_moe_runner(self, layer, config):
            captured.append(config)

    wrapper = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    wrapper.gpu_method = _GpuMethod()
    wrapper.num_gpu_experts = resident_experts
    wrapper._uses_compact_gpu_expert_rows = True
    wrapper.override_num_local_experts = True
    config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=8,
        routed_scaling_factor=2.5,
    )

    wrapper.create_moe_runner(object(), config)

    if resident_experts == 0:
        assert captured == []
        assert wrapper.runner is None
    else:
        assert len(captured) == 1
        assert captured[0].num_experts == 8
        assert captured[0].num_local_experts == resident_experts
        assert captured[0].routed_scaling_factor is None


def test_kt_wrapper_propagates_compact_capability_to_delegated_scheme():
    scheme = SimpleNamespace(supports_kt_compact_expert_rows=True)
    wrapper = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    wrapper._uses_compact_gpu_expert_rows = False

    wrapper._enable_compact_gpu_expert_rows(scheme)

    assert wrapper._uses_compact_gpu_expert_rows is True
    assert scheme._kt_compact_expert_rows is True


def test_k3_situ_maps_to_explicit_kt_activation_contract():
    kwargs = resolve_kt_cpu_activation(
        SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            swiglu_limit=None,
        ),
        "MXFP4",
    )

    assert kwargs == {
        "activation": "situ",
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "swiglu_alpha": 0.0,
        "swiglu_limit": 0.0,
    }


def test_kt_v4_resident_wave_forwards_the_shared_activation_contract(
    monkeypatch,
):
    import sglang.srt.layers.moe.topk as topk_mod
    import sglang.srt.layers.quantization.v4_marlin_moe as v4_mod

    captured = {}

    def fake_apply(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["hidden_states"])

    monkeypatch.setattr(v4_mod, "apply_v4_marlin_moe", fake_apply)
    monkeypatch.setattr(
        topk_mod.TopKOutputChecker,
        "format_is_standard",
        staticmethod(lambda _value: True),
    )

    method = Mxfp4MarlinMoEMethod.__new__(Mxfp4MarlinMoEMethod)
    method._kt_layerwise_enabled = True
    method._kt_activation_kwargs = {
        "activation": "situ",
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "swiglu_alpha": 0.0,
        "swiglu_limit": 0.0,
    }
    method.moe_runner_config = SimpleNamespace(routed_scaling_factor=None)
    layer = SimpleNamespace(
        _v4_marlin_path=True,
        _v4_marlin_weights=object(),
        w13_weight=torch.empty((2, 8, 2)),
        should_fuse_routed_scaling_factor_in_topk=False,
    )
    topk_output = SimpleNamespace(
        topk_ids=torch.tensor([[0]], dtype=torch.int32),
        topk_weights=torch.ones((1, 1)),
    )
    dispatch_output = SimpleNamespace(
        hidden_states=torch.ones((1, 4)),
        topk_output=topk_output,
    )

    result = method.apply(layer, dispatch_output)

    assert tuple(result.hidden_states.shape) == (1, 4)
    assert captured["activation"] == "situ"
    assert captured["situ_beta"] == 4.0
    assert captured["situ_linear_beta"] == 25.0
    assert captured["swiglu_alpha"] == 0.0
    assert captured["swiglu_limit"] == 0.0


def test_kt_v4_nonplain_activation_requires_split_gate_up_layout():
    with pytest.raises(ValueError, match="gate_up_interleaved=False"):
        validate_kt_v4_mxfp4_activation(
            SimpleNamespace(gate_up_interleaved=True),
            {"activation": "situ"},
        )

    validate_kt_v4_mxfp4_activation(
        SimpleNamespace(gate_up_interleaved=False),
        {"activation": "swiglu_oai"},
    )


@pytest.mark.parametrize(
    "method",
    [
        "RAWINT4",
        "FP8",
        "BF16",
        "FP8_PERCHANNEL",
        "GPTQ_INT4",
        "MXFP4",
        "NVFP4",
        "MXFP8",
    ],
)
def test_situ_is_available_to_all_shared_native_cpu_activation_methods(method):
    kwargs = resolve_kt_cpu_activation(
        SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            swiglu_limit=None,
        ),
        method,
    )

    assert kwargs["activation"] == "situ"
    assert kwargs["situ_beta"] == 4.0
    assert kwargs["situ_linear_beta"] == 25.0


def test_situ_zero_linear_beta_disables_the_up_softcap():
    kwargs = resolve_kt_cpu_activation(
        SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=0.0,
            swiglu_limit=None,
        ),
        "BF16",
    )

    assert kwargs["situ_linear_beta"] == 0.0


def test_minimax_swiglu_oai_and_v4_clamp_keep_distinct_contracts():
    for method in (
        "RAWINT4",
        "FP8",
        "BF16",
        "FP8_PERCHANNEL",
        "GPTQ_INT4",
        "SYCL_GPTQ_INT4",
        "MXFP4",
        "NVFP4",
        "MXFP8",
        "LLAMAFILE",
    ):
        oai = resolve_kt_cpu_activation(
            SimpleNamespace(
                activation="silu",
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7.0,
                swiglu_limit=None,
            ),
            method,
        )
        assert oai["activation"] == "swiglu_oai"
        assert oai["swiglu_alpha"] == 1.702
        assert oai["swiglu_limit"] == 7.0
        assert oai["situ_beta"] is None

    v4 = resolve_kt_cpu_activation(
        SimpleNamespace(
            activation="silu",
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=10.0,
        ),
        "MXFP4",
    )
    assert v4["activation"] == "silu"
    assert v4["swiglu_alpha"] == 0.0
    assert v4["swiglu_limit"] == 10.0


@pytest.mark.parametrize(
    "config, method, match",
    [
        (
            SimpleNamespace(activation="situ", gemm1_alpha=4.0),
            "SYCL_GPTQ_INT4",
            "shared CPU activation path",
        ),
        (
            SimpleNamespace(
                activation="silu",
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7.0,
            ),
            "AMXINT4",
            "cannot preserve",
        ),
        (
            SimpleNamespace(activation="situ", gemm1_alpha=float("nan")),
            "MXFP4",
            "finite positive beta",
        ),
        (
            SimpleNamespace(activation="silu", gemm1_alpha=-1.0),
            "BF16",
            "finite and non-negative",
        ),
        (
            SimpleNamespace(
                activation="silu",
                gemm1_alpha=1.702,
                gemm1_beta=2.0,
            ),
            "MXFP4",
            "gemm1_beta=1.0",
        ),
        (
            SimpleNamespace(
                activation="silu",
                gemm1_alpha=None,
                gemm1_clamp_limit=7.0,
            ),
            "MXFP4",
            "post-SiLU clamp",
        ),
        (SimpleNamespace(activation="gelu"), "MXFP4", "do not support"),
        (
            SimpleNamespace(
                activation="situ",
                gemm1_alpha=4.0,
                gemm1_clamp_limit=-1.0,
            ),
            "MXFP4",
            "linear_beta",
        ),
    ],
)
def test_invalid_kt_cpu_activation_contracts_fail_fast(config, method, match):
    with pytest.raises(ValueError, match=match):
        resolve_kt_cpu_activation(config, method)


def test_kt_submit_accepts_situ_after_kernel_contract_is_explicit():
    wrapper = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    wrapper.moe_runner_config = SimpleNamespace(activation="situ")
    wrapper.tp_rank = 1
    wrapper.wrapper = None

    wrapper.submit(object(), SimpleNamespace())
