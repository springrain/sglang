"""CPU contracts for KTransformers compact expert rows in quant schemes."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


GLOBAL_EXPERTS = 8
HIDDEN_SIZE = 16
INTERMEDIATE_SIZE = 16


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # KTEP deliberately leaves the layer's logical/global geometry intact.
        self.num_experts = GLOBAL_EXPERTS
        self.num_local_experts = GLOBAL_EXPERTS
        self.moe_ep_rank = 3
        self.intermediate_size_per_partition = INTERMEDIATE_SIZE


def _make_ct_nvfp4_method():
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
        CompressedTensorsW4A4Nvfp4MoE,
    )

    method = CompressedTensorsW4A4Nvfp4MoE.__new__(CompressedTensorsW4A4Nvfp4MoE)
    method.group_size = 16
    method.use_flashinfer_trtllm = False
    return method


def _make_ct_fp8_method(*, block_quant: bool = False):
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
        CompressedTensorsW8A8Fp8MoE,
    )

    method = CompressedTensorsW8A8Fp8MoE.__new__(CompressedTensorsW8A8Fp8MoE)
    method.weight_quant = SimpleNamespace(
        strategy=(
            QuantizationStrategy.BLOCK if block_quant else QuantizationStrategy.TENSOR
        )
    )
    method.input_quant = SimpleNamespace(
        strategy=QuantizationStrategy.TENSOR,
        dynamic=False,
    )
    method.weight_block_size = (16, 16) if block_quant else None
    method.block_quant = block_quant
    method.static_input_scales = True
    method.use_flashinfer_trtllm = block_quant
    return method


def _make_quark_fp8_method():
    from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8_moe import (
        QuarkW8A8FP8MoE,
    )

    return QuarkW8A8FP8MoE(
        weight_config={"qscheme": "per_tensor"},
        input_config={"is_dynamic": False, "qscheme": "per_tensor"},
    )


def _make_quark_int4fp8_method():
    from sglang.srt.layers.quantization.quark_int4fp8_moe import (
        QuarkInt4Fp8MoEMethod,
    )

    return QuarkInt4Fp8MoEMethod.__new__(QuarkInt4Fp8MoEMethod)


def _create_weights(method, num_physical_experts: int) -> _Layer:
    layer = _Layer()
    method.create_weights(
        layer=layer,
        num_experts=num_physical_experts,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        params_dtype=torch.bfloat16,
    )
    return layer


@pytest.mark.parametrize("num_physical_experts", [0, 1, 3])
@pytest.mark.parametrize(
    "method_factory",
    [_make_ct_nvfp4_method, _make_ct_fp8_method, _make_quark_fp8_method],
)
def test_create_weights_uses_compact_physical_rows(
    method_factory, num_physical_experts
):
    method = method_factory()
    layer = _create_weights(method, num_physical_experts)

    assert method.supports_kt_compact_expert_rows
    assert layer.num_experts == GLOBAL_EXPERTS
    assert layer.num_local_experts == GLOBAL_EXPERTS
    for name, parameter in layer.named_parameters():
        assert parameter.shape[0] == num_physical_experts, name


@pytest.mark.parametrize("num_physical_experts", [0, 1, 3])
def test_ct_nvfp4_postprocess_uses_physical_rows(monkeypatch, num_physical_experts):
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4_moe as nvfp4_mod,
    )

    method = _make_ct_nvfp4_method()
    layer = _create_weights(method, num_physical_experts)
    with torch.no_grad():
        for name in (
            "w13_weight_global_scale",
            "w2_weight_global_scale",
            "w13_input_global_scale",
            "w2_input_global_scale",
        ):
            getattr(layer, name).fill_(1)

    monkeypatch.setattr(nvfp4_mod, "swizzle_blockscale", lambda value: value)
    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")
    assert layer.w13_weight.shape[0] == num_physical_experts
    assert layer.w2_weight.shape[0] == num_physical_experts
    if num_physical_experts:
        assert layer.g1_alphas.shape == (num_physical_experts,)
        assert layer.g2_alphas.shape == (num_physical_experts,)


@pytest.mark.parametrize("num_physical_experts", [0, 1, 3])
def test_ct_fp8_postprocess_uses_physical_rows(monkeypatch, num_physical_experts):
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_fp8_moe as fp8_mod,
    )

    method = _make_ct_fp8_method()
    layer = _create_weights(method, num_physical_experts)
    quantized_shards = []

    monkeypatch.setattr(fp8_mod, "is_fp8_fnuz", lambda: False)
    monkeypatch.setattr(fp8_mod, "per_tensor_dequantize", lambda weight, scale: weight)

    def fake_scaled_fp8_quant(weight, scale):
        quantized_shards.append(weight)
        return weight, None

    monkeypatch.setattr(fp8_mod, "scaled_fp8_quant", fake_scaled_fp8_quant)
    method.process_weights_after_loading(layer)

    assert len(quantized_shards) == 2 * num_physical_experts
    if num_physical_experts:
        assert layer.w13_weight_scale.shape == (num_physical_experts,)


@pytest.mark.parametrize("num_physical_experts", [0, 1, 3])
def test_quark_fp8_postprocess_uses_physical_rows(monkeypatch, num_physical_experts):
    from sglang.srt.layers.quantization.quark.schemes import (
        quark_w8a8_fp8_moe as fp8_mod,
    )

    method = _make_quark_fp8_method()
    layer = _create_weights(method, num_physical_experts)
    quantized_shards = []

    monkeypatch.setattr(fp8_mod, "_is_fp8_fnuz", False)
    monkeypatch.setattr(fp8_mod, "per_tensor_dequantize", lambda weight, scale: weight)

    def fake_scaled_fp8_quant(weight, scale):
        quantized_shards.append(weight)
        return weight, None

    monkeypatch.setattr(fp8_mod, "scaled_fp8_quant", fake_scaled_fp8_quant)
    method.process_weights_after_loading(layer)

    assert len(quantized_shards) == 2 * num_physical_experts
    if num_physical_experts:
        assert layer.w13_weight_scale.shape == (num_physical_experts,)


@pytest.mark.parametrize("num_physical_experts", [0, 1, 3])
def test_quark_int4fp8_postprocess_uses_physical_rows(
    monkeypatch, num_physical_experts
):
    from sglang.srt.layers.quantization import quark_int4fp8_moe as int4fp8_mod

    method = _make_quark_int4fp8_method()
    assert method.supports_kt_compact_expert_rows
    layer = _Layer()
    layer.w13_weight = torch.nn.Parameter(
        torch.ones(num_physical_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 8),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.ones(num_physical_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE // 8),
        requires_grad=False,
    )
    layer.w13_fp8_scale = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0]]).expand(num_physical_experts, -1).clone(),
        requires_grad=False,
    )
    layer.w2_fp8_scale = torch.nn.Parameter(
        torch.ones(num_physical_experts), requires_grad=False
    )
    layer.w13_int4_scale = torch.nn.Parameter(
        torch.ones(num_physical_experts, 2 * INTERMEDIATE_SIZE),
        requires_grad=False,
    )
    layer.w2_int4_scale = torch.nn.Parameter(
        torch.ones(num_physical_experts, HIDDEN_SIZE), requires_grad=False
    )

    monkeypatch.setattr(int4fp8_mod, "_is_hip", False)
    monkeypatch.setattr(
        int4fp8_mod, "shuffle_weight", lambda weight, shape: weight, raising=False
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape[0] == num_physical_experts
    assert layer.w2_weight.shape[0] == num_physical_experts
    if num_physical_experts:
        assert layer.w13_fp8_scale.shape == (num_physical_experts,)
        torch.testing.assert_close(
            layer.w13_int4_scale[:, :INTERMEDIATE_SIZE],
            torch.ones(num_physical_experts, INTERMEDIATE_SIZE),
        )
        torch.testing.assert_close(
            layer.w13_int4_scale[:, INTERMEDIATE_SIZE:],
            torch.full((num_physical_experts, INTERMEDIATE_SIZE), 2.0),
        )


def test_quark_int4fp8_online_loader_maps_before_compact_scale_access(monkeypatch):
    from sglang.srt.layers.quantization import quark_int4fp8_moe as int4fp8_mod

    method = _make_quark_int4fp8_method()
    method._kt_compact_expert_rows = True
    method.w13_shard_size = 2
    method.w2_shard_size = 2
    method.tp_rank = 0
    method.online_quant_progress_bar = SimpleNamespace(update=lambda _count: None)

    layer = SimpleNamespace(
        quant_method=SimpleNamespace(
            gpu_experts_mask=torch.tensor([False, True, False, False]),
            logical_to_gpu_index=torch.tensor([-1, 0, -1, -1]),
        ),
        use_presharded_weights=True,
        w13_int4_scale=torch.zeros(1, 4),
        w13_fp8_scale=torch.zeros(1, 2),
        w2_int4_scale=torch.zeros(1, 4),
        w2_fp8_scale=torch.zeros(1),
    )
    param = torch.nn.Parameter(
        torch.zeros(1, 4, 2, dtype=torch.uint32), requires_grad=False
    )
    loaded_weight = torch.ones(2, 16)
    original_calls = []

    monkeypatch.setattr(
        int4fp8_mod,
        "quantize_fp8_scale_tensorwise",
        lambda _weight: (None, torch.tensor(3.0)),
    )
    monkeypatch.setattr(
        int4fp8_mod,
        "quantize_int4_scale_columnwise",
        lambda _weight: (torch.ones(2, 2, dtype=torch.uint32), torch.full((2,), 6.0)),
    )
    monkeypatch.setattr(int4fp8_mod, "pack_int4_to_int32", lambda weight: weight)

    loader = method.get_weight_loader(
        layer,
        lambda _param, _weight, **kwargs: original_calls.append(kwargs),
    )
    loader(param, loaded_weight, "weight", "w1", 0)
    assert original_calls == []

    loader(param, loaded_weight, "weight", "w1", 1)
    assert original_calls == [
        {"shard_id": "w1", "weight_name": "weight", "expert_id": 1}
    ]
    torch.testing.assert_close(layer.w13_int4_scale[0, :2], torch.full((2,), 2.0))
    assert layer.w13_fp8_scale[0, 0].item() == 3.0


class _CapturingRunner:
    def __init__(self):
        self.quant_info = None
        self.runner_backend = SimpleNamespace(is_aiter=lambda: False)

    def run(self, dispatch_output, quant_info):
        self.quant_info = quant_info
        return SimpleNamespace(hidden_states=dispatch_output.hidden_states)


def test_ct_nvfp4_compact_runner_forces_routed_trtllm(monkeypatch):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4_moe as nvfp4_mod,
    )

    captured = {}
    monkeypatch.setattr(
        nvfp4_mod,
        "MoeRunner",
        lambda backend, config: (
            captured.update(backend=backend, config=config) or SimpleNamespace()
        ),
    )
    method = _make_ct_nvfp4_method()
    method.use_flashinfer_trtllm = True
    method._kt_compact_expert_rows = True
    config = SimpleNamespace()

    method.create_moe_runner(SimpleNamespace(), config)

    assert captured == {
        "backend": MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        "config": config,
    }


def test_ct_fp8_compact_runner_forces_routed_trtllm(monkeypatch):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_fp8_moe as fp8_mod,
    )

    captured = {}
    monkeypatch.setattr(
        fp8_mod,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_TRTLLM,
    )
    monkeypatch.setattr(
        fp8_mod,
        "MoeRunner",
        lambda backend, config: (
            captured.update(backend=backend, config=config) or SimpleNamespace()
        ),
    )
    method = _make_ct_fp8_method(block_quant=True)
    method._kt_compact_expert_rows = True
    config = SimpleNamespace()

    method.create_moe_runner(SimpleNamespace(), config)

    assert captured == {
        "backend": MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        "config": config,
    }


@pytest.mark.parametrize("num_physical_experts", [1, 3])
def test_ct_nvfp4_flashinfer_geometry_is_compact(num_physical_experts):
    method = _make_ct_nvfp4_method()
    method.use_flashinfer_trtllm = True
    method._kt_compact_expert_rows = True
    method.runner = _CapturingRunner()

    layer = SimpleNamespace(
        w13_weight=torch.empty(num_physical_experts, 2, 2),
        w2_weight=torch.empty(num_physical_experts, 2, 2),
        w13_weight_scale=torch.empty(num_physical_experts, 2, 2),
        w2_weight_scale=torch.empty(num_physical_experts, 2, 2),
        g1_scale_c=torch.empty(num_physical_experts),
        g1_alphas=torch.empty(num_physical_experts),
        g2_alphas=torch.empty(num_physical_experts),
        w13_input_scale_quant=torch.empty(num_physical_experts),
        num_experts=GLOBAL_EXPERTS,
        num_local_experts=GLOBAL_EXPERTS,
        moe_ep_rank=3,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        routing_method_type=1,
    )
    dispatch_output = SimpleNamespace(hidden_states=torch.empty(1, HIDDEN_SIZE))

    method.apply_weights(layer, dispatch_output)
    quant_info = method.runner.quant_info
    assert quant_info.global_num_experts == GLOBAL_EXPERTS
    assert quant_info.local_expert_offset == 0
    assert quant_info.local_num_experts == num_physical_experts


@pytest.mark.parametrize("num_physical_experts", [1, 3])
def test_ct_nvfp4_cutlass_geometry_is_compact(num_physical_experts):
    method = _make_ct_nvfp4_method()
    method._kt_compact_expert_rows = True
    method.runner = _CapturingRunner()
    method.moe_runner_config = SimpleNamespace(apply_router_weight_on_input=False)

    layer = SimpleNamespace(
        w13_weight=torch.empty(num_physical_experts, 2, 2),
        w2_weight=torch.empty(num_physical_experts, 2, 2),
        w13_weight_scale=torch.empty(num_physical_experts, 2, 2),
        w2_weight_scale=torch.empty(num_physical_experts, 2, 2),
        g1_alphas=torch.empty(num_physical_experts),
        g2_alphas=torch.empty(num_physical_experts),
        w13_input_scale_quant=torch.empty(num_physical_experts),
        w2_input_scale_quant=torch.empty(num_physical_experts),
        moe_ep_size=4,
        moe_ep_rank=3,
        moe_tp_size=2,
        moe_tp_rank=1,
    )
    dispatch_output = SimpleNamespace(hidden_states=torch.empty(1, HIDDEN_SIZE))

    method.apply_weights(layer, dispatch_output)
    quant_info = method.runner.quant_info
    assert quant_info.moe_ep_size == 1
    assert quant_info.moe_ep_rank == 0
    assert quant_info.moe_tp_size == 2
    assert quant_info.moe_tp_rank == 1


@pytest.mark.parametrize("num_physical_experts", [1, 3])
def test_ct_fp8_flashinfer_geometry_is_compact(monkeypatch, num_physical_experts):
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

    monkeypatch.setattr(
        flashinfer_trtllm, "get_activation_type", lambda *args, **kwargs: 0
    )
    method = _make_ct_fp8_method(block_quant=True)
    method._kt_compact_expert_rows = True
    method.runner = _CapturingRunner()
    method.moe_runner_config = SimpleNamespace(
        activation="silu",
        is_gated=True,
    )

    layer = SimpleNamespace(
        w13_weight=torch.empty(num_physical_experts, 2, 2),
        w2_weight=torch.empty(num_physical_experts, 2, 2),
        w13_weight_scale=torch.empty(num_physical_experts, 2, 2),
        w2_weight_scale=torch.empty(num_physical_experts, 2, 2),
        w13_input_scale=None,
        w2_input_scale=None,
        num_experts=GLOBAL_EXPERTS,
        num_local_experts=GLOBAL_EXPERTS,
        moe_ep_rank=3,
        routing_method_type=1,
    )
    dispatch_output = SimpleNamespace(
        hidden_states=torch.empty(1, HIDDEN_SIZE),
        topk_output=None,
    )

    method.apply_weights(layer, dispatch_output)
    quant_info = method.runner.quant_info
    assert quant_info.global_num_experts == GLOBAL_EXPERTS
    assert quant_info.local_expert_offset == 0
    assert quant_info.local_num_experts == num_physical_experts
