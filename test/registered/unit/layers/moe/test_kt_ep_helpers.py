"""Unit coverage for the SGLang-side KT Q8/P1 routing helpers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

kt_ep = pytest.importorskip("sglang.srt.layers.moe.kt_ep_wrapper")


def _parse_server_args(extra: list[str]) -> ServerArgs:
    import argparse

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    # The modular resolver deliberately stops before model-derived defaults
    # for a dummy model. Supply the two values generic validation reads so this
    # helper can exercise the real resolve -> check lifecycle without loading a
    # model solely to test KT's argument contract.
    namespace = parser.parse_args(
        [
            "--model",
            "dummy",
            "--served-model-name",
            "dummy",
            "--chunked-prefill-size",
            "-1",
            *extra,
        ]
    )
    server_args = ServerArgs.from_cli_args(namespace)
    server_args.resolve_once()
    server_args.check_server_args()
    return server_args


@pytest.mark.parametrize(
    ("method", "threshold", "dynamic", "expected"),
    [
        ("RAWINT4", 4096, False, True),
        ("rawint4", 1, False, True),
        ("RAWINT4", 0, False, False),
        ("RAWINT4", None, False, False),
        ("MXFP4", 4096, False, False),
        ("MXFP4", None, True, True),
    ],
)
def test_pack_all_native_experts_for_dynamic_or_rawint4_full_gpu_prefill(
    method, threshold, dynamic, expected
):
    config = SimpleNamespace(
        method=method,
        gpu_prefill_token_threshold=threshold,
        kt_enable_dynamic_expert_update=dynamic,
    )

    assert kt_ep._should_pack_all_experts_on_load(config) is expected


def test_uniform_masks_keep_dense_layers_on_gpu():
    masks = kt_ep.generate_uniform_masks(
        num_layers=5,
        num_experts=8,
        num_gpu_experts=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
    )
    assert masks.dtype == torch.bool
    assert masks.shape == (5, 8)
    assert masks[0].all()
    assert [int(row.sum()) for row in masks[1:]] == [2, 2, 2, 2]


def test_front_loading_masks_are_deterministic_and_bounded():
    masks = kt_ep.generate_front_loading_masks(
        num_layers=4,
        num_experts=4,
        num_gpu_experts=5,
        first_k_dense_replace=0,
        moe_layer_freq=1,
    )
    assert [int(row.sum()) for row in masks] == [4, 1, 0, 0]
    assert torch.equal(masks, kt_ep.generate_front_loading_masks(4, 4, 5, 0, 1))


def test_frequency_masks_do_not_spend_budget_on_pinned_layers():
    activation_freq = torch.zeros(6, 4)
    activation_freq[:2] = 1000
    activation_freq[5, 3] = 10

    masks = kt_ep.generate_frequency_masks(
        activation_freq=activation_freq,
        num_gpu_experts=8,
        first_k_dense_replace=2,
        moe_layer_freq=1,
    )

    # The first two rows bypass KTEP and keep their all-GPU sentinel masks.
    assert masks[:2].all()
    # All eight budgeted experts must belong to the four remaining MoE layers,
    # even when the bypassed layers have much larger recorded counts.
    assert int(masks[2:].sum()) == 8
    assert masks[5, 3]


def test_frequency_masks_clamp_budget_to_eligible_moe_experts():
    activation_freq = torch.zeros(5, 3)

    masks = kt_ep.generate_frequency_masks(
        activation_freq=activation_freq,
        num_gpu_experts=100,
        first_k_dense_replace=1,
        moe_layer_freq=2,
    )

    # With the global modulo convention used by the wrapper, only layers 2 and
    # 4 are eligible MoE layers. The oversized budget clamps to their six rows.
    assert masks[[0, 1, 3]].all()
    assert masks[2].all()
    assert masks[4].all()
    assert int(masks[[2, 4]].sum()) == 6


def test_random_masks_are_reproducible():
    first = kt_ep.generate_random_masks(4, 8, 6, 0, 1, seed=17)
    second = kt_ep.generate_random_masks(4, 8, 6, 0, 1, seed=17)
    assert torch.equal(first, second)
    assert int(first.sum()) == 6


def test_mask_and_remap_preserves_gpu_order_and_masks_cpu_experts():
    mask = torch.tensor([True, False, True, False])
    logical_to_gpu = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    topk_ids = torch.tensor([[0, 1, 2], [3, 2, 0]], dtype=torch.int64)

    eager = getattr(kt_ep.mask_and_remap_expert_ids, "__wrapped__", None)
    remap = eager or kt_ep.mask_and_remap_expert_ids
    output = remap(topk_ids, mask, logical_to_gpu)
    expected = torch.tensor([[0, -1, 1], [-1, 1, 0]], dtype=torch.int32)
    assert torch.equal(output, expected)


def test_mask_and_remap_preserves_invalid_routing_sentinels():
    mask = torch.tensor([True, False, True, False])
    logical_to_gpu = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    topk_ids = torch.tensor([[-1, 4, 2]], dtype=torch.int64)

    eager = getattr(kt_ep.mask_and_remap_expert_ids, "__wrapped__", None)
    remap = eager or kt_ep.mask_and_remap_expert_ids
    output = remap(topk_ids, mask, logical_to_gpu)

    assert torch.equal(output, torch.tensor([[-1, -1, 1]], dtype=torch.int32))


def test_materialize_kt_topk_output_resolves_bypassed_routing_once(monkeypatch):
    import sglang.srt.layers.moe.topk as topk_mod

    expected = topk_mod.StandardTopKOutput(
        topk_weights=torch.tensor([[0.75, 0.25]]),
        topk_ids=torch.tensor([[3, 1]]),
        router_logits=torch.zeros(1, 4),
    )
    calls = []

    def fake_select_experts(**kwargs):
        calls.append(kwargs["layer_id"])
        return expected

    monkeypatch.setattr(topk_mod, "select_experts", fake_select_experts)
    bypassed = topk_mod.BypassedTopKOutput(
        hidden_states=torch.zeros(1, 2),
        router_logits=torch.zeros(1, 4),
        topk_config=topk_mod.TopKConfig(top_k=2),
    )

    result = kt_ep.materialize_kt_topk_output(bypassed, layer_idx=7)

    assert result is expected
    assert calls == [7]
    assert kt_ep.materialize_kt_topk_output(expected, layer_idx=7) is expected


def test_materialize_kt_topk_output_rejects_triton_kernel_carrier(monkeypatch):
    import sglang.srt.layers.moe.topk as topk_mod

    carrier = object()
    monkeypatch.setattr(
        topk_mod.TopKOutputChecker,
        "format_is_bypassed",
        staticmethod(lambda value: False),
    )
    monkeypatch.setattr(
        topk_mod.TopKOutputChecker,
        "format_is_triton_kernels",
        staticmethod(lambda value: value is carrier),
    )

    with pytest.raises(ValueError, match="triton_kernel"):
        kt_ep.materialize_kt_topk_output(carrier, layer_idx=7)


def test_weight_loader_without_metadata_uses_kt_compact_mapping(monkeypatch):
    import sglang.srt.layers.moe.fused_moe_triton.layer as fused_moe_layer

    moe = fused_moe_layer.FusedMoE.__new__(fused_moe_layer.FusedMoE)
    moe.quant_config = None
    moe._expert_storage_rank = 0
    moe._num_local_routed = 6
    moe._has_fused_shared = False
    moe.num_local_experts = 6

    wrapper = kt_ep.KTEPWrapperMethod.__new__(kt_ep.KTEPWrapperMethod)
    wrapper.num_gpu_experts = 2
    wrapper.gpu_experts_mask = torch.tensor(
        [False, True, False, False, True, False], dtype=torch.bool
    )
    wrapper.logical_to_gpu_index = torch.tensor(
        [-1, 0, -1, -1, 1, -1], dtype=torch.int32
    )
    moe.quant_method = wrapper

    writes = []

    def record_write(**kwargs):
        writes.append(kwargs["expert_id"])

    moe._weight_loader_impl = record_write
    monkeypatch.setattr(
        fused_moe_layer, "get_global_expert_location_metadata", lambda: None
    )
    param = SimpleNamespace(_sglang_require_global_experts=False)
    loaded_weight = torch.ones(1)

    moe.weight_loader(param, loaded_weight, "weight", "w1", expert_id=4)
    moe.weight_loader(param, loaded_weight, "weight", "w1", expert_id=1)
    moe.weight_loader(param, loaded_weight, "weight", "w1", expert_id=2)

    assert writes == [1, 0]


def test_fused_mxfp4_loader_selects_only_resident_rows():
    import sglang.srt.layers.moe.fused_moe_triton.layer as fused_moe_layer

    class _StaticMxfp4Config:
        @staticmethod
        def get_name():
            return "mxfp4"

        @staticmethod
        def is_static_cfg():
            return True

    moe = fused_moe_layer.FusedMoE.__new__(fused_moe_layer.FusedMoE)
    moe.quant_config = _StaticMxfp4Config()
    wrapper = kt_ep.KTEPWrapperMethod.__new__(kt_ep.KTEPWrapperMethod)
    wrapper.gpu_index_to_logical = torch.tensor([1, 4], dtype=torch.int32)
    moe.quant_method = wrapper

    param = SimpleNamespace(data=torch.zeros((2, 2, 2), dtype=torch.float32))
    loaded_weight = torch.stack(
        [torch.full((2, 2), float(expert_id)) for expert_id in range(6)]
    )

    moe.weight_loader(param, loaded_weight, "weight", "w1", expert_id=None)

    assert torch.equal(param.data[0], torch.full((2, 2), 1.0))
    assert torch.equal(param.data[1], torch.full((2, 2), 4.0))


def test_select_top_experts_ignores_invalid_ids_and_is_stable():
    topk_ids = torch.tensor([[2, 2, -1], [4, 2, 1], [4, 9, 1]])
    selected = kt_ep.select_top_experts_from_batch(
        topk_ids=topk_ids, num_experts=5, num_gpu_experts=3
    )
    assert torch.equal(selected, torch.tensor([1, 2, 4]))


def test_update_gpu_expert_mappings_round_trip():
    mask, logical_to_gpu, gpu_to_logical = kt_ep.update_gpu_expert_mappings(
        selected_experts=torch.tensor([5, 1, 3]),
        num_experts=8,
        device=torch.device("cpu"),
    )
    assert torch.equal(
        mask, torch.tensor([False, True, False, True, False, True, False, False])
    )
    assert torch.equal(
        logical_to_gpu, torch.tensor([-1, 1, -1, 2, -1, 0, -1, -1], dtype=torch.int32)
    )
    assert torch.equal(gpu_to_logical, torch.tensor([5, 1, 3], dtype=torch.int32))


def test_kt_cli_matrix_is_accepted():
    server_args = _parse_server_args(
        [
            "--kt-weight-path",
            "/tmp/q8.gguf",
            "--kt-method",
            "LLAMAFILE",
            "--kt-cpuinfer",
            "128",
            "--kt-threadpool-count",
            "2",
            "--kt-num-gpu-experts",
            "64",
            "--kt-max-deferred-experts-per-token",
            "2",
        ]
    )
    assert server_args.kt_method == "LLAMAFILE"
    assert server_args.kt_cpuinfer == 128
    assert server_args.kt_threadpool_count == 2
    assert server_args.kt_num_gpu_experts == 64
    assert server_args.kt_max_deferred_experts_per_token == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--kt-weight-path", "/tmp/q8.gguf", "--kt-cpuinfer", "0"],
        [
            "--kt-weight-path",
            "/tmp/q8.gguf",
            "--kt-cpuinfer",
            "2",
            "--kt-threadpool-count",
            "0",
        ],
        [
            "--kt-weight-path",
            "/tmp/q8.gguf",
            "--kt-cpuinfer",
            "2",
            "--moe-a2a-backend",
            "deepep",
        ],
        [
            "--kt-weight-path",
            "/tmp/q8.gguf",
            "--kt-cpuinfer",
            "2",
            "--ep-size",
            "2",
        ],
        [
            "--kt-weight-path",
            "/tmp/q8.gguf",
            "--kt-cpuinfer",
            "2",
            "--moe-runner-backend",
            "triton_kernel",
        ],
        [
            "--kt-weight-path",
            "/tmp/q8.gguf",
            "--kt-cpuinfer",
            "2",
            "--moe-runner-backend",
            "hpc_ops",
        ],
    ],
)
def test_kt_invalid_runtime_combinations_fail_fast(extra):
    with pytest.raises(ValueError):
        _parse_server_args(extra)
