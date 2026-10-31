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


def test_uniform_masks_reserve_shared_experts_before_routed_ids():
    masks = kt_ep.generate_uniform_masks(
        num_layers=3,
        num_experts=8,
        num_gpu_experts=8,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        num_fused_shared_experts=2,
    )

    assert [int(row.sum()) for row in masks] == [3, 3, 2]
    assert masks[0, [6, 7, 0]].all()
    assert masks[1, [6, 7, 0]].all()
    assert masks[2, [6, 7]].all()


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


def test_global_gec_slots_fill_layer_major_without_averaging():
    masks = kt_ep.generate_global_slot_masks(
        num_layers=60,
        num_experts=256,
        global_slots=2560,
        first_k_dense_replace=0,
        moe_layer_freq=1,
    )

    assert [int(masks[layer].sum()) for layer in range(12)] == [
        *([256] * 10),
        0,
        0,
    ]
    assert int(masks.sum()) == 2560


def test_gec_slot_masks_support_uniform_and_front_loading():
    uniform = kt_ep.generate_gec_slot_masks(
        num_layers=4,
        num_experts=8,
        global_slots=8,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        placement_strategy="uniform",
    )
    front_loading = kt_ep.generate_gec_slot_masks(
        num_layers=4,
        num_experts=8,
        global_slots=10,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        placement_strategy="front-loading",
    )

    assert [int(row.sum()) for row in uniform] == [2, 2, 2, 2]
    assert [int(row.sum()) for row in front_loading] == [8, 2, 0, 0]
    assert int(front_loading.sum()) == 10
    with pytest.raises(ValueError, match="uniform.*front-loading"):
        kt_ep.generate_gec_slot_masks(
            num_layers=4,
            num_experts=8,
            global_slots=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            placement_strategy="random",
        )


def test_gec_front_loading_5000_slots_fills_13_moe_layers_after_pinned_range():
    masks = kt_ep.generate_gec_slot_masks(
        num_layers=40,
        num_experts=384,
        global_slots=5000,
        first_k_dense_replace=5,
        moe_layer_freq=1,
        placement_strategy="front-loading",
    )

    assert masks[:5].all()
    assert [int(row.sum()) for row in masks[5:20]] == [
        *([384] * 13),
        8,
        0,
    ]
    assert int(masks[5:].sum()) == 5000


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


def test_mask_gpu_expert_ids_for_cpu_keeps_only_fallback_routes():
    mask = torch.tensor([True, False, True, False])
    topk_ids = torch.tensor([[0, 1, 2], [3, -1, 4]], dtype=torch.int64)

    output = kt_ep.mask_gpu_expert_ids_for_cpu(topk_ids, mask)

    assert torch.equal(output, torch.tensor([[-1, 1, -1], [3, -1, -1]]))


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


def test_mxfp4_prepared_only_enabled_by_positive_gec_slots():
    assert not kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_expert_gpu_slots=None)
    )
    assert not kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_expert_gpu_slots=0)
    )
    assert kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_expert_gpu_slots=2048)
    )
    assert not kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_expert_gpu_slots="invalid")
    )


def test_mxfp4_layerwise_reservation_skips_gec_layers(monkeypatch):
    method = SimpleNamespace(_gec_enabled=True)
    layer = SimpleNamespace()
    monkeypatch.setattr(
        kt_ep,
        "_MXFP4_PREFILL_LAYER_REGISTRY",
        {("gec",): {7: (method, layer)}},
    )
    monkeypatch.setattr(kt_ep, "_MXFP4_LAYERWISE_MANAGERS", {})
    monkeypatch.setattr(kt_ep, "_MXFP4_LAYERWISE_DISABLED_REASONS", {})

    def unexpected_probe(*args, **kwargs):
        raise AssertionError("GEC layers must not reserve full-layer slots")

    monkeypatch.setattr(
        kt_ep, "_mxfp4_pipeline_runtime_supported", unexpected_probe
    )

    assert kt_ep.get_mxfp4_layerwise_prefill_reservation_bytes() == 0


def test_mxfp4_prepared_only_layer_accepts_cpu_or_released_raw_images():
    names = kt_ep._MXFP4_RAW_WEIGHT_NAMES
    layer = SimpleNamespace(
        _v4_marlin_weights=object(),
        **{name: torch.empty(1) for name in names},
    )
    assert kt_ep._mxfp4_prepared_only_layer(layer)

    delattr(layer, names[0])
    assert not kt_ep._mxfp4_prepared_only_layer(layer)


def test_mxfp4_release_drops_raw_parameters_and_keeps_metadata():
    names = kt_ep._MXFP4_RAW_WEIGHT_NAMES
    layer = SimpleNamespace(
        _v4_marlin_weights=object(),
        _kt_mxfp4_raw_weights={
            name: torch.empty(2, 3, 4) for name in names
        },
        **{name: torch.empty(2, 3, 4) for name in names},
    )

    released = kt_ep._drop_mxfp4_raw_parameters(layer)

    # These test tensors are CPU-resident, so no device bytes are reported.
    assert released == 0
    assert not hasattr(layer, "_kt_mxfp4_raw_weights")
    assert all(not hasattr(layer, name) for name in names)
    assert kt_ep._mxfp4_prepared_only_layer(layer)
    assert kt_ep._mxfp4_raw_metadata(layer) == {
        name: ((2, 3, 4), torch.float32) for name in names
    }

    scratch = kt_ep._get_mxfp4_raw_scratch(layer, torch.device("cpu"), 5)
    assert scratch["w13_weight"].shape == (5, 3, 4)
    assert scratch["w13_weight"].dtype == torch.float32


def test_mxfp4_release_requires_a_prepared_image():
    names = kt_ep._MXFP4_RAW_WEIGHT_NAMES
    layer = SimpleNamespace(**{name: torch.empty(2, 3, 4) for name in names})

    assert kt_ep._drop_mxfp4_raw_parameters(layer) == 0
    assert all(hasattr(layer, name) for name in names)
    assert not hasattr(layer, kt_ep._MXFP4_RAW_RELEASED_ATTR)


def test_resident_host_ids_filter_canonical_rows():
    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=False,
        prepared_source=None,
        gpu_expert_ids=[1, 2, 3],
        canonical_ids={1},
    ) == [2, 3]

    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=True,
        prepared_source=None,
        gpu_expert_ids=[1, 2, 3],
        canonical_ids={1, 2, 3},
    ) == [1, 2, 3]

    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=True,
        prepared_source=object(),
        gpu_expert_ids=[1, 2, 3, 4],
        canonical_ids={1, 4},
    ) == [2, 3]

    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=True,
        prepared_source=object(),
        gpu_expert_ids=[1, 2, 3],
        canonical_ids={1, 2, 3},
    ) == []


def test_kt_layer_cuda_device_prefers_prepared_image_and_keeps_rank_device():
    prepared = SimpleNamespace(w13=SimpleNamespace(device=torch.device("cuda:1")))
    layer = SimpleNamespace(
        _v4_marlin_weights=prepared,
        w13_weight=SimpleNamespace(device=torch.device("cuda:0")),
    )
    assert kt_ep._kt_layer_cuda_device(layer) == torch.device("cuda:1")

    cpu_layer = SimpleNamespace(
        _v4_marlin_weights=None,
        w13_weight=torch.empty(0),
        parameters=lambda: iter([torch.empty(0)]),
    )
    assert kt_ep._kt_layer_cuda_device(
        cpu_layer, fallback=torch.device("cuda:1")
    ) == torch.device("cuda:1")


def test_mxfp4_scratch_capacity_is_a_positive_row_count():
    layer = SimpleNamespace()
    assert kt_ep._mxfp4_scratch_capacity(layer, 32) == 32
    assert kt_ep._mxfp4_scratch_capacity(layer, 3) == 3
    assert kt_ep._mxfp4_scratch_capacity(layer, 0) == 1
    assert kt_ep._mxfp4_scratch_capacity(layer, -8) == 1


def test_mxfp4_raw_scratch_is_model_scoped_and_keeps_legacy_bank():
    owner = SimpleNamespace(
        **{
            name: torch.empty(0, 2, 2)
            for name in kt_ep._MXFP4_RAW_WEIGHT_NAMES
        }
    )
    legacy = {"raw": "legacy-raw", "prepared": object()}
    owner._kt_mxfp4_repack_scratch = legacy

    scratch = kt_ep._get_mxfp4_raw_scratch(
        owner, torch.device("cpu"), 4
    )
    assert scratch["w13_weight"].shape == (4, 2, 2)
    assert owner._kt_mxfp4_raw_h2d_scratch["capacity"] == 4
    assert owner._kt_mxfp4_repack_scratch is legacy
    assert (
        kt_ep._get_mxfp4_raw_scratch(owner, torch.device("cpu"), 2)
        is scratch
    )


def test_mxfp4_prefill_slot_allows_prepared_only_raw_scratch_rows(monkeypatch):
    class _Event:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kt_ep.torch.cuda, "Event", _Event)

    def make_slot(raw_rows, physical_rows, prepared_only):
        raw = {
            name: torch.empty(raw_rows, 2, 2)
            for name in kt_ep._Mxfp4PrefillSlot.RAW_NAMES
        }
        return kt_ep._Mxfp4PrefillSlot(
            0,
            raw,
            SimpleNamespace(num_experts=physical_rows),
            num_experts=8,
            physical_rows=physical_rows,
            prepared_only=prepared_only,
        )

    slot = make_slot(raw_rows=2, physical_rows=5, prepared_only=True)
    assert slot.raw_rows == 2
    assert slot.physical_rows == 5

    with pytest.raises(ValueError, match="same row count"):
        make_slot(raw_rows=2, physical_rows=5, prepared_only=False)
    with pytest.raises(ValueError, match="selected row layout"):
        make_slot(raw_rows=6, physical_rows=5, prepared_only=True)
