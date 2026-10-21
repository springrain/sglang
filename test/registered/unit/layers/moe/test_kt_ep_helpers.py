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


def _make_mxfp4_refresh_fixture(canonical_rows: int):
    raw_names = kt_ep._Mxfp4PrefillSlot.RAW_NAMES

    def rows(count: int, seed: float) -> torch.Tensor:
        return torch.arange(count * 2, dtype=torch.float32).reshape(count, 2) + seed

    layer_raw = {name: torch.zeros(2, 2) for name in raw_names}
    slot_raw = {
        name: rows(3, 10.0 * (index + 1))
        for index, name in enumerate(raw_names)
    }
    canonical = {
        name: rows(canonical_rows, 100.0 * (index + 1))
        for index, name in enumerate(raw_names)
    }
    layer = SimpleNamespace(**layer_raw)
    layer._kt_mxfp4_raw_weights = canonical
    layer._v4_marlin_weights = SimpleNamespace(
        w13=torch.zeros(2, 2),
        w13_scale=torch.zeros(2, 2),
        w2=torch.zeros(2, 2),
        w2_scale=torch.zeros(2, 2),
    )
    slot = SimpleNamespace(**slot_raw)
    slot.marlin_prepared = SimpleNamespace(
        w13=rows(3, 1.0),
        w13_scale=rows(3, 11.0),
        w2=rows(3, 21.0),
        w2_scale=rows(3, 31.0),
    )
    slot.num_experts = 4
    slot.logical_to_row = [-1, -1, -1, 2]
    manager = kt_ep._Mxfp4LayerwisePrefillManager.__new__(
        kt_ep._Mxfp4LayerwisePrefillManager
    )
    return manager, layer, slot, canonical, layer_raw, slot_raw


def test_boundary_refresh_leaves_the_full_canonical_shadow_unchanged():
    manager, layer, slot, canonical, layer_raw, slot_raw = (
        _make_mxfp4_refresh_fixture(canonical_rows=4)
    )
    before = {name: tensor.clone() for name, tensor in canonical.items()}

    manager._refresh_original_layer_rows(
        method=None,
        layer=layer,
        slot=slot,
        row_owner=[2, 1],
        pairs=[(2, 3)],
    )

    for name in kt_ep._Mxfp4PrefillSlot.RAW_NAMES:
        assert torch.equal(canonical[name], before[name])
        assert torch.equal(layer_raw[name][0], slot_raw[name][2])


def test_boundary_refresh_updates_a_compact_canonical_row():
    manager, layer, slot, canonical, layer_raw, _ = _make_mxfp4_refresh_fixture(
        canonical_rows=2
    )
    untouched = {name: tensor[1].clone() for name, tensor in canonical.items()}

    manager._refresh_original_layer_rows(
        method=None,
        layer=layer,
        slot=slot,
        row_owner=[2, 1],
        pairs=[(2, 3)],
    )

    for name in kt_ep._Mxfp4PrefillSlot.RAW_NAMES:
        assert torch.equal(canonical[name][0], layer_raw[name][0])
        assert torch.equal(canonical[name][1], untouched[name])


def test_boundary_promotions_commit_a_live_bank_and_queue_a_reused_bank(
    monkeypatch,
):
    class _History:
        @staticmethod
        def scores(layer_idx: int) -> torch.Tensor:
            del layer_idx
            return torch.tensor(
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                dtype=torch.float32,
            )

    manager = kt_ep._Mxfp4LayerwisePrefillManager.__new__(
        kt_ep._Mxfp4LayerwisePrefillManager
    )
    manager.planner = SimpleNamespace(history=_History())
    manager.shared_ids = []
    manager.telemetry = None
    manager.device = torch.device("cpu")
    manager._staging_hit_counts = {
        10: {2: 3, 3: 1},
        11: {6: 4},
    }
    manager._layer_staging_state = {
        10: (0, (0, 1), (2, 3)),
        11: (1, (4, 5), (6, 7)),
    }
    manager._pending_boundary_promotions = {}

    method10 = SimpleNamespace(
        _resident_row_owner=torch.tensor([0, 1], dtype=torch.int64)
    )
    method11 = SimpleNamespace(
        _resident_row_owner=torch.tensor([4, 5], dtype=torch.int64)
    )
    manager.signature = ("unit-test", "boundary-promotion")
    monkeypatch.setitem(
        kt_ep._MXFP4_PREFILL_LAYER_REGISTRY,
        manager.signature,
        {10: (method10, object()), 11: (method11, object())},
    )

    pool = kt_ep.StagingRowPool(
        8, resident_capacity=2, staging_capacity=2
    )
    before = pool.configure(10, [0, 1], [2, 3])
    slot10 = SimpleNamespace(
        index=0,
        loaded_layer_idx=10,
        row_pool=pool,
        consumed_event=SimpleNamespace(record=lambda stream: None),
        resident_ids=[0, 1],
        staged_ids=[2, 3],
        effective_ids=[0, 1, 2, 3],
        partial=True,
        effective_mask_cuda=object(),
        num_experts=8,
    )
    slot11 = SimpleNamespace(index=1, loaded_layer_idx=99)
    manager.slots = (slot10, slot11)
    manager.row_pools = (pool, object())

    refresh_calls = []
    publish_calls = []
    manager._refresh_original_layer_rows = lambda *args: refresh_calls.append(args)
    manager._publish_method_row_owner = (
        lambda method, row_owner: publish_calls.append((method, list(row_owner)))
    )
    monkeypatch.setattr(
        kt_ep.torch.cuda,
        "current_stream",
        lambda *args, **kwargs: object(),
    )

    manager._apply_boundary_promotions()

    assert len(refresh_calls) == 1
    assert refresh_calls[0][-1] == [(0, 2), (1, 3)]
    assert publish_calls == [(method10, [2, 3])]
    assert manager._pending_boundary_promotions == {11: [(4, 6)]}
    assert slot10.resident_ids == [2, 3]
    assert slot10.staged_ids == []
    assert manager._layer_staging_state[10] == (0, (2, 3), ())
    assert pool.mapping(10)[2] == before[2]
    assert pool.mapping(10)[3] == before[3]
    assert pool.mapping(10)[0] == -1
    assert pool.row_roles(10)[before[2]] == "resident"
    assert pool.row_roles(10)[before[3]] == "resident"
    assert manager._staging_hit_counts == {}


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


def test_mxfp4_prepared_only_requires_a_positive_staging_window(monkeypatch):
    monkeypatch.delenv(kt_ep._MXFP4_PREPARED_ONLY_ENV, raising=False)

    assert not kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_layer_h2d_slots=None)
    )
    assert not kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_layer_h2d_slots=0)
    )
    assert kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_layer_h2d_slots=32)
    )

    monkeypatch.setenv(kt_ep._MXFP4_PREPARED_ONLY_ENV, "0")
    assert not kt_ep._mxfp4_prepared_only_requested(
        SimpleNamespace(kt_layer_h2d_slots=32)
    )


def test_mxfp4_prepared_only_layer_requires_prepared_and_cpu_raw_images():
    names = kt_ep._MXFP4_RAW_WEIGHT_NAMES
    layer = SimpleNamespace(
        _v4_marlin_weights=object(),
        **{name: torch.empty(1) for name in names},
    )
    assert kt_ep._mxfp4_prepared_only_layer(layer)

    delattr(layer, names[0])
    assert not kt_ep._mxfp4_prepared_only_layer(layer)


def test_resident_host_ids_prefer_pending_promotions_and_canonical_rows():
    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=False,
        prepared_source=None,
        gpu_expert_ids=[1, 2, 3],
        pending_pairs=[],
        canonical_ids={1},
    ) == [2, 3]

    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=True,
        prepared_source=None,
        gpu_expert_ids=[1, 2, 3],
        pending_pairs=[],
        canonical_ids={1, 2, 3},
    ) == [1, 2, 3]

    assert kt_ep._mxfp4_resident_host_ids(
        prepared_only=True,
        prepared_source=object(),
        gpu_expert_ids=[1, 2, 3],
        pending_pairs=[(1, 7)],
        canonical_ids={1, 2, 3},
    ) == [7]


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


def test_mxfp4_scratch_capacity_is_bounded_by_the_staging_window():
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(
            kt_config=SimpleNamespace(kt_layer_h2d_slots=8)
        )
    )
    assert kt_ep._mxfp4_scratch_capacity(layer, 32) == 8
    assert kt_ep._mxfp4_scratch_capacity(layer, 3) == 3
    assert kt_ep._mxfp4_scratch_capacity(SimpleNamespace(), 32) == 32


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
