"""Unit coverage for the SGLang-side KT Q8/P1 routing helpers."""

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_prefill_num_tokens,
)
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


def test_num_gpu_layers_is_the_real_kt_initialization_boundary(monkeypatch):
    """Pinned prefix layers bypass KT and do not consume the cache budget."""

    num_layers = 6
    num_experts = 8
    num_gpu_layers = 2
    capacity = 3
    hf_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_local_experts=num_experts,
        first_k_dense_replace=0,
        moe_layer_freq=1,
    )
    server_args = SimpleNamespace(
        kt_num_gpu_layers=num_gpu_layers,
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=capacity,
        kt_expert_placement_strategy="decayed-lfu",
    )
    moe_config = SimpleNamespace(
        kt_weight_path="/weights",
        kt_num_gpu_layers=num_gpu_layers,
        kt_cpuinfer=4,
        kt_threadpool_count=1,
        kt_numa_nodes=None,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=None,
        kt_gpu_prefill_token_threshold=None,
        kt_enable_dynamic_expert_update=False,
        kt_expert_placement_strategy="decayed-lfu",
        kt_prefill_stream_top_n=capacity,
        kt_expert_lora_path=None,
    )

    captured_budget = None
    generate_uniform_masks = kt_ep.generate_uniform_masks

    def capture_uniform_budget(
        num_layers,
        num_experts,
        num_gpu_experts,
        first_k_dense_replace,
        moe_layer_freq,
    ):
        nonlocal captured_budget
        captured_budget = num_gpu_experts
        return generate_uniform_masks(
            num_layers,
            num_experts,
            num_gpu_experts,
            first_k_dense_replace,
            moe_layer_freq,
        )

    monkeypatch.setattr(kt_ep, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(
        kt_ep,
        "model_config_of",
        lambda _: SimpleNamespace(hf_config=hf_config),
    )
    monkeypatch.setattr(kt_ep, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kt_ep.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(kt_ep, "generate_uniform_masks", capture_uniform_budget)
    monkeypatch.setattr(
        kt_ep,
        "get_exec",
        lambda: SimpleNamespace(moe=moe_config),
    )
    monkeypatch.setattr(
        kt_ep,
        "get_schedule",
        lambda: SimpleNamespace(chunked_prefill_size=4096),
    )
    monkeypatch.setattr(
        "sglang.srt.layers.moe.utils.is_kt_ep_wrapper_disabled",
        lambda: False,
    )

    # L-1 is still native/all-GPU, so it must return before mask/cache init.
    assert (
        kt_ep.create_kt_config_from_server_args(
            server_args, layer_idx=num_gpu_layers - 1
        )
        is None
    )
    assert captured_budget is None

    # L is the first cache-managed layer and receives exactly C physical slots.
    config = kt_ep.create_kt_config_from_server_args(
        server_args, layer_idx=num_gpu_layers
    )
    assert config is not None
    assert int(config.gpu_experts_mask.sum()) == capacity

    masks = kt_ep._KT_GPU_EXPERTS_MASKS
    assert masks is not None
    assert masks[num_gpu_layers - 1].all()
    assert [int(row.sum()) for row in masks[num_gpu_layers:]] == [capacity] * (
        num_layers - num_gpu_layers
    )
    assert captured_budget == capacity * (num_layers - num_gpu_layers)


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


def test_transient_stream_ownership_splits_main_cpu_and_fallback_exactly_once():
    topk_ids = torch.tensor([[0, 4, 2], [5, 1, 4]], dtype=torch.int64)
    candidates = (4, 5)

    cpu_only = kt_ep.filter_expert_assignments(
        topk_ids, candidates, keep_selected=False
    )
    fallback = kt_ep.filter_expert_assignments(
        topk_ids, candidates, keep_selected=True
    )

    assert torch.equal(cpu_only, torch.tensor([[0, -1, 2], [-1, 1, -1]]))
    assert torch.equal(fallback, torch.tensor([[-1, 4, -1], [5, -1, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[0, 4, 2], [5, 1, 4]]))
    assert not ((cpu_only >= 0) & (fallback >= 0)).any()


def test_stream_execution_partition_rejects_duplicate_or_missing_ownership():
    kt_ep.validate_stream_execution_partition((4, 5, 6), (4, 6), (5,))

    with pytest.raises(RuntimeError, match="both GPU and CPU owned"):
        kt_ep.validate_stream_execution_partition((4, 5), (4,), (4, 5))
    with pytest.raises(RuntimeError, match="terminate exactly once"):
        kt_ep.validate_stream_execution_partition((4, 5), (4,), ())


def test_stream_plan_capacity_spills_tail_into_the_main_cpu_task():
    plan = kt_ep.KTExpertStreamPlan(
        layer_idx=8,
        epoch=1,
        stream_hotset=(2, 3, 4, 5),
        candidate_expert_ids=(2, 3, 4, 5),
        streamed_expert_ids=(2, 3, 4, 5),
        cpu_owned_candidate_ids=(),
        streamed_wave2_supported=True,
    )

    bounded = kt_ep.bound_stream_plan_to_prebegin_capacity(
        plan,
        2,
        fallback_reason="two staging slots",
    )

    assert bounded.stream_hotset == plan.stream_hotset
    assert bounded.candidate_expert_ids == plan.candidate_expert_ids
    assert bounded.streamed_expert_ids == (2, 3)
    assert bounded.cpu_owned_candidate_ids == (4, 5)
    assert bounded.fallback_reason == "two staging slots"


def test_stream_prebegin_is_reused_without_a_second_writer_submission():
    plan = kt_ep.KTExpertStreamPlan(
        layer_idx=8,
        epoch=1,
        stream_hotset=(2, 3, 4, 5),
        candidate_expert_ids=(2, 3, 4, 5),
        streamed_expert_ids=(2, 3, 4, 5),
        cpu_owned_candidate_ids=(),
        streamed_wave2_supported=True,
    )

    class FakeTicket:
        def __init__(self, expert_id):
            self.expert_id = expert_id
            self.released = False

        def wait_and_launch(self, **kwargs):
            return SimpleNamespace(output=torch.ones_like(kwargs["hidden_states"]))

        def release(self):
            self.released = True

    class FakeTransport:
        supported = True
        prebegin_capacity = 2

        def __init__(self):
            self.begin_calls = []
            self.tickets = ()

        def begin_candidates(self, _method, candidate_ids):
            ids = tuple(candidate_ids)
            self.begin_calls.append(ids)
            self.tickets = tuple(FakeTicket(expert_id) for expert_id in ids)
            return self.tickets

        def abort_candidates(self, _tickets=None):
            raise AssertionError("healthy prebegun execution must not abort")

    state = SimpleNamespace(
        suppressed_stream_candidates=0,
        last_stream_plan=plan,
        last_window_counts=(0, 0, 8, 7, 6, 5),
        plan_persistent_replacements=lambda *_args, **_kwargs: (),
    )
    transport = FakeTransport()
    method = SimpleNamespace(
        _expert_stream_transport=transport,
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=state,
        tp_rank=0,
        kt_config=SimpleNamespace(layer_idx=8),
    )

    prebegun = kt_ep.KTEPWrapperMethod._prebegin_stream_candidates(method, plan)
    assert prebegun.plan.streamed_expert_ids == (2, 3)
    assert prebegun.plan.cpu_owned_candidate_ids == (4, 5)
    assert transport.begin_calls == [(2, 3)]
    assert state.suppressed_stream_candidates == 2

    hidden_states = torch.zeros((2, 3), dtype=torch.float32)
    execution = kt_ep.KTEPWrapperMethod._execute_stream_candidates(
        method,
        layer=SimpleNamespace(moe_runner_config=SimpleNamespace(swiglu_limit=None)),
        plan=prebegun.plan,
        hidden_states=hidden_states,
        topk_ids=torch.tensor([[2, 3], [4, 5]]),
        topk_weights=torch.ones((2, 2)),
        prebegun_tickets=prebegun.tickets,
    )

    assert transport.begin_calls == [(2, 3)]
    assert execution.successful_expert_ids == (2, 3)
    assert torch.equal(execution.output, torch.full_like(hidden_states, 2.0))
    assert all(ticket.released for ticket in transport.tickets)


def test_stream_prebegin_failure_keeps_all_candidates_in_main_cpu_task():
    plan = kt_ep.KTExpertStreamPlan(
        layer_idx=8,
        epoch=1,
        stream_hotset=(2, 3),
        candidate_expert_ids=(2, 3),
        streamed_expert_ids=(2, 3),
        cpu_owned_candidate_ids=(),
        streamed_wave2_supported=True,
    )

    class FailingTransport:
        supported = True
        prebegin_capacity = 2

        def begin_candidates(self, _method, _candidate_ids):
            raise RuntimeError("synthetic writer submit failure")

    state = SimpleNamespace(
        suppressed_stream_candidates=0,
        last_stream_plan=plan,
    )
    method = SimpleNamespace(
        _expert_stream_transport=FailingTransport(),
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=state,
        tp_rank=0,
        kt_config=SimpleNamespace(layer_idx=8),
    )

    prebegun = kt_ep.KTEPWrapperMethod._prebegin_stream_candidates(method, plan)

    assert prebegun.tickets == ()
    assert prebegun.plan.streamed_expert_ids == ()
    assert prebegun.plan.cpu_owned_candidate_ids == (2, 3)
    assert state.last_stream_plan is prebegun.plan
    assert state.suppressed_stream_candidates == 2


def test_stream_transport_without_capacity_fails_closed_to_main_cpu():
    plan = kt_ep.KTExpertStreamPlan(
        layer_idx=8,
        epoch=1,
        stream_hotset=(2, 3),
        candidate_expert_ids=(2, 3),
        streamed_expert_ids=(2, 3),
        cpu_owned_candidate_ids=(),
        streamed_wave2_supported=True,
    )

    class CapacitylessTransport:
        supported = True

        def begin_candidates(self, *_args, **_kwargs):
            raise AssertionError("capacity-less transport must not submit writers")

    state = SimpleNamespace(
        suppressed_stream_candidates=0,
        last_stream_plan=plan,
    )
    method = SimpleNamespace(
        _expert_stream_transport=CapacitylessTransport(),
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=state,
    )

    bounded = kt_ep.KTEPWrapperMethod._bound_stream_plan_for_prebegin(
        method, plan
    )

    assert bounded.streamed_expert_ids == ()
    assert bounded.cpu_owned_candidate_ids == (2, 3)
    assert "does not declare prebegin capacity" in bounded.fallback_reason
    assert state.suppressed_stream_candidates == 2


def test_stream_prebegin_does_not_hide_a_fail_stopped_transport():
    plan = kt_ep.KTExpertStreamPlan(
        layer_idx=8,
        epoch=1,
        stream_hotset=(2,),
        candidate_expert_ids=(2,),
        streamed_expert_ids=(2,),
        cpu_owned_candidate_ids=(),
        streamed_wave2_supported=True,
    )

    class FatalTransport:
        supported = True
        prebegin_capacity = 2
        fail_stopped = True
        irreversible_pending = False

        def begin_candidates(self, *_args, **_kwargs):
            raise RuntimeError("unsafe resident state")

    method = SimpleNamespace(
        _expert_stream_transport=FatalTransport(),
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=SimpleNamespace(
            suppressed_stream_candidates=0,
            last_stream_plan=plan,
        ),
        tp_rank=0,
        kt_config=SimpleNamespace(layer_idx=8),
    )

    with pytest.raises(RuntimeError, match="unsafe resident state"):
        kt_ep.KTEPWrapperMethod._prebegin_stream_candidates(method, plan)


def test_apply_prebegins_writers_before_main_cpu_submit_and_reuses_tickets():
    source = inspect.getsource(kt_ep.KTEPWrapperMethod.apply)

    dispatch_commit = source.index("transport.confirm_dispatch_ready(")
    prebegin = source.index("self._prebegin_stream_candidates(stream_plan)")
    cpu_submit = source.index("self._submit_with_staged_input(", prebegin)
    execute = source.index("self._execute_stream_candidates(", cpu_submit)
    ticket_reuse = source.index("prebegun_tickets=prebegun_tickets", execute)

    assert dispatch_commit < prebegin < cpu_submit < execute < ticket_reuse


def test_apply_routes_merge_errors_through_layer_output_fail_stop():
    source = inspect.getsource(kt_ep.KTEPWrapperMethod.apply)

    stream_merge = source.index("stream_merge_error: Optional[Exception]")
    cpu_drain = source.index("# Step 7: Drain the main CPU task", stream_merge)
    layer_status = source.index('"LAYER_OUTPUT_READY"', cpu_drain)
    fail_stop = source.index("transport.mark_fatal(", layer_status)
    publish_guard = source.index("if layer_output_ready:", fail_stop)
    final_raise = source.index(
        '"KT layer output failed before resident publication"', publish_guard
    )

    assert stream_merge < cpu_drain < layer_status < fail_stop < publish_guard
    assert publish_guard < final_raise


def test_stream_dispatcher_returns_failed_tail_for_one_late_cpu_task():
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )
    plan = state.record_prefill_window(
        (0, 0, 7, 6), streamed_wave2_supported=True
    )

    class FakeTicket:
        def __init__(self, expert_id, fail=False):
            self.expert_id = expert_id
            self.fail = fail
            self.released = False

        def wait_and_launch(self, **kwargs):
            if self.fail:
                raise RuntimeError("synthetic H2D failure")
            return SimpleNamespace(
                output=torch.full_like(kwargs["hidden_states"], 2.0)
            )

        def release(self):
            self.released = True

    tickets = (FakeTicket(2), FakeTicket(3, fail=True))

    class FakeTransport:
        supported = True

        def __init__(self):
            self.aborted = None

        def begin_candidates(self, _method, candidate_ids):
            assert tuple(candidate_ids) == (2, 3)
            return tickets

        def abort_candidates(self, remaining=None):
            self.aborted = None if remaining is None else tuple(remaining)

    transport = FakeTransport()
    method = SimpleNamespace(
        _expert_stream_transport=transport,
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=state,
        tp_rank=0,
        kt_config=SimpleNamespace(layer_idx=8),
    )
    hidden_states = torch.zeros((2, 3), dtype=torch.float32)
    result = kt_ep.KTEPWrapperMethod._execute_stream_candidates(
        method,
        layer=SimpleNamespace(moe_runner_config=SimpleNamespace(swiglu_limit=None)),
        plan=plan,
        hidden_states=hidden_states,
        topk_ids=torch.tensor([[2, 3], [2, 1]]),
        topk_weights=torch.ones((2, 2)),
    )

    assert torch.equal(result.output, torch.full_like(hidden_states, 2.0))
    assert result.successful_expert_ids == (2,)
    assert result.failed_expert_ids == (3,)
    assert result.installed_replacements == ()
    assert tickets[0].released
    assert transport.aborted == (tickets[1],)


def test_stream_dispatcher_rejects_an_active_persistent_victim():
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )
    plan = state.record_prefill_window(
        (1, 0, 7, 0), streamed_wave2_supported=True
    )
    illegal = kt_ep.Replacement(
        candidate_expert_id=2,
        victim_expert_id=0,
        slot_id=0,
    )
    state.plan_persistent_replacements = lambda *_args, **_kwargs: (illegal,)

    ticket = SimpleNamespace(expert_id=2)

    class FakeTransport:
        supported = True

        def __init__(self):
            self.aborted = None

        def begin_candidates(self, _method, candidate_ids):
            assert tuple(candidate_ids) == (2,)
            return (ticket,)

        def abort_candidates(self, tickets):
            self.aborted = tuple(tickets)

    transport = FakeTransport()
    method = SimpleNamespace(
        _expert_stream_transport=transport,
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=state,
        tp_rank=0,
        kt_config=SimpleNamespace(layer_idx=8),
    )

    with pytest.raises(RuntimeError, match="expert active in the current window"):
        kt_ep.KTEPWrapperMethod._execute_stream_candidates(
            method,
            layer=SimpleNamespace(
                moe_runner_config=SimpleNamespace(swiglu_limit=None)
            ),
            plan=plan,
            hidden_states=torch.zeros((2, 3)),
            topk_ids=torch.tensor([[2, 0], [2, 2]]),
            topk_weights=torch.ones((2, 2)),
        )

    assert transport.aborted == (ticket,)


def test_stream_dispatcher_installs_then_confirms_mapping_at_publish(monkeypatch):
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )
    state.record_prefill_window(
        (0, 0, 7, 1), streamed_wave2_supported=True
    )
    plan = state.record_prefill_window(
        (0, 0, 8, 0), streamed_wave2_supported=True
    )

    class FakeStream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event)

    main_stream = FakeStream()
    monkeypatch.setattr(kt_ep.torch.cuda, "current_stream", lambda *_: main_stream)
    monkeypatch.setattr(kt_ep.dist, "is_initialized", lambda: False)

    install_event = object()
    victim_safe_event = object()

    class FakeTicket:
        expert_id = 2

        def __init__(self):
            self.install_kwargs = None
            self.confirmed = None

        def wait_and_launch(self, **kwargs):
            return SimpleNamespace(output=torch.ones_like(kwargs["hidden_states"]))

        def install(self, resident_prepared, resident_slot, **kwargs):
            self.install_kwargs = (resident_prepared, resident_slot, kwargs)
            return SimpleNamespace(
                ticket=self,
                install_event=install_event,
                mapping_publish_allowed=True,
            )

        def confirm_mapping_published(self, commit, **kwargs):
            self.confirmed = (commit, kwargs)

    ticket = FakeTicket()

    class FakeTransport:
        supported = True

        def begin_candidates(self, _method, candidate_ids):
            assert tuple(candidate_ids) == (2,)
            return (ticket,)

        def abort_candidates(self, _tickets=None):
            raise AssertionError("healthy promotion must not abort")

        def mark_fatal(self, *_args, **_kwargs):
            raise AssertionError("healthy publication must not fail-stop")

    raw = {name: object() for name in kt_ep._Mxfp4PrefillSlot.RAW_NAMES}
    method = SimpleNamespace(
        _expert_stream_transport=FakeTransport(),
        _streamed_wave2_fallback_reason="unsupported",
        expert_cache_state=state,
        tp_rank=0,
        kt_config=SimpleNamespace(layer_idx=8),
        gpu_experts_mask=torch.tensor([True, True, False, False]),
        logical_to_gpu_index=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        gpu_index_to_logical=torch.tensor([0, 1], dtype=torch.int32),
        gpu_experts_mask_cuda=torch.tensor([True, True, False, False]),
        logical_to_gpu_index_cuda=torch.tensor(
            [0, 1, -1, -1], dtype=torch.int32
        ),
        wrapper=SimpleNamespace(
            gpu_experts_mask=torch.tensor([True, True, False, False])
        ),
    )
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(swiglu_limit=None),
        _v4_marlin_weights=object(),
        _kt_mxfp4_raw_weights=raw,
    )
    hidden_states = torch.zeros((2, 3), dtype=torch.float32)

    execution = kt_ep.KTEPWrapperMethod._execute_stream_candidates(
        method,
        layer=layer,
        plan=plan,
        hidden_states=hidden_states,
        topk_ids=torch.tensor([[2, 3], [2, 1]]),
        topk_weights=torch.ones((2, 2)),
        victim_safe_event=victim_safe_event,
    )

    assert execution.successful_expert_ids == (2,)
    assert len(execution.installed_replacements) == 1
    assert len(execution.install_commits) == 1
    _, installed_slot, install_kwargs = ticket.install_kwargs
    assert installed_slot == execution.installed_replacements[0].slot_id
    assert install_kwargs["resident_raw"] is raw
    assert install_kwargs["victim_safe_event"] is victim_safe_event
    assert install_kwargs["install_stream"] is main_stream

    kt_ep.KTEPWrapperMethod._publish_stream_replacements(
        method,
        execution.installed_replacements,
        execution.install_commits,
    )

    assert main_stream.waited == [install_event]
    assert ticket.confirmed == (
        execution.install_commits[0],
        {"local_success": True, "local_error": None},
    )
    assert state.resident_expert_ids[installed_slot] == 2


def test_select_top_experts_ignores_invalid_ids_and_is_stable():
    topk_ids = torch.tensor([[2, 2, -1], [4, 2, 1], [4, 9, 1]])
    selected = kt_ep.select_top_experts_from_batch(
        topk_ids=topk_ids, num_experts=5, num_gpu_experts=3
    )
    assert torch.equal(selected, torch.tensor([1, 2, 4]))


def test_count_expert_route_assignments_ignores_invalid_ids():
    out = torch.full((5,), -1, dtype=torch.int64)
    counts = kt_ep.count_expert_route_assignments(
        torch.tensor([[2, 2, -1], [4, 2, 1], [4, 9, 1]]),
        num_experts=5,
        out=out,
    )
    assert counts.data_ptr() == out.data_ptr()
    assert torch.equal(counts, torch.tensor([0, 2, 3, 0, 2]))


def test_decayed_lfu_cache_state_keeps_unsupported_candidates_on_cpu():
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=8,
        capacity=4,
        stream_top_n=4,
        resident_expert_ids=(0, 1, 2, 3),
        reference_assignments=64,
    )

    plan = state.record_prefill_window(
        (10, 1, 0, 0, 9, 8, 7, 6),
        streamed_wave2_supported=False,
        fallback_reason="test backend has no wave-2 runner",
    )

    # The hotset is exactly Top-4. Expert 0 is already resident, so only the
    # other three are candidates; expert 7 must not backfill the resident hit.
    assert plan.stream_hotset == (0, 4, 5, 6)
    assert plan.candidate_expert_ids == (4, 5, 6)
    assert plan.streamed_expert_ids == ()
    assert plan.cpu_owned_candidate_ids == (4, 5, 6)
    assert state.suppressed_stream_candidates == 3
    assert state.lifetime_assignments == (10, 1, 0, 0, 9, 8, 7, 6)
    assert state.last_access_epoch == (1, 1, -1, -1, 1, 1, 1, 1)


def test_decayed_lfu_cache_state_claims_candidates_before_cpu_submit():
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )

    plan = state.record_prefill_window(
        (1, 0, 4, 3),
        streamed_wave2_supported=True,
    )

    assert plan.stream_hotset == (2, 3)
    assert plan.streamed_expert_ids == (2, 3)
    assert plan.cpu_owned_candidate_ids == ()
    assert state.epoch == 1
    assert state.suppressed_stream_candidates == 0


def test_decayed_lfu_promotion_is_hysteretic_and_incremental():
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=6,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )

    # The first window can stream current misses, but minimum residency keeps
    # both persistent slots stable.
    state.record_prefill_window(
        (0, 0, 7, 1, 0, 0), streamed_wave2_supported=True
    )
    assert state.plan_persistent_replacements(
        (2, 3), state.last_window_counts
    ) == ()

    # On the second window only the much hotter candidate clears 25% hysteresis.
    state.record_prefill_window(
        (0, 0, 8, 0, 0, 0), streamed_wave2_supported=True
    )
    replacements = state.plan_persistent_replacements(
        (2,), state.last_window_counts
    )
    assert len(replacements) == 1
    assert replacements[0].candidate_expert_id == 2
    assert replacements[0].slot_id in (0, 1)

    untouched_slot = 1 - replacements[0].slot_id
    untouched_expert = state.resident_expert_ids[untouched_slot]
    state.commit_persistent_replacements(replacements)
    assert state.resident_expert_ids[replacements[0].slot_id] == 2
    assert state.resident_expert_ids[untouched_slot] == untouched_expert
    assert state.resident_since_epoch[replacements[0].slot_id] == state.epoch


def test_stream_mapping_publish_is_in_place_and_commits_policy(monkeypatch):
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )
    state.record_prefill_window(
        (0, 0, 7, 1), streamed_wave2_supported=True
    )
    state.record_prefill_window(
        (0, 0, 8, 0), streamed_wave2_supported=True
    )
    replacement = state.plan_persistent_replacements(
        (2,), state.last_window_counts
    )[0]

    wrapper_mask = torch.tensor([True, True, False, False])
    method = SimpleNamespace(
        expert_cache_state=state,
        gpu_experts_mask=torch.tensor([True, True, False, False]),
        logical_to_gpu_index=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        gpu_index_to_logical=torch.tensor([0, 1], dtype=torch.int32),
        gpu_experts_mask_cuda=torch.tensor([True, True, False, False]),
        logical_to_gpu_index_cuda=torch.tensor(
            [0, 1, -1, -1], dtype=torch.int32
        ),
        tp_rank=0,
        wrapper=SimpleNamespace(gpu_experts_mask=wrapper_mask),
    )
    pointers = (
        method.gpu_experts_mask.data_ptr(),
        method.logical_to_gpu_index.data_ptr(),
        method.gpu_index_to_logical.data_ptr(),
        method.gpu_experts_mask_cuda.data_ptr(),
        method.logical_to_gpu_index_cuda.data_ptr(),
        wrapper_mask.data_ptr(),
    )
    monkeypatch.setattr(kt_ep.dist, "is_initialized", lambda: False)

    kt_ep.KTEPWrapperMethod._publish_stream_replacements(
        method, (replacement,)
    )

    assert state.resident_expert_ids[replacement.slot_id] == 2
    assert method.gpu_index_to_logical[replacement.slot_id] == 2
    assert method.gpu_experts_mask[2]
    assert not method.gpu_experts_mask[replacement.victim_expert_id]
    assert torch.equal(method.gpu_experts_mask, wrapper_mask)
    assert pointers == (
        method.gpu_experts_mask.data_ptr(),
        method.logical_to_gpu_index.data_ptr(),
        method.gpu_index_to_logical.data_ptr(),
        method.gpu_experts_mask_cuda.data_ptr(),
        method.logical_to_gpu_index_cuda.data_ptr(),
        wrapper_mask.data_ptr(),
    )


def test_stream_mapping_publish_failure_acks_every_commit(monkeypatch):
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=2,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )
    replacements = (
        kt_ep.Replacement(candidate_expert_id=2, victim_expert_id=0, slot_id=0),
        kt_ep.Replacement(candidate_expert_id=3, victim_expert_id=1, slot_id=1),
    )

    class FailingMask:
        def copy_(self, _value):
            raise RuntimeError("synthetic pinned-mask publish failure")

    acknowledgements = []

    class FakeTicket:
        def confirm_mapping_published(self, commit, **kwargs):
            acknowledgements.append((commit, kwargs))

    commits = tuple(
        SimpleNamespace(
            ticket=FakeTicket(),
            install_event=object(),
            mapping_publish_allowed=True,
        )
        for _ in replacements
    )

    class FakeStream:
        def wait_event(self, _event):
            pass

    monkeypatch.setattr(kt_ep.torch.cuda, "current_stream", lambda *_: FakeStream())
    method = SimpleNamespace(
        expert_cache_state=state,
        gpu_experts_mask=torch.tensor([True, True, False, False]),
        logical_to_gpu_index=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        gpu_index_to_logical=torch.tensor([0, 1], dtype=torch.int32),
        gpu_experts_mask_cuda=torch.tensor([True, True, False, False]),
        logical_to_gpu_index_cuda=torch.tensor(
            [0, 1, -1, -1], dtype=torch.int32
        ),
        tp_rank=0,
        wrapper=SimpleNamespace(gpu_experts_mask=FailingMask()),
    )

    with pytest.raises(RuntimeError, match="runtime is fail-stopped"):
        kt_ep.KTEPWrapperMethod._publish_stream_replacements(
            method, replacements, commits
        )

    assert len(acknowledgements) == 2
    assert all(not kwargs["local_success"] for _, kwargs in acknowledgements)
    assert all(
        isinstance(kwargs["local_error"], RuntimeError)
        for _, kwargs in acknowledgements
    )
    assert state.resident_expert_ids == (0, 1)


def test_decayed_lfu_zero_stream_width_still_records_history():
    state = kt_ep.KTExpertCacheState.create(
        layer_idx=8,
        num_experts=4,
        capacity=2,
        stream_top_n=0,
        resident_expert_ids=(0, 1),
        reference_assignments=8,
    )
    plan = state.record_prefill_window(
        (0, 0, 4, 2),
        streamed_wave2_supported=False,
        fallback_reason="disabled by stream_top_n=0",
    )

    assert plan.stream_hotset == ()
    assert plan.candidate_expert_ids == ()
    assert state.observed_windows == 1
    assert state.lifetime_assignments == (0, 0, 4, 2)
    assert state.reuse_signal[2] > state.reuse_signal[3] > 0


def test_decayed_lfu_uses_runtime_prefill_row_count(monkeypatch):
    from sglang.srt.layers import dp_attention

    method = object.__new__(kt_ep.KTEPWrapperMethod)
    method._decayed_lfu_enabled = True
    monkeypatch.setattr(dp_attention, "get_prefill_num_tokens", lambda: 3)
    assert kt_ep.KTEPWrapperMethod._prefill_cache_token_count(method, 8) == 3
    assert kt_ep.KTEPWrapperMethod._should_record_prefill_cache_window(method, 8)

    monkeypatch.setattr(dp_attention, "get_prefill_num_tokens", lambda: 0)
    assert not kt_ep.KTEPWrapperMethod._should_record_prefill_cache_window(method, 8)


def test_mixed_histogram_counts_only_prefill_prefix_rows():
    method = SimpleNamespace(
        expert_cache_state=SimpleNamespace(num_experts=5),
        _cache_policy_stream=None,
        _cache_policy_counts=None,
    )
    counts = kt_ep.KTEPWrapperMethod._submit_prefill_cache_histogram(
        method,
        torch.tensor([[2, 1], [4, 2], [3, 3], [0, 0]]),
        prefill_num_tokens=2,
    )
    assert torch.equal(counts, torch.tensor([0, 1, 2, 0, 1]))


def test_mixed_batch_prefill_prefix_length_excludes_decode_tail():
    running_indices = torch.tensor([10, 11, 12], dtype=torch.int64)
    assert compute_prefill_num_tokens(ForwardMode.MIXED, 10, running_indices) == 7
    assert compute_prefill_num_tokens(ForwardMode.MIXED, 10, None) == 0
    assert compute_prefill_num_tokens(ForwardMode.EXTEND, 10, None) == 10
    assert compute_prefill_num_tokens(ForwardMode.DECODE, None, None) == 0


def test_non_dp_forward_stamps_prefill_runtime_flags(monkeypatch):
    from sglang.srt.layers import communicator, dp_attention

    forward_batch = object.__new__(ForwardBatch)
    forward_batch.is_extend_in_batch = True
    forward_batch.prefill_num_tokens = 5
    forward_batch.input_embeds = None
    forward_batch.input_ids = torch.zeros(5, dtype=torch.int64)
    runner = SimpleNamespace(attn_tp_sequence_sharded=lambda _: False)
    context = SimpleNamespace(use_input_scattered=lambda _: False)
    monkeypatch.setattr(communicator, "get_attn_tp_context", lambda: context)

    forward_batch.prepare_attn_tp_scatter_input(runner)

    assert dp_attention.get_is_extend_in_batch()
    assert dp_attention.get_prefill_num_tokens() == 5


def test_decayed_lfu_window_gate_freezes_decode(monkeypatch):
    from sglang.srt.layers import dp_attention

    method = object.__new__(kt_ep.KTEPWrapperMethod)
    method._decayed_lfu_enabled = True

    monkeypatch.setattr(dp_attention, "get_prefill_num_tokens", lambda: 0)
    assert not method._should_record_prefill_cache_window(num_tokens=32)

    monkeypatch.setattr(dp_attention, "get_prefill_num_tokens", lambda: 32)
    assert method._should_record_prefill_cache_window(num_tokens=32)
    assert not method._should_record_prefill_cache_window(num_tokens=0)


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
    ],
)
def test_kt_invalid_runtime_combinations_fail_fast(extra):
    with pytest.raises(ValueError):
        _parse_server_args(extra)
