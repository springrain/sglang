# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE = (
    Path(__file__).parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "moe"
    / "kt_mxfp4_stream_transport.py"
)

_SPEC = importlib.util.spec_from_file_location("_kt_mxfp4_stream_transport", MODULE)
assert _SPEC is not None and _SPEC.loader is not None
_TRANSPORT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRANSPORT
_SPEC.loader.exec_module(_TRANSPORT)
Mxfp4ExpertLayout = _TRANSPORT.Mxfp4ExpertLayout
Mxfp4StreamTicket = _TRANSPORT.Mxfp4StreamTicket
StreamTicketState = _TRANSPORT.StreamTicketState
validate_ticket_transition = _TRANSPORT.validate_ticket_transition
Mxfp4StreamTransport = _TRANSPORT.Mxfp4StreamTransport
StreamSlotState = _TRANSPORT.StreamSlotState
format_token_mismatches = _TRANSPORT._format_token_mismatches
validate_tp_layout_contract = _TRANSPORT._validate_tp_layout_contract


def test_ticket_forwards_the_complete_activation_contract():
    captured = {}
    expected = object()

    class Manager:
        def _wait_and_launch(self, ticket, **kwargs):
            captured["ticket"] = ticket
            captured.update(kwargs)
            return expected

    ticket = object.__new__(Mxfp4StreamTicket)
    ticket.manager = Manager()
    hidden_states = torch.zeros((1, 4))
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)
    topk_weights = torch.ones((1, 1))

    result = ticket.wait_and_launch(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    assert result is expected
    assert captured["ticket"] is ticket
    assert captured["activation"] == "situ"
    assert captured["situ_beta"] == 4.0
    assert captured["situ_linear_beta"] == 25.0
    assert captured["swiglu_alpha"] == 0.0
    assert captured["swiglu_limit"] is None


def _raw_tensors(experts: int = 3):
    hidden = 64
    intermediate = 128
    return {
        "w13_weight": torch.empty(
            experts, 2 * intermediate, hidden // 2, dtype=torch.int8
        ),
        "w13_weight_scale_inv": torch.empty(
            experts, 2 * intermediate, hidden // 32, dtype=torch.float32
        ),
        "w2_weight": torch.empty(
            experts, hidden, intermediate // 2, dtype=torch.int8
        ),
        "w2_weight_scale_inv": torch.empty(
            experts, hidden, intermediate // 32, dtype=torch.float32
        ),
    }


def test_layout_is_single_expert_and_writer_compatible():
    layout = Mxfp4ExpertLayout.from_tensors(_raw_tensors())

    assert layout.hidden_size == 64
    assert layout.intermediate_size == 128
    assert layout.spec("w13_weight").expert_shape == (256, 32)
    assert layout.spec("w13_weight").host_dtype == torch.int8
    assert layout.spec("w13_weight_scale_inv").host_dtype == torch.bfloat16
    assert layout.spec("w13_weight_scale_inv").gpu_dtype == torch.float32


def test_staging_depth_has_a_compatibility_prebegin_alias():
    manager = object.__new__(Mxfp4StreamTransport)
    manager.slots = (object(), object())

    assert manager.rolling_window_supported
    assert manager.staging_depth == 2
    assert manager.prebegin_capacity == 2


def test_cpu_main_ordering_commit_covers_ticket_order_and_failure():
    manager = object.__new__(Mxfp4StreamTransport)
    calls = []

    def stage_commit(op, stage, local_success, **kwargs):
        calls.append((op, stage, local_success, kwargs))
        return local_success

    manager._stage_commit = stage_commit
    ticket0 = SimpleNamespace(
        manager=manager,
        ticket_id=11,
        expert_id=7,
        state=StreamTicketState.WRITER_SUBMITTED,
        slot_index=0,
        slot_generation=3,
    )
    ticket1 = SimpleNamespace(
        manager=manager,
        ticket_id=12,
        expert_id=9,
        state=StreamTicketState.WRITER_SUBMITTED,
        slot_index=1,
        slot_generation=4,
    )
    manager.slots = (
        SimpleNamespace(
            ticket_id=11,
            generation=3,
            state=StreamSlotState.WRITING,
        ),
        SimpleNamespace(
            ticket_id=12,
            generation=4,
            state=StreamSlotState.WRITING,
        ),
    )
    manager._active = {11: ticket0, 12: ticket1}
    manager._pending = []

    assert manager.confirm_cpu_main_ordered(
        (ticket0, ticket1), local_success=True
    )
    assert calls == [
        (
            "PRELAUNCH",
            "CPU_MAIN_ORDERED",
            True,
            {
                "ticket": ticket0,
                "detail": (
                    (11, 12),
                    (7, 9),
                    ("WRITER_SUBMITTED", "WRITER_SUBMITTED"),
                    ((0, 3), (1, 4)),
                ),
            },
        )
    ]

    assert not manager.confirm_cpu_main_ordered(
        (ticket0, ticket1), local_success=False
    )
    assert calls[-1][2] is False

    ticket1.state = StreamTicketState.CREATED
    assert not manager.confirm_cpu_main_ordered(
        (ticket0, ticket1), local_success=True
    )
    assert calls[-1][2] is False


def test_begin_candidates_keeps_overflow_as_ordered_created_tickets(monkeypatch):
    manager = object.__new__(Mxfp4StreamTransport)
    manager.group_epoch = 0
    manager._closed = False
    manager._fatal_error = None
    manager._active = {}
    manager._pending = []
    manager._pending_commits = {}
    manager._next_ticket_id = 1
    manager.weight_namespace = "weights"
    manager.slots = tuple(
        SimpleNamespace(
            index=index,
            state=StreamSlotState.FREE,
            ticket_id=None,
            generation=10 + index,
        )
        for index in range(2)
    )
    manager._stage_commit = lambda _op, _stage, local_success, **_kwargs: (
        local_success
    )
    submissions = []

    def create_ticket(ticket_manager, ticket_id, method, expert_id):
        return SimpleNamespace(
            manager=ticket_manager,
            ticket_id=ticket_id,
            method=method,
            expert_id=expert_id,
            state=StreamTicketState.CREATED,
            slot_index=None,
            slot_generation=-1,
            writer_completion=None,
        )

    def submit_writer(ticket, slot):
        slot.state = StreamSlotState.WRITING
        slot.ticket_id = ticket.ticket_id
        slot.generation += 1
        ticket.slot_index = slot.index
        ticket.slot_generation = slot.generation
        ticket.state = StreamTicketState.WRITER_SUBMITTED
        submissions.append((ticket.ticket_id, slot.index, slot.generation))

    monkeypatch.setattr(_TRANSPORT, "Mxfp4StreamTicket", create_ticket)
    manager._submit_writer = submit_writer
    method = SimpleNamespace(
        global_num_experts=16,
        kt_config=SimpleNamespace(layer_idx=8),
    )

    tickets = manager.begin_candidates(method, (7, 9, 11, 13))

    assert len(tickets) == 4
    assert [ticket.state for ticket in tickets] == [
        StreamTicketState.WRITER_SUBMITTED,
        StreamTicketState.WRITER_SUBMITTED,
        StreamTicketState.CREATED,
        StreamTicketState.CREATED,
    ]
    assert submissions == [(1, 0, 11), (2, 1, 12)]
    assert tuple(manager._pending) == tickets[2:]
    assert tuple(manager._active.values()) == tickets


def test_cpu_main_ordering_accepts_eager_prefix_and_created_suffix():
    manager = object.__new__(Mxfp4StreamTransport)
    calls = []
    manager._stage_commit = lambda op, stage, local_success, **kwargs: calls.append(
        (op, stage, local_success, kwargs)
    ) or local_success
    tickets = tuple(
        SimpleNamespace(
            manager=manager,
            ticket_id=11 + index,
            expert_id=7 + 2 * index,
            state=(
                StreamTicketState.WRITER_SUBMITTED
                if index < 2
                else StreamTicketState.CREATED
            ),
            slot_index=index if index < 2 else None,
            slot_generation=3 + index if index < 2 else -1,
        )
        for index in range(4)
    )
    manager.slots = tuple(
        SimpleNamespace(
            ticket_id=tickets[index].ticket_id,
            generation=tickets[index].slot_generation,
            state=StreamSlotState.WRITING,
        )
        for index in range(2)
    )
    manager._active = {ticket.ticket_id: ticket for ticket in tickets}
    manager._pending = list(tickets[2:])

    assert manager.confirm_cpu_main_ordered(tickets, local_success=True)
    assert calls[-1][2] is True
    assert calls[-1][3]["detail"] == (
        (11, 12, 13, 14),
        (7, 9, 11, 13),
        (
            "WRITER_SUBMITTED",
            "WRITER_SUBMITTED",
            "CREATED",
            "CREATED",
        ),
        ((0, 3), (1, 4), (None, -1), (None, -1)),
    )

    manager._pending.reverse()
    assert not manager.confirm_cpu_main_ordered(tickets, local_success=True)
    assert calls[-1][2] is False


def test_each_wait_refills_a_slot_released_by_the_previous_candidate():
    manager = object.__new__(Mxfp4StreamTransport)
    tickets = [
        SimpleNamespace(
            manager=manager,
            ticket_id=index + 1,
            expert_id=20 + index,
            state=(
                StreamTicketState.WRITER_SUBMITTED
                if index < 2
                else StreamTicketState.CREATED
            ),
            slot_index=index if index < 2 else None,
            slot_generation=5 + index if index < 2 else -1,
        )
        for index in range(4)
    ]
    manager.slots = tuple(
        SimpleNamespace(
            index=index,
            state=StreamSlotState.WRITING,
            ticket_id=tickets[index].ticket_id,
            generation=tickets[index].slot_generation,
            reuse_event=object(),
        )
        for index in range(2)
    )
    manager._active = {ticket.ticket_id: ticket for ticket in tickets}
    manager._pending = tickets[2:].copy()
    submissions = []
    refill_stages = []

    def stage_commit(op, stage, local_success, **kwargs):
        refill_stages.append((op, stage, local_success, kwargs))
        return local_success

    manager._stage_commit = stage_commit

    def submit_writer(ticket, slot):
        slot.state = StreamSlotState.WRITING
        slot.ticket_id = ticket.ticket_id
        slot.generation += 1
        ticket.slot_index = slot.index
        ticket.slot_generation = slot.generation
        ticket.state = StreamTicketState.WRITER_SUBMITTED
        submissions.append((ticket.ticket_id, slot.index, slot.generation))

    manager._submit_writer = submit_writer

    # Both slots are occupied at ticket 0's entry, so no deferred writer is
    # submitted yet.
    manager._refill_writer_submissions_for_wait(tickets[0])
    assert submissions == []

    # Finalizing ticket 0 frees slot 0.  Ticket 1 is already writer-submitted,
    # but its wait entry must immediately backfill slot 0 with ticket 2.
    manager._retire_ticket(tickets[0], manager.slots[0])
    manager._refill_writer_submissions_for_wait(tickets[1])
    assert submissions == [(3, 0, 6)]
    assert manager.slots[0].ticket_id == 3
    assert manager.slots[1].ticket_id == 2
    assert manager._pending == [tickets[3]]

    # The same rolling rule keeps two in flight after ticket 1 finalizes.
    manager._retire_ticket(tickets[1], manager.slots[1])
    manager._refill_writer_submissions_for_wait(tickets[2])

    assert submissions == [(3, 0, 6), (4, 1, 7)]
    assert [slot.ticket_id for slot in manager.slots] == [3, 4]
    assert [ticket.slot_generation for ticket in tickets[2:]] == [6, 7]
    assert manager._pending == []
    assert set(manager._active) == {3, 4}
    assert [stage[:3] for stage in refill_stages] == [
        ("WAIT_LAUNCH", "REFILL_PLAN", True),
        ("WAIT_LAUNCH", "REFILL_PLAN", True),
        ("WAIT_LAUNCH", "REFILL_PLAN", True),
    ]
    assert refill_stages[1][3]["detail"] == (
        (3, 4),
        (22, 23),
        ((0, 5),),
    )


def test_refill_consensus_precedes_writer_pump():
    source = MODULE.read_text(encoding="utf-8")
    refill = source[
        source.index("    def _refill_writer_submissions_for_wait(") : source.index(
            "    def _submit_writer("
        )
    ]

    assert refill.index('"REFILL_PLAN"') < refill.index(
        "self._pump_writer_submissions()"
    )


def test_refill_rejects_an_out_of_order_created_ticket_without_pumping():
    manager = object.__new__(Mxfp4StreamTransport)
    head = SimpleNamespace(
        manager=manager,
        ticket_id=3,
        expert_id=22,
        state=StreamTicketState.CREATED,
        slot_index=None,
        slot_generation=-1,
    )
    later = SimpleNamespace(
        manager=manager,
        ticket_id=4,
        expert_id=23,
        state=StreamTicketState.CREATED,
        slot_index=None,
        slot_generation=-1,
    )
    manager._active = {3: head, 4: later}
    manager._pending = [head, later]
    manager.slots = (
        SimpleNamespace(index=0, generation=7, state=StreamSlotState.FREE),
        SimpleNamespace(index=1, generation=8, state=StreamSlotState.WRITING),
    )
    pumped = []
    manager._pump_writer_submissions = lambda: pumped.append(True)
    stages = []
    manager._stage_commit = lambda op, stage, local_success, **kwargs: stages.append(
        (op, stage, local_success, kwargs)
    ) or local_success

    with pytest.raises(RuntimeError, match="rolling refill turn"):
        manager._refill_writer_submissions_for_wait(later)

    assert pumped == []
    assert stages[0][:3] == ("WAIT_LAUNCH", "REFILL_PLAN", False)


def test_dispatch_ready_commit_precedes_ticket_creation():
    manager = object.__new__(Mxfp4StreamTransport)
    calls = []
    manager._stage_commit = lambda *args, **kwargs: calls.append(
        (args, kwargs)
    ) or kwargs.get("local_success", args[2])

    assert manager.confirm_dispatch_ready(
        layer_idx=8,
        candidate_ids=(7, 9),
        local_success=True,
    )
    assert calls == [
        (
            ("PREBEGIN", "DISPATCH_READY", True),
            {"detail": (8, (7, 9))},
        )
    ]


def test_wrapper_phase_commit_is_slot_state_independent():
    manager = object.__new__(Mxfp4StreamTransport)
    calls = []
    manager._stage_commit = lambda *args, **kwargs: calls.append(
        (args, kwargs)
    ) or args[2]
    tickets = (
        SimpleNamespace(ticket_id=11, expert_id=7),
        SimpleNamespace(ticket_id=12, expert_id=9),
    )

    assert manager.confirm_wrapper_phase(
        "LAYER_OUTPUT_READY",
        tickets,
        local_success=True,
    )
    assert calls == [
        (
            ("WRAPPER", "LAYER_OUTPUT_READY", True),
            {"detail": ((11, 12), (7, 9))},
        )
    ]
    with pytest.raises(ValueError, match="unsupported MXFP4 wrapper phase"):
        manager.confirm_wrapper_phase(
            "UNKNOWN",
            tickets,
            local_success=True,
        )


def test_ticket_creation_failure_aborts_before_any_writer_is_pumped(monkeypatch):
    manager = object.__new__(Mxfp4StreamTransport)
    manager.group_epoch = 0
    manager._closed = False
    manager._fatal_error = None
    manager._active = {}
    manager._pending = []
    manager._pending_commits = {}
    manager._next_ticket_id = 1
    manager.weight_namespace = "weights"
    stages = []
    aborted = []
    pumped = []
    created = []

    def stage_commit(op, stage, local_success, **_kwargs):
        stages.append((op, stage, local_success))
        return local_success

    manager._stage_commit = stage_commit

    def abort_partial_group():
        aborted.append(True)
        manager._active.clear()
        manager._pending.clear()

    manager._abort_candidates_internal = abort_partial_group
    manager._pump_writer_submissions = lambda: pumped.append(True)

    def create_ticket(_manager, ticket_id, _method, expert_id):
        if created:
            raise RuntimeError("synthetic CUDA event allocation failure")
        ticket = SimpleNamespace(ticket_id=ticket_id, expert_id=expert_id)
        created.append(ticket)
        return ticket

    monkeypatch.setattr(
        _TRANSPORT,
        "Mxfp4StreamTicket",
        create_ticket,
    )
    method = SimpleNamespace(
        global_num_experts=16,
        kt_config=SimpleNamespace(layer_idx=8),
    )

    with pytest.raises(RuntimeError, match="ticket creation failed"):
        manager.begin_candidates(method, (7, 9))

    assert stages == [
        ("BEGIN", "ENTER", True),
        ("BEGIN", "TICKETS_CREATED", False),
    ]
    assert aborted == [True]
    assert pumped == []
    assert manager._next_ticket_id == 3


def test_layout_rejects_a_cross_tensor_shape_mismatch():
    tensors = _raw_tensors()
    tensors["w2_weight_scale_inv"] = torch.empty(3, 64, 5)

    with pytest.raises(ValueError, match="inconsistent DeepSeek-V4 MXFP4"):
        Mxfp4ExpertLayout.from_tensors(tensors)


def test_tp_compatibility_signature_ignores_local_cuda_ordinal():
    base = Mxfp4ExpertLayout.from_tensors(_raw_tensors())
    rank0 = replace(
        base,
        device=torch.device("cuda:0"),
        compute_capability=(8, 9),
    )
    rank1 = replace(
        base,
        device=torch.device("cuda:1"),
        compute_capability=(8, 9),
    )

    assert rank0.compatibility_signature == rank1.compatibility_signature
    assert rank0.local_registry_signature != rank1.local_registry_signature
    assert validate_tp_layout_contract(
        [
            ("cuda:0", rank0.compatibility_signature),
            ("cuda:1", rank1.compatibility_signature),
        ]
    ) == rank0.compatibility_signature


def test_tp_compatibility_signature_rejects_real_layout_or_dtype_difference():
    base = Mxfp4ExpertLayout.from_tensors(_raw_tensors())
    cuda_layout = replace(
        base,
        device=torch.device("cuda:0"),
        compute_capability=(8, 9),
    )
    different_shape = replace(cuda_layout, hidden_size=128)
    different_specs = list(cuda_layout.raw_specs)
    different_specs[0] = replace(different_specs[0], gpu_dtype=torch.int32)
    different_dtype = replace(cuda_layout, raw_specs=tuple(different_specs))
    different_compute_capability = replace(
        cuda_layout, compute_capability=(12, 0)
    )

    assert cuda_layout.compatibility_signature != different_shape.compatibility_signature
    assert cuda_layout.compatibility_signature != different_dtype.compatibility_signature
    assert (
        cuda_layout.compatibility_signature
        != different_compute_capability.compatibility_signature
    )


def test_tp_layout_contract_reports_heterogeneous_gpu_capability_clearly():
    base = Mxfp4ExpertLayout.from_tensors(_raw_tensors())
    sm89 = replace(
        base,
        device=torch.device("cuda:0"),
        compute_capability=(8, 9),
    )
    sm120 = replace(
        base,
        device=torch.device("cuda:1"),
        compute_capability=(12, 0),
    )

    with pytest.raises(RuntimeError, match="homogeneous TP group") as exc_info:
        validate_tp_layout_contract(
            [
                ("cuda:0", sm89.compatibility_signature),
                ("cuda:1", sm120.compatibility_signature),
            ]
        )

    message = str(exc_info.value)
    assert "compute capability" in message
    assert "cuda:1" in message
    assert "(12, 0)" in message


def test_token_mismatch_diagnostics_report_peer_rank_and_token():
    reports = [
        (("factory", "layout", "canonical"), True),
        (("factory", "layout", "peer-local-device"), True),
    ]

    assert format_token_mismatches(reports) == [
        (1, ("factory", "layout", "peer-local-device"))
    ]


def test_different_local_weight_paths_produce_the_same_operation_token():
    layout = Mxfp4ExpertLayout.from_tensors(_raw_tensors())
    canonical_identity = (
        "/tp0/canonical/model",
        2,
        layout.compatibility_signature,
    )

    def make_token(local_weight_path: str):
        manager = object.__new__(Mxfp4StreamTransport)
        manager.transport_identity = canonical_identity
        manager.weight_namespace = canonical_identity[0]
        manager.registry_key = (
            local_weight_path,
            2,
            layout.local_registry_signature,
        )
        manager.group_epoch = 4
        manager._collective_seq = 0
        manager.slots = ()
        ticket = SimpleNamespace(
            layer_idx=7,
            weight_namespace=manager.weight_namespace,
            ticket_id=11,
            expert_id=23,
            state=StreamTicketState.CREATED,
            slot_index=None,
            slot_generation=-1,
        )
        return manager._operation_token("WAIT_LAUNCH", "ENTER", ticket=ticket)

    assert make_token("/rank0/model") == make_token("/rank1/other-mount/model")


def test_ticket_namespace_comes_from_manager_not_rank_local_method_path():
    source = MODULE.read_text(encoding="utf-8")
    ticket = source[
        source.index("class Mxfp4StreamTicket:") : source.index(
            "class Mxfp4StreamTransport:"
        )
    ]
    factory = source[
        source.index("def get_or_create_mxfp4_stream_transport(") : source.index(
            "def close_mxfp4_stream_transports("
        )
    ]

    assert "self.weight_namespace = manager.weight_namespace" in ticket
    assert "method.kt_config, \"weight_path\"" not in ticket
    assert "_broadcast_object_from_tp0(local_weight_namespace)" in factory
    assert "local_weight_namespace," in factory


def test_ticket_state_machine_has_explicit_recovery_terminal():
    validate_ticket_transition(
        StreamTicketState.WRITER_SUBMITTED, StreamTicketState.FAILED
    )
    validate_ticket_transition(StreamTicketState.FAILED, StreamTicketState.ABORTED)
    validate_ticket_transition(
        StreamTicketState.WAVE2_LAUNCHED, StreamTicketState.INSTALL_COMMITTED
    )
    validate_ticket_transition(
        StreamTicketState.INSTALL_COMMITTED, StreamTicketState.INSTALLED
    )

    with pytest.raises(RuntimeError, match="invalid streamed ticket transition"):
        validate_ticket_transition(
            StreamTicketState.INSTALLED, StreamTicketState.WRITER_SUBMITTED
        )


def test_transport_source_has_the_complete_single_expert_chain():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    assert "begin_candidates" in names
    assert "wait_and_launch" in names
    assert "install" in names
    assert "abort_candidates" in names
    assert "submit_write_weight_scale_to_buffer" in source
    assert "sync_write_weight_scale_to_buffer" in source
    assert "prepare_v4_mxfp4_marlin_expert" in source
    assert "apply_v4_mxfp4_marlin_streamed_experts" in source
    assert "view_v4_mxfp4_marlin_slot" in source


def test_install_waits_for_wave_before_any_resident_copy():
    source = MODULE.read_text(encoding="utf-8")
    install = source[
        source.index("    def _install(") : source.index(
            "    def _detach_staging_after_install("
        )
    ]

    wave_wait = install.index("install_stream.wait_event(ticket.wave_done_event)")
    victim_wait = install.index("install_stream.wait_event(victim_safe_event)")
    first_resident_copy = install.index("resident_view.w13.copy_(")
    assert wave_wait < first_resident_copy
    assert victim_wait < first_resident_copy
    assert "resident_raw_views[name].copy_(slot.raw[name]" in install
    assert install.index("resident_raw_views[name].copy_(") < install.index(
        "ticket.install_done_event.record(install_stream)"
    )


def test_recovery_fences_work_and_clears_the_whole_group():
    source = MODULE.read_text(encoding="utf-8")
    recovery = source[
        source.index("    def _abort_candidates_internal(") : source.index(
            "    def _ensure_owned("
        )
    ]

    assert "sync_write_weight_scale_to_buffer" in recovery
    assert "self.transfer_stream.synchronize()" in recovery
    assert "self.prepare_stream.synchronize()" in recovery
    assert "torch.cuda.synchronize(self.layout.device)" in recovery
    assert "self._active.clear()" in recovery
    assert "self._pending.clear()" in recovery
    assert "slot.state = StreamSlotState.FREE" in recovery


def test_transport_never_publishes_resident_mapping_metadata():
    source = MODULE.read_text(encoding="utf-8")

    assert "gpu_experts_mask" not in source
    assert "logical_to_gpu_index" not in source
    assert "update_kt_wrapper_masks" not in source


def test_host_slot_reuse_consensus_precedes_tp0_writer_submit():
    source = MODULE.read_text(encoding="utf-8")
    submit = source[
        source.index("    def _submit_writer(") : source.index(
            "    def _wait_and_launch("
        )
    ]

    local_fence = submit.index("slot.host_dma_done_event.synchronize()")
    all_rank_fence = submit.index('"HOST_REUSE_READY"')
    writer_submit = submit.index("wrapper.submit_write_weight_scale_to_buffer(")
    assert local_fence < all_rank_fence < writer_submit


def test_control_token_covers_transport_layer_ticket_and_generation():
    source = MODULE.read_text(encoding="utf-8")
    token = source[
        source.index("    def _operation_token(") : source.index(
            "    def _stage_commit("
        )
    ]
    stage = source[
        source.index("    def _stage_commit(") : source.index(
            "    def _public_ticket_enter("
        )
    ]

    for field in (
        "self.transport_identity",
        "ticket.layer_idx",
        "ticket.weight_namespace",
        "ticket.ticket_id",
        "ticket.expert_id",
        "slot_index",
        "generation",
        "op",
        "stage",
    ):
        assert field in token
    assert "_tp_all_gather_objects" in stage
    assert "_broadcast_object_from_tp0" not in stage
    assert "_tp_all_succeeded" not in stage


def test_external_completion_events_are_ticket_owned_not_slot_reused():
    source = MODULE.read_text(encoding="utf-8")
    ticket_init = source[
        source.index("class Mxfp4StreamTicket:") : source.index(
            "class Mxfp4StreamTransport:"
        )
    ]
    wave = source[
        source.index("    def _wait_and_launch(") : source.index(
            "    def _install("
        )
    ]

    assert "self.wave_done_event = torch.cuda.Event()" in ticket_init
    assert "self.install_done_event = torch.cuda.Event()" in ticket_init
    assert "self.release_done_event = torch.cuda.Event()" in ticket_init
    assert "completion_event=ticket.wave_done_event" in wave
    assert "completion_event=slot.wave_done_event" not in wave


def test_two_commits_release_both_staging_slots_for_candidates_three_and_four():
    manager = object.__new__(Mxfp4StreamTransport)
    tickets = [SimpleNamespace(ticket_id=index) for index in range(1, 5)]
    manager._active = {ticket.ticket_id: ticket for ticket in tickets}
    manager._pending_commits = {}
    slots = [
        SimpleNamespace(
            state=StreamSlotState.INSTALLING,
            ticket_id=tickets[index].ticket_id,
            reuse_event=None,
        )
        for index in range(2)
    ]
    commits = [
        SimpleNamespace(commit_token=("commit", index)) for index in range(2)
    ]
    tickets[0].install_done_event = object()
    tickets[1].install_done_event = object()

    manager._detach_staging_after_install(tickets[0], slots[0], commits[0])
    manager._detach_staging_after_install(tickets[1], slots[1], commits[1])

    assert [slot.state for slot in slots] == [
        StreamSlotState.FREE,
        StreamSlotState.FREE,
    ]
    assert set(manager._active) == {3, 4}
    assert len(manager._pending_commits) == 2
    assert slots[0].reuse_event is tickets[0].install_done_event
    assert slots[1].reuse_event is tickets[1].install_done_event


def test_install_return_is_independent_of_next_writer_pump():
    source = MODULE.read_text(encoding="utf-8")
    install = source[
        source.index("    def _install(") : source.index(
            "    def _detach_staging_after_install("
        )
    ]
    retire = source[
        source.index("    def _retire_ticket(") : source.index(
            "    def _fail_ticket("
        )
    ]

    assert "_pump_writer_submissions" not in install
    assert "_pump_writer_submissions" not in retire
    assert "ResidentInstallCommit(" in install


def test_close_fences_before_host_unregister_and_retains_on_failure():
    source = MODULE.read_text(encoding="utf-8")
    close = source[
        source.index("    def close(self, *, coordinated: bool = True)") : source.index(
            "@dataclass(frozen=True)\nclass UnsupportedMxfp4StreamTransport"
        )
    ]

    writer_abort = close.index("self._abort_candidates_internal()")
    transfer_fence = close.index("self.transfer_stream.synchronize()")
    device_fence = close.index("torch.cuda.synchronize(self.layout.device)")
    host_close = close.index("self.host_pool.close()")
    assert writer_abort < transfer_fence < device_fence < host_close
    assert "return False" in close


def test_abort_preserves_irreversible_commits_for_later_publish(monkeypatch):
    manager = object.__new__(Mxfp4StreamTransport)
    commit = object()
    manager._fatal_error = None
    manager._active = {}
    manager._pending = []
    manager._pending_commits = {("commit", 1): commit}
    manager.tp_rank = 1
    manager.transfer_stream = SimpleNamespace(synchronize=lambda: None)
    manager.prepare_stream = SimpleNamespace(synchronize=lambda: None)
    manager.layout = SimpleNamespace(device=torch.device("cpu"))
    manager.slots = ()
    stages = []
    manager._stage_commit = lambda op, stage, *_args, **_kwargs: stages.append(
        (op, stage)
    ) or True
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)

    result = manager._abort_candidates_internal()

    assert result.recovered
    assert result.pending_commits == (commit,)
    assert result.mapping_publish_required
    assert manager._pending_commits == {("commit", 1): commit}
    assert stages == [("ABORT", "RECOVERY_FENCED")]


def test_irreversible_commit_is_registered_before_active_ticket_is_removed():
    source = MODULE.read_text(encoding="utf-8")
    start = source.index("    def _detach_staging_after_install(")
    detach = source[
        start : source.index("    def confirm_mapping_published(", start)
    ]

    assert detach.index("self._pending_commits[commit.commit_token] = commit") < (
        detach.index("self._active.pop(ticket.ticket_id, None)")
    )


def test_factory_two_phase_commit_precedes_registry_publication():
    source = MODULE.read_text(encoding="utf-8")
    factory = source[
        source.index("def get_or_create_mxfp4_stream_transport(") : source.index(
            "def close_mxfp4_stream_transports("
        )
    ]

    construct = factory.index('("FACTORY", "CONSTRUCT", transport_identity)')
    publish = factory.index("_TRANSPORTS[key] = manager")
    assert construct < publish
    assert "construction_error" in factory
    assert "manager.close(coordinated=False)" in factory


def test_host_unregister_checks_return_code_before_releasing_shm():
    source = MODULE.read_text(encoding="utf-8")
    host_close = source[
        source.index("    def close(self) -> bool:") : source.index(
            "class _GpuStreamSlot:"
        )
    ]

    assert "result = torch.cuda.cudart().cudaHostUnregister(pointer)" in host_close
    assert "if int(result) != 0:" in host_close
    assert host_close.index("if unregister_error is not None:") < host_close.index(
        "self.buffers.clear()"
    )
