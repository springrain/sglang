"""CPU-only unit checks for the MXFP4 event-fence ring transport.

Loads kt_ep_wrapper.py with its sglang.srt dependencies stubbed, so no
sglang install, GPU, or kt_kernel_ext build is required:

  F1  _ring_blocks partition algebra: ceil-split, short tail, empty input.
  F2  Ring consensus count is exactly 2 x ceil(E / e_chunk) per layer
      (chunk 1 reproduces the legacy 2 x E cadence), and the ring branch
      is wired between the window branch and the legacy loop.
  F3  Per-block op order is the golden sequence; tp_rank 1 submits
      nothing; a sticky ring error pops at the next block's reuse
      consensus; transported destination bytes match the ring rows.
  F4  --kt-prefill-event-fence=0 structural identity: one intent MIN,
      no geometry, and the frozen allocation stays at two rows
      (_ring_row_count).
  F5  Capacity ladder: failed probes drop to the floor, higher rungs are
      skipped above the request, and a peer MIN negotiates the local
      step down with one warning.
  F6  (deleted with the env->CLI migration: the no-device-sync inert warn
      moved to ServerArgs validation, and the frozen CLI parameter has
      no sub-env ignore channel for the check to observe)
  F7  --kt-prefill-fence-debug freezes via an all-rank MAX; the digest
      hook runs once per block in single-rank mode, and a cross-rank
      mismatch parks a sticky error in the ring channel instead of
      raising in place.
  F8  chunk 0 folds the fence intent off, and the staging-window mutex
      freezes the fence off with one warning (the drift segments left
      with the env->CLI migration: a frozen CLI parameter has no runtime
      drift channel for the checks to observe).

Run: python third_party/sglang/test/manual/kt_event_fence_ring_test.py
"""

import contextlib
import importlib.util
import inspect
import logging
import os
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER_PATH = (
    REPO_ROOT
    / "third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
)
# The staging window stays an env (tier-1 knob); every other ring knob
# of this batch is now a frozen --kt-* CLI parameter mirrored by the
# exec bag stubbed into runtime_context inside _load_wrapper().
ENV_WINDOW = "SGLANG_KT_PREFILL_STAGE_LAYER_WINDOW"
RING_ENVS = (ENV_WINDOW,)

# Toy per-bank single-expert payload bytes; every value is 16B aligned and
# far away from any production shape.
BANK_NBYTES = (32, 16, 48, 16)


def _stub(fullname, **attrs):
    mod = types.ModuleType(fullname)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[fullname] = mod
    return mod


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_wrapper():
    """Exec kt_ep_wrapper.py with every sglang.srt import satisfied by stubs."""
    _stub("sglang")
    _stub("sglang.srt")
    _stub("sglang.srt.arg_groups")
    _stub("sglang.srt.arg_groups.overrides", model_config_of=lambda *a, **k: None)
    _stub(
        "sglang.srt.distributed",
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tp_group=lambda: None,
    )
    _stub("sglang.srt.layers")
    _stub("sglang.srt.layers.moe")
    _stub("sglang.srt.layers.quantization")
    _stub("sglang.srt.layers.quantization.base_config", FusedMoEMethodBase=object)
    _stub(
        "sglang.srt.layers.quantization.marlin_utils",
        marlin_permute_scales=lambda *a, **k: None,
    )
    # Shared exec bag: the frozen CLI defaults for this batch, mutated
    # per test and restored in finally blocks (setattr precedent).
    exec_bag = types.SimpleNamespace(
        moe=types.SimpleNamespace(
            kt_prefill_event_fence=1,
            kt_prefill_stage_chunk_experts=64,
            kt_prefill_no_device_sync=0,
            kt_prefill_fence_debug=0,
        )
    )
    _stub(
        "sglang.srt.runtime_context",
        get_exec=lambda: exec_bag,
        get_schedule=lambda: None,
    )
    # is_cuda() False skips the gptq_marlin_repack import branch; the "eager"
    # backend string keeps module-level torch.compile decorators inductor-free.
    _stub(
        "sglang.srt.utils", get_compiler_backend=lambda: "eager", is_cuda=lambda: False
    )
    mod = _load_module("kt_ep_wrapper_iso", WRAPPER_PATH)
    mod._TEST_EXEC_BAG = exec_bag
    return mod


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _attach_handler(w, level=logging.WARNING):
    handler = _RecordHandler()
    wrapper_logger = logging.getLogger(w.__name__)
    old_level = wrapper_logger.level
    wrapper_logger.setLevel(level)
    wrapper_logger.addHandler(handler)
    return handler, wrapper_logger, old_level


def _detach_handler(wrapper_logger, handler, old_level):
    wrapper_logger.removeHandler(handler)
    wrapper_logger.setLevel(old_level)


def _warnings(handler, needle):
    return [
        r
        for r in handler.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


def _save_ring_envs():
    return {name: os.environ.pop(name, None) for name in RING_ENVS}


def _restore_envs(saved):
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _reset_fence_latches(w):
    # The drift and sub-env latches left with the env->CLI migration;
    # only the mutex warn survives (the window knob is still an env).
    w._event_fence_mutex_warned = False


class _FakeEvent:
    """Stands in for torch.cuda.Event; every call lands on the op log."""

    def __init__(self, op_log, name):
        self.op_log = op_log
        self.name = name

    def synchronize(self):
        self.op_log.append(f"{self.name}.synchronize")

    def record(self, stream):
        self.op_log.append(f"{self.name}.record")


class _FakeStream:
    def __init__(self, op_log, name):
        self.op_log = op_log
        self.name = name

    def synchronize(self):
        self.op_log.append(f"{self.name}.synchronize")


class _FakeCpuInfer:
    """One submit op per expert, one sync op per block."""

    def __init__(self, op_log, raise_on_submit=False):
        self.op_log = op_log
        self.raise_on_submit = raise_on_submit

    def submit_write_weight_scale_to_buffer(self, *args):
        self.op_log.append("submit")
        if self.raise_on_submit:
            raise ValueError("submit failed")

    def sync_write_weight_scale_to_buffer(self):
        self.op_log.append("sync")


class _cuda_stream_patch:
    """CPU torch has no usable CUDA stream context; swap in a nullcontext."""

    def __enter__(self):
        self.saved = torch.cuda.stream
        torch.cuda.stream = lambda stream: contextlib.nullcontext()

    def __exit__(self, *exc):
        torch.cuda.stream = self.saved
        return False


def _raw_names(w):
    return w._Mxfp4PrefillSlot.RAW_NAMES


def _ring_geometry(w, e_chunk, num_experts=8, debug_rows=False):
    return w._Mxfp4RingGeometry(
        e_chunk=e_chunk,
        num_slots=2,
        ring_rows=2 * e_chunk,
        num_experts=num_experts,
        bank_expert_nbytes=BANK_NBYTES,
        total_nbytes=2 * e_chunk * sum(BANK_NBYTES),
        debug_rows=debug_rows,
    )


def _ring_buffers(w, geometry, num_experts=8):
    # cpu_buffers rows follow the fill pattern (row * 29 + 3) % 211; the
    # destinations start zeroed so a transported row is easy to identify.
    buffers = {}
    dests = {}
    for name, nbytes in zip(_raw_names(w), BANK_NBYTES):
        buf = torch.empty(geometry.ring_rows, nbytes, dtype=torch.uint8)
        for row in range(geometry.ring_rows):
            buf[row].fill_((row * 29 + 3) % 211)
        buffers[name] = buf
        dests[name] = torch.zeros(num_experts, nbytes, dtype=torch.uint8)
    return buffers, dests


def _make_ring_manager(w, geometry, tp_rank=0, level=2, num_experts=8):
    """Bypass __init__ (needs CUDA streams); install ring-mode state."""
    mgr = w._Mxfp4LayerwisePrefillManager.__new__(w._Mxfp4LayerwisePrefillManager)
    op_log = []
    buffers, dests = _ring_buffers(
        w=w, geometry=geometry, num_experts=num_experts
    )
    mgr.epoch = 3
    mgr._ownership_level = level
    mgr._ownership_warned = set()
    mgr._ring_generation = 0
    mgr._ring_free_events = [
        _FakeEvent(op_log, f"ring_free[{i}]")
        for i in range(geometry.num_slots)
    ]
    # was_used starts True so every block observes the reuse sync, the
    # discipline the begin fence is built around.
    mgr._ring_was_used = [True] * geometry.num_slots
    mgr._ring_owner = [None] * geometry.num_slots
    mgr._ring_freed = [True] * geometry.num_slots
    mgr._ring_last_chunk_key = None
    mgr._ring_skip_note_logged = False
    mgr._pending_ring_error = None
    mgr._stats_chunk = None
    mgr._stats_span_layer = None
    mgr.context = types.SimpleNamespace(
        _ring_geometry=geometry,
        cpu_buffers=buffers,
        all_rank_buffer_ptrs={name: [4096] for name in _raw_names(w)},
    )
    mgr.transfer_stream = _FakeStream(op_log, "transfer")
    weight_infos = [
        (name, buffers[name], dests[name]) for name in _raw_names(w)
    ]
    real_commit = w._Mxfp4LayerwisePrefillManager._commit_tp_device_runtime_phase

    def commit_device(local_error, phase):
        op_log.append("consensus:" + phase)
        return real_commit(mgr, local_error, phase)

    mgr._commit_tp_device_runtime_phase = commit_device
    real_ownership = w._Mxfp4LayerwisePrefillManager._ownership_record_ring

    def ownership(layer_idx, generation, ring_slot):
        op_log.append("ownershipR")
        return real_ownership(
            mgr, layer_idx=layer_idx, generation=generation, ring_slot=ring_slot
        )

    mgr._ownership_record_ring = ownership
    method = types.SimpleNamespace(
        tp_rank=tp_rank, wrapper=_FakeCpuInfer(op_log)
    )
    return mgr, method, op_log, buffers, dests, weight_infos


def _make_ring_context(w, num_experts=8):
    """SharedFullContext fake for the negotiation choreography."""
    ctx = w.SharedFullContext.__new__(w.SharedFullContext)
    ctx._ring_geometry = None
    ctx._event_fence_frozen = False
    ctx._staging_window_mode_cache = None
    ctx._is_mxfp4_quant = False
    gpu_layer = types.SimpleNamespace(num_experts=num_experts)
    for name, nbytes in zip(_raw_names(w), BANK_NBYTES):
        setattr(
            gpu_layer,
            name,
            torch.zeros(num_experts, nbytes, dtype=torch.uint8),
        )
    ctx.gpu_layer = gpu_layer
    return ctx


def f1_ring_blocks(w):
    assert w._ring_blocks(7, 3) == [(0, 3), (3, 3), (6, 1)]
    assert w._ring_blocks(4, 2) == [(0, 2), (2, 2)]
    assert w._ring_blocks(1, 1) == [(0, 1)]
    assert w._ring_blocks(5, 64) == [(0, 5)]
    assert w._ring_blocks(0, 64) == []
    print("F1 PASS: _ring_blocks ceil-splits with a short tail, empty is []")


def f2_consensus_count(w):
    for e_chunk, expected in ((2, 6), (1, 10), (64, 2)):
        geometry = _ring_geometry(w=w, e_chunk=e_chunk)
        mgr, method, op_log, _, _, weight_infos = _make_ring_manager(
            w=w, geometry=geometry
        )
        with _cuda_stream_patch():
            err = mgr._load_ring_cpu_experts(
                method=method,
                cpu_expert_ids=[0, 1, 2, 3, 4],
                weight_infos=weight_infos,
                layer_idx=0,
                stats_chunk=None,
                geometry=geometry,
            )
        assert err is None, err
        count = sum(1 for op in op_log if op.startswith("consensus:"))
        assert count == expected, (e_chunk, expected, count, op_log)
    # Dispatch wiring: the ring branch sits between the window branch and
    # the legacy per-expert loop inside _load_slot.
    src = inspect.getsource(w._Mxfp4LayerwisePrefillManager._load_slot)
    window_idx = src.find("if geometry is not None:")
    ring_idx = src.find("elif ring_geometry is not None:")
    legacy_idx = src.find("batch_dma = _kt_prefill_batch_dma_enabled()")
    assert 0 < window_idx < ring_idx < legacy_idx, (
        window_idx,
        ring_idx,
        legacy_idx,
    )
    print(
        "F2 PASS: consensus count is 2 x ceil(E/e_chunk); chunk=1 mirrors "
        "legacy 2 x E; dispatch order audited"
    )


def f3_op_order_and_bytes(w):
    geometry = _ring_geometry(w=w, e_chunk=2)
    mgr, method, op_log, buffers, dests, weight_infos = _make_ring_manager(
        w=w, geometry=geometry
    )
    with _cuda_stream_patch():
        err = mgr._load_ring_cpu_experts(
            method=method,
            cpu_expert_ids=[0, 1, 2, 3, 4],
            weight_infos=weight_infos,
            layer_idx=0,
            stats_chunk=None,
            geometry=geometry,
        )
    assert err is None, err
    expected = [
        "ring_free[0].synchronize",
        "consensus:ring slot 0 reuse for layer 0 generation 0",
        "ownershipR",
        "submit", "submit", "sync",
        "consensus:host writes for layer 0 block 0",
        "ring_free[0].record",
        "ring_free[1].synchronize",
        "consensus:ring slot 1 reuse for layer 0 generation 1",
        "ownershipR",
        "submit", "submit", "sync",
        "consensus:host writes for layer 0 block 1",
        "ring_free[1].record",
        "ring_free[0].synchronize",
        "consensus:ring slot 0 reuse for layer 0 generation 2",
        "ownershipR",
        "submit", "sync",
        "consensus:host writes for layer 0 block 2",
        "ring_free[0].record",
    ]
    assert op_log == expected, op_log
    # Byte contract: expert rows land exactly from the ring rows that the
    # frozen geometry assigned them (slot 0 rows 0/1, slot 1 rows 2/3,
    # then slot 0 row 0 again for the tail block).
    src_rows = (0, 1, 2, 3, 0)
    for name in _raw_names(w):
        for expert_id, row in enumerate(src_rows):
            assert torch.equal(dests[name][expert_id], buffers[name][row]), (
                name,
                expert_id,
            )

    # A non-TP0 rank runs the identical spine minus the host packing.
    mgr1, method1, op1, buf1, dst1, infos1 = _make_ring_manager(
        w=w, geometry=geometry, tp_rank=1
    )
    with _cuda_stream_patch():
        err = mgr1._load_ring_cpu_experts(
            method=method1,
            cpu_expert_ids=[0, 1, 2, 3, 4],
            weight_infos=infos1,
            layer_idx=0,
            stats_chunk=None,
            geometry=geometry,
        )
    assert err is None, err
    assert "submit" not in op1 and "sync" not in op1, op1
    expected_tp1 = [op for op in expected if op not in ("submit", "sync")]
    assert op1 == expected_tp1, op1
    for expert_id, row in enumerate(src_rows):
        assert torch.equal(dst1["w13_weight"][expert_id], buf1["w13_weight"][row])

    # A sticky ring error pops at the next block's reuse consensus and
    # raises out of the loop; the channel is drained exactly once.
    mgr2, method2, op2, _, _, infos2 = _make_ring_manager(
        w=w, geometry=geometry
    )
    mgr2._pending_ring_error = RuntimeError("stale ring error")
    try:
        mgr2._load_ring_cpu_experts(
            method=method2,
            cpu_expert_ids=[0, 1, 2, 3, 4],
            weight_infos=infos2,
            layer_idx=0,
            stats_chunk=None,
            geometry=geometry,
        )
        raise AssertionError("a sticky ring error must raise at consensus")
    except RuntimeError as exc:
        assert "stale ring error" in str(exc.__cause__), exc
    assert mgr2._pending_ring_error is None
    assert op2 == [
        "ring_free[0].synchronize",
        "consensus:ring slot 0 reuse for layer 0 generation 0",
    ], op2
    print(
        "F3 PASS: golden per-block op order, tp1 spends no submit, sticky "
        "pops once, ring bytes land intact"
    )


def f4_fence_off_identity(w):
    assert w._ring_row_count(geometry=None) == 2
    assert w._ring_row_count(geometry=_ring_geometry(w=w, e_chunk=3)) == 6
    handler, wrapper_logger, old_level = _attach_handler(w)
    saved = _save_ring_envs()
    _reset_fence_latches(w)
    bag = w._TEST_EXEC_BAG.moe
    real_fence = bag.kt_prefill_event_fence
    bag.kt_prefill_event_fence = 0
    calls = []
    real_min = w._all_tp_ranks_succeeded

    def _spy(value):
        calls.append(value)
        return real_min(value)

    w._all_tp_ranks_succeeded = _spy
    try:
        ctx = _make_ring_context(w=w)
        ctx._negotiate_event_fence()
        assert ctx._ring_geometry is None and ctx._event_fence_frozen is False
        assert calls == [False], calls
        # The window mode cache is untouched when the intent is off.
        assert ctx._staging_window_mode_cache is None
        assert not handler.records, handler.records
    finally:
        bag.kt_prefill_event_fence = real_fence
        w._all_tp_ranks_succeeded = real_min
        _restore_envs(saved)
        _reset_fence_latches(w)
        _detach_handler(wrapper_logger, handler, old_level)
    print(
        "F4 PASS: fence parameter off runs one intent MIN, keeps no "
        "geometry, and leaves the two-row allocation"
    )


def f5_ladder_and_min(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    saved = _save_ring_envs()
    _reset_fence_latches(w)
    real_sources = w._staging_capacity_sources
    real_min_int = w._tp_int_min_all_reduce
    ctx = _make_ring_context(w=w)
    try:
        w._staging_capacity_sources = lambda: None
        assert (
            ctx._ring_chunk_candidates(requested=64, bank_nbytes=BANK_NBYTES)
            == 1
        )
        huge = (1 << 40, 1 << 40, 1 << 40)
        w._staging_capacity_sources = lambda: huge
        assert (
            ctx._ring_chunk_candidates(requested=64, bank_nbytes=BANK_NBYTES)
            == 64
        )
        assert (
            ctx._ring_chunk_candidates(requested=32, bank_nbytes=BANK_NBYTES)
            == 32
        )
        # Totals for 64/32/16 experts are 2*x*112; fail the top two rungs.
        w._staging_capacity_sources = lambda: (5000, 9000, 9000)
        assert (
            ctx._ring_chunk_candidates(requested=64, bank_nbytes=BANK_NBYTES)
            == 16
        )
        w._staging_capacity_sources = lambda: (2000, 2000, 2000)
        assert (
            ctx._ring_chunk_candidates(requested=64, bank_nbytes=BANK_NBYTES)
            == 1
        )

        # A peer MIN negotiates the local ladder step down with one warning.
        handler.records.clear()
        w._staging_capacity_sources = lambda: huge
        min_calls = []

        def _forced_min(value):
            min_calls.append(value)
            return 16

        w._tp_int_min_all_reduce = _forced_min
        # argparse choices reject invalid chunk values at startup, and the
        # default-on fence intent needs no setup here.
        assert w._TEST_EXEC_BAG.moe.kt_prefill_event_fence == 1
        ctx = _make_ring_context(w=w)
        ctx._negotiate_event_fence()
        assert ctx._event_fence_frozen is True
        geometry = ctx._ring_geometry
        assert geometry.e_chunk == 16 and geometry.ring_rows == 32
        assert geometry.num_slots == 2 and geometry.num_experts == 8
        assert geometry.bank_expert_nbytes == BANK_NBYTES
        assert geometry.total_nbytes == 32 * sum(BANK_NBYTES)
        assert geometry.debug_rows is False
        assert min_calls == [64], min_calls
        hits = _warnings(handler=handler, needle="negotiated down")
        assert len(hits) == 1, handler.records
        assert "64" in hits[0].getMessage() and "16" in hits[0].getMessage()
    finally:
        w._staging_capacity_sources = real_sources
        w._tp_int_min_all_reduce = real_min_int
        _restore_envs(saved)
        _reset_fence_latches(w)
        _detach_handler(wrapper_logger, handler, old_level)
    print(
        "F5 PASS: ladder floor on probe failure, rung skipping, peer MIN "
        "negotiates down with one line"
    )


def f7_debug_digest(w):
    handler, wrapper_logger, old_level = _attach_handler(w, logging.DEBUG)
    saved = _save_ring_envs()
    _reset_fence_latches(w)
    real_sources = w._staging_capacity_sources
    real_any = w._any_tp_rank_true
    real_dist = w.dist
    real_wsize = w.get_tensor_model_parallel_world_size
    real_group = w.get_tp_group
    try:
        w._staging_capacity_sources = lambda: (1 << 40, 1 << 40, 1 << 40)
        max_calls = []

        def _forced_max(value):
            max_calls.append(value)
            return True

        w._any_tp_rank_true = _forced_max
        # Default-on fence intent; local debug off, the all-rank MAX
        # forces it on.
        assert w._TEST_EXEC_BAG.moe.kt_prefill_event_fence == 1
        assert w._TEST_EXEC_BAG.moe.kt_prefill_fence_debug == 0
        ctx = _make_ring_context(w=w)
        ctx._negotiate_event_fence()
        assert ctx._ring_geometry.debug_rows is True
        assert max_calls == [False], max_calls
        w._any_tp_rank_true = real_any

        # Single-rank digest: exactly one debug line per block, advisory.
        geometry = _ring_geometry(w=w, e_chunk=2, debug_rows=True)
        mgr, method, op_log, buffers, dests, weight_infos = _make_ring_manager(
            w=w, geometry=geometry
        )
        with _cuda_stream_patch():
            err = mgr._load_ring_cpu_experts(
                method=method,
                cpu_expert_ids=[0, 1, 2, 3, 4],
                weight_infos=weight_infos,
                layer_idx=0,
                stats_chunk=None,
                geometry=geometry,
            )
        assert err is None, err
        debugs = [
            r
            for r in handler.records
            if r.levelno == logging.DEBUG and "head digest" in r.getMessage()
        ]
        assert len(debugs) == 3, handler.records

        # A cross-rank mismatch parks a sticky error in the ring channel.
        w.dist = types.SimpleNamespace(
            is_initialized=lambda: True,
            all_gather_object=lambda lst, obj, group=None: (
                lst.__setitem__(0, obj),
                lst.__setitem__(1, "f" * 64),
            ),
        )
        w.get_tensor_model_parallel_world_size = lambda: 2
        w.get_tp_group = lambda: types.SimpleNamespace(cpu_group=None)
        mgr._ring_debug_row(ring_slot=0, geometry=geometry)
        assert isinstance(mgr._pending_ring_error, RuntimeError)
        assert "mismatch" in str(mgr._pending_ring_error)
    finally:
        w._staging_capacity_sources = real_sources
        w._any_tp_rank_true = real_any
        w.dist = real_dist
        w.get_tensor_model_parallel_world_size = real_wsize
        w.get_tp_group = real_group
        _restore_envs(saved)
        _reset_fence_latches(w)
        _detach_handler(wrapper_logger, handler, old_level)
    print(
        "F7 PASS: debug MAX freeze, one digest per block, mismatch parks "
        "the sticky channel"
    )


def f8_chunk_zero_and_window_mutex(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    saved = _save_ring_envs()
    _reset_fence_latches(w)
    bag = w._TEST_EXEC_BAG.moe
    real_min = w._all_tp_ranks_succeeded
    real_chunk = bag.kt_prefill_stage_chunk_experts
    try:
        calls = []

        def _spy(value):
            calls.append(value)
            return real_min(value)

        w._all_tp_ranks_succeeded = _spy
        # chunk 0 folds the intent off before the MIN runs.
        assert bag.kt_prefill_event_fence == 1
        bag.kt_prefill_stage_chunk_experts = 0
        ctx = _make_ring_context(w=w)
        ctx._negotiate_event_fence()
        assert ctx._event_fence_frozen is False and ctx._ring_geometry is None
        assert calls == [False], calls

        # A live staging window freezes the fence off with one warning.
        handler.records.clear()
        bag.kt_prefill_stage_chunk_experts = 64
        os.environ[ENV_WINDOW] = "1"
        ctx = _make_ring_context(w=w)
        ctx._negotiate_event_fence()
        assert ctx._event_fence_frozen is False and ctx._ring_geometry is None
        assert calls == [False, True], calls
        hits = _warnings(handler=handler, needle="is inert while")
        assert len(hits) == 1, handler.records
        os.environ.pop(ENV_WINDOW)
    finally:
        bag.kt_prefill_stage_chunk_experts = real_chunk
        w._all_tp_ranks_succeeded = real_min
        _restore_envs(saved)
        _reset_fence_latches(w)
        _detach_handler(wrapper_logger, handler, old_level)
    print(
        "F8 PASS: chunk 0 folds the intent off; the window mutex freezes "
        "the fence off with one warning"
    )


def main():
    wrapper = _load_wrapper()
    f1_ring_blocks(w=wrapper)
    f2_consensus_count(w=wrapper)
    f3_op_order_and_bytes(w=wrapper)
    f4_fence_off_identity(w=wrapper)
    f5_ladder_and_min(w=wrapper)
    f7_debug_digest(w=wrapper)
    f8_chunk_zero_and_window_mutex(w=wrapper)
    print("ALL PASS: kt_event_fence_ring_test (F1-F8)")


if __name__ == "__main__":
    main()
