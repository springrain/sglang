"""CPU-only unit checks for the MXFP4 batch-DMA planner and staging windows.

Loads kt_ep_wrapper.py with its sglang.srt dependencies stubbed, so no
sglang install, GPU, or kt_kernel_ext build is required:

  N1  Run-merge planner: holes always break runs, run popcount equals the
      row budget, and the merge-off planner replays the legacy per-expert
      dst-major 4-bank order.
  N2  Staging geometry builder: the stride-helper equality, 16B alignment,
      undersized-bank warn-once, and tiling assertions all fire; expert
      count stays parameterized; bf16 scale tables match the hand formula.
  N3  Coverage guard: a dropped entry raises before any copy and the guard
      keeps its hands off every sticky error channel (structural).
  N4  w0 bit-equivalence: the flag toggles as a bool, and with merging off
      the plan is the legacy sequence (dst, bank) ascending, byte for byte.
  N5  w1 integration: op log order, window value bit-compare including
      untouched GPU-resident holes, one host sync, one event record, and
      the TP1 log without submits.
  N6  w2 parity: window alternation, generation-2 reuse check raising
      before any submit, and abort_round leaving generation and the
      release table un-rewound.
  N7  Twin managers: script-filtered op logs match line for line, TP0
      submit failure and rank-1 consensus failure abort at the same phase,
      and generation stays equal across both sides after abort.
  N8  Capacity predicate: pure four-input AND at the boundaries and free
      of env/CUDA/distributed reads (source audit).
  N9  Consumed-fence order: dynamic-update sequencing, exception path
      recording exactly once, record failure stream-sync fallback plus
      sticky, and the next-layer transport-setup pop ordering (source
      audit).
  N10 Assert W: unfreed same-epoch window reuse warns once, level 1
      returns an error object, level 0 is fully inert, and the A/B skip
      note logs exactly once; orthogonal to the WINDOW env (source audit).
  N11 Env semantics: WINDOW parse gate, cross-rank MIN negotiation with
      the "negotiated down" warning, restart-required drift warn-once,
      BATCH_DMA invalid gate, the dual-slot host-write batch assertion,
      and the debug digest hook (debug line per window, mismatch to the
      sticky channel).

Run: python third_party/sglang/test/manual/kt_batch_dma_window_test.py
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

ENV_BATCH_DMA = "SGLANG_KT_PREFILL_BATCH_DMA"
ENV_STAGE_WINDOW = "SGLANG_KT_PREFILL_STAGE_LAYER_WINDOW"


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
    _stub(
        "sglang.srt.runtime_context",
        get_exec=lambda: None,
        get_schedule=lambda: None,
    )
    _stub(
        "sglang.srt.utils", get_compiler_backend=lambda: "eager", is_cuda=lambda: False
    )
    return _load_module("kt_ep_wrapper_iso", WRAPPER_PATH)


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _warnings(handler, needle):
    return [
        r
        for r in handler.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


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


class _FakeEvent:
    """Stands in for torch.cuda.Event; every call lands on the op log."""

    def __init__(self, op_log, name, raise_on_record=False):
        self.op_log = op_log
        self.name = name
        self.raise_on_record = raise_on_record
        self.record_count = 0

    def synchronize(self):
        self.op_log.append(f"{self.name}.synchronize")

    def record(self, stream):
        self.record_count += 1
        if self.raise_on_record:
            raise ValueError("record failed")
        self.op_log.append(f"{self.name}.record")


class _FakeStream:
    def __init__(self, op_log, name):
        self.op_log = op_log
        self.name = name

    def wait_event(self, event):
        pass

    def synchronize(self):
        self.op_log.append(f"{self.name}.synchronize")


class _FakeCpuInfer:
    """One submit op per expert, one sync op per window."""

    def __init__(self, op_log, raise_on_submit=False):
        self.op_log = op_log
        self.raise_on_submit = raise_on_submit

    def submit_write_weight_scale_to_buffer(self, *args):
        self.op_log.append("submit")
        if self.raise_on_submit:
            raise ValueError("submit failed")

    def sync_write_weight_scale_to_buffer(self):
        self.op_log.append("sync")


def _raw_names(w):
    return w._Mxfp4PrefillSlot.RAW_NAMES


def _geometry(w, num_windows, expert_nbytes=64, merge=True, debug_rows=False):
    # Synthetic geometry: the num_experts=8 toy layout keeps every literal
    # far away from production shapes.
    num_experts = 8
    bank_count = len(_raw_names(w))
    strides = tuple(num_experts * expert_nbytes for _ in range(bank_count))
    return w._StagingWindowGeometry(
        mode=num_windows,
        num_windows=num_windows,
        num_experts=num_experts,
        bank_strides=strides,
        bank_expert_nbytes=tuple(expert_nbytes for _ in range(bank_count)),
        bank_merge_ok=tuple(merge for _ in range(bank_count)),
        total_nbytes=num_windows * sum(strides),
        debug_rows=debug_rows,
    )


def _staging_buffers(w, num_windows, shape=(6, 4)):
    rows = num_windows * 8
    buffers = {}
    for name in _raw_names(w):
        buf = torch.empty((rows,) + shape, dtype=torch.uint8)
        for row in range(rows):
            buf[row].fill_((row * 29 + 3) % 211)
        buffers[name] = buf
    return buffers


def _make_window_manager(w, geometry, tp_rank=0, level=2):
    """Bypass __init__ (needs CUDA streams); install window-mode state."""
    mgr = w._Mxfp4LayerwisePrefillManager.__new__(w._Mxfp4LayerwisePrefillManager)
    op_log = []
    n = geometry.num_windows
    mgr._window_generation = 0
    mgr._window_free_events = [
        _FakeEvent(op_log, "window_free") for _ in range(n)
    ]
    mgr._window_was_used = [True] * n
    mgr._window_release_consensus_gen = [0] * n
    mgr._window_owner = [None] * n
    mgr._window_freed = [True] * n
    mgr._host_slot_owner = [None, None]
    mgr._host_slot_freed = [True, True]
    mgr._window_skip_note_logged = False
    mgr._pending_window_error = None
    mgr._pending_fence_error = None
    mgr._ownership_level = level
    mgr._ownership_warned = set()
    mgr.epoch = 3
    mgr._h2d_planner = w._Mxfp4H2DBatchPlanner(
        bank_count=len(_raw_names(w)), bank_merge_ok=geometry.bank_merge_ok
    )
    mgr._stats_chunk = None
    mgr._stats_span_layer = None
    mgr.context = types.SimpleNamespace(
        _staging_geometry=geometry,
        staging_buffers=_staging_buffers(w, geometry.num_windows),
        all_rank_staging_ptrs={
            name: [4096] for name in _raw_names(w)
        },
    )
    mgr.transfer_stream = _FakeStream(op_log, "transfer")
    real_commit = w._Mxfp4LayerwisePrefillManager._commit_tp_device_runtime_phase

    def commit_device(local_error, phase):
        op_log.append("consensus:" + phase)
        return real_commit(mgr, local_error, phase)

    mgr._commit_tp_device_runtime_phase = commit_device
    real_ownership = w._Mxfp4LayerwisePrefillManager._ownership_record_window

    def ownership(layer_idx, generation, win):
        op_log.append("ownershipW")
        return real_ownership(
            mgr, layer_idx=layer_idx, generation=generation, win=win
        )

    mgr._ownership_record_window = ownership
    return mgr, op_log


class _cuda_stream_patch:
    """CPU torch has no usable CUDA stream context; swap in a nullcontext."""

    def __enter__(self):
        self.saved = torch.cuda.stream
        torch.cuda.stream = lambda stream: contextlib.nullcontext()

    def __exit__(self, *exc):
        torch.cuda.stream = self.saved
        return False


def n1_planner_runs(w):
    merge = w._Mxfp4H2DBatchPlanner(
        bank_count=4, bank_merge_ok=(True, True, True, True)
    )
    idle = w._Mxfp4H2DBatchPlanner(
        bank_count=4, bank_merge_ok=(False, False, False, False)
    )
    assert merge.plan_layer([]) == []
    assert idle.plan_layer([]) == []

    # Fully consecutive ids merge into one run per bank.
    pairs = [(row, row) for row in range(6)]
    entries = merge.plan_layer(pairs)
    assert len(entries) == 4, entries
    assert all(e.rows == 6 and e.src_row == 0 and e.dst_row == 0 for e in entries)
    assert [(e.dst_row, e.bank) for e in entries] == [(0, b) for b in range(4)]

    # A GPU-resident hole (missing dst) breaks every run; no run spans it.
    hole_pairs = [(0, 0), (1, 1), (2, 2), (3, 4), (4, 5)]
    entries = merge.plan_layer(hole_pairs)
    by_bank = {b: [] for b in range(4)}
    for e in entries:
        by_bank[e.bank].append((e.src_row, e.dst_row, e.rows))
    for runs in by_bank.values():
        assert runs == [(0, 0, 3), (3, 4, 2)], runs
    covered = set()
    for e in entries:
        covered.update(range(e.dst_row, e.dst_row + e.rows))
    assert covered == {0, 1, 2, 4, 5} and 3 not in covered

    # Alternating ids are all singletons; the union still covers every row.
    alt = [(row, 2 * row) for row in range(4)]
    entries = merge.plan_layer(alt)
    assert len(entries) == 16 and all(e.rows == 1 for e in entries)
    dst_counts = {}
    for e in entries:
        dst_counts[e.dst_row] = dst_counts.get(e.dst_row, 0) + 1
    assert dst_counts == {0: 4, 2: 4, 4: 4, 6: 4}
    print("N1 PASS: runs merge only across contiguous src+dst; holes break runs")


def n2_geometry_builder(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    saved_probe = w._staging_capacity_sources
    w._staging_capacity_sources = lambda: (2**62, 2**62, 2**62)
    try:
        gpu_layer = types.SimpleNamespace(
            num_experts=8,
            w13_weight=torch.zeros(8, 512, 512, dtype=torch.uint8),
            w13_weight_scale_inv=torch.zeros(8, 512, 512, dtype=torch.float32),
            w2_weight=torch.zeros(8, 1024, 512, dtype=torch.uint8),
            w2_weight_scale_inv=torch.zeros(8, 512, 512, dtype=torch.float32),
        )
        ctx = w.SharedFullContext.__new__(w.SharedFullContext)
        ctx.gpu_layer = gpu_layer
        ctx._is_mxfp4_quant = True
        _n2_formula_and_params(w, handler=handler, ctx=ctx)
        _n2_hard_asserts(w, handler=handler, ctx=ctx, gpu_layer=gpu_layer)
    finally:
        w._staging_capacity_sources = saved_probe
        _detach_handler(wrapper_logger, handler, old_level)
    print("N2 PASS: builder hard asserts, bf16 formula, parameterized experts")


def _n2_formula_and_params(w, handler, ctx):
    names = _raw_names(w)
    gpu_layer = ctx.gpu_layer
    geometry = ctx._build_staging_geometry(2)
    num_experts = gpu_layer.num_experts
    total_rows = 2 * num_experts
    # Derived property: tiling, strides, and totals recomputed from
    # scratch, not copied out of the constructor.
    for index, name in enumerate(names):
        gpu_tensor = getattr(gpu_layer, name)
        per_expert_numel = gpu_tensor.numel() // num_experts
        buf_dtype = w._shm_bank_buf_dtype(True, name, gpu_tensor.dtype)
        expected = per_expert_numel * torch.empty(
            (), dtype=buf_dtype
        ).element_size()
        assert geometry.bank_expert_nbytes[index] == expected, name
        assert geometry.bank_strides[index] == num_experts * expected, name
        assert (
            2 * geometry.bank_strides[index]
            == total_rows * geometry.bank_expert_nbytes[index]
        ), name
    assert geometry.total_nbytes == 2 * sum(geometry.bank_strides)
    assert geometry.bank_merge_ok == (True, True, True, True)
    assert geometry.mode == 2 and geometry.num_windows == 2
    assert geometry.debug_rows is False
    assert not _warnings(handler, "run threshold")
    # bf16 scale tables: stored bf16 in SHM regardless of the f32 source.
    assert w._shm_bank_buf_dtype(True, names[1], torch.float32) is torch.bfloat16
    assert geometry.bank_expert_nbytes[1] == (512 * 512) * 2

    # Expert count stays parameterized end to end.
    gpu_layer_small = types.SimpleNamespace(
        num_experts=4,
        w13_weight=torch.zeros(4, 512, 512, dtype=torch.uint8),
        w13_weight_scale_inv=torch.zeros(4, 512, 128, dtype=torch.float32),
        w2_weight=torch.zeros(4, 512, 512, dtype=torch.uint8),
        w2_weight_scale_inv=torch.zeros(4, 512, 128, dtype=torch.float32),
    )
    ctx.gpu_layer = gpu_layer_small
    geometry4 = ctx._build_staging_geometry(1)
    assert geometry4.num_experts == 4
    assert geometry4.bank_strides[0] == 4 * geometry4.bank_expert_nbytes[0]


def _n2_hard_asserts(w, handler, ctx, gpu_layer):
    # Hard assert 1: the shared helper must reproduce the hand formula.
    ctx.gpu_layer = gpu_layer
    saved_nbytes = w._shm_bank_expert_nbytes

    def broken_nbytes(buf_dtype, expert_numel):
        return saved_nbytes(buf_dtype, expert_numel) + 16

    w._shm_bank_expert_nbytes = broken_nbytes
    try:
        try:
            ctx._build_staging_geometry(1)
            raise AssertionError("stride mismatch must raise")
        except RuntimeError as exc:
            assert "staging stride mismatch" in str(exc), exc
    finally:
        w._shm_bank_expert_nbytes = saved_nbytes

    # Hard assert 2: 16-byte alignment on every bank.
    gpu_layer_uneven = types.SimpleNamespace(
        num_experts=8,
        w13_weight=torch.zeros(8, 511, dtype=torch.uint8),
        w13_weight_scale_inv=torch.zeros(8, 512, 128, dtype=torch.float32),
        w2_weight=torch.zeros(8, 512, 512, dtype=torch.uint8),
        w2_weight_scale_inv=torch.zeros(8, 512, 128, dtype=torch.float32),
    )
    ctx.gpu_layer = gpu_layer_uneven
    try:
        ctx._build_staging_geometry(1)
        raise AssertionError("unaligned bank must raise")
    except RuntimeError as exc:
        assert "not 16-byte aligned" in str(exc), exc

    # Hard assert 3: undersized banks merge into one warning line
    # naming every bank, and stay unmerged.
    handler.records.clear()
    gpu_layer_tiny = types.SimpleNamespace(
        num_experts=8,
        w13_weight=torch.zeros(8, 4096, dtype=torch.uint8),
        w13_weight_scale_inv=torch.zeros(8, 4096, dtype=torch.float32),
        w2_weight=torch.zeros(8, 4096, dtype=torch.uint8),
        w2_weight_scale_inv=torch.zeros(8, 4096, dtype=torch.float32),
    )
    ctx.gpu_layer = gpu_layer_tiny
    geometry_tiny = ctx._build_staging_geometry(2)
    hits = _warnings(handler, "run threshold")
    assert len(hits) == 1, handler.records
    assert "w13_weight" in hits[0].getMessage()
    assert "w2_weight_scale_inv" in hits[0].getMessage()
    assert geometry_tiny.bank_merge_ok == (False,) * 4


def n3_coverage_guard(w):
    planner = w._Mxfp4H2DBatchPlanner(
        bank_count=4, bank_merge_ok=(True, True, True, True)
    )
    entries = planner.plan_layer([(0, 3)])
    w._assert_h2d_plan_coverage(
        entries, bank_count=4, expected_rows=1
    )
    broken = entries[1:]
    try:
        w._assert_h2d_plan_coverage(
            broken, bank_count=4, expected_rows=1
        )
        raise AssertionError("a dropped entry must raise")
    except RuntimeError as exc:
        assert "H2D plan covers" in str(exc), exc
    # The guard raises instead of touching a sticky error channel, so peer
    # ranks never wait on a host that died before its first copy.
    src = inspect.getsource(w._assert_h2d_plan_coverage)
    assert "_pending" not in src
    print("N3 PASS: coverage guard raises pre-copy, sticky channels untouched")


def n4_w0_bit_equivalence(w):
    saved = os.environ.pop(ENV_BATCH_DMA, None)
    try:
        assert w._kt_prefill_batch_dma_enabled() is False
        os.environ[ENV_BATCH_DMA] = "1"
        assert w._kt_prefill_batch_dma_enabled() is True
        os.environ[ENV_BATCH_DMA] = "0"
        assert w._kt_prefill_batch_dma_enabled() is False
    finally:
        if saved is None:
            os.environ.pop(ENV_BATCH_DMA, None)
        else:
            os.environ[ENV_BATCH_DMA] = saved

    # Merge-off plan must replay the legacy per-expert enqueue: for each
    # ascending (src, dst) row, the four banks in order, all singleton.
    idle = w._Mxfp4H2DBatchPlanner(
        bank_count=4, bank_merge_ok=(False, False, False, False)
    )
    pairs = [(0, 1), (1, 2), (2, 5), (3, 6)]
    entries = idle.plan_layer(pairs)
    expected = []
    for src_row, dst_row in pairs:
        for bank in range(4):
            expected.append((bank, src_row, dst_row, 1))
    got = [(e.bank, e.src_row, e.dst_row, e.rows) for e in entries]
    assert got == expected, got
    # Union coverage is exact: every (bank, expert) cell enqueued once.
    cells = {(e.bank, e.dst_row) for e in entries}
    assert len(cells) == 4 * len(pairs)
    # The planner is a pure function: no collective or CUDA read is hidden
    # behind the batch-dma switch.
    src = inspect.getsource(w._Mxfp4H2DBatchPlanner.plan_layer)
    assert "dist" not in src and "cuda" not in src and "environ" not in src
    print("N4 PASS: w0 plan is the legacy (dst, bank) order, collectives 0")


def _run_window_load(mgr, op_log, w, geometry, method, layer_idx, ids):
    destination = {
        name: torch.full((8, 6, 4), 0xAB, dtype=torch.uint8)
        for name in _raw_names(w)
    }
    weight_infos = [(name, None, destination[name]) for name in _raw_names(w)]
    with _cuda_stream_patch():
        err = mgr._load_window_cpu_experts(
            method=method,
            cpu_expert_ids=ids,
            weight_infos=weight_infos,
            layer_idx=layer_idx,
            stats_chunk=None,
            geometry=geometry,
        )
    return err, destination


def n5_w1_integration(w):
    geometry = _geometry(w, 1)
    mgr, op_log = _make_window_manager(w, geometry)
    method = types.SimpleNamespace(tp_rank=0, wrapper=_FakeCpuInfer(op_log))
    ids = [1, 3, 4, 6, 7]
    err, destination = _run_window_load(mgr, op_log, w, geometry, method, 5, ids)
    assert err is None

    staging = mgr.context.staging_buffers
    # Window 0 rows are laid out by position; GPU-resident experts (holes)
    # must keep their pre-filled destination bytes.
    holes = set(range(8)) - set(ids)
    for name in _raw_names(w):
        for position, expert_id in enumerate(ids):
            assert torch.equal(
                destination[name][expert_id], staging[name][position]
            ), (name, expert_id)
        for hole in holes:
            assert torch.all(destination[name][hole] == 0xAB), (name, hole)

    assert op_log == (
        ["window_free.synchronize",
         "consensus:staging window 0 reuse for layer 5",
         "ownershipW"]
        + ["submit"] * len(ids)
        + ["sync",
           "consensus:host writes for layer 5",
           "window_free.record"]
    ), op_log
    assert mgr._window_generation == 1
    assert mgr._window_release_consensus_gen == [0]

    # Non-TP0 ranks share the consensus spine but never touch the wrapper.
    mgr1, log1 = _make_window_manager(w, geometry)
    method1 = types.SimpleNamespace(tp_rank=1, wrapper=None)
    err, _ = _run_window_load(mgr1, log1, w, geometry, method1, 5, ids)
    assert err is None
    spine = [line for line in op_log if line not in ("submit", "sync")]
    assert log1 == spine, (log1, spine)
    print("N5 PASS: w1 op order, value bit-compare, TP1 spine without submits")


def n6_w2_parity(w):
    geometry = _geometry(w, 2)
    mgr, op_log = _make_window_manager(w, geometry)
    method = types.SimpleNamespace(tp_rank=0, wrapper=_FakeCpuInfer(op_log))
    ids = list(range(8))
    for layer_idx in (5, 6):
        err, _ = _run_window_load(mgr, op_log, w, geometry, method, layer_idx, ids)
        assert err is None
    # Generation 2 reuses window 0; its gen-0 reuse consensus is on record.
    err, _ = _run_window_load(mgr, op_log, w, geometry, method, 7, ids)
    assert err is None
    assert mgr._window_generation == 3
    assert mgr._window_release_consensus_gen == [2, 1]
    wins = [line for line in op_log if "reuse for layer" in line]
    assert wins == [
        "consensus:staging window 0 reuse for layer 5",
        "consensus:staging window 1 reuse for layer 6",
        "consensus:staging window 0 reuse for layer 7",
    ]

    err, _ = _run_window_load(mgr, op_log, w, geometry, method, 9, ids)
    assert err is None
    # Forge a release lag: the next window-0 reuse must raise before any
    # submit of that generation.
    mgr._window_release_consensus_gen[0] = 1
    op_log.clear()
    try:
        _run_window_load(mgr, op_log, w, geometry, method, 11, ids)
        raise AssertionError("release lag must raise")
    except RuntimeError as exc:
        assert "recycled at generation 4" in str(exc), exc
    assert op_log == [
        "window_free.synchronize",
        "consensus:staging window 0 reuse for layer 11",
    ], op_log

    # Abort is round-scoped: generation and the release table never rewind.
    mgr.round_active = False
    mgr._pending_window_error = RuntimeError("stale")
    mgr._pending_fence_error = RuntimeError("stale")
    assert mgr._window_generation == 5
    mgr.abort_round()
    assert mgr._window_generation == 5
    assert mgr._window_release_consensus_gen == [1, 3]
    assert mgr._pending_window_error is None and mgr._pending_fence_error is None
    assert mgr._window_owner == [None, None]
    assert mgr._window_freed == [True, True]
    print("N6 PASS: w2 alternation, gen-2 reuse check pre-submit, no rewind")


def n7_twin_managers(w):
    geometry = _geometry(w, 2)
    mgr0, log0 = _make_window_manager(w, geometry, tp_rank=0)
    mgr1, log1 = _make_window_manager(w, geometry, tp_rank=1)
    method0 = types.SimpleNamespace(tp_rank=0, wrapper=_FakeCpuInfer(log0))
    method1 = types.SimpleNamespace(tp_rank=1, wrapper=None)
    ids = list(range(8))
    err0, _ = _run_window_load(mgr0, log0, w, geometry, method0, 5, ids)
    err1, _ = _run_window_load(mgr1, log1, w, geometry, method1, 5, ids)
    assert err0 is None and err1 is None
    # The only asymmetric section is TP0's host packing; every consensus
    # line and its order must match across ranks.
    spine0 = [line for line in log0 if line not in ("submit", "sync")]
    assert spine0 == log1, (spine0, log1)
    assert mgr0._window_generation == mgr1._window_generation == 1

    # TP0 submit failure joins the sticky channel and aborts at consensus 2.
    mgr_f, log_f = _make_window_manager(w, geometry, tp_rank=0)
    failing = types.SimpleNamespace(
        tp_rank=0, wrapper=_FakeCpuInfer(log_f, raise_on_submit=True)
    )
    try:
        _run_window_load(mgr_f, log_f, w, geometry, failing, 5, ids)
        raise AssertionError("submit failure must abort via consensus")
    except RuntimeError as exc:
        assert "host writes for layer 5" in str(exc), exc
    assert log_f[-1] == "consensus:host writes for layer 5", log_f
    assert mgr_f._pending_window_error is None

    # The peer script-injects the same phase failure: both sides abort at
    # the identical consensus line, generation still equal after abort.
    mgr_g, log_g = _make_window_manager(w, geometry, tp_rank=1)
    real_commit = w._Mxfp4LayerwisePrefillManager._commit_tp_device_runtime_phase

    def failing_commit(local_error, phase):
        log_g.append("consensus:" + phase)
        if "host writes" in phase:
            raise RuntimeError(f"MXFP4 {phase} failed on at least one TP rank")
        return real_commit(mgr_g, local_error, phase)

    mgr_g._commit_tp_device_runtime_phase = failing_commit
    try:
        _run_window_load(
            mgr_g, log_g, w, geometry,
            types.SimpleNamespace(tp_rank=1, wrapper=None), 5, ids,
        )
        raise AssertionError("peer consensus failure must raise")
    except RuntimeError as exc:
        assert "host writes for layer 5" in str(exc), exc
    assert log_g[-1] == "consensus:host writes for layer 5", log_g
    mgr_f.round_active = False
    mgr_g.round_active = False
    mgr_f.abort_round()
    mgr_g.abort_round()
    assert mgr_f._window_generation == mgr_g._window_generation == 1
    print("N7 PASS: twin spines equal; both failure modes abort at one phase")


def n8_capacity_predicate(w):
    budget = w._MIN_RUN_ENTRY_BYTES
    assert w._staging_window_capacity_ok(budget, budget, budget, budget) is True
    assert w._staging_window_capacity_ok(budget, budget - 1, budget, budget) is False
    assert w._staging_window_capacity_ok(budget, budget, budget - 1, budget) is False
    assert w._staging_window_capacity_ok(budget, budget, budget, budget - 1) is False
    assert w._staging_window_capacity_ok(budget, 0, 2 * budget, 2 * budget) is False
    # Source audit: the predicate must stay a pure four-input function; an
    # env or CUDA read here would fork behavior off the negotiated decision.
    src = inspect.getsource(w._staging_window_capacity_ok)
    assert "environ" not in src and "cuda" not in src and "dist" not in src
    print("N8 PASS: capacity predicate is pure AND at every shortfall edge")


def _make_apply_manager(w, dynamic, op_log, slot):
    mgr = w._Mxfp4LayerwisePrefillManager.__new__(w._Mxfp4LayerwisePrefillManager)
    mgr._pending_fence_error = None
    mgr.epoch = 3
    mgr.device = None
    mgr._stats_chunk = None
    mgr._stats_span_layer = None
    mgr.current_slot_index = None
    mgr.context = types.SimpleNamespace(
        gpu_method=types.SimpleNamespace(
            apply=lambda gpu_layer, dispatch_output: "result"
        ),
        gpu_layer=None,
    )
    mgr._acquire = lambda layer_idx, method, layer: (slot, False)
    mgr._bind_slot = lambda slot_: None
    mgr._record_prepared_backing_on_stream = lambda slot_, stream: None
    mgr._prefetch_successor = lambda slot_: op_log.append("prefetch_start")
    mgr.successor_layer_idx = lambda layer_idx: None
    mgr._stats_finalize_chunk = lambda: None
    real_commit = w._Mxfp4LayerwisePrefillManager._commit_tp_runtime_phase

    def commit(local_error, phase):
        if "compute launch" in phase:
            op_log.append("compute_commit")
        else:
            op_log.append("update_commit")
        return real_commit(mgr, local_error, phase)

    mgr._commit_tp_runtime_phase = commit
    config = types.SimpleNamespace(
        layer_idx=5, kt_enable_dynamic_expert_update=dynamic
    )
    method = types.SimpleNamespace(
        kt_config=config,
        tp_rank=1,
        _kt_stats_pending_churn=None,
        _update_gpu_experts_from_batch=lambda **kwargs: op_log.append(
            "hot_update_reads"
        ),
    )
    return mgr, method


class _torch_cuda_patch:
    def __init__(self, op_log, main_stream):
        self.op_log = op_log
        self.main_stream = main_stream

    def __enter__(self):
        self.saved_stream = torch.cuda.current_stream
        self.saved_sync = torch.cuda.synchronize
        torch.cuda.current_stream = lambda device=None: self.main_stream
        torch.cuda.synchronize = lambda device=None: self.op_log.append(
            "update_sync"
        )

    def __exit__(self, *exc):
        torch.cuda.current_stream = self.saved_stream
        torch.cuda.synchronize = self.saved_sync
        return False


def _apply_slot(w, op_log, raise_on_record=False):
    return types.SimpleNamespace(
        index=0,
        layer_idx=5,
        epoch=3,
        reuse_guard="ready",
        state="LOADED",
        has_consumed_event=False,
        consumed_event=_FakeEvent(
            op_log, "consumed", raise_on_record=raise_on_record
        ),
        ready_event=object(),
        **{name: torch.zeros(1) for name in _raw_names(w)},
    )


def n9_consumed_fence_order(w):
    # Dynamic update on: the fence is strictly after the update commit and
    # strictly before successor prefetch.
    op_log = []
    main_stream = _FakeStream(op_log, "main")
    slot = _apply_slot(w, op_log)
    mgr, method = _make_apply_manager(w, True, op_log, slot)
    with _torch_cuda_patch(op_log, main_stream):
        result = mgr.apply(method, None, None)
    assert result == "result"
    assert op_log == [
        "compute_commit",
        "update_sync",
        "hot_update_reads",
        "update_commit",
        "consumed.record",
        "prefetch_start",
    ], op_log
    assert slot.reuse_guard == "consumed" and slot.state == "IN_USE"
    assert slot.has_consumed_event is True
    assert mgr.current_slot_index == 0
    assert mgr._pending_fence_error is None

    # Dynamic update off: the fence still lands before prefetch. A real
    # stream object is required; apply waits on the ready event.
    op_log = []
    slot = _apply_slot(w, op_log)
    mgr, method = _make_apply_manager(w, False, op_log, slot)
    with _torch_cuda_patch(op_log, _FakeStream(op_log, "main")):
        mgr.apply(method, None, None)
    assert op_log == ["compute_commit", "consumed.record", "prefetch_start"]

    # Compute failure: the exception-path fence records exactly once, and
    # the guarded re-arm in the except block is a no-op.
    op_log = []
    slot = _apply_slot(w, op_log)
    mgr, method = _make_apply_manager(w, False, op_log, slot)

    def broken_apply(gpu_layer, dispatch_output):
        raise ValueError("boom")

    mgr.context.gpu_method = types.SimpleNamespace(apply=broken_apply)
    with _torch_cuda_patch(op_log, _FakeStream(op_log, "main")):
        try:
            mgr.apply(method, None, None)
            raise AssertionError("compute failure must surface")
        except RuntimeError as exc:
            assert "compute launch" in str(exc), exc
    assert slot.consumed_event.record_count == 1
    assert op_log == ["consumed.record", "compute_commit"], op_log

    # Fence failure: stream-sync fallback engages, the sticky slot carries
    # the cause, and abort_round clears it.
    op_log = []
    slot = _apply_slot(w, op_log, raise_on_record=True)
    mgr, method = _make_apply_manager(w, True, op_log, slot)
    with _torch_cuda_patch(op_log, _FakeStream(op_log, "main")):
        result = mgr.apply(method, None, None)
    assert result == "result"
    assert op_log[-2] == "main.synchronize", op_log
    assert op_log[-1] == "prefetch_start", op_log
    assert slot.reuse_guard == "synchronized"
    assert isinstance(mgr._pending_fence_error, ValueError)
    mgr.round_active = False
    mgr._host_slot_owner = [None, None]
    mgr._host_slot_freed = [True, True]
    mgr._window_owner = []
    mgr._window_freed = []
    mgr._pending_window_error = None
    mgr.abort_round()
    assert mgr._pending_fence_error is None

    # Source audit: the next-layer transport-setup consensus still pops the
    # sticky fence error before it commits.
    src = inspect.getsource(w._Mxfp4LayerwisePrefillManager._load_slot)
    pop_idx = src.find("setup_error = self._pending_fence_error")
    commit_idx = src.find("transport setup for layer")
    assert 0 < pop_idx < commit_idx, (pop_idx, commit_idx)
    print("N9 PASS: fence after update commit, before prefetch, once per slot")


def n10_assert_w(w):
    handler, wrapper_logger, old_level = _attach_handler(w, logging.DEBUG)
    try:
        geometry = _geometry(w, 2)
        mgr, _ = _make_window_manager(w, geometry, level=2)
        assert mgr._ownership_record_window(
            layer_idx=5, generation=0, win=0
        ) is None
        assert mgr._window_owner[0] == (3, 5, 0)
        assert mgr._window_freed[0] is False
        notes = [
            r for r in handler.records
            if r.levelno == logging.DEBUG
            and "host-slot sentinel inactive" in r.getMessage()
        ]
        assert len(notes) == 1, handler.records
        # Same-epoch overwrite without the free-sync fires W.
        assert mgr._ownership_record_window(
            layer_idx=7, generation=1, win=0
        ) is None
        hits = _warnings(handler, "invariant W")
        assert len(hits) == 1 and "generation=0" in hits[0].getMessage(), (
            handler.records
        )
        # The suspect coordinates are never written back: the same dedup
        # key (code, layer, slot, epoch) stays silent, and a different
        # layer key fires exactly one new line.
        mgr._ownership_record_window(layer_idx=7, generation=2, win=0)
        mgr._ownership_record_window(layer_idx=7, generation=3, win=0)
        assert len(_warnings(handler, "invariant W")) == 1, handler.records
        mgr._ownership_record_window(layer_idx=9, generation=4, win=0)
        hits = _warnings(handler, "invariant W")
        assert len(hits) == 2 and "generation=0" in hits[1].getMessage(), (
            handler.records
        )
        assert mgr._window_owner[0] == (3, 5, 0)
        assert mgr._window_freed[0] is False
        # A completed free-sync re-arms the window.
        mgr._window_freed[0] = True
        assert mgr._ownership_record_window(
            layer_idx=9, generation=5, win=0
        ) is None
        assert mgr._window_owner[0] == (3, 9, 5)
        # An epoch bump ages out the stale owner without a free-sync.
        mgr.epoch = 4
        assert mgr._ownership_record_window(
            layer_idx=6, generation=6, win=0
        ) is None
        assert mgr._window_owner[0] == (4, 6, 6)
        assert len(_warnings(handler, "invariant W")) == 2, handler.records
        # Level 1 returns the error object for the consensus channel; it
        # still warns once, and the live owner survives the violation.
        warns_before = len(_warnings(handler, "invariant W"))
        mgr_e, _ = _make_window_manager(w, geometry, level=1)
        mgr_e._ownership_record_window(layer_idx=5, generation=0, win=1)
        err = mgr_e._ownership_record_window(
            layer_idx=7, generation=1, win=1
        )
        assert isinstance(err, RuntimeError) and "invariant W" in str(err), err
        assert mgr_e._window_owner[1] == (3, 5, 0)
        assert len(_warnings(handler, "invariant W")) == warns_before + 1
        # Level 0 is fully inert: no owner, no shadow flag, no skip note.
        handler.records.clear()
        mgr_z, _ = _make_window_manager(w, geometry, level=0)
        assert mgr_z._ownership_record_window(
            layer_idx=5, generation=0, win=0
        ) is None
        assert mgr_z._window_owner[0] is None
        assert mgr_z._window_freed[0] is True
        assert mgr_z._window_skip_note_logged is False
        assert not handler.records
        # Orthogonal to the WINDOW env: the record reads no env at all.
        src = inspect.getsource(
            w._Mxfp4LayerwisePrefillManager._ownership_record_window
        )
        assert "environ" not in src and "STAGE_LAYER_WINDOW" not in src
    finally:
        _detach_handler(wrapper_logger, handler, old_level)
    print("N10 PASS: assert W fires once per key; level gate and skip note")


def n11_env_semantics(w):
    handler, wrapper_logger, old_level = _attach_handler(w, logging.DEBUG)
    saved_window = os.environ.pop(ENV_STAGE_WINDOW, None)
    saved_dma = os.environ.pop(ENV_BATCH_DMA, None)
    saved_dist = w.dist
    saved_world = w.get_tensor_model_parallel_world_size
    saved_group = w.get_tp_group
    try:
        _n11_env_gates(w, handler=handler)
        _n11_min_negotiation(
            w,
            handler=handler,
            saved_dist=saved_dist,
            saved_world=saved_world,
            saved_group=saved_group,
        )
        _n11_write_guards_and_digest(w, handler=handler)
    finally:
        w.dist = saved_dist
        w.get_tensor_model_parallel_world_size = saved_world
        w.get_tp_group = saved_group
        w._stage_window_env_warned = False
        w._stage_window_restart_warned = False
        w._batch_dma_env_warned = False
        if saved_window is None:
            os.environ.pop(ENV_STAGE_WINDOW, None)
        else:
            os.environ[ENV_STAGE_WINDOW] = saved_window
        if saved_dma is None:
            os.environ.pop(ENV_BATCH_DMA, None)
        else:
            os.environ[ENV_BATCH_DMA] = saved_dma
        _detach_handler(wrapper_logger, handler, old_level)
    print("N11 PASS: env gates, MIN negotiation, drift, dual-slot, digest")


def _n11_env_gates(w, handler):
    # WINDOW parse gate.
    assert w._stage_window_env_mode() == 0
    os.environ[ENV_STAGE_WINDOW] = "1"
    assert w._stage_window_env_mode() == 1
    os.environ[ENV_STAGE_WINDOW] = "2"
    assert w._stage_window_env_mode() == 2
    w._stage_window_env_warned = False
    os.environ[ENV_STAGE_WINDOW] = "9"
    assert w._stage_window_env_mode() == 0
    assert w._stage_window_env_mode() == 0
    hits = _warnings(handler, "Invalid SGLANG_KT_PREFILL_STAGE_LAYER_WINDOW")
    assert len(hits) == 1, handler.records

    # Restart-required drift: the frozen MIN value wins, warned once.
    w._stage_window_restart_warned = False
    os.environ[ENV_STAGE_WINDOW] = "2"
    w._check_stage_window_drift(1)
    w._check_stage_window_drift(1)
    hits = _warnings(handler, "stays in effect until restart")
    assert len(hits) == 1 and "from 1 to 2" in hits[0].getMessage(), (
        handler.records
    )
    w._check_stage_window_drift(2)
    assert len(_warnings(handler, "until restart")) == 1, handler.records

    # BATCH_DMA invalid value warns once and disables.
    w._batch_dma_env_warned = False
    os.environ[ENV_BATCH_DMA] = "9"
    assert w._kt_prefill_batch_dma_enabled() is False
    assert w._kt_prefill_batch_dma_enabled() is False
    hits = _warnings(handler, "Invalid SGLANG_KT_PREFILL_BATCH_DMA")
    assert len(hits) == 1, handler.records


def _n11_min_negotiation(w, handler, saved_dist, saved_world, saved_group):
    # Cross-rank MIN negotiation with a scripted peer at mode 1.
    handler.records.clear()
    ctx = w.SharedFullContext.__new__(w.SharedFullContext)
    w.dist = types.SimpleNamespace(
        is_initialized=lambda: True,
        all_reduce=lambda tensor, op=None, group=None: tensor.fill_(
            min(int(tensor.item()), 1)
        ),
        ReduceOp=types.SimpleNamespace(MIN=object()),
    )
    w.get_tensor_model_parallel_world_size = lambda: 2
    w.get_tp_group = lambda: types.SimpleNamespace(cpu_group=object())
    os.environ[ENV_STAGE_WINDOW] = "2"
    assert ctx._negotiate_staging_window_mode() == 1
    hits = _warnings(handler, "negotiated down")
    assert len(hits) == 1, handler.records
    handler.records.clear()
    os.environ[ENV_STAGE_WINDOW] = "1"
    assert ctx._negotiate_staging_window_mode() == 1
    assert not _warnings(handler, "negotiated down")
    # Without a process group the local value stands as-is.
    w.dist = saved_dist
    w.get_tensor_model_parallel_world_size = saved_world
    w.get_tp_group = saved_group
    assert ctx._negotiate_staging_window_mode() == 1
    os.environ[ENV_STAGE_WINDOW] = "2"
    assert ctx._negotiate_staging_window_mode() == 2


def _n11_write_guards_and_digest(w, handler):
    # Dual-slot guard: batched host writes must not run without
    # staging-window geometry.
    try:
        w._Mxfp4LayerwisePrefillManager._submit_window_writes(
            types.SimpleNamespace(),
            method=None,
            cpu_expert_ids=[],
            win=0,
            geometry=None,
        )
        raise AssertionError("geometry-less batch host write must raise")
    except AssertionError as exc:
        assert "staging-window geometry" in str(exc), exc

    # Debug digest: one debug line per observation without a process
    # group; a scripted mismatch lands on the sticky channel.
    geometry = _geometry(w, 2, debug_rows=True)
    mgr, _ = _make_window_manager(w, geometry)
    mgr._window_debug_row(win=0, geometry=geometry)
    mgr._window_debug_row(win=1, geometry=geometry)
    mgr._window_debug_row(win=0, geometry=geometry)
    digests = [
        r for r in handler.records
        if r.levelno == logging.DEBUG and "head digest" in r.getMessage()
    ]
    assert len(digests) == 3, handler.records

    def gather(lst, obj, group=None):
        lst[0] = obj
        lst[1] = "tampered"

    w.dist = types.SimpleNamespace(
        is_initialized=lambda: True, all_gather_object=gather
    )
    w.get_tensor_model_parallel_world_size = lambda: 2
    w.get_tp_group = lambda: types.SimpleNamespace(cpu_group=object())
    mgr._window_debug_row(win=1, geometry=geometry)
    assert isinstance(mgr._pending_window_error, RuntimeError)
    assert "mismatch" in str(mgr._pending_window_error)


def main():
    wrapper = _load_wrapper()
    n1_planner_runs(wrapper)
    n2_geometry_builder(wrapper)
    n3_coverage_guard(wrapper)
    n4_w0_bit_equivalence(wrapper)
    n5_w1_integration(wrapper)
    n6_w2_parity(wrapper)
    n7_twin_managers(wrapper)
    n8_capacity_predicate(wrapper)
    n9_consumed_fence_order(wrapper)
    n10_assert_w(wrapper)
    n11_env_semantics(wrapper)
    print("ALL PASS: kt_batch_dma_window_test (N1-N11)")


if __name__ == "__main__":
    main()
