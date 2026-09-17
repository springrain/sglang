"""CPU-only unit checks for the per-rank pinned bank (--kt-direct-bank-dma).

Loads kt_bank_dma.py directly, kt_ep_wrapper.py with its sglang.srt
dependencies stubbed, and bench/bench_fp4_moe.py with kt_kernel stubbed, so
no sglang install, GPU, NCCL, or kt_kernel_ext build is required:

  D1  manifest validation — each broken key group names its own reason
      literal; the tile loader raises tile_sha256/tile_unreadable/
      tile_missing against real files in a temp bank directory.
  D2  capacity precheck — the RAM, memlock and NUMA legs gate
      independently; the /proc and /sys probes fail closed where absent.
  D3  row-byte golden tuples — the deliberately duplicated sglang/bench
      formulas agree on two shapes and on every ValueError message.
  D4  flag freeze — kt_direct_bank_dma=0 freezes off silently;
      default-on with a missing manifest, a mutex_window and
      no_weight_path all degrade with exactly one [KT-DEGRADE] line.
  D5  _RankBank row views alias the pinned storage; the H2D planner keeps
      the legacy 4-entry no-merge contract and coverage raises on a hole.
  D6  cross-rank manifest digests disagree -> the same RuntimeError on
      every rank; a peer error payload degrades instead of raising.
  D7  (deleted with the env->CLI migration: a frozen CLI parameter has
      no runtime drift channel for the check to observe)
  D8  the pack-side task_tag ABI probe mirrors the forward_task probe
      (stale doc raises, failure does not latch, escape valve bypasses).
  D9  DUMP syncs the raw-ready event exactly once before readback.

Run: python third_party/sglang/test/manual/kt_bank_dma_test.py
"""

import hashlib
import importlib.util
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER_PATH = (
    REPO_ROOT
    / "third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
)
BANK_DMA_PATH = (
    REPO_ROOT
    / "third_party/sglang/python/sglang/srt/layers/moe/kt_bank_dma.py"
)
BENCH_PATH = REPO_ROOT / "kt-kernel/bench/bench_fp4_moe.py"


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
    _stub(
        "sglang.srt.layers.moe",
        kt_bank_dma=_load_module(name="kt_bank_dma_iso", path=BANK_DMA_PATH),
    )
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
            kt_direct_bank_dma=1,
            kt_prefill_event_fence=1,
            kt_prefill_stage_chunk_experts=64,
            kt_prefill_no_device_sync=0,
            kt_prefill_fence_debug=0,
            kt_dump_slot_bytes=0,
            kt_bank_dma_batch=0,
            kt_bank_dma_lean=0,
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
    mod = _load_module(name="kt_ep_wrapper_iso", path=WRAPPER_PATH)
    mod._TEST_EXEC_BAG = exec_bag
    return mod


def _load_bench():
    """Exec bench_fp4_moe.py; an empty kt_kernel_ext stub leaves BACKENDS None."""
    _stub("kt_kernel", kt_kernel_ext=types.SimpleNamespace(moe=types.SimpleNamespace()))
    return _load_module(name="bench_fp4_moe_iso", path=BENCH_PATH)


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _attach_handler(owner, level=logging.WARNING):
    handler = _RecordHandler()
    owner_logger = logging.getLogger(owner.__name__)
    old_level = owner_logger.level
    owner_logger.setLevel(level)
    owner_logger.addHandler(handler)
    return handler, owner_logger, old_level


def _detach_handler(owner_logger, handler, old_level):
    owner_logger.removeHandler(handler)
    owner_logger.setLevel(old_level)


def _warnings(handler, needle):
    return [
        r
        for r in handler.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


def _geometry(bank, num_layers=1, num_experts=3):
    # Magic-number-free toy shape; every row byte count stays 16B aligned.
    return bank._BankGeometry(
        gpu_tp_count=2,
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=512,
        intermediate_size=128,
        group_size=32,
    )


def _row_bytes(bank, geometry):
    return bank.bank_field_row_bytes(
        hidden=geometry.hidden_size,
        inter=geometry.intermediate_size,
        group_size=geometry.group_size,
        tp_count=geometry.gpu_tp_count,
    )


def _manifest_payload(bank, geometry, rank=0):
    """A fully legal manifest dict; each test breaks exactly one piece."""
    groups = bank._bank_expected_groups(geometry=geometry)
    row_bytes = _row_bytes(bank=bank, geometry=geometry)
    fields = [
        {
            "field": field,
            "rows": geometry.num_experts,
            "nbytes": row_bytes[field],
            "sha256": "0" * 64,
        }
        for field in bank.BANK_FIELDS
    ]
    tiles = {}
    for layer in range(geometry.num_layers):
        name = bank.BANK_TILE_NAMING.format(layer=layer, rank=rank)
        tiles[name] = {"layer": layer, "rank": rank, "fields": fields}
    return {
        "layout_version": bank.BANK_LAYOUT_VERSION,
        "backend": "toy",
        "topo": groups["topo"],
        "dims": groups["dims"],
        "dtype": groups["dtype"],
        "map": groups["map"],
        "keys_sha256": {
            group: bank._canonical_json_sha256(groups[group])
            for group in bank.BANK_KEY_GROUPS
        },
        "weights_sha256": "a" * 64,
        "tiles": tiles,
    }


def _validate_reason(bank, geometry, payload, rank=0):
    manifest = bank._bank_manifest_from_dict(payload)
    return bank._bank_manifest_validate(
        manifest=manifest, geometry=geometry, rank=rank
    )


def d1_manifest_validate(bank):
    geometry = _geometry(bank=bank)
    tile_name = bank.BANK_TILE_NAMING.format(layer=0, rank=0)
    assert _validate_reason(bank=bank, geometry=geometry,
                            payload=_manifest_payload(bank=bank, geometry=geometry)
                            ) is None

    cases = []

    def _case(reason, mutate):
        def run():
            payload = _manifest_payload(bank=bank, geometry=geometry)
            mutate(payload)
            return _validate_reason(bank=bank, geometry=geometry, payload=payload)

        cases.append((reason, run))

    _case("layout_version", lambda p: p.__setitem__("layout_version", 9))
    _case("topo", lambda p: p["topo"].__setitem__("gpu_tp_count", 8))
    _case("dims", lambda p: p["dims"].__setitem__("num_experts", 7))
    _case("dtype", lambda p: p["dtype"].__setitem__("w2_weight", "bfloat16"))
    _case("map", lambda p: p["map"].__setitem__("rows_per_tile", 5))
    _case("weights_sha256_malformed",
          lambda p: p.__setitem__("weights_sha256", "z" * 64))
    _case("tile_missing", lambda p: p.__setitem__("tiles", {}))
    _case("tile_field_shape",
          lambda p: p["tiles"][tile_name].__setitem__(
              "fields", list(reversed(p["tiles"][tile_name]["fields"]))))
    _case("tile_field_shape",
          lambda p: p["tiles"][tile_name]["fields"][0].__setitem__("rows", 1))
    _case("tile_field_shape",
          lambda p: p["tiles"][tile_name]["fields"][0].__setitem__(
              "nbytes", p["tiles"][tile_name]["fields"][0]["nbytes"] + 16))
    for reason, run in cases:
        assert run() == reason, reason
    print("D1 PASS: each broken key group names its exact reason literal")


def d1b_tile_loader(bank):
    geometry = _geometry(bank=bank)
    row_bytes = _row_bytes(bank=bank, geometry=geometry)
    payloads = {
        field: bytes(
            (offset + index) % 251
            for index in range(geometry.num_experts * row_bytes[field])
        )
        for offset, field in enumerate(bank.BANK_FIELDS)
    }
    tile_name = bank.BANK_TILE_NAMING.format(layer=0, rank=0)

    def _real_manifest():
        payload = _manifest_payload(bank=bank, geometry=geometry)
        for entry in payload["tiles"][tile_name]["fields"]:
            entry["sha256"] = hashlib.sha256(
                payloads[entry["field"]]
            ).hexdigest()
        return payload

    with tempfile.TemporaryDirectory() as root:
        tile_dir = os.path.join(root, "bank")
        os.makedirs(tile_dir)
        tile_path = os.path.join(tile_dir, tile_name)
        with open(tile_path, "wb") as handle:
            for field in bank.BANK_FIELDS:
                handle.write(payloads[field])
        manifest = bank._bank_manifest_from_dict(_real_manifest())
        loaded = bank._bank_load_layer_bytes(
            root=root, manifest=manifest, layer=0, rank=0
        )
        assert loaded == payloads, "tile bytes round-trip byte-identical"

        with open(tile_path, "r+b") as handle:
            probe = handle.read(1)
            handle.seek(0)
            handle.write(bytes([probe[0] ^ 0xFF]))
        _expect_bank_error(
            bank=bank,
            fn=lambda: bank._bank_load_layer_bytes(
                root=root, manifest=manifest, layer=0, rank=0),
            reason="tile_sha256",
        )
        os.remove(tile_path)
        _expect_bank_error(
            bank=bank,
            fn=lambda: bank._bank_load_layer_bytes(
                root=root, manifest=manifest, layer=0, rank=0),
            reason="tile_unreadable",
        )
        empty = bank._bank_manifest_from_dict({"tiles": {}})
        _expect_bank_error(
            bank=bank,
            fn=lambda: bank._bank_load_layer_bytes(
                root=root, manifest=empty, layer=0, rank=0),
            reason="tile_missing",
        )
    print("D1b PASS: tile loader pins sha256/unreadable/missing failures")


def _expect_bank_error(bank, fn, reason):
    try:
        fn()
    except bank._BankDegradeError as exc:
        assert exc.reason == reason, (exc.reason, reason, exc.detail)
        return
    raise AssertionError(f"expected _BankDegradeError({reason})")


def d2_capacity_precheck(bank):
    check = bank._bank_capacity_precheck
    assert check(total_nbytes=99, mem_available=100,
                 memlock_limit=100, numa_free=None) is True
    assert check(total_nbytes=101, mem_available=100,
                 memlock_limit=10**6, numa_free=None) is False
    assert check(total_nbytes=101, mem_available=10**6,
                 memlock_limit=100, numa_free=None) is False
    # NUMA leg compares the SUM of per-node free bytes (Linux spills
    # first-touch pages across nodes by default): 50+49=99 < 100 vetoes.
    assert check(total_nbytes=100, mem_available=100,
                 memlock_limit=100, numa_free=(50, 49)) is False
    assert check(total_nbytes=100, mem_available=100,
                 memlock_limit=100, numa_free=(60, 60)) is True
    # total=101: sums of 130/131 admit; a sum of 100 vetoes.
    assert check(total_nbytes=101, mem_available=10**6,
                 memlock_limit=10**6, numa_free=(50, 80)) is True
    assert check(total_nbytes=101, mem_available=10**6,
                 memlock_limit=10**6, numa_free=(51, 80)) is True
    assert check(total_nbytes=101, mem_available=10**6,
                 memlock_limit=10**6, numa_free=(20, 80)) is False
    # The discriminating case: one empty node, one rich node.  An
    # even-split or strictest-node rule would veto; the sum admits.
    assert check(total_nbytes=100, mem_available=10**6,
                 memlock_limit=10**6, numa_free=(0, 200)) is True
    # Elastic: probes are fail-closed, so None is always acceptable.
    sources = bank._bank_capacity_sources()
    if sources is not None:
        mem_available, memlock_limit = sources
        assert isinstance(mem_available, int) and mem_available > 0
        assert isinstance(memlock_limit, int) and memlock_limit > 0
    numa = bank._bank_numa_free_bytes()
    if numa is not None:
        assert numa and all(isinstance(free, int) and free >= 0 for free in numa)
    print("D2 PASS: RAM/memlock/NUMA legs gate independently; probes fail closed")


def _expect_value_error(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected ValueError")


def d3_row_bytes_golden(bank, bench):
    shapes = [
        (512, 128, 32, 2, {
            "w13_weight": 32768,
            "w13_weight_scale_inv": 4096,
            "w2_weight": 16384,
            "w2_weight_scale_inv": 2048,
        }),
        (768, 192, 64, 3, {
            "w13_weight": 49152,
            "w13_weight_scale_inv": 3072,
            "w2_weight": 24576,
            "w2_weight_scale_inv": 1536,
        }),
    ]
    for hidden, inter, group, tp, expected in shapes:
        sglang_rows = bank.bank_field_row_bytes(
            hidden=hidden, inter=inter, group_size=group, tp_count=tp)
        bench_rows = bench.bank_field_row_bytes(
            hidden=hidden, inter=inter, group_size=group, tp_count=tp)
        assert sglang_rows == bench_rows == expected
        assert all(nbytes % 16 == 0 for nbytes in expected.values())
    illegal = [
        (512, 130, 32, 4),   # inter not divisible by tp
        (510, 128, 32, 2),   # hidden not divisible by group
        (512, 96, 32, 2),    # inter/tp not divisible by group
        (4, 8, 2, 4),        # divisibility holds but rows are 16B misaligned
    ]
    for hidden, inter, group, tp in illegal:
        sglang_msg = _expect_value_error(
            fn=lambda: bank.bank_field_row_bytes(
                hidden=hidden, inter=inter, group_size=group, tp_count=tp))
        bench_msg = _expect_value_error(
            fn=lambda: bench.bank_field_row_bytes(
                hidden=hidden, inter=inter, group_size=group, tp_count=tp))
        assert sglang_msg == bench_msg, (sglang_msg, bench_msg)
    print("D3 PASS: dual-source formulas agree on tuples and error text")


def _make_bank_context(w, staging=None, weight_path=None):
    """Bypass __init__ (needs CUDA streams); install only sentinel state."""
    ctx = w.SharedFullContext.__new__(w.SharedFullContext)
    ctx._rank_bank = None
    ctx._bank_dma_frozen = None
    ctx._staging_geometry = staging
    ctx._kt_config = types.SimpleNamespace(
        weight_path=weight_path, num_layers=1
    )
    return ctx


def d4_flag_and_freeze(w, bank):
    w_handler, w_logger, w_level = _attach_handler(owner=w)
    saved = w._TEST_EXEC_BAG.moe.kt_direct_bank_dma
    try:
        # Forced off freezes silently and never consults the bank.
        w._TEST_EXEC_BAG.moe.kt_direct_bank_dma = 0
        w._KT_DEGRADE_EMITTED.clear()
        ctx = _make_bank_context(w=w)
        ctx._init_rank_bank()
        assert ctx._bank_dma_frozen is False and ctx._rank_bank is None
        assert not w_handler.records, w_handler.records

        # Default on: a missing manifest degrades with exactly one line.
        w._TEST_EXEC_BAG.moe.kt_direct_bank_dma = 1
        w_handler.records.clear()
        w._KT_DEGRADE_EMITTED.clear()
        ctx = _make_bank_context(w=w, weight_path="toy")
        ctx._init_rank_bank()
        assert ctx._bank_dma_frozen is False and ctx._rank_bank is None
        hits = _warnings(handler=w_handler, needle="manifest_unreadable")
        assert len(hits) == 1 and "[KT-DEGRADE]" in hits[0].getMessage(), (
            w_handler.records
        )

        # A live staging window vetoes the bank with exactly one warn.
        w_handler.records.clear()
        ctx = _make_bank_context(w=w, staging=object())
        ctx._init_rank_bank()
        assert ctx._bank_dma_frozen is False and ctx._rank_bank is None
        hits = _warnings(handler=w_handler, needle="mutex_window")
        assert len(hits) == 1 and "[KT-DEGRADE]" in hits[0].getMessage(), (
            w_handler.records
        )

        # No weight_path degrades identically via the first_error channel.
        w._KT_DEGRADE_EMITTED.clear()
        w_handler.records.clear()
        ctx = _make_bank_context(w=w)
        ctx._init_rank_bank()
        assert ctx._bank_dma_frozen is False and ctx._rank_bank is None
        hits = _warnings(handler=w_handler, needle="no_weight_path")
        assert len(hits) == 1, w_handler.records
    finally:
        w._TEST_EXEC_BAG.moe.kt_direct_bank_dma = saved
        w._KT_DEGRADE_EMITTED.clear()
        _detach_handler(w_logger, w_handler, w_level)
    print("D4 PASS: flag off silent; missing-manifest/mutex/path degrade once")


def d5_rank_bank_and_planner(w, bank):
    geometry = _geometry(bank=bank)
    source = torch.arange(3 * 8, dtype=torch.uint8).reshape(3, 8)
    rb = bank._RankBank(
        root="tmp",
        rank=0,
        geometry=geometry,
        row_nbytes={"w13_weight": 8},
        tensors={(0, "w13_weight"): source},
    )
    row = rb.row(0, "w13_weight", 2)
    # Row views alias the pinned arena; indexing mints a new Python wrapper
    # each call, so identity is checked on the storage pointer, not the object.
    assert torch.equal(row, source[2])
    assert row.data_ptr() == source[2].data_ptr()
    assert row.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()

    planner = w._Mxfp4H2DBatchPlanner(bank_count=4, bank_merge_ok=(False,) * 4)
    bank_entries = planner.plan_layer([(5, 7)])
    got = [(e.bank, e.src_row, e.dst_row, e.rows) for e in bank_entries]
    assert got == [(0, 5, 7, 1), (1, 5, 7, 1), (2, 5, 7, 1), (3, 5, 7, 1)]
    # Swapping the source row is the ONLY diff against the legacy slot feed.
    legacy_entries = planner.plan_layer([(1, 7)])
    got = [(e.bank, e.src_row, e.dst_row, e.rows) for e in legacy_entries]
    assert got == [(0, 1, 7, 1), (1, 1, 7, 1), (2, 1, 7, 1), (3, 1, 7, 1)]
    w._assert_h2d_plan_coverage(entries=bank_entries, bank_count=4,
                                expected_rows=1)
    merged = w._Mxfp4H2DBatchPlanner(bank_count=4, bank_merge_ok=(True,) * 4)
    runs = merged.plan_layer([(5, 7), (6, 8)])
    got = [(e.bank, e.src_row, e.dst_row, e.rows) for e in runs]
    assert got == [(0, 5, 7, 2), (1, 5, 7, 2), (2, 5, 7, 2), (3, 5, 7, 2)]
    w._assert_h2d_plan_coverage(entries=runs, bank_count=4, expected_rows=2)
    try:
        w._assert_h2d_plan_coverage(entries=bank_entries[:3], bank_count=4,
                                    expected_rows=1)
        raise AssertionError("a bank hole must fail coverage")
    except RuntimeError as exc:
        assert "H2D plan" in str(exc)
    print("D5 PASS: row views share storage; planner keeps the legacy contract")


def d6_digest_disagree(w):
    handler, w_logger, w_level = _attach_handler(owner=w)
    real = (w.dist, w.get_tensor_model_parallel_world_size, w.get_tp_group)
    # Default-on exec bag drives the bank negotiation in every case below.
    assert w._TEST_EXEC_BAG.moe.kt_direct_bank_dma == 1
    w._KT_DEGRADE_EMITTED.clear()
    try:

        def _gather_split(lst, obj, group=None):
            lst[0] = (None, "digestA")
            lst[1] = (None, "digestB")

        w.dist = types.SimpleNamespace(
            is_initialized=lambda: True,
            all_reduce=lambda tensor, op, group: None,
            all_gather_object=_gather_split,
            ReduceOp=types.SimpleNamespace(MIN=None),
        )
        w.get_tensor_model_parallel_world_size = lambda: 2
        w.get_tp_group = lambda: types.SimpleNamespace(cpu_group=None)
        ctx = _make_bank_context(w=w, weight_path="toy")
        ctx._load_bank_local = lambda: (object(), "digestA")
        try:
            ctx._init_rank_bank()
            raise AssertionError("split manifest digests must raise")
        except RuntimeError as exc:
            assert "manifests disagree" in str(exc), exc
        assert not handler.records, handler.records

        # A peer's error payload takes the degrade path instead of raising.
        def _gather_peer_error(lst, obj, group=None):
            lst[0] = (("no_weight_path", "peer"), "digestA")
            lst[1] = (None, "digestA")

        w.dist.all_gather_object = _gather_peer_error
        ctx = _make_bank_context(w=w, weight_path="toy")
        ctx._load_bank_local = lambda: (object(), "digestA")
        ctx._init_rank_bank()
        assert ctx._bank_dma_frozen is False and ctx._rank_bank is None
        hits = _warnings(handler=handler, needle="no_weight_path")
        assert len(hits) == 1, handler.records
    finally:
        w.dist, w.get_tensor_model_parallel_world_size, w.get_tp_group = real
        w._KT_DEGRADE_EMITTED.clear()
        _detach_handler(w_logger, handler, w_level)
    print("D6 PASS: split digests raise; a peer error degrades once instead")


def d8_pack_task_tag_abi(bench):
    class _OldPackMoe:
        def write_weight_scale_to_buffer_task(self, *args):
            raise AssertionError("not called")

    _OldPackMoe.write_weight_scale_to_buffer_task.__doc__ = (
        "write_weight_scale_to_buffer_task(gpu_tp_count, expert_id, "
        "w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs)"
    )

    class _NewPackMoe:
        def write_weight_scale_to_buffer_task(self, *args):
            raise AssertionError("not called")

    _NewPackMoe.write_weight_scale_to_buffer_task.__doc__ = (
        "write_weight_scale_to_buffer_task(gpu_tp_count, expert_id, "
        "w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs, "
        "task_tag: int = 0)"
    )

    bench._PACK_TASK_TAG_ABI_CHECKED = False
    try:
        try:
            bench._ensure_pack_task_tag_abi(moe=_OldPackMoe())
            raise AssertionError("a pre-task_tag binding must raise")
        except RuntimeError as exc:
            assert "task_tag" in str(exc) and "Rebuild kt-kernel" in str(exc)
        # A failed probe must not latch the once-flag; the next call re-checks.
        assert bench._PACK_TASK_TAG_ABI_CHECKED is False

        bench._ensure_pack_task_tag_abi(moe=_NewPackMoe())
        assert bench._PACK_TASK_TAG_ABI_CHECKED is True
        bench._ensure_pack_task_tag_abi(moe=_OldPackMoe())  # latched: no re-check
        assert bench._PACK_TASK_TAG_ABI_CHECKED is True

        # Escape hatch bypasses even a stale binding doc.
        bench._PACK_TASK_TAG_ABI_CHECKED = False
        os.environ["SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK"] = "1"
        try:
            bench._ensure_pack_task_tag_abi(moe=_OldPackMoe())
            assert bench._PACK_TASK_TAG_ABI_CHECKED is True
        finally:
            del os.environ["SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK"]
    finally:
        bench._PACK_TASK_TAG_ABI_CHECKED = False
    print("D8 PASS: task_tag ABI probe raise/pass/latch/escape all correct")


def d9_dump_slot_bytes(w):
    event = types.SimpleNamespace(calls=0)
    event.synchronize = lambda: setattr(event, "calls", event.calls + 1)
    tensors = {
        name: torch.arange(
            offset * 8, offset * 8 + 8, dtype=torch.float32
        ).reshape(2, 4).to(torch.bfloat16)
        for offset, name in enumerate(w._Mxfp4PrefillSlot.RAW_NAMES)
    }
    slot = types.SimpleNamespace(
        index=0, raw_ready_event=event, **tensors
    )
    mgr = w._Mxfp4LayerwisePrefillManager.__new__(
        w._Mxfp4LayerwisePrefillManager
    )
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as root:
        os.chdir(root)
        try:
            mgr._dump_slot_bytes(slot=slot, layer_idx=3)
        finally:
            os.chdir(cwd)
        # The raw fence syncs exactly once, before any readback (D9);
        # strict sync-before-read ordering is pinned by the code order
        # at the GPU site and covered by the xysa10 machine tail item.
        assert event.calls == 1
        for name in w._Mxfp4PrefillSlot.RAW_NAMES:
            path = os.path.join(
                root, "kt_slot_dump", f"layer3_rank0_slot0_{name}.bin"
            )
            with open(path, "rb") as handle:
                blob = handle.read()
            assert blob == (
                tensors[name].detach().cpu().view(torch.uint8)
                .numpy().tobytes()
            ), name
    print("D9 PASS: dump syncs the raw fence once before readback")


def main():
    wrapper = _load_wrapper()
    bank = wrapper._bank_dma
    bench = _load_bench()
    d1_manifest_validate(bank)
    d1b_tile_loader(bank)
    d2_capacity_precheck(bank)
    d3_row_bytes_golden(bank, bench)
    d4_flag_and_freeze(wrapper, bank)
    d5_rank_bank_and_planner(wrapper, bank)
    d6_digest_disagree(wrapper)
    d8_pack_task_tag_abi(bench)
    d9_dump_slot_bytes(wrapper)
    print("ALL PASS: kt_bank_dma_test (D1-D9)")


if __name__ == "__main__":
    main()
