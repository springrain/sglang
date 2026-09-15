"""CPU-only unit checks for the MXFP4 slot-ownership sentinel.

Loads kt_ep_wrapper.py with its sglang.srt dependencies stubbed, so no
sglang install, GPU, or kt_kernel_ext build is required:

  N1  Assert A - a parity-flipped host_slot warns once (dedup), returns
      None at warn-only level, and leaves the shadow table untouched.
  N2  Assert B/E - overwriting a live same-epoch owner without the
      free-sync warns once and does not replace the owner; at enforce
      level the check returns a RuntimeError for the consensus channel.
  N3  Assert D - epoch rewind and an out-of-automaton reuse_guard each
      fire; prime loads (no guard) pass with one debug skip-note total.
  N4  Default equivalence - env unset means warn-only (2), "0" is fully
      silent off (shadow never populated), invalid values warn once and
      fall back to 2.
  N5  Runtime env edits cannot retarget a constructed manager: behavior
      is governed solely by the level cached in the instance attribute.

Run: python third_party/sglang/test/manual/kt_slot_ownership_assert_test.py
"""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER_PATH = (
    REPO_ROOT
    / "third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
)
ENV_NAME = "SGLANG_KT_SLOT_OWNERSHIP_ASSERT"


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
    _stub("sglang.srt.runtime_context", get_exec=lambda: None, get_schedule=lambda: None)
    # is_cuda() False skips the gptq_marlin_repack import branch; the "eager"
    # backend string keeps module-level torch.compile decorators inductor-free.
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


def _make_manager(w, level, epoch=7):
    """Bypass __init__ (needs CUDA streams); install only sentinel state."""
    mgr = w._Mxfp4LayerwisePrefillManager.__new__(w._Mxfp4LayerwisePrefillManager)
    mgr._ownership_level = level
    mgr._host_slot_owner = [None, None]
    mgr._host_slot_freed = [True, True]
    mgr._ownership_warned = set()
    mgr._ownership_prime_logged = False
    mgr.epoch = epoch
    return mgr


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


def n1_assert_a(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    try:
        mgr = _make_manager(w, level=2)
        # Parity flip of the host_slot = position % 2 assignment.
        assert mgr._ownership_record_host_slot(3, 0, 1) is None
        hits = _warnings(handler, "invariant A")
        assert len(hits) == 1 and "host_slot=1" in hits[0].getMessage(), (
            handler.records
        )
        # Suspect coordinates must not be recorded into the shadow table.
        assert mgr._host_slot_owner == [None, None], mgr._host_slot_owner
        # Same dedup key (code, layer, slot, epoch): no repeated warning.
        assert mgr._ownership_record_host_slot(3, 0, 1) is None
        assert mgr._ownership_record_host_slot(3, 4, 1) is None
        assert len(_warnings(handler, "invariant A")) == 1, handler.records
        # Out-of-range slots are A violations too.
        assert mgr._ownership_record_host_slot(3, 0, 5) is None
        assert len(_warnings(handler, "host_slot=5")) == 1, handler.records
        assert mgr._host_slot_owner == [None, None], mgr._host_slot_owner
    finally:
        _detach_handler(wrapper_logger, handler, old_level)
    print("N1 PASS: assert A warns once per key and never touches the shadow")


def n2_assert_b_e(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    try:
        mgr = _make_manager(w, level=2)
        assert mgr._ownership_record_host_slot(3, 0, 0) is None
        assert mgr._host_slot_owner[0] == (7, 3, 0), mgr._host_slot_owner
        assert mgr._host_slot_freed[0] is False
        # Overwrite of a live same-epoch owner without a free-sync: this is
        # the consumed-before-overwrite family, fired via the freed twin.
        assert mgr._ownership_record_host_slot(3, 2, 0) is None
        hits = _warnings(handler, "invariant B")
        assert len(hits) == 1 and "position=0" in hits[0].getMessage(), (
            handler.records
        )
        # The live owner must survive; recording the violator would erase
        # the evidence pointed at by the dedup key.
        assert mgr._host_slot_owner[0] == (7, 3, 0), mgr._host_slot_owner
        assert mgr._ownership_record_host_slot(3, 4, 0) is None
        assert len(_warnings(handler, "invariant B")) == 1, handler.records
        # The host-slot sync releases the owner; the next record then passes.
        mgr._host_slot_freed[0] = True
        assert mgr._ownership_record_host_slot(3, 2, 0) is None
        assert mgr._host_slot_owner[0] == (7, 3, 2), mgr._host_slot_owner
        # Cross-epoch leftovers are legitimate (the round was dropped).
        mgr.epoch = 8
        assert mgr._ownership_record_host_slot(4, 0, 0) is None
        assert len(_warnings(handler, "invariant B")) == 1, handler.records
        # Enforce level returns an error object instead of raising in place.
        mgr = _make_manager(w, level=1)
        assert mgr._ownership_record_host_slot(3, 0, 0) is None
        err = mgr._ownership_record_host_slot(3, 2, 0)
        assert isinstance(err, RuntimeError), err
        assert "invariant B" in str(err)
        assert mgr._host_slot_owner[0] == (7, 3, 0), mgr._host_slot_owner
    finally:
        _detach_handler(wrapper_logger, handler, old_level)
    print("N2 PASS: B/E fires on unfreed overwrite; freed/cross-epoch paths pass")


def n3_assert_d(w):
    handler, wrapper_logger, old_level = _attach_handler(w, logging.DEBUG)
    try:
        mgr = _make_manager(w, level=2, epoch=3)
        slot = types.SimpleNamespace(index=0, epoch=5, reuse_guard="ready")
        err = mgr._ownership_state_check(slot, 3)
        assert err is None
        hits = _warnings(handler, "invariant D")
        assert len(hits) == 1 and "rewound" in hits[0].getMessage(), (
            handler.records
        )
        # Out-of-automaton guard (a "loading" remnant on a non-loading slot).
        mgr = _make_manager(w, level=2)
        slot = types.SimpleNamespace(index=1, epoch=0, reuse_guard="loading")
        assert mgr._ownership_state_check(slot, 3) is None
        hits = _warnings(handler, "unexpected reuse_guard")
        assert len(hits) == 1 and "'loading'" in hits[0].getMessage(), (
            handler.records
        )
        # Legitimate guards pass; prime loads log the skip-reason once.
        mgr = _make_manager(w, level=2)
        for guard in ("consumed", "ready", "raw", "synchronized"):
            slot = types.SimpleNamespace(index=0, epoch=6, reuse_guard=guard)
            assert mgr._ownership_state_check(slot, 3) is None, guard
        slot = types.SimpleNamespace(index=0, epoch=6, reuse_guard=None)
        assert mgr._ownership_state_check(slot, 3) is None
        assert mgr._ownership_state_check(slot, 3) is None
        debugs = [
            r
            for r in handler.records
            if r.levelno == logging.DEBUG and "prime loads" in r.getMessage()
        ]
        assert len(debugs) == 1, handler.records
        assert mgr._ownership_prime_logged is True
        # Legit guards and prime loads added nothing beyond the two D hits.
        assert len(_warnings(handler, "invariant D")) == 2, handler.records
        # Enforce level surfaces the D violation as an error object.
        mgr = _make_manager(w, level=1, epoch=3)
        slot = types.SimpleNamespace(index=0, epoch=5, reuse_guard="raw")
        err = mgr._ownership_state_check(slot, 3)
        assert isinstance(err, RuntimeError) and "invariant D" in str(err), err
    finally:
        _detach_handler(wrapper_logger, handler, old_level)
    print("N3 PASS: D catches epoch rewind and guard gaps; prime skip-note once")


def n4_default_equivalence(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    saved = os.environ.pop(ENV_NAME, None)
    try:
        assert w._kt_ownership_assert_level() == 2
        os.environ[ENV_NAME] = "0"
        assert w._kt_ownership_assert_level() == 0
        os.environ[ENV_NAME] = "1"
        assert w._kt_ownership_assert_level() == 1
        # Invalid values warn once per process and fall back to 2.
        w._ownership_env_warned = False
        os.environ[ENV_NAME] = "9"
        assert w._kt_ownership_assert_level() == 2
        assert w._kt_ownership_assert_level() == 2
        hits = _warnings(handler, "Invalid SGLANG_KT_SLOT_OWNERSHIP_ASSERT")
        assert len(hits) == 1 and "'9'" in hits[0].getMessage(), handler.records
        # Level 0 is fully silent: no warning, and the shadow is never even
        # populated by a clean record.
        handler.records.clear()
        os.environ[ENV_NAME] = "0"
        mgr = _make_manager(w, level=0)
        assert mgr._ownership_record_host_slot(3, 0, 1) is None
        assert mgr._ownership_record_host_slot(3, 0, 0) is None
        slot = types.SimpleNamespace(index=0, epoch=99, reuse_guard="loading")
        assert mgr._ownership_state_check(slot, 3) is None
        assert not handler.records, handler.records
        assert mgr._host_slot_owner == [None, None], mgr._host_slot_owner
    finally:
        _detach_handler(wrapper_logger, handler, old_level)
        if saved is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = saved
    print("N4 PASS: unset=2, 0=silent off, 1=enforce, invalid warns once to 2")


def n5_env_latched(w):
    handler, wrapper_logger, old_level = _attach_handler(w)
    saved = os.environ.pop(ENV_NAME, None)
    try:
        # The instance caches the level at construction; a later env edit
        # must not retarget an existing manager (structural constraint:
        # every rank evaluates the same level for the process lifetime).
        mgr = _make_manager(w, level=w._kt_ownership_assert_level())
        assert mgr._ownership_level == 2
        os.environ[ENV_NAME] = "1"
        assert mgr._ownership_record_host_slot(3, 0, 0) is None
        assert mgr._ownership_record_host_slot(3, 2, 0) is None
        hits = _warnings(handler, "invariant B")
        assert len(hits) == 1, handler.records
        assert w._kt_ownership_assert_level() == 1  # fresh reads stay env-live
    finally:
        _detach_handler(wrapper_logger, handler, old_level)
        if saved is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = saved
    print("N5 PASS: cached level governs the instance; runtime env edit inert")


def main():
    wrapper = _load_wrapper()
    n1_assert_a(wrapper)
    n2_assert_b_e(wrapper)
    n3_assert_d(wrapper)
    n4_default_equivalence(wrapper)
    n5_env_latched(wrapper)
    print("ALL PASS: kt_slot_ownership_assert_test (N1-N5)")


if __name__ == "__main__":
    main()
