"""CPU-only unit checks for the MXFP4 gate reason chain and the autofree ABI probe.

Loads kt_ep_wrapper.py with its sglang.srt dependencies stubbed, and
experts_base.py with kt_kernel stubbed, so no sglang install, GPU, or
kt_kernel_ext build is required:

  N4  DEVICE_CAPABILITY — an otherwise fully eligible method/layer pair
      reports exactly that reason on a non-SM89/120 capability.
  N5  ABI probe — _ensure_forward_task_abi raises a rebuild hint for a
      pre-autofree pybind doc, passes the current one, honors the skip valve.
  N6  dedup — one warning line per (family, code); suppress valve silences it.

Run: python third_party/sglang/test/manual/kt_degrade_reason_test.py
"""

import importlib.util
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
EXPERTS_BASE_PATH = REPO_ROOT / "kt-kernel/python/experts_base.py"


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


class _KtConfig:
    method = "MXFP4"
    layer_idx = 0
    weight_path = "toy"
    num_layers = 2


class _Layer:
    """Minimal MXFP4 layer carrying the DSV4 raw contract, 128/64 aligned."""

    def __init__(self):
        self.w13_weight = torch.zeros(2, 4, 128)
        self.w13_weight_scale_inv = torch.zeros(2, 4, 4)
        self.w2_weight = torch.zeros(2, 8, 64)
        self.w2_weight_scale_inv = torch.zeros(2, 8, 2)

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])


def _eligible_method():
    return types.SimpleNamespace(
        gpu_prefill_token_threshold=8,
        kt_config=_KtConfig(),
        gpu_method=type("Mxfp4ToyMoEMethod", (), {})(),
        moe_runner_config=None,
        tp_rank=0,
    )


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def n4_device_capability(w):
    method, layer = _eligible_method(), _Layer()
    w._MXFP4_V4_HELPERS_AVAILABLE = True
    w._V4_TRITON_ENV_LATCHED = None
    w._KT_DEGRADE_EMITTED.clear()
    handler = _RecordHandler()
    wrapper_logger = logging.getLogger(w.__name__)
    wrapper_logger.addHandler(handler)
    is_avail = torch.cuda.is_available
    get_cap = torch.cuda.get_device_capability
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda device: (9, 0)
        assert w._mxfp4_pipeline_backend_supported(method, layer) is False
        hits = [
            r for r in handler.records
            if r.levelno == logging.WARNING and "device_capability" in r.getMessage()
        ]
        assert len(hits) == 1, f"expected one device_capability warning, got {handler.records}"
        assert "sm90" in hits[0].getMessage()
        torch.cuda.get_device_capability = lambda device: (8, 9)
        assert w._mxfp4_pipeline_backend_supported(method, layer) is True
    finally:
        torch.cuda.is_available = is_avail
        torch.cuda.get_device_capability = get_cap
        wrapper_logger.removeHandler(handler)
    print("N4 PASS: device_capability reason emitted once; (8,9) admits")


def n6_emit_dedup(w):
    w._KT_DEGRADE_EMITTED.clear()
    handler = _RecordHandler()
    wrapper_logger = logging.getLogger(w.__name__)
    wrapper_logger.addHandler(handler)
    try:
        fam = "n6_family"
        w._kt_degrade_emit(fam, w._KT_DEGRADE_CUDA_UNAVAILABLE, "first")
        w._kt_degrade_emit(fam, w._KT_DEGRADE_CUDA_UNAVAILABLE, "second\nignored")
        assert len(handler.records) == 1, handler.records
        assert "first" in handler.records[0].getMessage()
        w._kt_degrade_emit(fam, w._KT_DEGRADE_DEVICE_CAPABILITY)
        assert len(handler.records) == 2, handler.records
        # Silent codes never log.
        w._kt_degrade_emit(fam, w._KT_DEGRADE_NOT_REQUESTED)
        assert len(handler.records) == 2, handler.records
        # Suppress valve is read on every call.
        os.environ["SGLANG_KT_SUPPRESS_DEGRADE_WARN"] = "1"
        w._kt_degrade_emit("n6_other", w._KT_DEGRADE_CUDA_UNAVAILABLE)
        assert len(handler.records) == 2, handler.records
        del os.environ["SGLANG_KT_SUPPRESS_DEGRADE_WARN"]

        # Shell passthrough gates: a non-MXFP4 method is silent, and the
        # helpers-missing path warns exactly once across repeated calls.
        w._KT_DEGRADE_EMITTED.clear()
        handler.records.clear()
        quiet = _eligible_method()
        quiet.kt_config.method = "INT4"
        assert w._mxfp4_pipeline_requested(quiet) is False
        assert not handler.records, handler.records
        w._MXFP4_V4_HELPERS_AVAILABLE = False
        method = _eligible_method()
        assert w._mxfp4_pipeline_requested(method) is False
        assert w._mxfp4_pipeline_requested(method) is False
        hits = [
            r for r in handler.records if "v4_helpers_missing" in r.getMessage()
        ]
        assert len(hits) == 1, handler.records
        assert "prepare_v4_mxfp4_marlin" in hits[0].getMessage()
    finally:
        wrapper_logger.removeHandler(handler)
        os.environ.pop("SGLANG_KT_SUPPRESS_DEGRADE_WARN", None)
        w._MXFP4_V4_HELPERS_AVAILABLE = None
    print("N6 PASS: dedup one line per (family, code); valve and gates hold")


def _load_experts_base():
    _stub("kt_kernel", kt_kernel_ext=types.SimpleNamespace())
    return _load_module("experts_base_iso", EXPERTS_BASE_PATH)


def n5_forward_task_abi(eb):
    class _OldMoe:
        def forward_task(self, *args):
            raise AssertionError("not called")

    _OldMoe.forward_task.__doc__ = (
        "forward_task(qlen, k, expert_ids, weights, input, output, incremental=False)"
    )

    class _NewMoe:
        def forward_task(self, *args):
            raise AssertionError("not called")

    _NewMoe.forward_task.__doc__ = (
        "forward_task(qlen, k, expert_ids, weights, input, output, "
        "incremental=False, autofree=False)"
    )

    eb._FORWARD_TASK_ABI_CHECKED = False
    try:
        eb._ensure_forward_task_abi(_OldMoe())
        raise AssertionError("pre-autofree doc must raise")
    except RuntimeError as exc:
        assert "autofree" in str(exc) and "Rebuild kt-kernel" in str(exc)
    # A failed probe must not latch the once-flag; the next call re-checks.
    assert eb._FORWARD_TASK_ABI_CHECKED is False

    eb._ensure_forward_task_abi(_NewMoe())
    assert eb._FORWARD_TASK_ABI_CHECKED is True
    eb._ensure_forward_task_abi(_OldMoe())  # latched: no re-check
    assert eb._FORWARD_TASK_ABI_CHECKED is True

    # Escape hatch bypasses even a stale binding doc.
    eb._FORWARD_TASK_ABI_CHECKED = False
    os.environ["SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK"] = "1"
    try:
        eb._ensure_forward_task_abi(_OldMoe())
        assert eb._FORWARD_TASK_ABI_CHECKED is True
    finally:
        del os.environ["SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK"]
    print("N5 PASS: ABI probe raise/pass/latch/escape-hatch all correct")


def main():
    wrapper = _load_wrapper()
    n4_device_capability(wrapper)
    n6_emit_dedup(wrapper)
    n5_forward_task_abi(_load_experts_base())
    print("ALL PASS: kt_degrade_reason_test (N4/N5/N6)")


if __name__ == "__main__":
    main()
