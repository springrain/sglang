"""CPU-only unit checks for the tier-3 decode-side guardrails (steps A + C).

Loads kt-kernel's experts_base.py and sglang's kt_ep_wrapper.py with their
native/sglang dependencies stubbed, so no kt_kernel_ext build, GPU, or sglang
install is required:

  W1  _get_cpu_infer passthrough - a truthy reserve_cores writes the
      WorkerPoolConfig field, None leaves the field unwritten (old-extension
      tolerance), a negative value raises ValueError before the ctor runs,
      and the CPUInfer ctor arity switches 1-arg vs 2-arg on the watchdog
      timeout alone.
  W2  The torch intra-op thread clamp runs only when reserve_cores is set
      (bit-exact default) and only ever clamps down.
  W3  sync_forward probes the latched watchdog before syncing: a tripped
      pool raises a tagged RuntimeError without touching cpu_infer.sync, an
      untripped pool behaves exactly as before, and an extension without the
      probe silently passes.
  W6  server_args.py (parsed via ast, no import) declares the three new
      --kt-cpuinfer-* switches with int+choices form, exec.moe namespace,
      the documented defaults, and raises ValueError on a <1000 ms budget.
  W7  kt_ep_wrapper's KTConfig really carries reserve_cores /
      watchdog_timeout_ms defaults and the get_exec().moe / wrapper-kwargs
      wiring exists in source.
  W8  Source-text guards prove the C++ poison/heartbeat channel, the
      watchdog thread, the pybind surface, and that the legacy
      cudaLaunchHostFunc submit/sync sites are still in place.

W4/W5/W9 (memop handshake) are deferred: cuStreamBatchMemOp has no ADD
primitive, so the deferred-expert handshake fork (replay-daemon vs
eager-only vs drop) is pending a ruling.

Run: python third_party/sglang/test/manual/kt_decode_side_test.py
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
KT_KERNEL_DIR = REPO_ROOT / "kt-kernel"
EXPERTS_BASE_PATH = KT_KERNEL_DIR / "python/experts_base.py"
WRAPPER_PATH = (
    REPO_ROOT / "third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
)
SERVER_ARGS_PATH = REPO_ROOT / "third_party/sglang/python/sglang/srt/server_args.py"

WATCHDOG_SWITCH = "kt_cpuinfer_watchdog"
WATCHDOG_TIMEOUT = "kt_cpuinfer_watchdog_timeout_ms"
RESERVE_CORES = "kt_cpuinfer_reserve_cores"


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


def _load_experts_base():
    """Exec experts_base.py with the native extension satisfied by a stub."""
    ext = types.ModuleType("kt_kernel_ext")
    sys.modules["kt_kernel_ext"] = ext
    _stub("kt_kernel", kt_kernel_ext=ext)
    return _load_module("experts_base_iso", EXPERTS_BASE_PATH), ext


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
    _stub(
        "sglang.srt.utils", get_compiler_backend=lambda: "eager", is_cuda=lambda: False
    )
    return _load_module("kt_ep_wrapper_iso", WRAPPER_PATH)


class _FakeWorkerPoolConfig:
    """Records every attribute the plumbing writes."""

    def __init__(self, record):
        object.__setattr__(self, "_record", record)

    def __setattr__(self, name, value):
        self._record[name] = value


def _install_fake_ext(ext):
    """Fresh recording fakes for one _get_cpu_infer case."""
    record = {"config": {}, "ctor_args": []}
    ext.WorkerPoolConfig = lambda: _FakeWorkerPoolConfig(record["config"])

    def _fake_cpuinfer(*args):
        record["ctor_args"].append(args)
        return object()

    ext.CPUInfer = _fake_cpuinfer
    return record


def _get(eb, threads=6, tp=2, reserve_cores=None, watchdog_ms=0):
    eb._MoEBase._cpu_infer_instance = None
    return eb._MoEBase._get_cpu_infer(
        threads,
        tp,
        None,
        reserve_cores=reserve_cores,
        watchdog_timeout_ms=watchdog_ms,
    )


def w1_pool_passthrough(eb, ext):
    rec = _install_fake_ext(ext)
    _get(eb, reserve_cores=2, watchdog_ms=0)
    assert rec["config"].get("reserve_cores_per_numa") == 2, rec["config"]
    assert rec["ctor_args"] and len(rec["ctor_args"][0]) == 1, rec["ctor_args"]

    rec = _install_fake_ext(ext)
    _get(eb, reserve_cores=None, watchdog_ms=1500)
    # Field untouched so extensions predating reserve_cores_per_numa work.
    assert "reserve_cores_per_numa" not in rec["config"], rec["config"]
    assert rec["ctor_args"] and len(rec["ctor_args"][0]) == 2, rec["ctor_args"]
    assert rec["ctor_args"][0][1] == 1500, rec["ctor_args"]

    # Zero keeps the legacy 1-arg ctor as well.
    rec = _install_fake_ext(ext)
    _get(eb, reserve_cores=0, watchdog_ms=0)
    assert len(rec["ctor_args"][0]) == 1, rec["ctor_args"]

    rec = _install_fake_ext(ext)
    try:
        _get(eb, reserve_cores=-1, watchdog_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative reserve_cores must raise ValueError")
    assert rec["ctor_args"] == [], rec["ctor_args"]
    assert "reserve_cores_per_numa" not in rec["config"], rec["config"]
    eb._MoEBase._cpu_infer_instance = None
    print("W1 PASS: reserve field set iff truthy; ctor arity on watchdog alone")


def w2_torch_clamp_gated(eb, ext):
    import torch

    calls = []
    orig = torch.set_num_threads
    torch.set_num_threads = lambda n: calls.append(n)
    try:
        _install_fake_ext(ext)
        _get(eb, threads=6, tp=2, reserve_cores=1, watchdog_ms=0)
        # threads=6, tp=2 -> subpool counts [3, 3]; clamp never raises the cap.
        expected = min(torch.get_num_threads(), 3)
        assert calls == [expected], calls
        _install_fake_ext(ext)
        _get(eb, threads=6, tp=2, reserve_cores=None, watchdog_ms=1500)
        assert calls == [expected], calls
    finally:
        torch.set_num_threads = orig
        eb._MoEBase._cpu_infer_instance = None
    print("W2 PASS: torch intra-op clamp only on reserve, and only downwards")


class _FakeCPUInferPool:
    def __init__(self, tripped=None, text=""):
        self._tripped = tripped
        self._text = text
        self.sync_calls = []

    def sync(self, allow_pending):
        self.sync_calls.append(allow_pending)

    def watchdog_tripped(self):
        assert self._tripped is not None, "old-extension fake lacks the probe"
        return self._tripped

    def watchdog_text(self):
        return self._text


class _OldCPUInferPool:
    """Pre-watchdog extension shape: sync only, no probes."""

    def __init__(self):
        self.sync_calls = []

    def sync(self, allow_pending):
        self.sync_calls.append(allow_pending)


def _run_sync_forward(eb, cpu_infer):
    import torch

    saved_pending = dict(eb.BaseMoEWrapper._layer_has_pending_deferred)
    saved_buffer = eb.KExpertsCPUBuffer
    saved_bypass = eb._should_bypass_stream_callback

    class _FakeBuffer:
        buffer_depth = 2

        @staticmethod
        def get_buffer(flat, topk):
            tokens, hidden = flat.shape
            return (
                flat,
                flat.new_zeros((tokens, topk), dtype=torch.int32),
                flat.new_zeros((tokens, topk), dtype=torch.int32),
                flat.new_zeros((tokens, topk)),
                torch.zeros((_FakeBuffer.buffer_depth, tokens, hidden)),
                torch.zeros((tokens,), dtype=torch.int32),
                torch.zeros((_FakeBuffer.buffer_depth, tokens, hidden)),
            )

    fake_self = types.SimpleNamespace(
        layer_idx=3,
        num_experts_per_tok=2,
        cpu_infer=cpu_infer,
        copy_forward_output_to_device=lambda hidden: None,
    )
    eb.KExpertsCPUBuffer = _FakeBuffer
    eb._should_bypass_stream_callback = lambda device: True
    eb.BaseMoEWrapper._layer_has_pending_deferred.clear()
    try:
        eb.BaseMoEWrapper.sync_forward(fake_self, torch.zeros((2, 4)), 0)
    finally:
        eb.KExpertsCPUBuffer = saved_buffer
        eb._should_bypass_stream_callback = saved_bypass
        eb.BaseMoEWrapper._layer_has_pending_deferred.clear()
        eb.BaseMoEWrapper._layer_has_pending_deferred.update(saved_pending)


def w3_sync_probe(eb, ext):
    pool = _FakeCPUInferPool(tripped=False)
    _run_sync_forward(eb, pool)
    assert pool.sync_calls == [0], pool.sync_calls

    pool = _FakeCPUInferPool(
        tripped=True, text="[kt watchdog] cpu pool no progress >1500ms task"
    )
    try:
        _run_sync_forward(eb, pool)
    except RuntimeError as err:
        assert "[kt watchdog]" in str(err), err
    else:
        raise AssertionError("tripped watchdog must raise in sync_forward")
    assert pool.sync_calls == [], pool.sync_calls

    old_pool = _OldCPUInferPool()
    _run_sync_forward(eb, old_pool)
    assert old_pool.sync_calls == [0], old_pool.sync_calls
    print("W3 PASS: tripped pool raises tagged error pre-sync; others unchanged")


def _annotations_map(tree, src, names):
    # The args live inside the ServerArgs dataclass body, not at module level.
    found = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in names
        ):
            found[node.target.id] = node
    return found


def _annotation_calls(ann, func_name):
    sl = ann.annotation.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    return [
        e
        for e in elts
        if isinstance(e, ast.Call) and getattr(e.func, "id", "") == func_name
    ]


def w6_server_args_ast():
    src = SERVER_ARGS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {WATCHDOG_SWITCH, WATCHDOG_TIMEOUT, RESERVE_CORES}
    ann = _annotations_map(tree, src, names)
    assert set(ann) == names, sorted(ann)
    for name in names:
        nss = _annotation_calls(ann[name], "NS")
        assert nss and nss[0].args[0].value == "exec.moe", name
        default = ann[name].value
        assert isinstance(default, ast.Constant) and isinstance(default.value, int), name

    assert ann[WATCHDOG_SWITCH].value.value == 0
    assert ann[WATCHDOG_TIMEOUT].value.value == 30000
    assert ann[RESERVE_CORES].value.value == 0

    choices = {}
    for name in names:
        args = _annotation_calls(ann[name], "Arg")
        for kw in (args[0].keywords if args else []):
            if kw.arg == "choices":
                choices[name] = [c.value for c in kw.value.elts]
    # Switch and reserve forms are int + Arg(choices=...), never store_true.
    assert choices[WATCHDOG_SWITCH] == [0, 1], choices
    assert choices[RESERVE_CORES] == [0, 1, 2], choices
    assert WATCHDOG_TIMEOUT not in choices, choices

    msgs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            call = node.exc
            if isinstance(call, ast.Call) and call.args:
                seg = ast.get_source_segment(src, call) or ""
                msgs.append(seg)
    assert any(
        "--kt-cpuinfer-watchdog-timeout-ms" in m and "1000" in m for m in msgs
    ), msgs
    print("W6 PASS: three exec.moe args, int+choices forms, gated validation")


def w7_wrapper_config():
    import dataclasses

    w = _load_wrapper()
    fields = {f.name: f for f in dataclasses.fields(w.KTConfig)}
    assert fields["reserve_cores"].default == 0, fields["reserve_cores"]
    assert fields["watchdog_timeout_ms"].default == 0, fields["watchdog_timeout_ms"]

    src = WRAPPER_PATH.read_text(encoding="utf-8")
    assert f"reserve_cores=get_exec().moe.{RESERVE_CORES}" in src
    assert f"if get_exec().moe.{WATCHDOG_SWITCH}" in src
    assert "reserve_cores=self.kt_config.reserve_cores" in src
    assert "watchdog_timeout_ms=self.kt_config.watchdog_timeout_ms" in src
    print("W7 PASS: KTConfig defaults + get_exec().moe wiring proven")


def w8_source_guards():
    def _hits(rel, needles):
        text = (KT_KERNEL_DIR.joinpath(*rel)).read_text(encoding="utf-8")
        missing = [n for n in needles if n not in text]
        assert not missing, f"{rel}: missing {missing}"

    _hits(
        ("cpu_backend", "task_queue.h"),
        ["poisoned_flag", "task_start_ns", "task_tag", "poison_text"],
    )
    _hits(
        ("cpu_backend", "task_queue.cpp"),
        ["poisoned_flag.load", "pending_task_tag", "poison_exception"],
    )
    _hits(
        ("cpu_backend", "cpuinfer.h"),
        [
            "watchdog_thread_",
            "start_watchdog_",
            "watchdog_tripped",
            "watchdog_text",
            # Legacy stream-callback submit/sync sites must survive the patch.
            "cudaLaunchHostFunc",
        ],
    )
    _hits(
        ("cpu_backend", "worker_pool.h"),
        ["reserve_cores_per_numa", "int reserve_cores = 0"],
    )
    _hits(("cpu_backend", "worker_pool.cpp"), ["reserve_cores", "pinned_per_numa"])
    _hits(
        ("ext_bindings.cpp",),
        ["reserve_cores_per_numa", "watchdog_tripped", "WorkerPoolConfig, int"],
    )
    _hits(
        ("python", "experts_base.py"),
        ["_cpuinfer_watchdog_raise", "reserve_cores_per_numa", "watchdog_timeout_ms"],
    )
    print("W8 PASS: poison/heartbeat/pybind guards + legacy launch sites intact")


def main():
    eb, ext = _load_experts_base()
    w1_pool_passthrough(eb, ext)
    w2_torch_clamp_gated(eb, ext)
    w3_sync_probe(eb, ext)
    w6_server_args_ast()
    w7_wrapper_config()
    w8_source_guards()
    print("ALL PASS: kt_decode_side_test (W1-W3, W6-W8)")


if __name__ == "__main__":
    main()
