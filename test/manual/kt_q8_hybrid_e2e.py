#!/usr/bin/env python3
"""Run the ARM + CUDA KTransformers Q8 hybrid test matrix.

This is intentionally a black-box test.  It starts a fresh SGLang server for
each placement/TP/launch-blocking combination, sends deterministic requests,
and records the response, latency, server log, and GPU utilization sample.

Run from the SGLang checkout on the ARM server, for example:

    python test/manual/kt_q8_hybrid_e2e.py \
        --model /data/.../Qwen3.6-35B-A3B \
        --kt-weight-path /data/.../Qwen3.6-35B-A3B-Q8_0.gguf \
        --kt-method LLAMAFILE \
        --kt-cpuinfer 128 \
        --kt-threadpool-count 2

The default matrix covers GPU experts 0/64/256, TP 1/2, batch 1/16, and both
values of CUDA_LAUNCH_BLOCKING.  Use the list options to reduce the matrix
while debugging a single case.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


@dataclasses.dataclass(frozen=True)
class TestCase:
    gpu_experts: int
    tensor_parallel: int
    batch_size: int
    launch_blocking: bool

    @property
    def name(self) -> str:
        blocking = "blocking" if self.launch_blocking else "async"
        return (
            f"gpu{self.gpu_experts}_tp{self.tensor_parallel}_"
            f"batch{self.batch_size}_{blocking}"
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black-box Q8 KTransformers CPU/GPU hybrid test matrix."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model directory")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--kt-weight-path", required=True, help="GGUF or KT CPU expert weights"
    )
    parser.add_argument("--kt-method", default="LLAMAFILE")
    parser.add_argument("--kt-cpuinfer", type=_positive_int, required=True)
    parser.add_argument("--kt-threadpool-count", type=_positive_int, default=2)
    parser.add_argument("--kt-numa-nodes", type=int, nargs="+")
    parser.add_argument(
        "--gpu-experts",
        type=_non_negative_int,
        nargs="+",
        default=[0, 64, 256],
        help="GPU experts per MoE layer",
    )
    parser.add_argument(
        "--tp-sizes",
        type=_positive_int,
        nargs="+",
        default=[1, 2],
        help="Tensor parallel sizes",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_positive_int,
        nargs="+",
        default=[1, 16],
        help="Number of concurrent /generate requests per iteration",
    )
    parser.add_argument(
        "--launch-blocking",
        choices=("off", "on", "both"),
        default="both",
        help="Values of CUDA_LAUNCH_BLOCKING to test",
    )
    parser.add_argument("--warmup", type=_non_negative_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=10)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=16)
    parser.add_argument("--request-timeout", type=_positive_int, default=180)
    parser.add_argument("--startup-timeout", type=_positive_int, default=600)
    parser.add_argument("--port", type=_positive_int, default=3000)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("test-logs/kt-q8-hybrid"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--sampling-backend", default="pytorch")
    parser.add_argument("--chunked-prefill-size", type=_positive_int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--max-deferred-experts-per-token", type=_non_negative_int)
    parser.add_argument("--gpu-prefill-token-threshold", type=_non_negative_int)
    parser.add_argument("--enable-dynamic-expert-update", action="store_true")
    parser.add_argument("--record-expert-distribution", action="store_true")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Also run the eager baseline for the same matrix",
    )
    parser.add_argument(
        "--strict-consistency",
        action="store_true",
        help="Fail when blocking and non-blocking responses differ",
    )
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        help="Run only the first N generated cases",
    )
    parser.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        help="One additional launch_server argument; repeat for multiple arguments",
    )
    return parser.parse_args()


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _tail(path: pathlib.Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _response_text(response: Any) -> str:
    if isinstance(response, dict):
        text = response.get("text")
        if isinstance(text, str):
            return text
        outputs = response.get("output")
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            return str(outputs[0].get("text", ""))
    return ""


def _response_signature(response: Any) -> str:
    if isinstance(response, dict):
        stable_response = {
            "text": _response_text(response),
            "output_ids": response.get("output_ids"),
        }
    else:
        stable_response = response
    canonical = json.dumps(
        stable_response, sort_keys=True, ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _generated_tokens(response: Any) -> int | None:
    if not isinstance(response, dict):
        return None
    output_ids = response.get("output_ids")
    if isinstance(output_ids, list):
        return len(output_ids)
    return None


def _http_json(
    url: str,
    payload: dict[str, Any] | None,
    timeout: int,
) -> Any:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


class ServerProcess:
    def __init__(self, args: argparse.Namespace, case: TestCase, log_path: pathlib.Path):
        self.args = args
        self.case = case
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None

    def command(self) -> list[str]:
        args = self.args
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            args.model,
            "--kt-weight-path",
            args.kt_weight_path,
            "--kt-method",
            args.kt_method,
            "--kt-cpuinfer",
            str(args.kt_cpuinfer),
            "--kt-threadpool-count",
            str(args.kt_threadpool_count),
            "--tensor-parallel-size",
            str(self.case.tensor_parallel),
            "--attention-backend",
            args.attention_backend,
            "--sampling-backend",
            args.sampling_backend,
            "--chunked-prefill-size",
            str(args.chunked_prefill_size),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
        ]
        if args.trust_remote_code:
            command.append("--trust-remote-code")
        command.extend(["--kt-num-gpu-experts", str(self.case.gpu_experts)])
        if args.kt_numa_nodes is not None:
            command.extend(["--kt-numa-nodes", *map(str, args.kt_numa_nodes)])
        if args.max_deferred_experts_per_token is not None:
            command.extend(
                [
                    "--kt-max-deferred-experts-per-token",
                    str(args.max_deferred_experts_per_token),
                ]
            )
        if args.gpu_prefill_token_threshold is not None:
            command.extend(
                [
                    "--kt-gpu-prefill-token-threshold",
                    str(args.gpu_prefill_token_threshold),
                ]
            )
        if args.enable_dynamic_expert_update:
            command.append("--kt-enable-dynamic-expert-update")
        if args.record_expert_distribution:
            command.extend(
                [
                    "--record-kt-gpu-expert-distribution",
                    "--expert-distribution-recorder-mode",
                    "stat",
                ]
            )
        if args.disable_cuda_graph:
            command.append("--disable-cuda-graph")
        command.extend(args.extra_server_arg)
        return command

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        source_python = str(_repo_root() / "python")
        env["PYTHONPATH"] = os.pathsep.join(
            [source_python, env["PYTHONPATH"]]
            if env.get("PYTHONPATH")
            else [source_python]
        )
        if self.case.launch_blocking:
            env["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            env.pop("CUDA_LAUNCH_BLOCKING", None)

        log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command(),
            cwd=_repo_root(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        # Keep the file descriptor alive until the child exits, but do not
        # retain a second handle in the test process.
        log_file.close()

    def wait_ready(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        health_url = f"http://{self.args.host}:{self.args.port}/health"
        last_error = ""
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.process.returncode}\n"
                    f"{_tail(self.log_path)}"
                )
            try:
                _http_json(health_url, None, timeout=5)
                return
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = str(exc)
                time.sleep(1)
        raise TimeoutError(
            f"server did not become ready within {self.args.startup_timeout}s: "
            f"{last_error}\n{_tail(self.log_path)}"
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def _request_one(
    host: str,
    port: int,
    prompt: str,
    max_new_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    response = _http_json(f"http://{host}:{port}/generate", payload, timeout)
    if not isinstance(response, dict):
        raise TypeError(f"unexpected /generate response: {type(response).__name__}")
    if _contains_non_finite(response):
        raise FloatingPointError("response contains NaN or Inf")
    if not _response_text(response):
        raise RuntimeError("response contains no generated text")
    return response


def _request_batch(args: argparse.Namespace, case: TestCase, iteration: int):
    prompts = [
        (
            "Return exactly one short sentence containing the number "
            f"{iteration}-{index}. Do not explain the task."
        )
        for index in range(case.batch_size)
    ]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=case.batch_size
    ) as executor:
        futures = [
            executor.submit(
                _request_one,
                args.host,
                args.port,
                prompt,
                args.max_new_tokens,
                args.request_timeout,
            )
            for prompt in prompts
        ]
        responses = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    token_count = sum(
        tokens or 0 for tokens in (_generated_tokens(response) for response in responses)
    )
    return responses, elapsed, token_count


def _sample_gpu_utilization() -> list[int]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        output = subprocess.check_output(
            [
                executable,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    values = []
    for line in output.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return values


def run_case(args: argparse.Namespace, case: TestCase) -> dict[str, Any]:
    log_path = args.output_dir / f"{case.name}.log"
    server = ServerProcess(args, case, log_path)
    result: dict[str, Any] = dataclasses.asdict(case)
    result["case"] = case.name
    result["log"] = str(log_path)
    started = time.perf_counter()
    try:
        server.start()
        server.wait_ready()
        for iteration in range(args.warmup):
            _request_batch(args, case, -(args.warmup - iteration))
        latencies = []
        total_tokens = 0
        signatures = []
        for iteration in range(args.iterations):
            responses, elapsed, tokens = _request_batch(args, case, iteration)
            latencies.append(elapsed)
            total_tokens += tokens
            signatures.extend(_response_signature(response) for response in responses)

        result.update(
            {
                "status": "pass",
                "elapsed_s": time.perf_counter() - started,
                "mean_batch_latency_ms": 1000.0 * sum(latencies) / len(latencies),
                "requests_per_second": case.batch_size * len(latencies) / sum(latencies),
                "generated_tokens": total_tokens,
                "token_per_second": (
                    total_tokens / sum(latencies) if total_tokens else None
                ),
                "response_signatures": signatures,
                "gpu_utilization_sample": _sample_gpu_utilization(),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.perf_counter() - started,
            }
        )
    finally:
        server.stop()
    return result


def _cases(args: argparse.Namespace) -> Iterable[TestCase]:
    blocking_values = {
        "off": [False],
        "on": [True],
        "both": [False, True],
    }[args.launch_blocking]
    generated = (
        TestCase(gpu, tp, batch, blocking)
        for gpu in args.gpu_experts
        for tp in args.tp_sizes
        for batch in args.batch_sizes
        for blocking in blocking_values
    )
    if args.max_cases is None:
        return generated
    return list(generated)[: args.max_cases]


def _check_consistency(results: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[int, int, int], dict[bool, dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "pass":
            continue
        key = (
            int(result["gpu_experts"]),
            int(result["tensor_parallel"]),
            int(result["batch_size"]),
        )
        grouped.setdefault(key, {})[bool(result["launch_blocking"])] = result

    mismatches = []
    for key, variants in grouped.items():
        if False not in variants or True not in variants:
            continue
        if variants[False]["response_signatures"] != variants[True]["response_signatures"]:
            mismatches.append(
                f"gpu={key[0]} tp={key[1]} batch={key[2]} "
                "blocking/non-blocking response mismatch"
            )
    return mismatches


def main() -> int:
    args = parse_args()
    if args.kt_numa_nodes is not None and len(args.kt_numa_nodes) != args.kt_threadpool_count:
        raise SystemExit(
            "--kt-numa-nodes must contain exactly --kt-threadpool-count values"
        )
    if args.enable_dynamic_expert_update and not (
        args.gpu_prefill_token_threshold and args.gpu_prefill_token_threshold > 0
    ):
        raise SystemExit(
            "--enable-dynamic-expert-update requires --gpu-prefill-token-threshold > 0"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    cases = list(_cases(args))
    print(f"Running {len(cases)} KT Q8 hybrid cases")
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        result = run_case(args, case)
        results.append(result)
        if result["status"] == "pass":
            print(
                "  PASS "
                f"latency={result['mean_batch_latency_ms']:.2f}ms "
                f"tok/s={result.get('token_per_second')}",
                flush=True,
            )
        else:
            print(f"  FAIL {result['error']}", flush=True)

    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    failures = [result for result in results if result["status"] != "pass"]
    mismatches = _check_consistency(results)
    print(f"Results written to {results_path}")
    if mismatches:
        print("Consistency warnings:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
    if failures:
        print(f"{len(failures)} case(s) failed. Inspect the per-case logs.")
        return 1
    if mismatches and args.strict_consistency:
        return 2
    print("All requested KT Q8 hybrid cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
