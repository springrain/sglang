# SPDX-License-Identifier: Apache-2.0
"""KTransformers server-argument normalization and validation."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    with_phase,
)

logger = logging.getLogger(__name__)


def handle_kt_compatibility(server_args: Any) -> None:
    """Normalize legacy KT inputs without mutating the sealed input record.

    This runs before CUDA-graph parsing so the legacy LoRA alias can disable
    graph capture through the same declaration stream read by later hooks.
    The Kimi-K3 environment variables predate their CLI equivalents and remain
    supported as one-way, deprecated opt-ins.
    """

    cfg = resolving_view(server_args)
    declared: dict[str, Any] = {}

    if cfg.kt_lora_path:
        if cfg.kt_expert_lora_path and cfg.kt_expert_lora_path != cfg.kt_lora_path:
            raise ValueError(
                "--kt-lora-path and --kt-expert-lora-path cannot point to "
                "different adapters in the static single-adapter implementation."
            )
        declared["kt_expert_lora_path"] = cfg.kt_lora_path
        logger.warning(
            "--kt-lora-path is treated as the KT CPU-expert adapter in this "
            "forward-port; use --kt-expert-lora-path explicitly."
        )

    if (cfg.kt_lora_path or cfg.kt_expert_lora_path) and not cfg.disable_cuda_graph:
        logger.warning(
            "CUDA graph is disabled because KT expert LoRA uses host-side "
            "input copies in the KT SFT path."
        )
        declared["disable_cuda_graph"] = True

    for env_name, env_field, arg_name in (
        (
            "SGLANG_K3_SHARED_EXPERTS_ATTN_TP",
            envs.SGLANG_K3_SHARED_EXPERTS_ATTN_TP,
            "enable_shared_experts_attn_tp",
        ),
        (
            "SGLANG_K3_DENSE_MLP_ATTN_TP",
            envs.SGLANG_K3_DENSE_MLP_ATTN_TP,
            "enable_dense_mlp_attn_tp",
        ),
    ):
        if not env_field.is_set():
            continue
        logger.warning(
            "%s is deprecated; use --%s instead.",
            env_name,
            arg_name.replace("_", "-"),
        )
        if env_field.get() and not getattr(cfg, arg_name):
            declared[arg_name] = True

    if declared:
        declare_resolution(server_args, "handle_kt_compatibility", **declared)


def enforce_kt_cuda_graph_compatibility(server_args: Any) -> None:
    """Apply the CUDA-graph restrictions of KT's host-side expert path.

    CPU experts submit work and copy results through host-side state that is
    unsafe to capture for prefill. Decode capture remains enabled for ordinary
    KT weights. KT expert LoRA additionally uses host-side input copies, so it
    disables both phases even when an explicit CUDA-graph config was supplied.
    """

    cfg = resolving_view(server_args)
    if not (cfg.kt_weight_path or cfg.kt_lora_path or cfg.kt_expert_lora_path):
        return

    cuda_graph_config = cfg.cuda_graph_config
    if not isinstance(cuda_graph_config, CudaGraphConfig):
        # Dummy-model resolution returns before CUDA-graph parsing.
        return

    new_config = cuda_graph_config
    if cfg.kt_weight_path and cuda_graph_config.prefill.backend != Backend.DISABLED:
        logger.warning(
            "Prefill CUDA graph is disabled for KT CPU experts; the KT "
            "host-side expert path is not safe to capture. Decode CUDA "
            "graph remains enabled."
        )
        new_config = with_phase(
            new_config,
            Phase.PREFILL,
            backend=Backend.DISABLED,
        )

    declared: dict[str, Any] = {}
    if cfg.kt_lora_path or cfg.kt_expert_lora_path:
        declared["disable_cuda_graph"] = True
        if new_config.decode.backend != Backend.DISABLED:
            new_config = with_phase(
                new_config,
                Phase.DECODE,
                backend=Backend.DISABLED,
            )
        if new_config.prefill.backend != Backend.DISABLED:
            new_config = with_phase(
                new_config,
                Phase.PREFILL,
                backend=Backend.DISABLED,
            )

    if new_config is not cuda_graph_config:
        declared["cuda_graph_config"] = new_config
    if declared:
        declare_resolution(
            server_args,
            "enforce_kt_cuda_graph_compatibility",
            **declared,
        )


def validate_kt_args(server_args: Any) -> None:
    """Validate KT expert placement and CPU runtime options."""

    cfg = resolving_view(server_args)

    if cfg.kt_threadpool_count <= 0:
        raise ValueError("--kt-threadpool-count must be positive.")
    if cfg.kt_numa_nodes is not None:
        if len(cfg.kt_numa_nodes) != cfg.kt_threadpool_count:
            raise ValueError(
                "--kt-numa-nodes must contain exactly --kt-threadpool-count "
                f"entries (got {len(cfg.kt_numa_nodes)} and "
                f"{cfg.kt_threadpool_count})."
            )
        if any(node < 0 for node in cfg.kt_numa_nodes):
            raise ValueError("--kt-numa-nodes values must be non-negative.")

    if cfg.kt_num_gpu_experts is not None and cfg.kt_num_gpu_experts < 0:
        raise ValueError("--kt-num-gpu-experts must be non-negative.")
    if cfg.kt_gpu_experts_ratio is not None and not (
        0.0 <= cfg.kt_gpu_experts_ratio <= 1.0
    ):
        raise ValueError("--kt-gpu-experts-ratio must be between 0 and 1.")
    if cfg.kt_num_gpu_layers is not None and cfg.kt_num_gpu_layers < 0:
        raise ValueError("--kt-num-gpu-layers must be non-negative.")
    if (
        cfg.kt_max_deferred_experts_per_token is not None
        and cfg.kt_max_deferred_experts_per_token < 0
    ):
        raise ValueError("--kt-max-deferred-experts-per-token must be non-negative.")
    if (
        cfg.kt_gpu_prefill_token_threshold is not None
        and cfg.kt_gpu_prefill_token_threshold < 0
    ):
        raise ValueError("--kt-gpu-prefill-token-threshold must be non-negative.")

    placement_strategies = {"frequency", "front-loading", "uniform", "random"}
    if cfg.kt_expert_placement_strategy not in placement_strategies:
        raise ValueError(
            "--kt-expert-placement-strategy must be one of "
            f"{sorted(placement_strategies)}, got "
            f"{cfg.kt_expert_placement_strategy!r}."
        )

    if cfg.kt_weight_path is not None and (
        cfg.kt_num_gpu_experts is None and cfg.kt_gpu_experts_ratio is None
    ):
        raise ValueError(
            "--kt-weight-path requires --kt-num-gpu-experts or --kt-gpu-experts-ratio."
        )

    if cfg.kt_weight_path is not None:
        if cfg.ep_size != 1:
            raise ValueError(
                "--kt-weight-path currently requires --ep-size 1; compact "
                "resident expert rows are rank-local and do not yet implement "
                "the EP>1 kernel ABI."
            )
        if cfg.moe_a2a_backend != "none":
            raise ValueError(
                "--kt-weight-path currently requires --moe-a2a-backend none; "
                "KTEP needs the standard routed dispatch output and cannot "
                "consume A2A dispatcher outputs."
            )
        if cfg.moe_runner_backend in ("triton_kernel", "hpc_ops"):
            raise ValueError(
                "--kt-weight-path is incompatible with --moe-runner-backend "
                f"{cfg.moe_runner_backend}: that runner consumes global expert "
                "routing metadata rather than KTEP's compact resident-row ids."
            )
        if cfg.kt_cpuinfer is None or cfg.kt_cpuinfer <= 0:
            raise ValueError(
                "--kt-weight-path requires --kt-cpuinfer to be a positive integer."
            )
        if cfg.kt_cpuinfer < cfg.kt_threadpool_count:
            raise ValueError(
                "--kt-cpuinfer must be at least --kt-threadpool-count "
                f"(got {cfg.kt_cpuinfer} and {cfg.kt_threadpool_count})."
            )

    if cfg.kt_enable_dynamic_expert_update and not (
        cfg.kt_gpu_prefill_token_threshold and cfg.kt_gpu_prefill_token_threshold > 0
    ):
        raise ValueError(
            "--kt-enable-dynamic-expert-update requires a positive "
            "--kt-gpu-prefill-token-threshold."
        )

    if cfg.kt_lora_path or cfg.kt_expert_lora_path:
        if cfg.kt_weight_path is None:
            raise ValueError("KT expert LoRA requires --kt-weight-path.")
        if cfg.kt_gpu_experts_ratio is not None or cfg.kt_num_gpu_experts != 0:
            raise ValueError(
                "KT expert LoRA currently supports CPU experts only: set "
                "--kt-num-gpu-experts 0 and omit --kt-gpu-experts-ratio."
            )
        if cfg.kt_gpu_prefill_token_threshold:
            raise ValueError(
                "KT expert LoRA cannot be combined with "
                "--kt-gpu-prefill-token-threshold."
            )
        if cfg.kt_enable_dynamic_expert_update:
            raise ValueError(
                "KT expert LoRA cannot be combined with "
                "--kt-enable-dynamic-expert-update."
            )
        if cfg.kt_method.upper() not in {
            "AMXBF16",
            "BF16",
            "AMXINT8",
            "AMXINT4",
        }:
            raise ValueError(
                "KT expert LoRA supports only AMXBF16, BF16, AMXINT8, "
                f"or AMXINT4 CPU methods (got {cfg.kt_method!r})."
            )

    if (cfg.kt_lora_path or cfg.kt_expert_lora_path) and (
        cfg.max_running_requests is not None and cfg.max_running_requests < 2
    ):
        raise ValueError(
            "KT expert LoRA serving requires --max-running-requests >= 2 "
            f"(got {cfg.max_running_requests})."
        )
