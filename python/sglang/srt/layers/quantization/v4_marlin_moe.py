# SPDX-License-Identifier: Apache-2.0
"""Prepared-weight Marlin backend for DeepSeek V4 MXFP4 MoE.

The preparation API is deliberately out-of-place and stream ordered. A
layerwise loader can target either resident weight buffer without changing
tensor addresses. The compute API caches shape-dependent workspaces per CUDA
stream so steady-state prefill does not allocate or transpose weights.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import mxfp4_marlin_repack


def make_kt_mxfp4_marlin_method(gpu_method, prefix: str = ""):
    """Select the current Marlin method for KT's MXFP4 layerwise path.

    New SGLang may select FlashInfer CUTLASS for MXFP4 by default.  KT's
    layerwise slots need the deterministic Marlin representation, so unwrap
    that method's shared FP8 loader and create the current native Marlin
    method.  The regular non-KT selection remains untouched.
    """
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
        Mxfp4MarlinMoEMethod,
    )

    # Match the old KT layerwise backend whitelist.  SM90 keeps the current
    # FlashInfer backend; KT's direct prepared-weight Marlin path was validated
    # for Ada (SM89) and consumer Blackwell (SM120).
    if not get_v4_mxfp4_marlin_streaming_capability().available:
        return gpu_method

    if isinstance(gpu_method, Mxfp4MarlinMoEMethod):
        gpu_method._kt_layerwise_enabled = True
        return gpu_method

    fp8_method = getattr(gpu_method, "_fp8", None)
    if fp8_method is None and getattr(gpu_method, "is_fp4_expert", False):
        # Some current configurations expose the shared FP8 loader directly
        # instead of wrapping it in an MXFP4-specific method.
        fp8_method = gpu_method
    if fp8_method is None:
        # The generic serialized-MXFP4 method has a different weight contract
        # (w13_weight_scale/w2_weight_scale) and is intentionally left on its
        # native backend.  The KT V4 layerwise path is only for the
        # DeepSeek-style `_scale_inv` contract.
        return gpu_method
    method = Mxfp4MarlinMoEMethod(fp8_method, prefix=prefix)
    method._kt_layerwise_enabled = True
    return method


V4_FP4_GROUP_SIZE = 32
_MARLIN_TILE = 16
_MAX_THREAD_N = 256
_KT_V4_MARLIN_STREAMING_COMPUTE_CAPABILITIES = frozenset({(8, 9), (12, 0)})
_V4_MARLIN_ACTIVATION_TYPES = {"silu": 0, "swiglu_oai": 1, "situ": 2}


def normalize_v4_mxfp4_activation(
    *,
    activation: str | None = None,
    swiglu_limit: float | None = None,
    swiglu_alpha: float = 0.0,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
) -> dict[str, float | int]:
    """Validate and normalize the shared KT CPU/GPU activation contract."""

    limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
    alpha = float(swiglu_alpha)
    beta = 0.0 if situ_beta is None else float(situ_beta)
    linear_beta = (
        0.0 if situ_linear_beta is None else float(situ_linear_beta)
    )
    activation = activation or ("swiglu_oai" if alpha > 0.0 else "silu")

    if activation not in _V4_MARLIN_ACTIVATION_TYPES:
        raise ValueError(f"unsupported V4 Marlin activation: {activation!r}")
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("swiglu_limit must be finite and non-negative")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("swiglu_alpha must be finite and non-negative")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("situ_beta must be finite and non-negative")
    if not math.isfinite(linear_beta) or linear_beta < 0.0:
        raise ValueError("situ_linear_beta must be finite and non-negative")

    if activation == "situ":
        if beta <= 0.0:
            raise ValueError("SiTU requires a finite positive situ_beta")
        if alpha != 0.0 or limit != 0.0:
            raise ValueError("SiTU must not reuse SwiGLU-OAI alpha/clamp fields")
    elif situ_beta is not None or situ_linear_beta is not None:
        raise ValueError("SiTU beta parameters require activation='situ'")
    elif activation == "swiglu_oai" and alpha <= 0.0:
        raise ValueError("SwiGLU-OAI requires a finite positive swiglu_alpha")
    elif activation == "silu" and alpha != 0.0:
        raise ValueError("plain SiLU cannot use swiglu_alpha")

    return {
        "activation_type": _V4_MARLIN_ACTIVATION_TYPES[activation],
        "swiglu_limit": limit,
        "swiglu_alpha": alpha,
        "situ_beta": beta,
        "situ_linear_beta": linear_beta,
    }


@dataclass(frozen=True)
class V4MarlinStreamingCapability:
    """Runtime capability reported to KT's streamed-expert scheduler.

    The first implementation deliberately exposes only the backend contract
    that is already implemented and numerically covered here.  Scheduling,
    staging-pool depth, and persistent resident publication remain owned by
    the KT wrapper.
    """

    available: bool
    reason: str
    single_expert_prepare: bool = True
    caller_owned_output: bool = True
    streamed_ready_group: bool = True


def get_v4_mxfp4_marlin_streaming_capability(
    device: torch.device | str | int | None = None,
) -> V4MarlinStreamingCapability:
    """Return an explicit, non-destructive streaming capability result."""
    if not torch.cuda.is_available():
        return V4MarlinStreamingCapability(False, "CUDA is not available")

    if isinstance(device, int):
        resolved_device = torch.device("cuda", device)
    elif device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    else:
        resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return V4MarlinStreamingCapability(
            False, f"V4 MXFP4 Marlin requires a CUDA device, got {resolved_device}"
        )

    compute_capability = torch.cuda.get_device_capability(resolved_device)
    if compute_capability not in _KT_V4_MARLIN_STREAMING_COMPUTE_CAPABILITIES:
        supported = ", ".join(
            f"SM{major}{minor}"
            for major, minor in sorted(_KT_V4_MARLIN_STREAMING_COMPUTE_CAPABILITIES)
        )
        return V4MarlinStreamingCapability(
            False,
            "KT V4 MXFP4 streamed Marlin is validated only on "
            f"{supported}; got SM{compute_capability[0]}{compute_capability[1]}",
        )
    return V4MarlinStreamingCapability(True, "supported")


def require_v4_mxfp4_marlin_streaming_support(
    device: torch.device | str | int | None = None,
) -> V4MarlinStreamingCapability:
    """Fail fast unless the validated KT streamed-Marlin path is available."""
    capability = get_v4_mxfp4_marlin_streaming_capability(device)
    if not capability.available:
        raise RuntimeError(capability.reason)
    return capability


@dataclass
class V4MarlinPreparedWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int


@triton.jit
def _swizzle_e8m0_scales_kernel(
    src,
    dst,
    total,
    per_expert,
    n,
    k_groups,
    SRC_IS_E8M0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    expert = offsets // per_expert
    out_index = offsets - expert * per_expert

    # Marlin scale order is an 8x8 transpose followed by a 4-lane swap.
    marlin_index = (out_index // 4) * 4
    lane4 = out_index % 4
    marlin_index += tl.where(lane4 == 1, 2, tl.where(lane4 == 2, 1, lane4))
    tile = marlin_index // 64
    lane64 = marlin_index - tile * 64
    transposed_index = tile * 64 + (lane64 % 8) * 8 + lane64 // 8

    group = transposed_index // n
    col = transposed_index - group * n
    src_index = expert * per_expert + col * k_groups + group
    value = tl.load(src + src_index, mask=mask, other=0)
    if SRC_IS_E8M0:
        bits = value.to(tl.uint8)
    else:
        exponent = tl.floor(tl.log2(value.to(tl.float32)) + 0.5) + 127.0
        exponent = tl.where(value > 0, exponent, 0.0)
        bits = tl.maximum(0.0, tl.minimum(255.0, exponent)).to(tl.uint8)
    tl.store(dst + offsets, bits, mask=mask)


def _swizzle_e8m0_scales(
    src: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    experts = src.shape[0]
    expected = (experts, size_n, size_k // V4_FP4_GROUP_SIZE)
    if tuple(src.shape) != expected:
        raise ValueError(f"expected scales {expected}, got {tuple(src.shape)}")
    output_shape = (experts, size_k // V4_FP4_GROUP_SIZE, size_n)
    if out is None:
        out = torch.empty(output_shape, dtype=torch.float8_e8m0fnu, device=src.device)
    elif (
        tuple(out.shape) != output_shape
        or out.dtype != torch.float8_e8m0fnu
        or out.device != src.device
    ):
        raise ValueError(
            f"scale out must be float8_e8m0fnu {output_shape} on {src.device}, "
            f"got {out.dtype} {tuple(out.shape)} on {out.device}"
        )

    src_is_bits = src.dtype in (torch.int8, torch.uint8, torch.float8_e8m0fnu)
    src_arg = src.view(torch.uint8) if src_is_bits else src
    total = out.numel()
    _swizzle_e8m0_scales_kernel[(triton.cdiv(total, 256),)](
        src_arg,
        out.view(torch.uint8),
        total,
        (size_k // V4_FP4_GROUP_SIZE) * size_n,
        size_n,
        size_k // V4_FP4_GROUP_SIZE,
        SRC_IS_E8M0=src_is_bits,
        BLOCK=256,
    )
    return out


def _prepared_shapes(num_experts: int, hidden_size: int, intermediate_size: int):
    return (
        (num_experts, hidden_size // _MARLIN_TILE, 4 * intermediate_size),
        (num_experts, hidden_size // V4_FP4_GROUP_SIZE, 2 * intermediate_size),
        (num_experts, intermediate_size // _MARLIN_TILE, 2 * hidden_size),
        (num_experts, intermediate_size // V4_FP4_GROUP_SIZE, hidden_size),
    )


def _validate_dimensions(num_experts: int, hidden_size: int, intermediate_size: int):
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if hidden_size % 64 or intermediate_size % 64:
        raise ValueError(
            "Marlin requires hidden/intermediate multiples of 64, got "
            f"{hidden_size}/{intermediate_size}"
        )


def get_v4_mxfp4_marlin_storage_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """Return bytes required by one prepared V4 MXFP4 image."""
    _validate_dimensions(num_experts, hidden_size, intermediate_size)
    w13, w13_scale, w2, w2_scale = _prepared_shapes(
        num_experts, hidden_size, intermediate_size
    )
    return (
        math.prod(w13) * torch.tensor([], dtype=torch.int32).element_size()
        + math.prod(w13_scale)
        * torch.tensor([], dtype=torch.float8_e8m0fnu).element_size()
        + math.prod(w2) * torch.tensor([], dtype=torch.int32).element_size()
        + math.prod(w2_scale)
        * torch.tensor([], dtype=torch.float8_e8m0fnu).element_size()
    )


def allocate_v4_mxfp4_marlin(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> V4MarlinPreparedWeights:
    """Allocate stable caller-owned Marlin storage without reading weights."""
    _validate_dimensions(num_experts, hidden_size, intermediate_size)
    w13, w13_scale, w2, w2_scale = _prepared_shapes(
        num_experts, hidden_size, intermediate_size
    )
    return V4MarlinPreparedWeights(
        w13=torch.empty(w13, dtype=torch.int32, device=device),
        w13_scale=torch.empty(w13_scale, dtype=torch.float8_e8m0fnu, device=device),
        w2=torch.empty(w2, dtype=torch.int32, device=device),
        w2_scale=torch.empty(w2_scale, dtype=torch.float8_e8m0fnu, device=device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )


def prepare_v4_mxfp4_marlin(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    out: V4MarlinPreparedWeights | None = None,
) -> V4MarlinPreparedWeights:
    """Prepare native DSV4 weights on the current CUDA stream."""
    raw_tensors = (w13, w13_scale, w2, w2_scale)
    if any(tensor.device != w13.device for tensor in raw_tensors):
        raise ValueError("all raw V4 MXFP4 tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in raw_tensors):
        raise ValueError("all raw V4 MXFP4 tensors must be contiguous")
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("V4 expert weights must be rank 3")
    experts = w13.shape[0]
    hidden_size = w13.shape[2] * 2
    intermediate_size = w2.shape[2] * 2
    if tuple(w13.shape) != (experts, 2 * intermediate_size, hidden_size // 2):
        raise ValueError(f"inconsistent w13 shape {tuple(w13.shape)}")
    if tuple(w2.shape) != (experts, hidden_size, intermediate_size // 2):
        raise ValueError(f"inconsistent w2 shape {tuple(w2.shape)}")
    _validate_dimensions(experts, hidden_size, intermediate_size)
    expected_w13_scale = (experts, 2 * intermediate_size, hidden_size // 32)
    expected_w2_scale = (experts, hidden_size, intermediate_size // 32)
    if (
        tuple(w13_scale.shape) != expected_w13_scale
        or tuple(w2_scale.shape) != expected_w2_scale
    ):
        raise ValueError(
            f"unexpected scale shapes {tuple(w13_scale.shape)}/{tuple(w2_scale.shape)}"
        )

    if out is None:
        out = allocate_v4_mxfp4_marlin(
            num_experts=experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=w13.device,
        )
    else:
        shapes = _prepared_shapes(experts, hidden_size, intermediate_size)
        actual = (
            tuple(out.w13.shape),
            tuple(out.w13_scale.shape),
            tuple(out.w2.shape),
            tuple(out.w2_scale.shape),
        )
        if actual != tuple(map(tuple, shapes)):
            raise ValueError(f"prepared output shapes {actual} do not match {shapes}")
        if (
            out.hidden_size != hidden_size
            or out.intermediate_size != intermediate_size
            or out.num_experts != experts
        ):
            raise ValueError("prepared output metadata does not match raw weights")
        if any(
            t.device != w13.device
            for t in (out.w13, out.w13_scale, out.w2, out.w2_scale)
        ):
            raise ValueError("prepared output must be on the raw weight device")

    mxfp4_marlin_repack(w13, hidden_size, 2 * intermediate_size, out.w13)
    mxfp4_marlin_repack(w2, intermediate_size, hidden_size, out.w2)
    _swizzle_e8m0_scales(
        w13_scale,
        size_k=hidden_size,
        size_n=2 * intermediate_size,
        out=out.w13_scale,
    )
    _swizzle_e8m0_scales(
        w2_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        out=out.w2_scale,
    )
    return out


def view_v4_mxfp4_marlin_slot(
    prepared: V4MarlinPreparedWeights, slot_index: int
) -> V4MarlinPreparedWeights:
    """Return a one-expert view without allocating or changing addresses."""
    if not 0 <= slot_index < prepared.num_experts:
        raise IndexError(
            f"slot_index {slot_index} is outside [0, {prepared.num_experts})"
        )
    expected = _prepared_shapes(
        prepared.num_experts,
        prepared.hidden_size,
        prepared.intermediate_size,
    )
    actual = (
        tuple(prepared.w13.shape),
        tuple(prepared.w13_scale.shape),
        tuple(prepared.w2.shape),
        tuple(prepared.w2_scale.shape),
    )
    if actual != tuple(map(tuple, expected)):
        raise ValueError(
            f"prepared storage shapes {actual} do not match metadata {expected}"
        )
    tensors = (
        prepared.w13,
        prepared.w13_scale,
        prepared.w2,
        prepared.w2_scale,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "prepared storage must be contiguous before taking a slot view"
        )

    return V4MarlinPreparedWeights(
        w13=prepared.w13[slot_index : slot_index + 1],
        w13_scale=prepared.w13_scale[slot_index : slot_index + 1],
        w2=prepared.w2[slot_index : slot_index + 1],
        w2_scale=prepared.w2_scale[slot_index : slot_index + 1],
        hidden_size=prepared.hidden_size,
        intermediate_size=prepared.intermediate_size,
        num_experts=1,
    )


def _add_single_expert_dimension(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(
        f"{name} must describe exactly one expert as rank 2 or [1, ...], "
        f"got {tuple(tensor.shape)}"
    )


def prepare_v4_mxfp4_marlin_expert(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    out: V4MarlinPreparedWeights,
    slot_index: int = 0,
) -> V4MarlinPreparedWeights:
    """Prepare one raw expert directly into one caller-owned prepared slot.

    ``out`` may be the full resident/staging image.  Only ``slot_index`` is
    written; the returned object is a one-expert view suitable for a streamed
    ready-group runner.  The operation is ordered on the current CUDA stream.
    """
    slot = view_v4_mxfp4_marlin_slot(out, slot_index)
    return prepare_v4_mxfp4_marlin(
        _add_single_expert_dimension("w13", w13),
        _add_single_expert_dimension("w13_scale", w13_scale),
        _add_single_expert_dimension("w2", w2),
        _add_single_expert_dimension("w2_scale", w2_scale),
        out=slot,
    )


@triton.jit
def _sanitize_topk_kernel(
    ids_in, weights_in, ids_out, weights_out, total, experts, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    ids = tl.load(ids_in + offsets, mask=mask, other=-1).to(tl.int32)
    weights = tl.load(weights_in + offsets, mask=mask, other=0.0).to(tl.float32)
    valid = (ids >= 0) & (ids < experts)
    tl.store(ids_out + offsets, tl.where(valid, ids, 0), mask=mask)
    tl.store(weights_out + offsets, tl.where(valid, weights, 0.0), mask=mask)


@triton.jit
def _gated_activation_kernel(
    inp,
    out,
    total,
    n,
    swiglu_limit,
    swiglu_alpha,
    situ_beta,
    situ_linear_beta,
    ACTIVATION_TYPE: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    HAS_SITU_LINEAR_BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    row = offsets // n
    col = offsets - row * n
    gate = tl.load(inp + row * (2 * n) + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(inp + row * (2 * n) + n + col, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION_TYPE == 2:
        gate_tanh = 2.0 * tl.sigmoid(2.0 * gate / situ_beta) - 1.0
        gate = situ_beta * gate_tanh * tl.sigmoid(gate)
        if HAS_SITU_LINEAR_BETA:
            up_tanh = 2.0 * tl.sigmoid(2.0 * up / situ_linear_beta) - 1.0
            up = situ_linear_beta * up_tanh
        value = gate * up
    else:
        if HAS_LIMIT:
            gate = tl.minimum(gate, swiglu_limit)
            up = tl.maximum(-swiglu_limit, tl.minimum(up, swiglu_limit))
        if ACTIVATION_TYPE == 1:
            value = gate * tl.sigmoid(gate * swiglu_alpha) * (up + 1.0)
        else:
            value = gate * tl.sigmoid(gate) * up
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _topk_reduce_kernel(
    inp, out, m, k, routed_scale, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < m) & (cols < k)
    acc = tl.zeros((BLOCK,), tl.float32)
    for topk_index in range(TOPK):
        acc += tl.load(
            inp + (row * TOPK + topk_index) * k + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(out + row * k + cols, acc * routed_scale, mask=mask)


def _select_block_size(m: int, topk: int, experts: int) -> int:
    average = m * topk / max(experts, 1)
    for block in (8, 16, 32, 48, 64):
        if average / block < 0.9:
            return block
    return 64


@dataclass
class _V4MarlinWorkspace:
    capacity_m: int
    block_size: int
    ids: torch.Tensor
    weights: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_pad: torch.Tensor
    cumsum: torch.Tensor
    locks: torch.Tensor
    intermediate13: torch.Tensor
    intermediate2: torch.Tensor
    c_tmp: torch.Tensor
    output: torch.Tensor
    empty: torch.Tensor


_WORKSPACES: dict[tuple, _V4MarlinWorkspace] = {}


def _routing_capacity(m: int, topk: int, experts: int, block: int) -> int:
    tokens = m * topk
    if tokens < experts + 1:
        return tokens * block
    return tokens + (experts + 1) * (block - 1)


def _get_workspace(
    hidden_states: torch.Tensor,
    weights: V4MarlinPreparedWeights,
    topk: int,
) -> _V4MarlinWorkspace:
    m = hidden_states.shape[0]
    capacity_m = max(1, triton.next_power_of_2(m))
    block = _select_block_size(capacity_m, topk, weights.num_experts)
    stream_id = torch.cuda.current_stream(hidden_states.device).cuda_stream
    key = (
        hidden_states.device.index,
        stream_id,
        hidden_states.dtype,
        weights.hidden_size,
        weights.intermediate_size,
        weights.num_experts,
        topk,
        capacity_m,
        block,
    )
    cached = _WORKSPACES.get(key)
    if cached is not None:
        return cached

    device = hidden_states.device
    k = weights.hidden_size
    n = weights.intermediate_size
    routed = _routing_capacity(capacity_m, topk, weights.num_experts, block)
    route_blocks = triton.cdiv(routed, block)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    locks_size = max(1, min((max(2 * n, k) // 64) * route_blocks, sms * 4))
    c_tmp_first = min(2 * n * routed, sms * 4 * block * _MAX_THREAD_N)
    c_tmp_second = min(k * routed, sms * 4 * block * _MAX_THREAD_N)
    if block == 8:
        c_tmp_first *= 2
        c_tmp_second *= 2
    c_tmp_size = max(1, c_tmp_first, c_tmp_second)
    routed_rows = capacity_m * topk

    cached = _V4MarlinWorkspace(
        capacity_m=capacity_m,
        block_size=block,
        ids=torch.empty((capacity_m, topk), dtype=torch.int32, device=device),
        weights=torch.empty((capacity_m, topk), dtype=torch.float32, device=device),
        sorted_ids=torch.empty(routed, dtype=torch.int32, device=device),
        expert_ids=torch.empty(route_blocks, dtype=torch.int32, device=device),
        num_tokens_post_pad=torch.empty(1, dtype=torch.int32, device=device),
        cumsum=torch.empty(weights.num_experts + 2, dtype=torch.int32, device=device),
        locks=torch.zeros(locks_size, dtype=torch.int32, device=device),
        intermediate13=torch.empty(
            routed_rows * max(2 * n, k), dtype=hidden_states.dtype, device=device
        ),
        intermediate2=torch.empty(
            (routed_rows, n), dtype=hidden_states.dtype, device=device
        ),
        c_tmp=torch.empty(c_tmp_size, dtype=torch.float32, device=device),
        output=torch.empty((capacity_m, k), dtype=hidden_states.dtype, device=device),
        empty=torch.empty(0, dtype=hidden_states.dtype, device=device),
    )
    _WORKSPACES[key] = cached
    return cached


def apply_v4_marlin_moe(
    *,
    hidden_states: torch.Tensor,
    prepared: V4MarlinPreparedWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float | None = None,
    activation: str | None = None,
    swiglu_alpha: float = 0.0,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
) -> torch.Tensor:
    """Execute DSV4 MXFP4 MoE with deterministic non-atomic Marlin GEMMs."""
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            f"V4 Marlin requires BF16 activations, got {hidden_states.dtype}"
        )
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    m, k = hidden_states.shape
    if k != prepared.hidden_size:
        raise ValueError(f"hidden size {k} != prepared {prepared.hidden_size}")
    if m == 0 or prepared.num_experts == 0:
        return torch.zeros_like(hidden_states)
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk ids/weights must be matching rank-2 tensors")
    activation_contract = normalize_v4_mxfp4_activation(
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    topk = topk_ids.shape[1]
    workspace = _get_workspace(hidden_states, prepared, topk)
    n = prepared.intermediate_size
    routed_rows = m * topk

    ids = workspace.ids[:m]
    routing_weights = workspace.weights[:m]
    _sanitize_topk_kernel[(triton.cdiv(routed_rows, 256),)](
        topk_ids,
        topk_weights,
        ids,
        routing_weights,
        routed_rows,
        prepared.num_experts,
        BLOCK=256,
    )

    from sgl_kernel import moe_align_block_size
    from sgl_kernel.scalar_type import scalar_types

    moe_align_block_size(
        ids,
        prepared.num_experts + 1,
        workspace.block_size,
        workspace.sorted_ids,
        workspace.expert_ids,
        workspace.num_tokens_post_pad,
        workspace.cumsum,
        True,
    )

    intermediate1 = workspace.intermediate13[: routed_rows * 2 * n].view(
        routed_rows, 2 * n
    )
    intermediate3 = workspace.intermediate13[: routed_rows * k].view(routed_rows, k)
    intermediate2 = workspace.intermediate2[:routed_rows].view(routed_rows, n)
    fp4_type = scalar_types.float4_e2m1f

    common = {
        "b_bias_or_none": None,
        "global_scale_or_none": None,
        "b_zeros_or_none": None,
        "g_idx_or_none": None,
        "perm_or_none": None,
        "workspace": workspace.locks,
        "sorted_token_ids": workspace.sorted_ids,
        "expert_ids": workspace.expert_ids,
        "num_tokens_post_padded": workspace.num_tokens_post_pad,
        "use_atomic_add": False,
        "use_fp32_reduce": True,
        "c_tmp_or_none": workspace.c_tmp,
        "empty_tensor_or_none": workspace.empty,
        "initialize_output": True,
    }
    moe_wna16_marlin_gemm(
        hidden_states,
        intermediate1,
        prepared.w13,
        b_scales=prepared.w13_scale,
        topk_weights=routing_weights,
        moe_block_size=workspace.block_size,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=fp4_type,
        size_m=m,
        size_n=2 * n,
        size_k=k,
        **common,
    )

    _gated_activation_kernel[(triton.cdiv(routed_rows * n, 256),)](
        intermediate1,
        intermediate2,
        routed_rows * n,
        n,
        activation_contract["swiglu_limit"],
        activation_contract["swiglu_alpha"],
        activation_contract["situ_beta"],
        activation_contract["situ_linear_beta"],
        ACTIVATION_TYPE=activation_contract["activation_type"],
        HAS_LIMIT=activation_contract["swiglu_limit"] > 0.0,
        HAS_SITU_LINEAR_BETA=activation_contract["situ_linear_beta"] > 0.0,
        BLOCK=256,
    )

    moe_wna16_marlin_gemm(
        intermediate2,
        intermediate3,
        prepared.w2,
        b_scales=prepared.w2_scale,
        topk_weights=routing_weights,
        moe_block_size=workspace.block_size,
        top_k=1,
        mul_topk_weights=True,
        is_ep=False,
        b_q_type=fp4_type,
        size_m=routed_rows,
        size_n=k,
        size_k=n,
        **common,
    )

    output = workspace.output[:m]
    _topk_reduce_kernel[(m, triton.cdiv(k, 256))](
        intermediate3,
        output,
        m,
        k,
        routed_scaling_factor,
        TOPK=topk,
        BLOCK=256,
    )
    return output


def remap_v4_mxfp4_streamed_assignments(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    logical_expert_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter logical assignments and remap them to staging-local IDs.

    Candidate order is authoritative: ``logical_expert_ids[i]`` must refer to
    prepared staging slot ``i``.  Non-candidate assignments become ``(-1, 0)``
    so this wave contributes only the assignments claimed by the stream
    ticket and can be added to the resident/CPU outputs exactly once.
    """
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk ids/weights must be matching rank-2 tensors")
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk ids/weights must be on the same device")

    candidate_ids = tuple(int(expert_id) for expert_id in logical_expert_ids)
    if not candidate_ids:
        raise ValueError("logical_expert_ids must not be empty")
    if any(expert_id < 0 for expert_id in candidate_ids):
        raise ValueError("logical_expert_ids must be non-negative")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("logical_expert_ids must be unique")

    candidates = torch.tensor(
        candidate_ids, dtype=topk_ids.dtype, device=topk_ids.device
    )
    matches = topk_ids.unsqueeze(-1).eq(candidates.view(1, 1, -1))
    claimed = matches.any(dim=-1)
    local_ids = matches.to(torch.int32).argmax(dim=-1)
    local_ids = local_ids.masked_fill(~claimed, -1)
    streamed_weights = topk_weights.masked_fill(~claimed, 0)
    return local_ids, streamed_weights


def apply_v4_mxfp4_marlin_streamed_experts(
    *,
    hidden_states: torch.Tensor,
    prepared_staging: V4MarlinPreparedWeights,
    logical_expert_ids: Sequence[int],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float | None = None,
    activation: str | None = None,
    swiglu_alpha: float = 0.0,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a candidate-private MXFP4 wave from prepared staging weights.

    This is the first-version Python-level streamed runner.  It reuses the
    existing deterministic Marlin MoE kernels, while making the logical to
    staging-local mapping explicit.  The result owns its storage (or is copied
    into caller-owned ``out``), so a later ready-group cannot overwrite it via
    the internal workspace cache.
    """
    candidate_ids = tuple(int(expert_id) for expert_id in logical_expert_ids)
    if len(candidate_ids) != prepared_staging.num_experts:
        raise ValueError(
            "logical_expert_ids count must match prepared staging slots: "
            f"{len(candidate_ids)} != {prepared_staging.num_experts}"
        )
    if any(
        tensor.device != hidden_states.device
        for tensor in (
            prepared_staging.w13,
            prepared_staging.w13_scale,
            prepared_staging.w2,
            prepared_staging.w2_scale,
            topk_ids,
            topk_weights,
        )
    ):
        raise ValueError(
            "activations, routing tensors, and staging must share a device"
        )

    local_ids, streamed_weights = remap_v4_mxfp4_streamed_assignments(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        logical_expert_ids=candidate_ids,
    )
    result = apply_v4_marlin_moe(
        hidden_states=hidden_states,
        prepared=prepared_staging,
        topk_weights=streamed_weights,
        topk_ids=local_ids,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if out is None:
        return result.clone()
    if (
        tuple(out.shape) != tuple(hidden_states.shape)
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
    ):
        raise ValueError(
            "streamed output must match hidden_states shape/dtype/device, got "
            f"{tuple(out.shape)} {out.dtype} on {out.device}"
        )
    out.copy_(result)
    return out
