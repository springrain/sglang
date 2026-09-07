"""GLM-5-Next native zero-RoPE sparse MLA (SM120 DSA decode/extend).

FlashInfer's H512 sparse-MLA kernel does not run on SM120, and no existing
DeepSeek DSA impl accepts GLM's index_kpool>1 physical topk indices.  This
model-local path is graph-safe: decode is a BF16-query / FP8-KV Triton kernel
with online softmax; extend is a chunked PyTorch reference.

GLM-5.3-Flash's pool rows are exactly 512 fp8 bytes: the model gets no
``override_kv_cache_dim``, so ``MLATokenToKVPool`` writes the latent with a
plain ``.to(float8_e4m3fn)`` cast and no per-group descales exist anywhere.
Both entries therefore run with ``kv_scale=None`` (the USE_KV_SCALE constexpr
branch stays for pools that do carry a separate [slots, 4] fp32 scale buffer).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GLM5_NEXT_SPARSE_HEAD_DIM = 512
GLM5_NEXT_LATENT_SCALE_GROUP_SIZE = 128
GLM5_NEXT_LATENT_SCALE_GROUPS = (
    GLM5_NEXT_SPARSE_HEAD_DIM // GLM5_NEXT_LATENT_SCALE_GROUP_SIZE
)
# Dispatch keys off the EntryClass arch (models/glm5_next.py); older GlmMoeDsa
# checkpoints carry RoPE and keep the flashinfer_sparse_mla path.
GLM5_NEXT_MODEL_ARCHS = ("Glm5NextForConditionalGeneration",)


def glm5_next_sparse_mla_launch_config(
    capability: tuple[int, int],
) -> tuple[int, int, int]:
    """Return static decode geometry tuned per supported GPU family."""

    if capability == (8, 6):
        # Ampere has a smaller register file per SM; halve the selected-token
        # tile so the 512-wide online-softmax accumulator remains resident.
        return 8, 4, 2
    if capability == (8, 9):
        return 16, 8, 2
    # SM120 (and anything newer without a tuned entry) keeps this geometry.
    return 16, 8, 2


@triton.jit
def _glm5_next_sparse_mla_decode_kernel(
    query_ptr,
    kv_ptr,
    kv_scale_ptr,
    indices_ptr,
    output_ptr,
    sm_scale: tl.float32,
    num_kv_tokens: tl.int64,
    topk: tl.int32,
    stride_query_row: tl.constexpr,
    stride_query_head: tl.constexpr,
    stride_query_dim: tl.constexpr,
    stride_kv_row: tl.constexpr,
    stride_kv_dim: tl.constexpr,
    stride_kv_scale_row: tl.constexpr,
    stride_kv_scale_group: tl.constexpr,
    stride_indices_row: tl.constexpr,
    stride_indices_col: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_head: tl.constexpr,
    stride_output_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    USE_KV_SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    dims = tl.arange(0, HEAD_DIM)
    query = tl.load(
        query_ptr
        + row * stride_query_row
        + head * stride_query_head
        + dims * stride_query_dim
    ).to(tl.float32)

    running_max: tl.float32 = -1.0e30
    running_sum: tl.float32 = 0.0
    accumulator = tl.zeros([HEAD_DIM], tl.float32)
    token_offsets = tl.arange(0, BLOCK_T)

    # ``topk`` is a static tensor width at graph capture (2051 for the released
    # checkpoint), while validity remains data-driven through the -1 sentinel.
    for start in range(0, topk, BLOCK_T):
        cols = start + token_offsets
        in_bounds = cols < topk
        physical = tl.load(
            indices_ptr + row * stride_indices_row + cols * stride_indices_col,
            mask=in_bounds,
            other=-1,
        ).to(tl.int64)
        valid = in_bounds & (physical >= 0) & (physical < num_kv_tokens)
        safe_physical = tl.where(valid, physical, 0)

        kv = tl.load(
            kv_ptr
            + safe_physical[:, None] * stride_kv_row
            + dims[None, :] * stride_kv_dim,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        if USE_KV_SCALE:
            kv_scale = tl.load(
                kv_scale_ptr
                + safe_physical[:, None] * stride_kv_scale_row
                + (dims[None, :] // SCALE_GROUP_SIZE) * stride_kv_scale_group,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            kv *= kv_scale
        scores = tl.sum(kv * query[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, -1.0e30)

        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)
        probabilities = tl.where(valid, probabilities, 0.0)

        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * kv, axis=0
        )
        running_max = next_max

    denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
    output = accumulator / denominator
    tl.store(
        output_ptr
        + row * stride_output_row
        + head * stride_output_head
        + dims * stride_output_dim,
        output.to(tl.bfloat16),
    )


def glm5_next_sparse_mla_decode(
    query: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    sm_scale: float,
) -> torch.Tensor:
    """Graph-safe decode: [tokens, heads, 512] bf16 q, strided fp8 pool views.

    ``indices`` are physical token indices [tokens, topk] with -1 padding, as
    produced by the fused IndexerKPool top-k.  ``kv_scale=None`` reads pool
    values at face value (GLM-5.3-Flash's pool has no descales).
    """

    capability = torch.cuda.get_device_capability(query.device)
    query_3d = query.contiguous()
    indices_2d = indices.contiguous()
    use_kv_scale = kv_scale is not None
    if use_kv_scale:
        kv_scale_2d = kv_scale.contiguous()
    else:
        # Triton requires a pointer argument even when the constexpr branch is
        # dead.  Reuse a live input pointer; its strides are never consumed.
        kv_scale_2d = kv_fp8
    output = torch.empty_like(query_3d, dtype=torch.bfloat16)
    grid = (query_3d.shape[0], query_3d.shape[1])
    block_t, num_warps, num_stages = glm5_next_sparse_mla_launch_config(capability)
    _glm5_next_sparse_mla_decode_kernel[grid](
        query_3d,
        kv_fp8,
        kv_scale_2d,
        indices_2d,
        output,
        float(sm_scale),
        kv_fp8.shape[0],
        indices_2d.shape[1],
        query_3d.stride(0),
        query_3d.stride(1),
        query_3d.stride(2),
        kv_fp8.stride(0),
        kv_fp8.stride(1),
        kv_scale_2d.stride(0),
        kv_scale_2d.stride(1),
        indices_2d.stride(0),
        indices_2d.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=GLM5_NEXT_SPARSE_HEAD_DIM,
        SCALE_GROUP_SIZE=GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
        USE_KV_SCALE=use_kv_scale,
        BLOCK_T=block_t,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _apply_current_chunk_kv(
    selected_kv: torch.Tensor,
    safe_indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    current_chunk_kv: torch.Tensor,
    sorted_current_locs: torch.Tensor,
    current_order: torch.Tensor,
) -> torch.Tensor:
    # After allocator reuse, current locs are not monotonic; a sorted table plus
    # searchsorted replaces a pool-sized BF16 lookup (cache stays in FP8).
    insertion = torch.searchsorted(
        sorted_current_locs, safe_indices.to(torch.int64)
    )
    bounded = insertion.clamp_max(sorted_current_locs.numel() - 1)
    is_current = valid & (
        sorted_current_locs[bounded] == safe_indices.to(torch.int64)
    )
    current_rows = current_order[bounded]
    return torch.where(
        is_current.unsqueeze(-1),
        current_chunk_kv[current_rows],
        selected_kv,
    )


def glm5_next_sparse_mla_extend(
    query: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    current_chunk_kv: torch.Tensor,
    current_chunk_locs: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    """EXTEND/prefill reference: chunked bmm+softmax, BF16 rounding aligned.

    Selected history rows are gathered from the FP8 pool view (dequantized per
    128-channel group when ``kv_scale`` is given); rows selected at
    current-chunk locs use the pre-quantization BF16 ``current_chunk_kv``.
    """

    if query.ndim != 3 or query.shape[-1] != GLM5_NEXT_SPARSE_HEAD_DIM:
        raise ValueError(
            "GLM-5-Next sparse MLA query must have shape [tokens, heads, 512], "
            f"got {tuple(query.shape)}"
        )
    if indices.ndim != 2 or indices.shape[0] != query.shape[0]:
        raise ValueError(
            "GLM-5-Next sparse MLA indices must have shape [tokens, capacity], "
            f"got {tuple(indices.shape)} for {query.shape[0]} query tokens"
        )
    if indices.dtype != torch.int32:
        raise TypeError(
            f"GLM-5-Next sparse MLA indices must be int32, got {indices.dtype}"
        )
    if current_chunk_kv.ndim != 2 or current_chunk_kv.shape != (
        current_chunk_locs.numel(),
        GLM5_NEXT_SPARSE_HEAD_DIM,
    ):
        raise ValueError(
            "GLM-5-Next current-chunk KV must be [extend_tokens, 512] and match "
            "current_chunk_locs, got "
            f"{tuple(current_chunk_kv.shape)} / {tuple(current_chunk_locs.shape)}"
        )
    if current_chunk_kv.dtype != torch.bfloat16:
        raise TypeError(
            f"GLM-5-Next current-chunk KV must be BF16, got {current_chunk_kv.dtype}"
        )

    sorted_current_locs, current_order = torch.sort(
        current_chunk_locs.to(torch.int64)
    )
    num_kv_tokens = kv_fp8.shape[0]
    output = torch.empty(
        (*query.shape[:-1], GLM5_NEXT_SPARSE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=query.device,
    )

    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        chunk_indices = indices[start:end]
        valid = chunk_indices >= 0
        if bool(torch.any(chunk_indices[valid] >= num_kv_tokens).item()):
            bad_index = int(chunk_indices[valid].max().item())
            raise IndexError(
                "GLM-5-Next sparse MLA index exceeds the KV cache: "
                f"max index {bad_index}, cache tokens {num_kv_tokens}"
            )

        nonempty = valid.any(dim=-1)
        safe_indices = chunk_indices.masked_fill(~valid, 0).to(torch.long)
        selected_kv = kv_fp8[safe_indices].reshape(
            *safe_indices.shape, GLM5_NEXT_SPARSE_HEAD_DIM
        )
        if kv_scale is not None:
            selected_kv = (
                selected_kv.float()
                .reshape(
                    *selected_kv.shape[:-1],
                    GLM5_NEXT_LATENT_SCALE_GROUPS,
                    GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
                )
                .mul(kv_scale[safe_indices].float().unsqueeze(-1))
                .reshape(*selected_kv.shape[:-1], GLM5_NEXT_SPARSE_HEAD_DIM)
                .to(torch.bfloat16)
            )
        else:
            selected_kv = selected_kv.to(torch.bfloat16)
        selected_kv = _apply_current_chunk_kv(
            selected_kv,
            safe_indices,
            valid,
            current_chunk_kv=current_chunk_kv,
            sorted_current_locs=sorted_current_locs,
            current_order=current_order,
        )

        scores = torch.bmm(
            query[start:end].to(torch.bfloat16), selected_kv.transpose(1, 2)
        ).float()
        scores.mul_(float(sm_scale))
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        # Empty rows occur only for padded eager work; give softmax a finite
        # dummy slot and zero the resulting output below.
        if not bool(torch.all(nonempty).item()):
            scores[~nonempty] = 0.0

        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        chunk_output = torch.bmm(probabilities, selected_kv)
        chunk_output[~nonempty] = 0
        output[start:end] = chunk_output

    return output


__all__ = [
    "glm5_next_sparse_mla_decode",
    "glm5_next_sparse_mla_extend",
]
