"""Optional fused copy backends for KT prefill staging.

The prefill manager stages several MXFP4 raw banks with the same logical expert
rows.  The scalar fallback issues one Torch copy per expert/bank (D2D) or one
per contiguous run/bank (H2D).  This module exposes two equivalent batched
paths when the CUDA JIT and runtime support them:

* a multi-bank index-copy kernel for resident D2D rows, including the optional
  W13 gate/up half swap; and
* one ``cudaMemcpyBatchAsync`` call for a whole pinned host generation.

Both helpers are opportunistic: they return ``False`` before launching any
copy when the layout is unsupported, so callers keep their existing loop as a
correct fallback.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence

import torch

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


_DISABLE_ENV = "SGLANG_KT_DISABLE_PREFILL_FAST_COPY"
_TRUE_VALUES = {"1", "true", "yes", "on"}

_D2D_RUNTIME_FAILED = False
_BATCH_RUNTIME_FAILED = False


@dataclass(frozen=True)
class RowCopyBank:
    """One raw bank participating in the fused resident D2D copy."""

    source: torch.Tensor
    destination: torch.Tensor
    swap_halves: bool = False


@dataclass(frozen=True)
class BatchCopy:
    """One independent H2D span in a pinned host generation."""

    source: torch.Tensor
    source_row: int
    destination: torch.Tensor
    destination_row: int
    row_bytes: int
    rows: int


@dataclass(frozen=True)
class RowCopyPlan:
    """Resolved banks and index rows for :func:`try_multi_row_copy`."""

    banks: tuple[RowCopyBank, ...]
    source_rows: tuple[int, ...]
    destination_rows: tuple[int, ...]


def _fast_path_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUE_VALUES


def row_nbytes(tensor: torch.Tensor) -> Optional[int]:
    """Return the byte size of one contiguous row, or ``None`` if unusable."""

    if tensor.dim() == 0 or tensor.numel() == 0 or tensor.shape[0] == 0:
        return None
    if not tensor.is_contiguous():
        return None
    return int(tensor.numel() // tensor.shape[0] * tensor.element_size())


def build_row_copy_plan(
    named_tensors: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    resident_rows: Sequence[tuple[int, int]],
    *,
    num_experts: int,
    old_row_owner: Sequence[int],
    w13_up_first: bool,
) -> Optional[RowCopyPlan]:
    """Resolve the shared source rows and per-bank descriptors for D2D."""

    if not named_tensors or not resident_rows:
        return None
    banks: list[RowCopyBank] = []
    shared_source_rows: Optional[list[int]] = None
    for name, source, destination in named_tensors:
        try:
            if source.shape[0] == int(num_experts):
                source_rows = [expert_id for expert_id, _ in resident_rows]
            else:
                source_rows = [
                    old_row_owner.index(expert_id)
                    for expert_id, _ in resident_rows
                ]
        except ValueError:
            return None
        if shared_source_rows is None:
            shared_source_rows = source_rows
        elif shared_source_rows != source_rows:
            return None
        banks.append(
            RowCopyBank(
                source=source,
                destination=destination,
                swap_halves=(
                    bool(w13_up_first)
                    and name in ("w13_weight", "w13_weight_scale_inv")
                ),
            )
        )
    if shared_source_rows is None:
        return None
    return RowCopyPlan(
        banks=tuple(banks),
        source_rows=tuple(shared_source_rows),
        destination_rows=tuple(destination_row for _, destination_row in resident_rows),
    )


@lru_cache(maxsize=1)
def _load_module() -> Optional[Module]:
    if _fast_path_disabled() or not torch.cuda.is_available():
        return None
    try:
        from sglang.kernels.jit.utils import load_jit

        module = load_jit(
            "kt_prefill_copy",
            cuda_files=["moe/kt_prefill_copy.cuh"],
            cuda_wrappers=[
                ("multi_row_copy", "&kt::MultiRowCopy::run"),
                ("batch_memcpy", "&kt::BatchMemcpy::run"),
            ],
            extra_cuda_cflags=["-O3"],
        )
    except Exception as exc:
        logger.warning(
            "KT prefill fused-copy kernels are unavailable; using the scalar "
            "copy loops: %s",
            exc,
            exc_info=True,
        )
        return None
    logger.info("KT prefill fused-copy kernels loaded")
    return module


def _cuda_major_version() -> Optional[int]:
    version = torch.version.cuda
    if not version:
        return None
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _probe_batch_memcpy(function) -> None:
    source = torch.arange(16, dtype=torch.uint8).pin_memory()
    destination = torch.zeros(16, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream(device=destination.device)
    function(
        torch.tensor([destination.data_ptr()], dtype=torch.int64),
        torch.tensor([source.data_ptr()], dtype=torch.int64),
        torch.tensor([16], dtype=torch.int64),
        stream.cuda_stream,
    )
    stream.synchronize()
    if not torch.equal(destination.cpu(), source):
        raise RuntimeError("cudaMemcpyBatchAsync probe copied the wrong bytes")


@lru_cache(maxsize=1)
def _batch_function():
    if _fast_path_disabled() or not torch.cuda.is_available():
        return None
    if (_cuda_major_version() or 0) < 13:
        logger.info("KT prefill batch H2D disabled: CUDA 13.0 or newer is required")
        return None
    module = _load_module()
    if module is None:
        return None
    try:
        function = module.batch_memcpy
        _probe_batch_memcpy(function)
    except Exception as exc:
        logger.warning(
            "KT prefill batch H2D probe failed; keeping the per-run copies: %s",
            exc,
            exc_info=True,
        )
        return None
    logger.info("KT prefill batch H2D enabled (cudaMemcpyBatchAsync)")
    return function


def _row_bank_supported(bank: RowCopyBank) -> bool:
    source = bank.source
    destination = bank.destination
    if not source.is_cuda or not destination.is_cuda:
        return False
    if source.device != destination.device:
        return False
    if source.dtype != destination.dtype:
        return False
    if not source.is_contiguous() or not destination.is_contiguous():
        return False
    if tuple(source.shape[1:]) != tuple(destination.shape[1:]):
        return False
    source_bytes = row_nbytes(source)
    destination_bytes = row_nbytes(destination)
    if source_bytes is None or source_bytes != destination_bytes:
        return False
    if source_bytes % 16 != 0:
        return False
    if bank.swap_halves and source_bytes % 32 != 0:
        return False
    if source.data_ptr() % 16 != 0 or destination.data_ptr() % 16 != 0:
        return False
    return True


def try_multi_row_copy(
    banks: Sequence[RowCopyBank],
    source_rows: Sequence[int],
    destination_rows: Sequence[int],
    device: torch.device,
) -> bool:
    """Copy the same row set for every bank with one CUDA kernel launch."""

    global _D2D_RUNTIME_FAILED

    if (
        _D2D_RUNTIME_FAILED
        or _fast_path_disabled()
        or not banks
        or not source_rows
        or len(source_rows) != len(destination_rows)
    ):
        return False
    if any(not _row_bank_supported(bank) for bank in banks):
        return False
    for bank in banks:
        if any(row < 0 or row >= bank.source.shape[0] for row in source_rows):
            return False
        if any(row < 0 or row >= bank.destination.shape[0] for row in destination_rows):
            return False

    module = _load_module()
    if module is None:
        return False
    try:
        dst_ptrs = torch.tensor(
            [bank.destination.data_ptr() for bank in banks],
            dtype=torch.int64,
            device=device,
        )
        src_ptrs = torch.tensor(
            [bank.source.data_ptr() for bank in banks],
            dtype=torch.int64,
            device=device,
        )
        feat_bytes = torch.tensor(
            [row_nbytes(bank.source) for bank in banks],
            dtype=torch.int64,
            device=device,
        )
        swap_halves = torch.tensor(
            [int(bank.swap_halves) for bank in banks],
            dtype=torch.int32,
            device=device,
        )
        dst_rows = torch.tensor(
            [int(row) for row in destination_rows],
            dtype=torch.int32,
            device=device,
        )
        src_rows = torch.tensor(
            [int(row) for row in source_rows],
            dtype=torch.int32,
            device=device,
        )
        module.multi_row_copy(
            dst_ptrs,
            src_ptrs,
            feat_bytes,
            swap_halves,
            dst_rows,
            src_rows,
        )
    except Exception as exc:
        _D2D_RUNTIME_FAILED = True
        logger.warning(
            "KT prefill fused D2D copy failed; falling back to scalar copies: %s",
            exc,
            exc_info=True,
        )
        return False
    return True


def _batch_copy_supported(copy: BatchCopy) -> bool:
    source = copy.source
    destination = copy.destination
    if source.is_cuda or not destination.is_cuda:
        return False
    if source.dtype != destination.dtype:
        return False
    if not source.is_contiguous() or not destination.is_contiguous():
        return False
    if tuple(source.shape[1:]) != tuple(destination.shape[1:]):
        return False
    source_bytes = row_nbytes(source)
    destination_bytes = row_nbytes(destination)
    if source_bytes is None or destination_bytes is None:
        return False
    if source_bytes != copy.row_bytes or destination_bytes != copy.row_bytes:
        return False
    if copy.rows <= 0:
        return False
    if copy.source_row < 0 or copy.source_row + copy.rows > source.shape[0]:
        return False
    if (
        copy.destination_row < 0
        or copy.destination_row + copy.rows > destination.shape[0]
    ):
        return False
    return True


def try_batch_memcpy(copies: Sequence[BatchCopy], stream: torch.cuda.Stream) -> bool:
    """Submit one host-to-device batch for all independent generation spans."""

    global _BATCH_RUNTIME_FAILED

    if (
        _BATCH_RUNTIME_FAILED
        or _fast_path_disabled()
        or not copies
        or getattr(stream, "cuda_stream", 0) == 0
    ):
        return False
    if any(not _batch_copy_supported(copy) for copy in copies):
        return False

    function = _batch_function()
    if function is None:
        return False

    destination_ptrs = []
    source_ptrs = []
    sizes = []
    for copy in copies:
        source_offset = copy.source_row * copy.row_bytes
        destination_offset = copy.destination_row * copy.row_bytes
        destination_ptrs.append(copy.destination.data_ptr() + destination_offset)
        source_ptrs.append(copy.source.data_ptr() + source_offset)
        sizes.append(copy.rows * copy.row_bytes)

    try:
        function(
            torch.tensor(destination_ptrs, dtype=torch.int64),
            torch.tensor(source_ptrs, dtype=torch.int64),
            torch.tensor(sizes, dtype=torch.int64),
            stream.cuda_stream,
        )
    except Exception as exc:
        _BATCH_RUNTIME_FAILED = True
        logger.warning(
            "KT prefill batch H2D failed; falling back to per-run copies: %s",
            exc,
            exc_info=True,
        )
        return False
    return True


def reset_fast_path_cache() -> None:
    """Clear one-time load/probe state (used by unit tests)."""

    global _D2D_RUNTIME_FAILED, _BATCH_RUNTIME_FAILED
    _load_module.cache_clear()
    _batch_function.cache_clear()
    _D2D_RUNTIME_FAILED = False
    _BATCH_RUNTIME_FAILED = False
