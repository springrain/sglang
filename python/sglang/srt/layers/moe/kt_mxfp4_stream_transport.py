# SPDX-License-Identifier: Apache-2.0
"""Two-slot streamed-expert transport for KT DeepSeek-V4 MXFP4.

This module owns weight transport only.  It deliberately does not update the
resident logical mappings: the KT wrapper publishes a new mapping only after
``ticket.install(...)`` has returned an install event and the wrapper reaches
its safe boundary.

The first implementation is intentionally narrow:

* DeepSeek-V4's native ``*_scale_inv`` MXFP4 contract;
* KT's caller-owned Marlin prepared weights;
* two POSIX-SHM pinned host slots and two fixed-address GPU staging slots;
* TP0 is the sole producer and writes every TP rank's SHM mapping;
* one streamed candidate per ticket (H2D -> prepare -> private wave-2);
* optional D2D installation into a fixed resident Marlin slot.

The pool is shared by compatible layers on one device.  A wrapper must finish
each ticket with ``install`` or ``release`` before it can reuse that staging
slot for a later candidate.
"""

from __future__ import annotations

import atexit
import ctypes
import gc
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from multiprocessing import shared_memory
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

RAW_NAMES = (
    "w13_weight",
    "w13_weight_scale_inv",
    "w2_weight",
    "w2_weight_scale_inv",
)
SCALE_NAMES = frozenset(("w13_weight_scale_inv", "w2_weight_scale_inv"))
_POOL_DEPTH = 2


def _device_compute_capability(
    device: torch.device,
) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return int(major), int(minor)


class StreamSlotState(str, Enum):
    FREE = "FREE"
    WRITING = "WRITING"
    HOST_READY = "HOST_READY"
    TRANSFERRING = "TRANSFERRING"
    READY = "READY"
    IN_USE = "IN_USE"
    INSTALLING = "INSTALLING"
    POISONED = "POISONED"


class StreamTicketState(str, Enum):
    CREATED = "CREATED"
    WRITER_SUBMITTED = "WRITER_SUBMITTED"
    HOST_READY = "HOST_READY"
    TRANSFER_SUBMITTED = "TRANSFER_SUBMITTED"
    READY = "READY"
    WAVE2_LAUNCHED = "WAVE2_LAUNCHED"
    INSTALL_COMMITTED = "INSTALL_COMMITTED"
    INSTALLED = "INSTALLED"
    RELEASED = "RELEASED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


_TICKET_TRANSITIONS = {
    StreamTicketState.CREATED: {
        StreamTicketState.WRITER_SUBMITTED,
        StreamTicketState.ABORTED,
        StreamTicketState.FAILED,
    },
    StreamTicketState.WRITER_SUBMITTED: {
        StreamTicketState.HOST_READY,
        StreamTicketState.ABORTED,
        StreamTicketState.FAILED,
    },
    StreamTicketState.HOST_READY: {
        StreamTicketState.TRANSFER_SUBMITTED,
        StreamTicketState.ABORTED,
        StreamTicketState.FAILED,
    },
    StreamTicketState.TRANSFER_SUBMITTED: {
        StreamTicketState.READY,
        StreamTicketState.ABORTED,
        StreamTicketState.FAILED,
    },
    StreamTicketState.READY: {
        StreamTicketState.WAVE2_LAUNCHED,
        StreamTicketState.ABORTED,
        StreamTicketState.FAILED,
    },
    StreamTicketState.WAVE2_LAUNCHED: {
        StreamTicketState.INSTALL_COMMITTED,
        StreamTicketState.RELEASED,
        StreamTicketState.ABORTED,
        StreamTicketState.FAILED,
    },
    StreamTicketState.INSTALL_COMMITTED: {StreamTicketState.INSTALLED},
    StreamTicketState.INSTALLED: set(),
    StreamTicketState.RELEASED: set(),
    StreamTicketState.ABORTED: set(),
    StreamTicketState.FAILED: {StreamTicketState.ABORTED},
}


def validate_ticket_transition(
    current: StreamTicketState, target: StreamTicketState
) -> None:
    """Validate a ticket transition without requiring CUDA (unit-testable)."""

    if target not in _TICKET_TRANSITIONS[current]:
        raise RuntimeError(f"invalid streamed ticket transition: {current} -> {target}")


@dataclass(frozen=True)
class Mxfp4StreamCapability:
    supported: bool
    reason: str


@dataclass(frozen=True)
class RawTensorSpec:
    name: str
    expert_shape: tuple[int, ...]
    gpu_dtype: torch.dtype
    host_dtype: torch.dtype

    @property
    def host_expert_nbytes(self) -> int:
        elements = 1
        for dim in self.expert_shape:
            elements *= dim
        return elements * torch.empty((), dtype=self.host_dtype).element_size()


@dataclass(frozen=True)
class Mxfp4ExpertLayout:
    device: torch.device
    hidden_size: int
    intermediate_size: int
    raw_specs: tuple[RawTensorSpec, ...]
    compute_capability: tuple[int, int] | None = None

    @property
    def compatibility_signature(self) -> tuple:
        """Cross-rank layout identity without the process-local CUDA ordinal."""

        return (
            self.device.type,
            self.compute_capability,
            self.hidden_size,
            self.intermediate_size,
            tuple(
                (spec.name, spec.expert_shape, str(spec.gpu_dtype), str(spec.host_dtype))
                for spec in self.raw_specs
            ),
        )

    @property
    def local_registry_signature(self) -> tuple:
        """Process-local key that keeps distinct CUDA devices separated."""

        return (str(self.device), self.compatibility_signature)

    @property
    def signature(self) -> tuple:
        """Backward-compatible alias for the process-local registry signature."""

        return self.local_registry_signature

    def spec(self, name: str) -> RawTensorSpec:
        for spec in self.raw_specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    @classmethod
    def from_tensors(
        cls, tensors: Mapping[str, torch.Tensor]
    ) -> Mxfp4ExpertLayout:
        missing = [name for name in RAW_NAMES if name not in tensors]
        if missing:
            raise ValueError(f"missing DeepSeek-V4 MXFP4 raw tensors: {missing}")

        raw = {name: tensors[name] for name in RAW_NAMES}
        for name, tensor in raw.items():
            if tensor.ndim != 3 or tensor.shape[0] <= 0:
                raise ValueError(
                    f"{name} must have at least one [expert, ...] image, got "
                    f"{tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        devices = {tensor.device for tensor in raw.values()}
        if len(devices) != 1:
            raise ValueError("all MXFP4 raw tensors must share one device")

        w13 = raw["w13_weight"]
        w2 = raw["w2_weight"]
        hidden_size = int(w13.shape[2]) * 2
        intermediate_size = int(w2.shape[2]) * 2
        expected = {
            "w13_weight": (2 * intermediate_size, hidden_size // 2),
            "w13_weight_scale_inv": (
                2 * intermediate_size,
                hidden_size // 32,
            ),
            "w2_weight": (hidden_size, intermediate_size // 2),
            "w2_weight_scale_inv": (hidden_size, intermediate_size // 32),
        }
        for name, shape in expected.items():
            actual = tuple(raw[name].shape[1:])
            if actual != shape:
                raise ValueError(
                    f"inconsistent DeepSeek-V4 MXFP4 {name}: {actual} != {shape}"
                )
        if hidden_size % 64 or intermediate_size % 64:
            raise ValueError(
                "KT streamed Marlin requires hidden/intermediate multiples of 64, "
                f"got {hidden_size}/{intermediate_size}"
            )

        specs = tuple(
            RawTensorSpec(
                name=name,
                expert_shape=tuple(raw[name].shape[1:]),
                gpu_dtype=raw[name].dtype,
                # kt-kernel exports numeric E8M0 values into BF16 staging;
                # the GPU raw destination performs the final dtype conversion.
                host_dtype=torch.bfloat16 if name in SCALE_NAMES else raw[name].dtype,
            )
            for name in RAW_NAMES
        )
        return cls(
            device=w13.device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            raw_specs=specs,
            compute_capability=_device_compute_capability(w13.device),
        )

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Mxfp4ExpertLayout:
        raw_source = getattr(layer, "_kt_mxfp4_raw_weights", None)
        if raw_source is None:
            raw_source = {
                name: getattr(layer, name)
                for name in RAW_NAMES
                if hasattr(layer, name)
            }
        return cls.from_tensors(raw_source)


@dataclass
class StreamedWaveResult:
    output: torch.Tensor
    completion_event: torch.cuda.Event


@dataclass(frozen=True)
class ResidentInstallCommit:
    """An irreversible resident-byte commit awaiting metadata publication."""

    ticket: Mxfp4StreamTicket
    install_event: torch.cuda.Event
    commit_token: tuple
    irreversible: bool = True
    mapping_publish_required: bool = True

    @property
    def mapping_publish_allowed(self) -> bool:
        manager = self.ticket.manager
        return (
            not manager.fail_stopped
            and self.ticket.state == StreamTicketState.INSTALL_COMMITTED
        )

    @property
    def fail_stop(self) -> bool:
        return self.ticket.manager.fail_stopped


class Mxfp4IrreversibleCommitError(RuntimeError):
    """Resident bytes may have changed; old metadata must never be reused."""

    irreversible = True
    fail_stop = True
    mapping_publish_allowed = False


@dataclass(frozen=True)
class AbortResult:
    recovered: bool
    pending_commits: tuple[ResidentInstallCommit, ...]

    @property
    def mapping_publish_required(self) -> bool:
        return bool(self.pending_commits)


class _PinnedHostPool:
    """Two per-rank POSIX-SHM slots; TP0 maps every rank's buffers."""

    def __init__(self, layout: Mxfp4ExpertLayout, tp_rank: int, tp_world: int):
        self.layout = layout
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.buffers: dict[str, torch.Tensor] = {}
        self._owned_shm: dict[str, shared_memory.SharedMemory] = {}
        self._opened_shm: dict[str, shared_memory.SharedMemory] = {}
        self._registered_ptrs: list[int] = []
        self.all_rank_ptrs: dict[str, list[int]] = {}
        self._closed = False

        unique_id = uuid.uuid4().hex[:12] if tp_rank == 0 else None
        unique_id = _broadcast_object_from_tp0(unique_id)
        if not unique_id:
            raise RuntimeError("TP0 did not publish an MXFP4 SHM identifier")
        self.unique_id = str(unique_id)

        allocation_error: Exception | None = None
        try:
            for spec in layout.raw_specs:
                nbytes = _POOL_DEPTH * spec.host_expert_nbytes
                shm_name = self._shm_name(spec.name, tp_rank)
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=nbytes
                )
                self._owned_shm[spec.name] = shm
                buffer = torch.frombuffer(
                    shm.buf, dtype=spec.host_dtype
                ).reshape((_POOL_DEPTH,) + spec.expert_shape)
                result = torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), nbytes, 0
                )
                if int(result) != 0:
                    raise RuntimeError(
                        f"cudaHostRegister failed for {spec.name}: {int(result)}"
                    )
                self._registered_ptrs.append(buffer.data_ptr())
                self.buffers[spec.name] = buffer
        except Exception as exc:  # noqa: BLE001 - peers must still commit/abort
            allocation_error = exc

        if not _tp_named_stage_commit(
            (
                "HOST_POOL",
                "ALLOCATED",
                self.unique_id,
                layout.compatibility_signature,
            ),
            allocation_error is None,
        ):
            self.close()
            message = "MXFP4 streamed SHM allocation failed on at least one TP rank"
            if allocation_error is not None:
                raise RuntimeError(message) from allocation_error
            raise RuntimeError(message)

        pointer_error: Exception | None = None
        try:
            self._collect_rank_pointers()
        except Exception as exc:  # noqa: BLE001 - peers must still commit/abort
            pointer_error = exc
        if not _tp_named_stage_commit(
            (
                "HOST_POOL",
                "POINTERS_MAPPED",
                self.unique_id,
                layout.compatibility_signature,
            ),
            pointer_error is None,
        ):
            self.close()
            message = "MXFP4 streamed SHM pointer mapping failed on at least one TP rank"
            if pointer_error is not None:
                raise RuntimeError(message) from pointer_error
            raise RuntimeError(message)

        # Names are no longer needed once TP0 has opened every peer mapping.
        for shm in self._owned_shm.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._pointer_snapshot = {
            name: tensor.data_ptr() for name, tensor in self.buffers.items()
        }

    def _shm_name(self, name: str, rank: int) -> str:
        return f"kt_mxfp4_stream_{name}_r{rank}_{self.unique_id}"

    def _collect_rank_pointers(self) -> None:
        for spec in self.layout.raw_specs:
            pointers: list[int] = []
            for rank in range(self.tp_world):
                if rank == self.tp_rank:
                    pointer = self.buffers[spec.name].data_ptr()
                elif self.tp_rank == 0:
                    shm_name = self._shm_name(spec.name, rank)
                    shm = shared_memory.SharedMemory(name=shm_name)
                    self._opened_shm[f"{spec.name}:r{rank}"] = shm
                    pointer = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                else:
                    pointer = 0
                pointers.append(pointer)
            self.all_rank_ptrs[spec.name] = pointers
        if self.tp_rank == 0 and not all(
            len(pointers) == self.tp_world and all(pointer > 0 for pointer in pointers)
            for pointers in self.all_rank_ptrs.values()
        ):
            raise RuntimeError("TP0 could not map every TP rank's streamed SHM")

    def rank_slot_pointers(self, name: str, host_slot: int) -> list[int]:
        if not 0 <= host_slot < _POOL_DEPTH:
            raise IndexError(host_slot)
        spec = self.layout.spec(name)
        return [
            pointer + host_slot * spec.host_expert_nbytes
            for pointer in self.all_rank_ptrs[name]
        ]

    def assert_fixed_addresses(self) -> None:
        for name, expected in self._pointer_snapshot.items():
            actual = self.buffers[name].data_ptr()
            if actual != expected:
                raise RuntimeError(
                    f"pinned host staging {name} moved: {expected} -> {actual}"
                )

    def close(self) -> bool:
        if self._closed:
            return True
        unregister_error: Exception | None = None
        for pointer in self._registered_ptrs:
            try:
                result = torch.cuda.cudart().cudaHostUnregister(pointer)
                if int(result) != 0:
                    raise RuntimeError(
                        "cudaHostUnregister failed with error code "
                        f"{int(result)}"
                    )
            except Exception as exc:
                unregister_error = exc
                logger.debug("failed to unregister streamed host buffer", exc_info=True)
        if unregister_error is not None:
            # Keep tensors and SHM mappings alive.  Closing exported buffers
            # after an uncertain unregister result is less safe than leaking
            # them until process teardown.
            return False
        self._closed = True
        self._registered_ptrs.clear()
        self.buffers.clear()
        # ``torch.frombuffer`` keeps an exported Py_buffer alive.  Drop every
        # tensor and collect it before SharedMemory.close(), otherwise Python
        # can raise BufferError during shutdown.
        gc.collect()
        for shm in self._opened_shm.values():
            try:
                shm.close()
            except Exception:
                logger.debug("failed to close peer streamed SHM", exc_info=True)
        self._opened_shm.clear()
        for shm in self._owned_shm.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("failed to unlink streamed SHM", exc_info=True)
            try:
                shm.close()
            except Exception:
                logger.debug("failed to close streamed SHM", exc_info=True)
        self._owned_shm.clear()
        return True


class _GpuStreamSlot:
    def __init__(self, index: int, layout: Mxfp4ExpertLayout):
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            allocate_v4_mxfp4_marlin,
        )

        self.index = index
        self.state = StreamSlotState.FREE
        self.ticket_id: int | None = None
        self.generation = 0
        self.reuse_event: torch.cuda.Event | None = None
        self.host_dma_was_used = False
        self.raw = {
            spec.name: torch.empty(
                (1,) + spec.expert_shape,
                dtype=spec.gpu_dtype,
                device=layout.device,
            )
            for spec in layout.raw_specs
        }
        self.prepared = allocate_v4_mxfp4_marlin(
            num_experts=1,
            hidden_size=layout.hidden_size,
            intermediate_size=layout.intermediate_size,
            device=layout.device,
        )
        self.host_dma_done_event = torch.cuda.Event()
        self.raw_ready_event = torch.cuda.Event()
        self.prepared_ready_event = torch.cuda.Event()
        self._pointer_snapshot = self._capture_pointers()

    def _capture_pointers(self) -> tuple[int, ...]:
        return tuple(
            [self.raw[name].data_ptr() for name in RAW_NAMES]
            + [
                self.prepared.w13.data_ptr(),
                self.prepared.w13_scale.data_ptr(),
                self.prepared.w2.data_ptr(),
                self.prepared.w2_scale.data_ptr(),
            ]
        )

    def assert_fixed_addresses(self) -> None:
        actual = self._capture_pointers()
        if actual != self._pointer_snapshot:
            raise RuntimeError(
                f"MXFP4 GPU staging slot {self.index} changed addresses"
            )


class Mxfp4StreamTicket:
    """One logical candidate and its exclusive two-slot transport claim."""

    def __init__(
        self,
        manager: Mxfp4StreamTransport,
        ticket_id: int,
        method: Any,
        expert_id: int,
    ):
        self.manager = manager
        self.ticket_id = ticket_id
        self.method = method
        self.expert_id = expert_id
        self.layer_idx = int(getattr(method.kt_config, "layer_idx", -1))
        self.weight_namespace = manager.weight_namespace
        self.state = StreamTicketState.CREATED
        self.slot_index: int | None = None
        self.writer_completion: Any | None = None
        self.failure: Exception | None = None
        self.recoverable = True
        self.epoch = manager.group_epoch
        self.slot_generation = -1
        with torch.cuda.device(manager.layout.device):
            # These events are never re-recorded by a later slot generation.
            self.wave_done_event = torch.cuda.Event()
            self.install_done_event = torch.cuda.Event()
            self.release_done_event = torch.cuda.Event()
        self.install_commit: ResidentInstallCommit | None = None

    @property
    def slot(self) -> _GpuStreamSlot:
        if self.slot_index is None:
            raise RuntimeError(f"stream ticket {self.ticket_id} has no staging slot")
        return self.manager.slots[self.slot_index]

    def _transition(self, target: StreamTicketState) -> None:
        validate_ticket_transition(self.state, target)
        self.state = target

    def wait_and_launch(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
    ) -> StreamedWaveResult:
        return self.manager._wait_and_launch(
            self,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            out=out,
            routed_scaling_factor=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
        )

    def install(
        self,
        resident_prepared: Any,
        resident_slot: int,
        *,
        resident_raw: Mapping[str, torch.Tensor] | None = None,
        victim_safe_event: torch.cuda.Event,
        install_stream: torch.cuda.Stream,
    ) -> ResidentInstallCommit:
        return self.manager._install(
            self,
            resident_prepared,
            resident_slot,
            resident_raw=resident_raw,
            victim_safe_event=victim_safe_event,
            install_stream=install_stream,
        )

    def release(self) -> torch.cuda.Event:
        return self.manager._release(self)

    def abort(self) -> AbortResult:
        """Synchronously fence and retire every unfinished ticket in the group."""

        return self.manager.abort_candidates()

    def confirm_mapping_published(
        self,
        commit: ResidentInstallCommit,
        *,
        local_success: bool = True,
        local_error: Exception | None = None,
    ) -> None:
        self.manager.confirm_mapping_published(
            commit,
            local_success=local_success,
            local_error=local_error,
        )


class Mxfp4StreamTransport:
    """Reusable fixed-address transport pool for compatible KT MXFP4 layers."""

    supported = True
    reason = "supported"

    def __init__(
        self,
        *,
        layout: Mxfp4ExpertLayout,
        registry_key: tuple,
        transport_identity: tuple,
    ):
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            require_v4_mxfp4_marlin_streaming_support,
        )

        require_v4_mxfp4_marlin_streaming_support(layout.device)
        self.layout = layout
        self.registry_key = registry_key
        self.transport_identity = transport_identity
        self.weight_namespace = str(transport_identity[0])
        self.tp_rank, self.tp_world = _tp_rank_world()
        self.host_pool = _PinnedHostPool(layout, self.tp_rank, self.tp_world)
        gpu_init_error: Exception | None = None
        try:
            with torch.cuda.device(layout.device):
                self.transfer_stream = torch.cuda.Stream(device=layout.device)
                self.prepare_stream = torch.cuda.Stream(device=layout.device)
                self.slots = tuple(
                    _GpuStreamSlot(index, layout) for index in range(_POOL_DEPTH)
                )
        except Exception as exc:  # noqa: BLE001 - factory commits across TP
            gpu_init_error = exc
        if gpu_init_error is not None:
            # No kernels have been enqueued yet.  Release this rank's host
            # mappings before every rank reports construction failure in the
            # factory's two-phase commit.
            self.host_pool.close()
            raise RuntimeError("local MXFP4 GPU staging allocation failed") from gpu_init_error
        self._next_ticket_id = 1
        self.group_epoch = 0
        self._collective_seq = 0
        self._active: dict[int, Mxfp4StreamTicket] = {}
        self._pending: list[Mxfp4StreamTicket] = []
        self._pending_commits: dict[tuple, ResidentInstallCommit] = {}
        self._closed = False
        self._fatal_error: Exception | None = None

    @property
    def fail_stopped(self) -> bool:
        return self._fatal_error is not None

    @property
    def fatal_reason(self) -> Exception | None:
        return self._fatal_error

    @property
    def irreversible_pending(self) -> bool:
        return bool(self._pending_commits)

    @property
    def pending_install_commits(self) -> tuple[ResidentInstallCommit, ...]:
        return tuple(self._pending_commits.values())

    def _operation_token(
        self,
        op: str,
        stage: str,
        *,
        ticket: Mxfp4StreamTicket | None = None,
        slot: _GpuStreamSlot | None = None,
        detail: tuple = (),
    ) -> tuple:
        sequence = self._collective_seq
        self._collective_seq += 1
        if slot is not None:
            slot_index = slot.index
            generation = slot.generation
        elif ticket is not None:
            slot_index = -1 if ticket.slot_index is None else ticket.slot_index
            generation = ticket.slot_generation
        else:
            slot_index = -1
            generation = -1
        return (
            self.transport_identity,
            self.group_epoch,
            sequence,
            -1 if ticket is None else ticket.layer_idx,
            "" if ticket is None else ticket.weight_namespace,
            -1 if ticket is None else ticket.ticket_id,
            -1 if ticket is None else ticket.expert_id,
            "" if ticket is None else ticket.state.value,
            slot_index,
            generation,
            op,
            stage,
            detail,
        )

    def _stage_commit(
        self,
        op: str,
        stage: str,
        local_success: bool,
        *,
        ticket: Mxfp4StreamTicket | None = None,
        slot: _GpuStreamSlot | None = None,
        detail: tuple = (),
    ) -> bool:
        """Token-match one TP stage before reducing its success status."""

        token = self._operation_token(
            op, stage, ticket=ticket, slot=slot, detail=detail
        )
        reports = _tp_all_gather_objects((token, bool(local_success)))
        expected = reports[0][0]
        if any(report_token != expected for report_token, _ in reports):
            mismatches = _format_token_mismatches(reports)
            error = RuntimeError(
                "MXFP4 TP control token mismatch; refusing potentially "
                "mispaired collective "
                f"(rank0_expected={expected}, mismatches={mismatches})"
            )
            self._fatal_error = error
            raise error
        return all(success for _, success in reports)

    def _public_ticket_enter(
        self,
        op: str,
        ticket: Mxfp4StreamTicket,
        allowed_states: tuple[StreamTicketState, ...],
    ) -> None:
        slot_valid = ticket.slot_index is None and ticket.state == StreamTicketState.CREATED
        if ticket.slot_index is not None and 0 <= ticket.slot_index < len(self.slots):
            assigned_slot = self.slots[ticket.slot_index]
            slot_valid = (
                assigned_slot.ticket_id == ticket.ticket_id
                and assigned_slot.generation == ticket.slot_generation
            )
        local_valid = (
            not self._closed
            and self._fatal_error is None
            and ticket.manager is self
            and ticket.ticket_id in self._active
            and ticket.state in allowed_states
            and slot_valid
        )
        if not self._stage_commit(
            op,
            "ENTER",
            local_valid,
            ticket=ticket,
            detail=(ticket.state.value,),
        ):
            raise RuntimeError(
                f"MXFP4 {op} entry validation failed on at least one TP rank"
            )

    def begin_candidates(
        self, method: Any, candidate_ids: Sequence[int]
    ) -> tuple[Mxfp4StreamTicket, ...]:
        """Create tickets and immediately submit writers for the first two."""

        self.group_epoch += 1
        validation_error: Exception | None = None
        candidates: tuple[int, ...] = ()
        try:
            candidates = tuple(int(expert_id) for expert_id in candidate_ids)
            if self._closed:
                raise RuntimeError("MXFP4 stream transport is closed")
            if self._fatal_error is not None:
                raise RuntimeError(
                    "MXFP4 stream transport is fail-stopped after an unsafe error"
                ) from self._fatal_error
            if len(set(candidates)) != len(candidates):
                raise ValueError("stream candidate IDs must be unique")
            if any(expert_id < 0 for expert_id in candidates):
                raise ValueError("stream candidate IDs must be non-negative")
            num_experts = int(getattr(method, "global_num_experts", 0))
            if num_experts and any(
                expert_id >= num_experts for expert_id in candidates
            ):
                raise ValueError(
                    f"stream candidate is outside [0, {num_experts}): {candidates}"
                )
        except Exception as exc:  # noqa: BLE001 - TP validates as one group
            validation_error = exc
        if not self._stage_commit(
            "BEGIN",
            "ENTER",
            validation_error is None
            and not self._active
            and not self._pending
            and not self._pending_commits,
            detail=(
                int(getattr(getattr(method, "kt_config", None), "layer_idx", -1)),
                self.weight_namespace,
                candidates,
            ),
        ):
            message = "invalid MXFP4 stream candidate list on at least one TP rank"
            if validation_error is not None:
                raise ValueError(message) from validation_error
            raise ValueError(message)

        tickets = []
        for expert_id in candidates:
            ticket = Mxfp4StreamTicket(
                self, self._next_ticket_id, method, expert_id
            )
            self._next_ticket_id += 1
            self._active[ticket.ticket_id] = ticket
            self._pending.append(ticket)
            tickets.append(ticket)
        try:
            self._pump_writer_submissions()
        except Exception:
            # ``begin_candidates`` has not returned its ticket objects yet.
            # Recover the partially submitted group here so a later window is
            # not stranded behind invisible WRITING/CREATED tickets.
            self._abort_candidates_internal()
            raise
        return tuple(tickets)

    def _pump_writer_submissions(self) -> None:
        while self._pending:
            slot = next(
                (slot for slot in self.slots if slot.state == StreamSlotState.FREE),
                None,
            )
            if slot is None:
                return
            ticket = self._pending.pop(0)
            self._submit_writer(ticket, slot)

    def _submit_writer(
        self, ticket: Mxfp4StreamTicket, slot: _GpuStreamSlot
    ) -> None:
        slot.state = StreamSlotState.WRITING
        slot.ticket_id = ticket.ticket_id
        slot.generation += 1
        ticket.slot_index = slot.index
        ticket.slot_generation = slot.generation
        preflight_error: Exception | None = None
        try:
            self.host_pool.assert_fixed_addresses()
            slot.assert_fixed_addresses()
        except Exception as exc:  # noqa: BLE001 - TP entry consensus
            preflight_error = exc
        if not self._stage_commit(
            "SUBMIT_WRITER",
            "ENTER",
            preflight_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(
                ticket, preflight_error or RuntimeError("peer preflight failed")
            )
            raise RuntimeError("MXFP4 writer entry failed on a TP peer") from preflight_error

        host_reuse_error: Exception | None = None
        try:
            if slot.host_dma_was_used:
                # Every rank must finish reading its own SHM mapping before
                # TP0 is allowed to overwrite any rank's corresponding slot.
                slot.host_dma_done_event.synchronize()
        except Exception as exc:  # noqa: BLE001 - consensus before TP0 write
            host_reuse_error = exc
        if not self._stage_commit(
            "SUBMIT_WRITER",
            "HOST_REUSE_READY",
            host_reuse_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(
                ticket, host_reuse_error or RuntimeError("peer host slot busy")
            )
            raise RuntimeError(
                "MXFP4 host staging reuse was not safe on every TP rank"
            ) from host_reuse_error

        submit_error: Exception | None = None
        completion = None
        if self.tp_rank == 0:
            try:
                wrapper = getattr(ticket.method, "wrapper", None)
                if wrapper is None:
                    raise RuntimeError("TP0 has no KT MXFP4 writer wrapper")
                completion = wrapper.submit_write_weight_scale_to_buffer(
                    self.tp_world,
                    ticket.expert_id,
                    self.host_pool.rank_slot_pointers("w13_weight", slot.index),
                    self.host_pool.rank_slot_pointers(
                        "w13_weight_scale_inv", slot.index
                    ),
                    self.host_pool.rank_slot_pointers("w2_weight", slot.index),
                    self.host_pool.rank_slot_pointers(
                        "w2_weight_scale_inv", slot.index
                    ),
                )
                if completion is None:
                    raise RuntimeError(
                        "KT extension lacks task-specific tracked writer completion"
                    )
            except Exception as exc:  # noqa: BLE001 - TP fail-stop propagation
                submit_error = exc
        if not self._stage_commit(
            "SUBMIT_WRITER",
            "WRITER_SUBMITTED",
            submit_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(ticket, submit_error or RuntimeError("peer submit failed"))
            message = (
                f"MXFP4 writer submission failed for expert {ticket.expert_id} "
                "on at least one TP rank"
            )
            if submit_error is not None:
                raise RuntimeError(message) from submit_error
            raise RuntimeError(message)
        ticket.writer_completion = completion
        ticket._transition(StreamTicketState.WRITER_SUBMITTED)

    def _wait_and_launch(
        self,
        ticket: Mxfp4StreamTicket,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None,
        routed_scaling_factor: float,
        swiglu_limit: float | None,
    ) -> StreamedWaveResult:
        self._public_ticket_enter(
            "WAIT_LAUNCH",
            ticket,
            (
                StreamTicketState.CREATED,
                StreamTicketState.WRITER_SUBMITTED,
            ),
        )
        backend_error: Exception | None = None
        try:
            from sglang.srt.layers.quantization.v4_marlin_moe import (
                apply_v4_mxfp4_marlin_streamed_experts,
                prepare_v4_mxfp4_marlin_expert,
            )
        except Exception as exc:  # noqa: BLE001 - TP backend consensus
            backend_error = exc
        if not self._stage_commit(
            "WAIT_LAUNCH",
            "BACKEND_READY",
            backend_error is None,
            ticket=ticket,
        ):
            self._fail_ticket(
                ticket, backend_error or RuntimeError("peer backend import failed")
            )
            self._abort_candidates_internal()
            raise RuntimeError("MXFP4 streamed backend is unavailable") from backend_error
        if ticket.state == StreamTicketState.CREATED:
            # More than two candidates wait here until an earlier ticket has
            # released one of the fixed slots.
            self._pump_writer_submissions()
        if not self._stage_commit(
            "WAIT_LAUNCH",
            "WRITER_AVAILABLE",
            ticket.state == StreamTicketState.WRITER_SUBMITTED,
            ticket=ticket,
        ):
            raise RuntimeError(
                f"ticket {ticket.ticket_id} is not writer-ready: {ticket.state}"
            )
        slot = ticket.slot

        writer_error: Exception | None = None
        if self.tp_rank == 0:
            try:
                ticket.method.wrapper.sync_write_weight_scale_to_buffer(
                    ticket.writer_completion
                )
            except Exception as exc:  # noqa: BLE001 - TP fail-stop propagation
                writer_error = exc
        if not self._stage_commit(
            "WAIT_LAUNCH",
            "WRITER_DONE",
            writer_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(ticket, writer_error or RuntimeError("peer writer failed"))
            self._abort_candidates_internal()
            message = (
                f"MXFP4 tracked writer failed for expert {ticket.expert_id} "
                "on at least one TP rank"
            )
            if writer_error is not None:
                raise RuntimeError(message) from writer_error
            raise RuntimeError(message)
        ticket._transition(StreamTicketState.HOST_READY)
        slot.state = StreamSlotState.HOST_READY

        transport_error: Exception | None = None
        try:
            with torch.cuda.stream(self.transfer_stream):
                if slot.reuse_event is not None:
                    self.transfer_stream.wait_event(slot.reuse_event)
                for name in RAW_NAMES:
                    slot.raw[name][0].copy_(
                        self.host_pool.buffers[name][slot.index],
                        non_blocking=True,
                    )
                slot.host_dma_done_event.record(self.transfer_stream)
                slot.raw_ready_event.record(self.transfer_stream)
                slot.host_dma_was_used = True
            ticket._transition(StreamTicketState.TRANSFER_SUBMITTED)
            slot.state = StreamSlotState.TRANSFERRING

            with torch.cuda.stream(self.prepare_stream):
                self.prepare_stream.wait_event(slot.raw_ready_event)
                prepare_v4_mxfp4_marlin_expert(
                    slot.raw["w13_weight"],
                    slot.raw["w13_weight_scale_inv"],
                    slot.raw["w2_weight"],
                    slot.raw["w2_weight_scale_inv"],
                    out=slot.prepared,
                )
                slot.prepared_ready_event.record(self.prepare_stream)
        except Exception as exc:  # noqa: BLE001 - TP fail-stop propagation
            transport_error = exc
        if not self._stage_commit(
            "WAIT_LAUNCH",
            "H2D_PREPARE_ENQUEUED",
            transport_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(
                ticket, transport_error or RuntimeError("peer transport failed")
            )
            self._abort_candidates_internal()
            message = (
                f"MXFP4 H2D/prepare enqueue failed for expert {ticket.expert_id} "
                "on at least one TP rank"
            )
            if transport_error is not None:
                raise RuntimeError(message) from transport_error
            raise RuntimeError(message)
        ticket._transition(StreamTicketState.READY)
        slot.state = StreamSlotState.READY

        launch_error: Exception | None = None
        output = None
        try:
            main_stream = torch.cuda.current_stream(self.layout.device)
            main_stream.wait_event(slot.prepared_ready_event)
            hidden_pad = self.layout.hidden_size - int(hidden_states.shape[-1])
            if hidden_pad < 0:
                raise ValueError(
                    f"hidden size {hidden_states.shape[-1]} exceeds streamed "
                    f"weight size {self.layout.hidden_size}"
                )
            padded_hidden = hidden_states
            if hidden_pad:
                padded_hidden = torch.nn.functional.pad(hidden_states, (0, hidden_pad))
            streamed = apply_v4_mxfp4_marlin_streamed_experts(
                hidden_states=padded_hidden,
                prepared_staging=slot.prepared,
                logical_expert_ids=(ticket.expert_id,),
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                routed_scaling_factor=routed_scaling_factor,
                swiglu_limit=swiglu_limit,
                out=None if hidden_pad else out,
            )
            if hidden_pad:
                streamed = streamed[..., : hidden_states.shape[-1]]
                if out is not None:
                    if (
                        out.shape != hidden_states.shape
                        or out.dtype != hidden_states.dtype
                        or out.device != hidden_states.device
                    ):
                        raise ValueError("streamed output must match hidden_states")
                    out.copy_(streamed)
                    streamed = out
            output = streamed
            ticket.wave_done_event.record(main_stream)
        except Exception as exc:  # noqa: BLE001 - TP fail-stop propagation
            launch_error = exc
        if not self._stage_commit(
            "WAIT_LAUNCH",
            "WAVE_ENQUEUED",
            launch_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(ticket, launch_error or RuntimeError("peer wave-2 failed"))
            self._abort_candidates_internal()
            message = (
                f"MXFP4 wave-2 launch failed for expert {ticket.expert_id} "
                "on at least one TP rank"
            )
            if launch_error is not None:
                raise RuntimeError(message) from launch_error
            raise RuntimeError(message)
        ticket._transition(StreamTicketState.WAVE2_LAUNCHED)
        slot.state = StreamSlotState.IN_USE
        assert output is not None
        return StreamedWaveResult(
            output=output, completion_event=ticket.wave_done_event
        )

    def _install(
        self,
        ticket: Mxfp4StreamTicket,
        resident_prepared: Any,
        resident_slot: int,
        *,
        resident_raw: Mapping[str, torch.Tensor] | None,
        victim_safe_event: torch.cuda.Event,
        install_stream: torch.cuda.Stream,
    ) -> ResidentInstallCommit:
        self._public_ticket_enter(
            "INSTALL", ticket, (StreamTicketState.WAVE2_LAUNCHED,)
        )
        backend_error: Exception | None = None
        try:
            from sglang.srt.layers.quantization.v4_marlin_moe import (
                view_v4_mxfp4_marlin_slot,
            )
        except Exception as exc:  # noqa: BLE001 - TP backend consensus
            backend_error = exc
        if not self._stage_commit(
            "INSTALL",
            "BACKEND_READY",
            backend_error is None,
            ticket=ticket,
        ):
            self._fail_ticket(
                ticket, backend_error or RuntimeError("peer backend import failed")
            )
            self._abort_candidates_internal()
            raise RuntimeError("MXFP4 install backend is unavailable") from backend_error
        slot = ticket.slot
        validation_error: Exception | None = None
        resident_view = None
        resident_raw_views: dict[str, torch.Tensor] = {}
        try:
            current_stream = torch.cuda.current_stream(self.layout.device)
            if install_stream.device != self.layout.device:
                raise ValueError(
                    "install_stream must be on the MXFP4 transport device"
                )
            if install_stream.cuda_stream != current_stream.cuda_stream:
                raise ValueError(
                    "resident install must be invoked on the explicitly supplied "
                    "current main stream"
                )
            if victim_safe_event is None:
                raise ValueError(
                    "resident install requires the wave-1 victim-safe event"
                )
            event_device = getattr(victim_safe_event, "device", self.layout.device)
            if event_device != self.layout.device:
                raise ValueError(
                    "victim-safe event must be recorded on the transport device"
                )
            resident_view = view_v4_mxfp4_marlin_slot(
                resident_prepared, int(resident_slot)
            )
            if (
                resident_view.hidden_size != self.layout.hidden_size
                or resident_view.intermediate_size != self.layout.intermediate_size
            ):
                raise ValueError("resident and streamed Marlin metadata differ")
            staged_tensors = (
                slot.prepared.w13,
                slot.prepared.w13_scale,
                slot.prepared.w2,
                slot.prepared.w2_scale,
            )
            resident_tensors = (
                resident_view.w13,
                resident_view.w13_scale,
                resident_view.w2,
                resident_view.w2_scale,
            )
            for source, target in zip(staged_tensors, resident_tensors):
                if (
                    source.shape != target.shape
                    or source.dtype != target.dtype
                    or source.device != target.device
                ):
                    raise ValueError("resident Marlin slot is incompatible with staging")
            if resident_raw is not None:
                missing = [name for name in RAW_NAMES if name not in resident_raw]
                if missing:
                    raise ValueError(
                        f"resident canonical raw image is missing {missing}"
                    )
                for name in RAW_NAMES:
                    target = resident_raw[name]
                    if not 0 <= int(resident_slot) < target.shape[0]:
                        raise IndexError(
                            f"raw resident slot {resident_slot} is outside "
                            f"{name} capacity {target.shape[0]}"
                        )
                    target_view = target[int(resident_slot) : int(resident_slot) + 1]
                    source = slot.raw[name]
                    if (
                        source.shape != target_view.shape
                        or source.dtype != target_view.dtype
                        or source.device != target_view.device
                    ):
                        raise ValueError(
                            f"resident canonical raw slot {name} is incompatible "
                            "with streamed staging"
                        )
                    resident_raw_views[name] = target_view
        except Exception as exc:  # noqa: BLE001 - TP fail-stop propagation
            validation_error = exc
        if not self._stage_commit(
            "INSTALL",
            "INSTALL_VALIDATED",
            validation_error is None,
            ticket=ticket,
            slot=slot,
            detail=(int(resident_slot),),
        ):
            self._fail_ticket(
                ticket, validation_error or RuntimeError("peer install invalid")
            )
            self._abort_candidates_internal()
            message = "MXFP4 resident install validation failed on at least one TP rank"
            if validation_error is not None:
                raise RuntimeError(message) from validation_error
            raise RuntimeError(message)

        install_error: Exception | None = None
        try:
            assert resident_view is not None
            install_stream.wait_event(slot.prepared_ready_event)
            # Do not rely on the wrapper using the same stream as wave-2.
            # Every resident write is ordered after both the staging reader
            # and the current resident wave-1 reader selected by the wrapper.
            install_stream.wait_event(ticket.wave_done_event)
            install_stream.wait_event(victim_safe_event)
            slot.state = StreamSlotState.INSTALLING
            resident_view.w13.copy_(slot.prepared.w13, non_blocking=True)
            resident_view.w13_scale.copy_(slot.prepared.w13_scale, non_blocking=True)
            resident_view.w2.copy_(slot.prepared.w2, non_blocking=True)
            resident_view.w2_scale.copy_(slot.prepared.w2_scale, non_blocking=True)
            for name in RAW_NAMES:
                if name in resident_raw_views:
                    resident_raw_views[name].copy_(slot.raw[name], non_blocking=True)
            ticket.install_done_event.record(install_stream)
        except Exception as exc:  # noqa: BLE001 - TP fail-stop propagation
            install_error = exc
        install_consensus_error: Exception | None = None
        try:
            install_committed_on_all_ranks = self._stage_commit(
                "INSTALL",
                "INSTALL_ENQUEUED",
                install_error is None,
                ticket=ticket,
                slot=slot,
                detail=(int(resident_slot),),
            )
        except Exception as exc:  # noqa: BLE001 - post-enqueue fail-stop
            install_consensus_error = exc
            install_committed_on_all_ranks = False
        if not install_committed_on_all_ranks:
            fatal = install_error or RuntimeError("peer install launch failed")
            if install_consensus_error is not None:
                fatal = install_consensus_error
            ticket.recoverable = False
            self._fatal_error = fatal
            self._fail_ticket(ticket, fatal)
            message = (
                "MXFP4 resident D2D install failed after TP validation; the "
                "runtime must fail-stop without publishing a new mapping"
            )
            error = Mxfp4IrreversibleCommitError(message)
            if fatal is not None:
                raise error from fatal
            raise error

        commit_registration_error: Exception | None = None
        commit: ResidentInstallCommit | None = None
        try:
            ticket._transition(StreamTicketState.INSTALL_COMMITTED)
            commit_token = (
                self.transport_identity,
                ticket.epoch,
                ticket.layer_idx,
                ticket.weight_namespace,
                ticket.ticket_id,
                ticket.expert_id,
                slot.index,
                slot.generation,
                int(resident_slot),
            )
            commit = ResidentInstallCommit(
                ticket=ticket,
                install_event=ticket.install_done_event,
                commit_token=commit_token,
            )
            ticket.install_commit = commit
            self._detach_staging_after_install(ticket, slot, commit)
        except Exception as exc:  # noqa: BLE001 - irreversible commit boundary
            commit_registration_error = exc
        try:
            commit_registered_on_all_ranks = self._stage_commit(
                "INSTALL",
                "COMMIT_REGISTERED",
                commit_registration_error is None,
                ticket=ticket,
                detail=(int(resident_slot),),
            )
        except Exception as exc:  # noqa: BLE001 - irreversible commit boundary
            commit_registration_error = exc
            commit_registered_on_all_ranks = False
        if not commit_registered_on_all_ranks:
            self.mark_fatal(
                "failed to register an irreversible resident install commit",
                cause=commit_registration_error,
            )
            raise Mxfp4IrreversibleCommitError(
                "resident bytes were installed but their commit could not be "
                "registered; runtime is fail-stopped"
            ) from commit_registration_error
        assert commit is not None
        return commit

    def _detach_staging_after_install(
        self,
        ticket: Mxfp4StreamTicket,
        slot: _GpuStreamSlot,
        commit: ResidentInstallCommit,
    ) -> None:
        """Free the two-slot transport while retaining an independent commit."""

        self._pending_commits[commit.commit_token] = commit
        self._active.pop(ticket.ticket_id, None)
        slot.reuse_event = ticket.install_done_event
        slot.ticket_id = None
        slot.state = StreamSlotState.FREE

    def confirm_mapping_published(
        self,
        commit: ResidentInstallCommit,
        *,
        local_success: bool = True,
        local_error: Exception | None = None,
    ) -> None:
        """Collectively acknowledge metadata publication.

        Every TP rank must call this from a ``finally`` block, passing
        ``local_success=False`` and its exception when local publication
        failed.  This prevents a failing rank from abandoning peers inside the
        publish acknowledgement collective.
        """

        ticket = commit.ticket
        try:
            local_valid = (
                not self._closed
                and self._fatal_error is None
                and ticket.manager is self
                and ticket.state == StreamTicketState.INSTALL_COMMITTED
                and self._pending_commits.get(commit.commit_token) is commit
                and ticket.install_commit is commit
                and commit.mapping_publish_required
            )
            if not self._stage_commit(
                "PUBLISH_ACK",
                "ENTER",
                local_valid,
                ticket=ticket,
                detail=(commit.commit_token,),
            ):
                raise RuntimeError(
                    "resident mapping publication entry failed on a TP peer"
                )
            if not self._stage_commit(
                "PUBLISH_ACK",
                "MAPPING_PUBLISHED",
                local_valid and local_success and local_error is None,
                ticket=ticket,
                detail=(commit.commit_token,),
            ):
                if local_error is not None:
                    raise RuntimeError(
                        "local resident mapping publication failed"
                    ) from local_error
                raise RuntimeError(
                    "resident mapping publication acknowledgement failed on a TP peer"
                )
        except Exception as exc:
            self.mark_fatal(
                "mapping publication failed after irreversible resident install",
                commit=commit,
                cause=exc,
            )
            raise Mxfp4IrreversibleCommitError(
                "resident bytes were installed but mapping publication was not "
                "confirmed; runtime is fail-stopped"
            ) from exc

        ticket._transition(StreamTicketState.INSTALLED)
        self._pending_commits.pop(commit.commit_token, None)

    def mark_fatal(
        self,
        reason: str,
        *,
        commit: ResidentInstallCommit | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Permanently stop transport after an unsafe metadata/byte divergence."""

        error = RuntimeError(reason)
        if cause is not None:
            error.__cause__ = cause
        self._fatal_error = error
        if commit is not None:
            ticket = commit.ticket
            ticket.recoverable = False
            ticket.failure = error

    def _release(self, ticket: Mxfp4StreamTicket) -> torch.cuda.Event:
        self._public_ticket_enter(
            "RELEASE", ticket, (StreamTicketState.WAVE2_LAUNCHED,)
        )
        slot = ticket.slot
        release_stream = torch.cuda.current_stream(self.layout.device)
        release_error: Exception | None = None
        try:
            release_stream.wait_event(ticket.wave_done_event)
            ticket.release_done_event.record(release_stream)
        except Exception as exc:  # noqa: BLE001 - TP stage propagation
            release_error = exc
        if not self._stage_commit(
            "RELEASE",
            "RELEASE_ENQUEUED",
            release_error is None,
            ticket=ticket,
            slot=slot,
        ):
            self._fail_ticket(
                ticket, release_error or RuntimeError("peer release failed")
            )
            self._abort_candidates_internal()
            raise RuntimeError("MXFP4 release failed on at least one TP rank")
        slot.reuse_event = ticket.release_done_event
        ticket._transition(StreamTicketState.RELEASED)
        self._retire_ticket(ticket, slot)
        return ticket.release_done_event

    def _retire_ticket(
        self, ticket: Mxfp4StreamTicket, slot: _GpuStreamSlot
    ) -> None:
        self._active.pop(ticket.ticket_id, None)
        slot.ticket_id = None
        slot.state = StreamSlotState.FREE
        # Do not pump here: install/release must return independently of the
        # next candidate's writer outcome.  A CREATED ticket pumps itself when
        # its wait_and_launch entry begins.

    def _fail_ticket(self, ticket: Mxfp4StreamTicket, error: Exception) -> None:
        ticket.failure = error
        if ticket.state not in (
            StreamTicketState.INSTALLED,
            StreamTicketState.RELEASED,
            StreamTicketState.FAILED,
        ):
            ticket.state = StreamTicketState.FAILED
        if ticket.slot_index is not None:
            slot = self.slots[ticket.slot_index]
            slot.state = StreamSlotState.POISONED

    def abort_candidates(
        self, tickets: Sequence[Mxfp4StreamTicket] | None = None
    ) -> AbortResult:
        """Exception-path recovery for an unfinished candidate group.

        All tracked writers are consumed, CUDA work is fenced, and both fixed
        slots are returned to FREE before pending tickets are discarded.  A
        resident D2D launch failure is intentionally not recoverable because
        the old resident slot may already contain partial new bytes; callers
        must fail-stop without publishing a mapping in that case.
        """

        foreign = (
            []
            if tickets is None
            else [ticket for ticket in tickets if ticket.manager is not self]
        )
        if not self._stage_commit(
            "ABORT",
            "ENTER",
            not foreign,
            detail=(
                tuple(
                    sorted(
                        ticket.ticket_id
                        for ticket in (tickets or tuple(self._active.values()))
                    )
                ),
            ),
        ):
            raise ValueError("cannot abort tickets owned by another manager")
        return self._abort_candidates_internal()

    def _abort_candidates_internal(self) -> AbortResult:
        if self._fatal_error is not None:
            raise RuntimeError(
                "MXFP4 transport cannot recover after a resident install failure; "
                "the process must fail-stop"
            ) from self._fatal_error
        if not self._active and not self._pending:
            return AbortResult(
                recovered=True,
                pending_commits=self.pending_install_commits,
            )
        irreversible = [
            ticket
            for ticket in self._active.values()
            if ticket.state == StreamTicketState.INSTALL_COMMITTED
        ]
        if irreversible:
            error = RuntimeError(
                "cannot abort after resident bytes were installed and before "
                "their mapping was published"
            )
            self._fatal_error = error
            raise Mxfp4IrreversibleCommitError(str(error)) from error

        writer_error: Exception | None = None
        if self.tp_rank == 0:
            for ticket in tuple(self._active.values()):
                if ticket.writer_completion is None:
                    continue
                try:
                    ticket.method.wrapper.sync_write_weight_scale_to_buffer(
                        ticket.writer_completion
                    )
                except Exception as exc:  # noqa: BLE001 - completion is fenced
                    # A tracked task can report its own writer error while still
                    # guaranteeing that it is no longer touching the host slot.
                    if writer_error is None:
                        writer_error = exc
        cuda_error: Exception | None = None
        try:
            self.transfer_stream.synchronize()
            self.prepare_stream.synchronize()
            # A wave-2 launch failure can leave partial kernels on the caller's
            # stream.  Full-device sync is exception-only and establishes the
            # overwrite fence required for a safe late-CPU fallback.
            torch.cuda.synchronize(self.layout.device)
        except Exception as exc:  # noqa: BLE001 - recovery decides fail-stop
            cuda_error = exc

        # A writer's reported task error is recoverable once completion.wait()
        # returned.  Only failure to establish CUDA quiescence makes reuse
        # unsafe.  Peers therefore commit based on the stream-fence result.
        cuda_safe = cuda_error is None
        if not self._stage_commit(
            "ABORT",
            "RECOVERY_FENCED",
            cuda_safe,
            detail=(tuple(sorted(self._active)),),
        ):
            fatal = RuntimeError(
                "MXFP4 transport could not establish a common recovery fence"
            )
            self._fatal_error = fatal
            raise fatal from cuda_error

        if writer_error is not None:
            logger.warning(
                "Recovered MXFP4 stream slots after a tracked writer error: %s",
                writer_error,
            )

        for ticket in tuple(self._active.values()):
            if ticket.state not in (
                StreamTicketState.INSTALLED,
                StreamTicketState.RELEASED,
                StreamTicketState.ABORTED,
            ):
                if ticket.state == StreamTicketState.FAILED:
                    ticket._transition(StreamTicketState.ABORTED)
                else:
                    ticket._transition(StreamTicketState.ABORTED)
        self._active.clear()
        self._pending.clear()
        for slot in self.slots:
            slot.ticket_id = None
            slot.state = StreamSlotState.FREE
            slot.reuse_event = None
            slot.host_dma_was_used = False
        return AbortResult(
            recovered=True,
            pending_commits=self.pending_install_commits,
        )

    def _ensure_owned(self, ticket: Mxfp4StreamTicket) -> None:
        self._ensure_open()
        if ticket.manager is not self or ticket.ticket_id not in self._active:
            raise RuntimeError("stream ticket is not active in this manager")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MXFP4 stream transport is closed")
        if self._fatal_error is not None:
            raise RuntimeError(
                "MXFP4 stream transport is fail-stopped after an unsafe error"
            ) from self._fatal_error

    def close(self, *, coordinated: bool = True) -> bool:
        """Fence all producers/consumers before releasing registered SHM.

        Returns ``True`` only when resources were actually released.  During
        interpreter teardown ``coordinated=False`` avoids distributed calls;
        active or unsafe pools are intentionally retained for OS cleanup.
        """

        if self._closed:
            return True
        if not coordinated and (
            self._active
            or self._pending_commits
            or self._fatal_error is not None
        ):
            return False
        if coordinated:
            if not self._stage_commit(
                "CLOSE",
                "ENTER",
                self._fatal_error is None and not self.irreversible_pending,
                detail=(tuple(sorted(self._active)),),
            ):
                self.mark_fatal("coordinated MXFP4 transport close was rejected")
                return False
            if self._active or self._pending:
                try:
                    self._abort_candidates_internal()
                except Exception as exc:  # noqa: BLE001 - close retains resources
                    self.mark_fatal(
                        "failed to consume active MXFP4 work before close",
                        cause=exc,
                    )
                    return False

        fence_error: Exception | None = None
        try:
            self.transfer_stream.synchronize()
            self.prepare_stream.synchronize()
            # Includes wave-2, install, release, and any caller/main stream that
            # recorded a per-ticket event consumed by this pool.
            torch.cuda.synchronize(self.layout.device)
        except Exception as exc:  # noqa: BLE001 - unsafe resources stay retained
            fence_error = exc
        if coordinated:
            try:
                fenced = self._stage_commit(
                    "CLOSE",
                    "SHUTDOWN_FENCED",
                    fence_error is None,
                )
            except Exception as exc:  # noqa: BLE001 - close retains resources
                self.mark_fatal("MXFP4 close token mismatch", cause=exc)
                return False
            if not fenced:
                self.mark_fatal(
                    "MXFP4 close could not fence work on every TP rank",
                    cause=fence_error,
                )
                return False
        elif fence_error is not None:
            return False

        if not self.host_pool.close():
            self.mark_fatal(
                "MXFP4 host staging could not be safely unregistered during close"
            )
            return False
        self._active.clear()
        self._pending.clear()
        self.slots = ()
        self._closed = True
        return True


@dataclass(frozen=True)
class UnsupportedMxfp4StreamTransport:
    supported: bool
    reason: str

    def begin_candidates(self, method: Any, candidate_ids: Sequence[int]):
        raise RuntimeError(self.reason)


_TRANSPORTS: dict[tuple, Mxfp4StreamTransport] = {}


def probe_mxfp4_stream_transport(
    method: Any, layer: torch.nn.Module
) -> Mxfp4StreamCapability:
    if (getattr(getattr(method, "kt_config", None), "method", "") or "").upper() != "MXFP4":
        return Mxfp4StreamCapability(False, "KT CPU method is not MXFP4")
    if not torch.cuda.is_available():
        return Mxfp4StreamCapability(False, "CUDA is not available")
    if method.gpu_method.__class__.__name__ != "Mxfp4MarlinMoEMethod":
        return Mxfp4StreamCapability(
            False, "streamed experts require KT's Mxfp4MarlinMoEMethod"
        )
    if not getattr(method.gpu_method, "_kt_layerwise_enabled", False):
        return Mxfp4StreamCapability(
            False, "MXFP4 Marlin method is not in caller-owned KT mode"
        )
    if not hasattr(layer, "_v4_marlin_weights"):
        return Mxfp4StreamCapability(
            False, "resident caller-owned V4 Marlin weights are unavailable"
        )
    try:
        layout = Mxfp4ExpertLayout.from_layer(layer)
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            get_v4_mxfp4_marlin_streaming_capability,
        )

        backend = get_v4_mxfp4_marlin_streaming_capability(layout.device)
        if not backend.available:
            return Mxfp4StreamCapability(False, backend.reason)
    except Exception as exc:  # noqa: BLE001 - capability must fail closed
        return Mxfp4StreamCapability(False, f"invalid V4 MXFP4 layout: {exc}")

    tp_rank, _ = _tp_rank_world()
    if tp_rank == 0:
        wrapper = getattr(method, "wrapper", None)
        moe = getattr(wrapper, "moe", None)
        if wrapper is None or not hasattr(
            moe, "write_weight_scale_to_buffer_tracked_task"
        ):
            return Mxfp4StreamCapability(
                False,
                "kt-kernel lacks task-specific MXFP4 writer completion",
            )
    return Mxfp4StreamCapability(True, "supported")


def get_or_create_mxfp4_stream_transport(
    method: Any, layer: torch.nn.Module
) -> Mxfp4StreamTransport | UnsupportedMxfp4StreamTransport:
    """Return a cross-layer transport pool or a fail-closed capability object."""

    capability = probe_mxfp4_stream_transport(method, layer)
    if not _tp_named_stage_commit(("FACTORY", "CAPABILITY"), capability.supported):
        reason = (
            capability.reason
            if not capability.supported
            else "at least one TP peer rejected streamed MXFP4"
        )
        return UnsupportedMxfp4StreamTransport(
            False,
            reason,
        )

    layout = Mxfp4ExpertLayout.from_layer(layer)
    _, tp_world = _tp_rank_world()
    local_weight_namespace = str(
        getattr(getattr(method, "kt_config", None), "weight_path", None) or ""
    )
    # Paths may be rank-local mount points.  The first implementation uses
    # TP0's path as the distributed namespace; a content/manifest fingerprint
    # can replace this value later without changing runtime token semantics.
    weight_namespace = str(
        _broadcast_object_from_tp0(local_weight_namespace) or ""
    )
    transport_identity = (
        weight_namespace,
        tp_world,
        layout.compatibility_signature,
    )
    layout_reports = _tp_all_gather_objects(
        (str(layout.device), layout.compatibility_signature)
    )
    _validate_tp_layout_contract(layout_reports)
    key = (
        local_weight_namespace,
        tp_world,
        layout.local_registry_signature,
    )
    manager = _TRANSPORTS.get(key)
    registry_reports = _tp_all_gather_objects(
        (("FACTORY", "REGISTRY", transport_identity), manager is not None)
    )
    if any(token != registry_reports[0][0] for token, _ in registry_reports):
        raise RuntimeError(
            "MXFP4 transport factory token mismatch across TP ranks: "
            f"{_format_token_mismatches(registry_reports)}"
        )
    all_have_manager = all(present for _, present in registry_reports)
    any_have_manager = any(present for _, present in registry_reports)
    if any_have_manager and not all_have_manager:
        return UnsupportedMxfp4StreamTransport(
            False, "MXFP4 transport registry is inconsistent across TP ranks"
        )
    if all_have_manager:
        assert manager is not None
        return manager
    construction_error: Exception | None = None
    try:
        manager = Mxfp4StreamTransport(
            layout=layout,
            registry_key=key,
            transport_identity=transport_identity,
        )
    except Exception as exc:  # noqa: BLE001 - all-rank construction commit
        construction_error = exc
        manager = None
    if not _tp_named_stage_commit(
        ("FACTORY", "CONSTRUCT", transport_identity),
        construction_error is None,
    ):
        if manager is not None:
            manager.close(coordinated=False)
        message = "MXFP4 stream transport construction failed on a TP rank"
        if construction_error is not None:
            raise RuntimeError(message) from construction_error
        raise RuntimeError(message)
    assert manager is not None
    _TRANSPORTS[key] = manager
    return manager


def close_mxfp4_stream_transports(*, coordinated: bool = True) -> None:
    retained: dict[tuple, Mxfp4StreamTransport] = {}
    for key, manager in tuple(_TRANSPORTS.items()):
        try:
            released = manager.close(coordinated=coordinated)
        except Exception:
            logger.debug("failed to close MXFP4 stream transport", exc_info=True)
            released = False
        if not released:
            # Do not drop the final Python references: SharedMemory destructors
            # must not unregister pages that active writer/DMA work may touch.
            retained[key] = manager
    _TRANSPORTS.clear()
    _TRANSPORTS.update(retained)


def _atexit_close_mxfp4_stream_transports() -> None:
    close_mxfp4_stream_transports(coordinated=False)


def _tp_rank_world() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    return (
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
    )


def _tp_cpu_group():
    from sglang.srt.distributed import get_tp_group

    return get_tp_group().cpu_group


def _tp_first_rank() -> int:
    from sglang.srt.distributed import get_tp_group

    return get_tp_group().first_rank


def _tp_all_succeeded(local_success: bool) -> bool:
    _, world = _tp_rank_world()
    if world == 1:
        return local_success
    status = torch.tensor([int(local_success)], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=_tp_cpu_group())
    return bool(status.item())


def _tp_boolean_consensus(local_value: bool) -> tuple[bool, bool]:
    """Return ``(all_true, any_true)`` for one TP-local boolean."""

    _, world = _tp_rank_world()
    if world == 1:
        return local_value, local_value
    count = torch.tensor([int(local_value)], dtype=torch.int32, device="cpu")
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=_tp_cpu_group())
    true_count = int(count.item())
    return true_count == world, true_count > 0


def _tp_all_gather_objects(value: Any) -> list[Any]:
    _, world = _tp_rank_world()
    if world == 1:
        return [value]
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, value, group=_tp_cpu_group())
    return gathered


def _tp_named_stage_commit(token: tuple, local_success: bool) -> bool:
    reports = _tp_all_gather_objects((token, bool(local_success)))
    expected = reports[0][0]
    if any(report_token != expected for report_token, _ in reports):
        raise RuntimeError(
            "MXFP4 TP factory/control token mismatch: "
            f"rank0_expected={expected}, "
            f"mismatches={_format_token_mismatches(reports)}"
        )
    return all(success for _, success in reports)


def _format_token_mismatches(reports: Sequence[tuple[Any, bool]]) -> list[tuple[int, Any]]:
    expected = reports[0][0]
    return [
        (rank, report_token)
        for rank, (report_token, _) in enumerate(reports)
        if report_token != expected
    ]


def _validate_tp_layout_contract(
    reports: Sequence[tuple[str, tuple]],
) -> tuple:
    """Require homogeneous GPU capability and expert layout across TP ranks."""

    expected = reports[0][1]
    mismatches = [
        (rank, device, signature)
        for rank, (device, signature) in enumerate(reports)
        if signature != expected
    ]
    if mismatches:
        rank0_device, rank0_signature = reports[0]
        raise RuntimeError(
            "KT MXFP4 streamed transport requires a homogeneous TP group: "
            "all ranks must have the same GPU compute capability and identical "
            "expert shapes/dtypes. "
            f"rank0=({rank0_device}, {rank0_signature}); "
            f"mismatches={mismatches}"
        )
    return expected


def _broadcast_object_from_tp0(value: Any) -> Any:
    rank, world = _tp_rank_world()
    if world == 1:
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(
        payload,
        src=_tp_first_rank(),
        group=_tp_cpu_group(),
    )
    return payload[0]


atexit.register(_atexit_close_mxfp4_stream_transports)

