# SPDX-License-Identifier: Apache-2.0
"""GEC-driven GPU-expert residency for the KT EP wrapper.

Implements the admission/eviction half of doc/KT-CPU-PICE-GPU.md on top of the
existing KT dynamic expert update machinery:

- ``kt_kernel.gec`` (the C++ Global Expert Cache) owns every decision:
  residency lookup, candidate filter, priority, Top-N admission, LRU eviction
  with probation protection.
- The physical side (weight copies, routing-table updates, TP broadcast) keeps
  using the proven dynamic update path in ``kt_ep_wrapper``.

Parameter mapping (all counted in Logical Experts, per the doc):

- ``--kt-expert-gpu-slots``: global resident capacity across MoE layers. It
  sizes the static GPU-expert rows at init (see
  ``_init_kt_gpu_experts_masks``) and, on its own, activates one model-scoped
  GEC cache shared by every wrapped layer. Each layer keeps a physical row
  view of that cache while admission and eviction use the global capacity. It
  does not require the legacy
  ``--kt-enable-dynamic-expert-update`` /
  ``--kt-gpu-prefill-token-threshold`` pair. The threshold remains an
  independent legacy full-GPU prefill control.
- ``--kt-layer-h2d-slots``: per-layer per-round admission budget, i.e. how
  many experts may be swapped into the cache in one update round. Defaults to
  the layer capacity (unbounded, legacy-like) when unset.
- ``--kt-layer-h2d-batch-size``: maximum experts per H2D copy batch. A batch
  is ``min(available, batch_size)`` and never waits to be filled. Defaults to
  a single batch when unset.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class GecCudaPipeline:
    """Device-local H2D pipeline used by the runtime update path.

    The callable runs on a dedicated copy stream.  The source stream is
    captured with an event and the caller's current stream waits only for the
    submitted copy event.  This keeps the dependency local to the layer and
    avoids a process-wide CUDA synchronization.
    """

    def __init__(self, device: torch.device, depth: int = 1) -> None:
        self.device = torch.device(device)
        self.depth = max(1, int(depth))
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if torch.cuda.is_available()
            else None
        )
        self._pending: List[torch.cuda.Event] = []
        self._slot_events: List[Optional[torch.cuda.Event]] = [None] * self.depth
        self._next_slot = 0
        self.pool = GecBufferPool(self.depth)
        self._ready_slots: List[int] = []
        self._gpu_slots: List[int] = []
        self.last_queue_us = 0
        self.last_submit_us = 0
        self.last_event_wait_us = 0

    @property
    def enabled(self) -> bool:
        return self.stream is not None

    def submit(
        self,
        operation: Callable[[], None],
        *,
        wait_on_current: bool = True,
    ) -> Optional[torch.cuda.Event]:
        """Enqueue one copy with bounded, reusable in-flight slots."""
        # A slot is reusable only after the GPU stream consuming its payload
        # has completed. The copy event in ``_slot_events`` is not a release
        # signal, so acquire through the pool before submitting a new copy.
        self.pool.poll()
        while True:
            try:
                slot_buffer = self.pool.acquire()
                break
            except RuntimeError:
                if not self.pool.wait_for_free():
                    raise RuntimeError(
                        "GEC H2D buffer pool has no free slot; consume the "
                        "pending H2D window before submitting another copy"
                    )
        slot = slot_buffer.slot_id
        if self.stream is None:
            self._next_slot = (self._next_slot + 1) % self.depth
            try:
                operation()
            except Exception:
                self.pool.release_unstarted(slot_buffer)
                raise
            self.pool.mark_ready(slot_buffer, None)
            self._ready_slots.append(slot)
            return None

        submit_start = time.perf_counter_ns()
        current = torch.cuda.current_stream(self.device)
        self._next_slot = (self._next_slot + 1) % self.depth
        previous = self._slot_events[slot]
        if previous is not None and not previous.query():
            wait_start = time.perf_counter_ns()
            current.wait_event(previous)
            self.last_queue_us = (time.perf_counter_ns() - wait_start) // 1000
        else:
            self.last_queue_us = 0
        source_ready = torch.cuda.Event(enable_timing=False)
        source_ready.record(current)
        self.stream.wait_event(source_ready)
        try:
            with torch.cuda.stream(self.stream):
                operation()
                complete = torch.cuda.Event(enable_timing=False)
                complete.record(self.stream)
        except Exception:
            # A copy may already have been queued before the callable raises.
            # Keep the slot occupied until that device-local work drains, then
            # let pool.poll() recycle it without exposing it to compute.
            cleanup = torch.cuda.Event(enable_timing=False)
            cleanup.record(self.stream)
            self._slot_events[slot] = cleanup
            self.pool.defer_release(slot_buffer, cleanup)
            self._pending.append(cleanup)
            raise
        if wait_on_current:
            current.wait_event(complete)
        self.last_submit_us = (time.perf_counter_ns() - submit_start) // 1000
        self.last_event_wait_us = 0
        self._slot_events[slot] = complete
        self.pool.mark_ready(slot_buffer, complete)
        if slot not in self._ready_slots:
            self._ready_slots.append(slot)
        self._pending.append(complete)
        if len(self._pending) > self.depth:
            self._pending = [event for event in self._pending if not event.query()]
        return complete

    def begin_compute(self) -> None:
        """Move all completed H2D slots into GPU_IN_USE."""
        for slot_id in self._ready_slots:
            self.pool.begin_gpu_use(self.pool.slots[slot_id])
            self._gpu_slots.append(slot_id)
        self._ready_slots.clear()

    def complete_compute(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Release slots only after the consuming GPU stream completes."""
        if not self._gpu_slots:
            return
        if self.stream is None:
            for slot_id in self._gpu_slots:
                self.pool.gpu_complete(self.pool.slots[slot_id], None)
            self._gpu_slots.clear()
            return
        completion = torch.cuda.Event(enable_timing=False)
        completion.record(stream or torch.cuda.current_stream(self.device))
        for slot_id in self._gpu_slots:
            self.pool.gpu_complete(self.pool.slots[slot_id], completion)
        self._gpu_slots.clear()

    def release_ready(self) -> None:
        """Release copy slots that are not consumed by a GPU GEMM."""
        for slot_id in self._ready_slots:
            self.pool.release_unstarted(self.pool.slots[slot_id])
        self._ready_slots.clear()

class GecExecutionLease:
    """A cache refcount lease released after a CUDA completion event."""

    def __init__(self, planner: "GecResidencyPlanner", expert_ids: Iterable[int]):
        self._planner = planner
        self._ids = tuple(dict.fromkeys(int(expert_id) for expert_id in expert_ids))
        self._event: Optional[torch.cuda.Event] = None
        self._released = False
        self._submitted = False
        # Reserve the lease before H2D publishes READY.  This keeps a newly
        # admitted row eviction-proof while its H2D dependency is pending.
        acquire = getattr(planner._cache, "acquire_pending", planner._cache.acquire)
        self._acquired = [
            expert_id
            for expert_id in self._ids
            if acquire(planner._id(expert_id))
        ]

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        if self._released:
            return
        if torch.cuda.is_available():
            self._event = torch.cuda.Event(enable_timing=False)
            self._event.record(stream or torch.cuda.current_stream())
            self._submitted = True
        else:
            self.release()

    def poll(self) -> bool:
        if self._released:
            return True
        if not self._submitted:
            return False
        if self._event is not None and not self._event.query():
            return False
        self.release()
        return True

    def release(self) -> None:
        if self._released:
            return
        for expert_id in self._acquired:
            self._planner._cache.release(self._planner._id(expert_id))
        self._released = True


class GecTransferLease:
    """Binds a logical admission to the completion of its copy event."""

    def __init__(
        self,
        planner: "GecResidencyPlanner",
        expert_ids: Iterable[int],
        event: Optional[torch.cuda.Event],
    ) -> None:
        self._planner = planner
        self._expert_ids = tuple(dict.fromkeys(int(item) for item in expert_ids))
        self._event = event
        self._completed = False
        self._wait_start_ns = time.perf_counter_ns()

    def poll(self) -> bool:
        if self._completed:
            return True
        if self._event is not None and not self._event.query():
            return False
        self._planner.record_wait(
            self._planner._gec.WaitReason.H2D_EVENT_WAIT,
            (time.perf_counter_ns() - self._wait_start_ns) // 1000,
        )
        self._planner.complete_h2d(self._expert_ids)
        self._completed = True
        return True


@dataclass
class _LayerState:
    num_experts: int
    capacity: int
    budget: int
    planner: Optional["GecResidencyPlanner"] = None
    resize: Optional[Callable[[int, List[int]], None]] = None


@dataclass
class GecBufferSlot:
    """Per-device temporary H2D slot lifecycle."""

    slot_id: int
    state: str = "FREE"
    event: Optional[torch.cuda.Event] = None


class GecBufferPool:
    """Bounded per-device pool used by the runtime copy pipeline."""

    def __init__(self, depth: int) -> None:
        self.slots = [GecBufferSlot(index) for index in range(max(1, int(depth)))]

    def acquire(self) -> GecBufferSlot:
        for slot in self.slots:
            if slot.state == "FREE":
                slot.state = "H2D"
                return slot
        # The copy stream is ordered; reclaim the oldest completed event.
        for slot in self.slots:
            if slot.state == "COMPLETION_WAIT" and (
                slot.event is None or slot.event.query()
            ):
                slot.state = "H2D"
                slot.event = None
                return slot
        raise RuntimeError("GEC H2D buffer pool exhausted")

    def wait_for_free(self) -> bool:
        """Wait for one device-local consumer event and free its slot.

        A READY_FOR_COMPUTE slot has no consumer completion event yet and must
        be consumed by the compute path before another copy is submitted.
        """
        pending = [
            slot.event
            for slot in self.slots
            if slot.state == "COMPLETION_WAIT" and slot.event is not None
        ]
        if not pending:
            return False
        synchronize = getattr(pending[0], "synchronize", None)
        if synchronize is None:
            return False
        synchronize()
        self.poll()
        return True

    def mark_ready(self, slot: GecBufferSlot, event) -> None:
        if slot.state != "H2D":
            raise RuntimeError("GEC H2D slot is not available for completion")
        slot.event = event
        slot.state = "READY_FOR_COMPUTE"

    def defer_release(self, slot: GecBufferSlot, event) -> None:
        """Release an H2D slot after failed work already queued on its stream."""
        if slot.state != "H2D":
            raise RuntimeError("GEC H2D slot is not pending a failed copy")
        slot.event = event
        slot.state = "COMPLETION_WAIT"

    def begin_gpu_use(self, slot: GecBufferSlot) -> None:
        if slot.state != "READY_FOR_COMPUTE":
            raise RuntimeError("GEC H2D slot is not ready for compute")
        slot.state = "GPU_IN_USE"

    def gpu_complete(self, slot: GecBufferSlot, event) -> None:
        if slot.state != "GPU_IN_USE":
            raise RuntimeError("GEC H2D slot is not in GPU use")
        slot.event = event
        slot.state = "COMPLETION_WAIT" if event is not None else "FREE"

    def release_unstarted(self, slot: GecBufferSlot) -> None:
        if slot.state not in {"H2D", "READY_FOR_COMPUTE"}:
            raise RuntimeError("GEC H2D slot has already entered GPU use")
        slot.event = None
        slot.state = "FREE"

    def poll(self) -> None:
        for slot in self.slots:
            if slot.state == "COMPLETION_WAIT" and (
                slot.event is None or slot.event.query()
            ):
                slot.state = "FREE"
                slot.event = None

    @property
    def free_count(self) -> int:
        self.poll()
        return sum(slot.state == "FREE" for slot in self.slots)


class GlobalGecCoordinator:
    """One persistent cache shared by every routed MoE layer in a model.

    The physical row budget is still registered per layer during model
    construction, but admission, eviction, hit accounting, and inflight
    ownership all use this single cache.  Logical ids retain their layer id,
    so a hot expert from one layer can compete with an idle expert from any
    other layer when a model has spare/rebalanced rows.
    """

    def __init__(
        self,
        gec_module,
        capacity: int,
        num_gpu_layers: int = 0,
        h2d_buffer_pool_depth: int = 1,
        tp_size: int = 1,
        max_running_requests: int = 0,
        h2d_batch_size: int = 1,
    ) -> None:
        if capacity < 1:
            raise ValueError("global GEC capacity must be >= 1")
        config = gec_module.GecConfig()
        config.num_gpu_layers = max(0, int(num_gpu_layers))
        config.expert_gpu_slots = int(capacity)
        config.layer_h2d_slots = int(capacity)
        # Each concurrently running request may own one current H2D batch;
        # batch size determines the number of Logical slots in that batch,
        # not the number of reusable request slots.
        derived_depth = max(0, int(max_running_requests))
        config.h2d_buffer_pool_depth = max(
            1, int(h2d_buffer_pool_depth), derived_depth
        )
        config.validate()
        self._gec = gec_module
        self.capacity = int(capacity)
        self.tp_size = max(1, int(tp_size))
        self.buffer_pool_depth = config.h2d_buffer_pool_depth
        self._cache = gec_module.GlobalExpertCache(
            config, gec_module.Topology(self.tp_size, 1, self.tp_size)
        )
        self._layers: Dict[int, _LayerState] = {}
        self._lock = threading.RLock()
        self.layout_epoch = 0

    @property
    def cache(self):
        return self._cache

    def register_layer(self, layer_idx: int, num_experts: int, capacity: int, budget: int) -> None:
        with self._lock:
            state = self._layers.get(int(layer_idx))
            if state is None:
                state = _LayerState(
                    num_experts=int(num_experts),
                    capacity=min(max(0, int(capacity)), int(num_experts)),
                    budget=max(0, int(budget)),
                )
                self._layers[int(layer_idx)] = state
            else:
                state.num_experts = int(num_experts)
                state.capacity = min(max(0, int(capacity)), int(num_experts))
                state.budget = max(0, int(budget))

    def register_physical_layer(
        self,
        layer_idx: int,
        planner: "GecResidencyPlanner",
        resize: Callable[[int, List[int]], None],
    ) -> None:
        with self._lock:
            state = self._layers[int(layer_idx)]
            state.planner = planner
            state.resize = resize

    def ensure_capacity(self, layer_idx: int, required: int) -> int:
        """Borrow physical rows from cold layers for a hot target layer.

        The operation only evicts experts that satisfy the normal cache gate
        (READY, no refcount, no inflight dependency). If a donor is busy, its
        rows remain in place and the target receives fewer rows this round.
        """
        with self._lock:
            target = self._layers.get(int(layer_idx))
            if target is None:
                return 0
            required = min(max(0, int(required)), target.num_experts)
            if required <= target.capacity:
                return target.capacity
            needed = required - target.capacity
            donors = [
                (idx, state)
                for idx, state in self._layers.items()
                if idx != int(layer_idx) and state.capacity > 0
            ]
            donors.sort(
                key=lambda item: (
                    self._layer_heat(item[1]),
                    item[0],
                )
            )
            for donor_idx, donor in donors:
                if needed <= 0:
                    break
                reducible = min(donor.capacity, needed)
                if donor.planner is None:
                    donor.capacity -= reducible
                    needed -= reducible
                    continue
                keep_ids = donor.planner._resident_ids()
                keep_ids.sort(
                    key=lambda item: (
                        donor.planner._latest_count(item.expert_id),
                        item,
                    )
                )
                removed = 0
                for victim in keep_ids:
                    if removed >= reducible:
                        break
                    if self._cache.evict(victim):
                        removed += 1
                if removed == 0:
                    continue
                new_capacity = donor.capacity - removed
                retained = [
                    item.expert_id
                    for item in donor.planner._resident_ids()
                ]
                if donor.resize is not None:
                    donor.resize(new_capacity, retained)
                donor.capacity = new_capacity
                donor.planner.capacity = new_capacity
                needed -= removed
            borrowed = required - target.capacity - max(0, needed)
            target.capacity += max(0, borrowed)
            if target.resize is not None and target.planner is not None:
                target.resize(
                    target.capacity,
                    [item.expert_id for item in target.planner._resident_ids()],
                )
            return target.capacity

    def export_layout(self) -> Dict[str, object]:
        """Export capacities and resident ids for a TP-rank layout broadcast."""
        with self._lock:
            layers: Dict[int, Dict[str, object]] = {}
            for idx, state in self._layers.items():
                residents = []
                if state.planner is not None:
                    residents = [
                        int(item.expert_id)
                        for item in state.planner._resident_ids()
                    ]
                layers[int(idx)] = {
                    "capacity": int(state.capacity),
                    "residents": residents,
                }
            self.layout_epoch += 1
            return {"epoch": self.layout_epoch, "layers": layers}

    def apply_layout(self, snapshot: Dict[str, object]) -> None:
        """Apply a rank-0 layout before local routing-table updates."""
        if not snapshot or not isinstance(snapshot.get("layers"), dict):
            raise ValueError("invalid GEC layout snapshot")
        with self._lock:
            self.layout_epoch = max(
                self.layout_epoch, int(snapshot.get("epoch", self.layout_epoch))
            )
            # Free all idle non-target residents first.  This makes the
            # mirror transaction independent of dictionary/layer iteration
            # order when capacity moves between layers.
            for raw_idx, raw_state in snapshot["layers"].items():
                state = self._layers.get(int(raw_idx))
                if state is None or not isinstance(raw_state, dict) or state.planner is None:
                    continue
                desired = {int(item) for item in raw_state.get("residents", [])}
                current = {
                    int(item.expert_id)
                    for item in state.planner._cache.resident_ids()
                    if item.layer_id == int(raw_idx)
                }
                for item in sorted(current - desired):
                    snapshot_state = state.planner._cache.snapshot(
                        state.planner._id(item)
                    )
                    if snapshot_state is not None and snapshot_state.refcount == 0:
                        state.planner._cache.evict(state.planner._id(item))
            for raw_idx, raw_state in snapshot["layers"].items():
                state = self._layers.get(int(raw_idx))
                if state is None or not isinstance(raw_state, dict):
                    continue
                capacity = int(raw_state.get("capacity", state.capacity))
                residents = [int(item) for item in raw_state.get("residents", [])]
                current = []
                if state.planner is not None:
                    current = [
                        int(item.expert_id)
                        for item in state.planner._resident_ids()
                    ]
                changed = capacity != state.capacity or current != residents
                removed = set(current) - set(residents)
                if state.planner is not None and removed:
                    busy = False
                    for item in removed:
                        snapshot_state = state.planner._cache.snapshot(
                            state.planner._id(item)
                        )
                        if snapshot_state is not None and snapshot_state.refcount > 0:
                            busy = True
                            break
                    if busy:
                        # A rank-local GPU lease is still active.  Keep the
                        # old rows until the next layout epoch rather than
                        # resizing storage underneath a live kernel.
                        continue
                if changed and state.resize is not None:
                    state.resize(capacity, residents)
                state.capacity = capacity
                if state.planner is not None:
                    state.planner.capacity = capacity
                    state.planner.sync_residents(residents)

    @staticmethod
    def _layer_heat(state: _LayerState) -> int:
        if state.planner is None:
            return 0
        return sum(state.planner._latest_counts)

    def layer_capacity(self, layer_idx: int) -> Optional[int]:
        with self._lock:
            state = self._layers.get(int(layer_idx))
            return None if state is None else state.capacity

    def resident_ids(self, layer_idx: Optional[int] = None):
        ids = self._cache.resident_ids()
        if layer_idx is None:
            return ids
        return [expert_id for expert_id in ids if expert_id.layer_id == int(layer_idx)]

    def summary(self) -> str:
        stats = self._cache.stats()
        return (
            f"GEC global: capacity={self.capacity} residents="
            f"{self._cache.resident_size()} hits={stats.hits} misses={stats.misses} "
            f"hit_rate={stats.hit_rate():.3f} admissions={stats.admissions} "
            f"evictions={stats.evictions} reh2d={stats.reh2d}"
        )


# Runtime model scope.  `create_kt_config_from_server_args` supplies the same
# coordinator object to every layer built from one ServerArgs instance.  The
# identity key prevents test/model reloads from sharing stale cache state.
_GEC_COORDINATOR_SCOPE: Optional[int] = None
_GEC_COORDINATOR_OWNER = None
_GEC_COORDINATOR: Optional[GlobalGecCoordinator] = None


def get_or_create_gec_coordinator(scope, kt_config, gec_module) -> GlobalGecCoordinator:
    global _GEC_COORDINATOR_SCOPE, _GEC_COORDINATOR_OWNER, _GEC_COORDINATOR
    scope_id = id(scope)
    if _GEC_COORDINATOR is None or _GEC_COORDINATOR_OWNER is not scope:
        slots = int(getattr(kt_config, "kt_expert_gpu_slots", 0) or 0)
        _GEC_COORDINATOR = GlobalGecCoordinator(
            gec_module,
            capacity=slots,
            num_gpu_layers=int(getattr(kt_config, "kt_num_gpu_layers", 0) or 0),
            tp_size=int(getattr(kt_config, "kt_tp_size", 1) or 1),
            max_running_requests=int(
                getattr(kt_config, "max_running_requests", 0) or 0
            ),
            h2d_batch_size=int(
                getattr(kt_config, "kt_layer_h2d_batch_size", 1) or 1
            ),
        )
        _GEC_COORDINATOR_SCOPE = scope_id
        _GEC_COORDINATOR_OWNER = scope
    return _GEC_COORDINATOR


def plan_stable_row_update(
    old_row_owner: torch.Tensor, new_residents: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diff a new resident set against current row ownership.

    Stable row assignment keeps each resident expert in its GPU row across
    rounds, so a round only copies the newly admitted experts into the rows
    freed by evicted ones instead of re-copying the whole resident set.

    Args:
        old_row_owner: int64 CPU tensor [capacity], row -> resident logical id.
        new_residents: int64 CPU tensor [capacity], sorted resident ids for
            the next round (the GEC plan result).

    Returns:
        ``(arrived, freed_rows, new_row_owner)``: ``arrived`` are the newly
        resident logical ids (ascending), ``freed_rows`` the rows whose
        previous resident departed (ascending, same length; ``arrived[k]``
        takes ``freed_rows[k]``), and ``new_row_owner`` the updated
        row -> logical id tensor.

    The result is deterministic given identical inputs, so every TP rank
    derives the same row assignment from the broadcast resident set.
    """
    arrived = new_residents[~torch.isin(new_residents, old_row_owner)]
    freed_rows = torch.nonzero(~torch.isin(old_row_owner, new_residents)).flatten()
    if arrived.numel() != freed_rows.numel():
        raise RuntimeError(
            "GEC row diff mismatch: "
            f"{arrived.numel()} arrivals vs {freed_rows.numel()} freed rows "
            "(resident set size invariant broken)"
        )
    new_row_owner = old_row_owner.clone()
    new_row_owner[freed_rows] = arrived
    return arrived, freed_rows, new_row_owner


def _load_gec_module():
    """Return the kt_kernel.g ext submodule, or None when unavailable."""
    try:
        import kt_kernel
    except ImportError:
        return None
    ext = getattr(kt_kernel, "kt_kernel_ext", None)
    return getattr(ext, "gec", None) if ext is not None else None


def gec_enabled(kt_config) -> bool:
    """Whether GEC residency planning is requested for this layer."""
    # GEC stands alone: --kt-expert-gpu-slots activates dynamic hot-expert
    # residency without the legacy --kt-enable-dynamic-expert-update /
    # --kt-gpu-prefill-token-threshold pair.
    slots = getattr(kt_config, "kt_expert_gpu_slots", None)
    return slots is not None and slots > 0


class GecResidencyPlanner:
    """Layer view over the model-scoped GEC admission/eviction planner.

    With a model coordinator, every TP rank maintains the same deterministic
    view so physical row reallocation happens before the selected set is
    broadcast. Direct construction without a coordinator remains useful for
    isolated unit tests and stays rank-0-only at the caller.
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        capacity: int,
        budget: int,
        gec_module,
        coordinator: Optional[GlobalGecCoordinator] = None,
    ) -> None:
        if capacity < 0 or (capacity == 0 and coordinator is None):
            raise ValueError("GEC capacity must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self._gec = gec_module
        self._coordinator = coordinator
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = min(capacity, num_experts)
        self.budget = max(0, budget)

        config = gec_module.GecConfig()
        config.expert_gpu_slots = (
            coordinator.capacity if coordinator is not None else self.capacity
        )
        config.layer_h2d_slots = self.budget
        config.validate()
        self._cache = (
            coordinator.cache
            if coordinator is not None
            else gec_module.GlobalExpertCache(config, gec_module.Topology(1, 1, 1))
        )
        self._scheduler = gec_module.H2DScheduler(config)
        self._telemetry = gec_module.TelemetryCollector()
        self._prefilled = False
        self._rounds = 0
        self._leases: List[GecExecutionLease] = []
        self._transfer_leases: List[GecTransferLease] = []
        self._inflight = gec_module.InflightRegistry()
        self._inflight_lock = threading.Lock()
        self._next_dependency_id = 1
        self._dependency_events: Dict[int, torch.cuda.Event] = {}
        self._defer_ready = coordinator is not None
        self._tp_size = coordinator.tp_size if coordinator is not None else 1
        self.buffer_pool_depth = (
            coordinator.buffer_pool_depth if coordinator is not None else 1
        )
        self._latest_counts: List[int] = [0] * num_experts
        if coordinator is not None:
            coordinator.register_layer(
                layer_idx, num_experts, self.capacity, self.budget
            )

    def _id(self, expert_id: int):
        return self._gec.LogicalExpertId(self.layer_idx, expert_id)

    def _latest_count(self, expert_id: int) -> int:
        if 0 <= int(expert_id) < len(self._latest_counts):
            return int(self._latest_counts[int(expert_id)])
        return 0

    def _admit_and_mark_ready(self, expert_id: int, now_us: int) -> bool:
        """Admit one expert and drive it to READY (evicting LRU on demand)."""
        eid = self._id(expert_id)
        if not self._cache.admit(eid, now_us):
            return False
        existing = self._cache.snapshot(eid)
        if existing is not None and existing.state in (
            self._gec.LifecycleState.LOADING,
            self._gec.LifecycleState.READY,
        ):
            return True
        self._cache.mark_scheduled(eid)
        self._cache.mark_loading(eid)
        if not self._defer_ready:
            for tp_rank in range(self._tp_size):
                self._cache.mark_shard_ready(eid, tp_rank, now_us)
            self._cache.record_h2d(eid)
        return True

    def initialize_residents(self, expert_ids: Iterable[int]) -> None:
        if self._coordinator is None:
            self._initialize_residents_locked(expert_ids)
            return
        with self._coordinator._lock:
            self._initialize_residents_locked(expert_ids)

    def _initialize_residents_locked(self, expert_ids: Iterable[int]) -> None:
        resident_ids = tuple(dict.fromkeys(int(item) for item in expert_ids))
        if any(item < 0 or item >= self.num_experts for item in resident_ids):
            raise ValueError('invalid initial GEC resident')
        now_us = time.perf_counter_ns() // 1000
        for expert_id in resident_ids:
            logical_id = self._id(expert_id)
            snapshot = self._cache.snapshot(logical_id)
            if snapshot is not None and snapshot.state == self._gec.LifecycleState.READY:
                continue
            if snapshot is not None and snapshot.state == self._gec.LifecycleState.LOADING:
                raise RuntimeError('initial GEC resident is still loading')
            if not self._cache.admit(logical_id, now_us):
                raise RuntimeError('global cache is full during initial GEC registration')
            self._cache.record_h2d(logical_id)
            self._cache.mark_scheduled(logical_id)
            self._cache.mark_loading(logical_id)
            for tp_rank in range(self._tp_size):
                self._cache.mark_shard_ready(logical_id, tp_rank, now_us)
        self._prefilled = True

    def plan(self, counts: List[int]) -> List[int]:
        # Cache admission and eviction are model-global. Serialize the
        # decision window so two concurrently executing layers cannot both
        # select the same free slot based on stale resident sets.
        if self._coordinator is None:
            return self._plan_locked(counts)
        with self._coordinator._lock:
            return self._plan_locked(counts)

    def _plan_locked(self, counts: List[int]) -> List[int]:
        """Compute the resident set for the next round.

        Args:
            counts: activation count per logical expert in the routed batch
                (length num_experts).

        Returns:
            Sorted list of resident logical expert ids; its length always
            equals ``self.capacity`` so the GPU routing tables stay dense.
        """
        round_start = time.perf_counter_ns()
        self._latest_counts = [
            max(0, int(counts[index])) if index < len(counts) else 0
            for index in range(self.num_experts)
        ]
        self.poll_gpu_completions()
        self.poll_h2d_completions()
        now_us = time.perf_counter_ns() // 1000
        cache = self._cache

        if not self._prefilled and self.capacity == 0:
            self._prefilled = True
        if not self._prefilled and self.capacity > 0:
            # Startup prefill: fill every resident row in one round, exempt
            # from the per-round H2D budget (doc: startup sequential prefill).
            # Startup prefill is deterministic layer/expert order.  Runtime
            # demand affects steady-state admission after the initial cache is
            # established, rather than changing the documented warm-start
            # layout.
            order = list(range(self.num_experts))
            for expert_id in order[: self.capacity]:
                self._admit_and_mark_ready(expert_id, now_us)
                if counts[expert_id]:
                    cache.record_use(self._id(expert_id), now_us)
            self._prefilled = True
        else:
            # Steady state: refresh routed residents, then admit the Top-N
            # misses within the per-round budget (doc sections 12-13).
            candidates = []
            for expert_id in range(self.num_experts):
                demand = counts[expert_id]
                if not demand:
                    continue
                eid = self._id(expert_id)
                lookup_start = time.perf_counter_ns()
                lookup_result = cache.lookup(eid, now_us)
                lookup_us = (time.perf_counter_ns() - lookup_start) // 1000
                if lookup_result == self._gec.LookupResult.HIT:
                    cache.record_use(eid, now_us)
                    self.record_cache_wait(expert_id, lookup_us)
                    continue
                snapshot = cache.snapshot(eid)
                candidate = self._gec.SchedulerCandidate()
                candidate.id = eid
                candidate.current_demand = demand
                candidate.historical_hit_count = (
                    snapshot.hit_count if snapshot is not None else 0
                )
                candidate.inflight = (
                    snapshot is not None
                    and snapshot.state == self._gec.LifecycleState.LOADING
                )
                if hasattr(candidate, "reuse_inflight"):
                    candidate.reuse_inflight = candidate.inflight
                candidate.cache_admission_locked = cache.admission_locked(eid)
                candidates.append(candidate)

            decision = self._scheduler.schedule(candidates, self.budget)
            if self._coordinator is not None and decision.h2d_selected:
                desired = min(
                    self.num_experts,
                    len(self._resident_ids()) + len(decision.h2d_selected),
                )
                self.capacity = self._coordinator.ensure_capacity(
                    self.layer_idx, desired
                )
            for eid in decision.h2d_selected:
                if self._admit_and_mark_ready(eid.expert_id, now_us):
                    cache.record_use(eid, now_us)

        self._rounds += 1
        self.record_router_dependency(
            (time.perf_counter_ns() - round_start) // 1000
        )
        return [eid.expert_id for eid in self._resident_ids()]

    def _resident_ids(self):
        if self._coordinator is not None:
            return self._coordinator.resident_ids(self.layer_idx)
        return [
            expert_id
            for expert_id in self._cache.resident_ids()
            if expert_id.layer_id == self.layer_idx
        ]

    def sync_residents(self, expert_ids: Iterable[int]) -> None:
        """Mirror rank-0 cache residency without re-running admission policy."""
        desired = {int(item) for item in expert_ids}
        current = {
            int(item.expert_id)
            for item in self._cache.resident_ids()
            if item.layer_id == self.layer_idx
        }
        for expert_id in sorted(current - desired):
            try:
                self._cache.evict(self._id(expert_id))
            except (AttributeError, RuntimeError):
                pass
        now_us = time.perf_counter_ns() // 1000
        for expert_id in sorted(desired):
            logical_id = self._id(expert_id)
            snapshot = self._cache.snapshot(logical_id)
            if snapshot is None:
                admitted = self._cache.admit(logical_id, now_us)
                if not admitted:
                    # A mirror rank must follow the broadcast layout even if
                    # its local probation state differs.  Recycle only a
                    # non-target, idle resident before retrying admission.
                    for victim in sorted(current - desired):
                        victim_snapshot = self._cache.snapshot(self._id(victim))
                        if (
                            victim_snapshot is not None
                            and victim_snapshot.refcount == 0
                            and self._cache.evict(self._id(victim))
                        ):
                            admitted = self._cache.admit(logical_id, now_us)
                            if admitted:
                                break
                if not admitted:
                    raise RuntimeError(
                        f"cannot mirror GEC resident {(self.layer_idx, expert_id)}"
                    )
                self._cache.mark_scheduled(logical_id)
                self._cache.mark_loading(logical_id)
            elif snapshot.state == self._gec.LifecycleState.Ready:
                continue
            elif snapshot.state == self._gec.LifecycleState.Absent:
                continue
            # The following copy event publishes READY after the current
            # update's H2D event, preserving the dependency boundary.

    def begin_gpu_use(
        self, expert_ids: Iterable[int], stream: Optional[torch.cuda.Stream] = None
    ) -> Optional[GecExecutionLease]:
        """Acquire resident experts until the supplied stream completes."""
        lease = GecExecutionLease(self, expert_ids)
        if not lease._acquired:
            return None
        self._leases.append(lease)
        if stream is not None or not torch.cuda.is_available():
            lease.record(stream)
        return lease

    def poll_gpu_completions(self) -> None:
        """Release only leases whose device event has completed."""
        if self._leases:
            self._leases = [lease for lease in self._leases if not lease.poll()]

    def _physical_shards(self, expert_id: int) -> Tuple[object, ...]:
        return tuple(
            self._gec.PhysicalShardId(self.layer_idx, int(expert_id), tp_rank)
            for tp_rank in range(self._tp_size)
        )

    def _ready_dependencies(self, expert_id: int) -> Tuple[int, ...]:
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= self.num_experts:
            return (0,) * self._tp_size
        logical_id = self._id(expert_id)
        if self._cache.snapshot(logical_id) is None:
            # Checkpoint-loaded resident rows can be used before the first GEC
            # admission round registers them in the cache.  Such rows have no
            # GEC-owned H2D event to wait for.
            return (0,) * self._tp_size
        dependencies = []
        for tp_rank in range(self._tp_size):
            dependency = self._cache.shard_dependency(logical_id, tp_rank)
            dependencies.append(0 if dependency is None else int(dependency))
        return tuple(dependencies)

    def reserve_h2d(
        self, expert_ids: Iterable[int]
    ) -> Tuple[List[int], List[int]]:
        with self._inflight_lock:
            return self._reserve_h2d_locked(expert_ids)

    def _reserve_h2d_locked(
        self, expert_ids: Iterable[int]
    ) -> Tuple[List[int], List[int]]:
        """Reserve each physical shard before submitting an H2D copy.

        Returns ``(new_ids, reused_ids)``.  Reused ids already have an active
        per-shard dependency and must wait for the original copy rather than
        submitting a duplicate transfer.
        """
        new_ids: List[int] = []
        reused_ids: List[int] = []
        now_us = time.perf_counter_ns() // 1000
        for expert_id in dict.fromkeys(int(item) for item in expert_ids):
            shards = self._physical_shards(expert_id)
            existing = [self._inflight.dependency_id(shard) for shard in shards]
            if any(dependency is not None for dependency in existing):
                if not all(dependency is not None for dependency in existing):
                    raise RuntimeError(
                        f"partial inflight registration for expert {expert_id}"
                    )
                reused_ids.append(expert_id)
                continue

            registered = []
            dependencies = []
            try:
                for shard in shards:
                    inserted, dependency = self._inflight.register_transfer(
                        shard, self._next_dependency_id, now_us
                    )
                    if not inserted:
                        raise RuntimeError(
                            f"inflight registration raced for expert {expert_id}"
                        )
                    registered.append(shard)
                    dependencies.append(int(dependency))
                    self._next_dependency_id += 1
            except Exception:
                for shard in registered:
                    self._inflight.complete(shard)
                raise
            logical_id = self._id(expert_id)
            try:
                for tp_rank, dependency in enumerate(dependencies):
                    self._cache.bind_shard_dependency(
                        logical_id, tp_rank, dependency
                    )
            except Exception:
                for shard in registered:
                    self._inflight.complete(shard)
                raise
            new_ids.append(expert_id)
        return new_ids, reused_ids

    def _bind_h2d_event(
        self, expert_ids: Iterable[int], event: Optional[torch.cuda.Event]
    ) -> None:
        if event is None:
            return
        for expert_id in expert_ids:
            dependencies = self._ready_dependencies(int(expert_id))
            for dependency in dependencies:
                if dependency:
                    self._dependency_events[dependency] = event

    def track_h2d(
        self, expert_ids: Iterable[int], event: Optional[torch.cuda.Event]
    ) -> None:
        """Register a copy completion dependency for newly admitted experts."""
        ids = tuple(dict.fromkeys(int(item) for item in expert_ids))
        with self._inflight_lock:
            for expert_id in ids:
                self._cache.record_h2d(self._id(expert_id))
            self._bind_h2d_event(ids, event)
        lease = GecTransferLease(self, ids, event)
        self._transfer_leases.append(lease)
        if event is None:
            lease.poll()

    def poll_h2d_completions(self) -> None:
        if self._transfer_leases:
            self._transfer_leases = [
                lease for lease in self._transfer_leases if not lease.poll()
            ]

    def complete_h2d(self, expert_ids: Iterable[int]) -> None:
        with self._inflight_lock:
            self._complete_h2d_locked(expert_ids)

    def _complete_h2d_locked(self, expert_ids: Iterable[int]) -> None:
        """Publish TP-atomic READY after the device copy event is complete."""
        now_us = time.perf_counter_ns() // 1000
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            dependencies = self._ready_dependencies(expert_id)
            shards = self._physical_shards(expert_id)
            for shard in shards:
                self._inflight.complete(shard)
            for dependency in dependencies:
                self._dependency_events.pop(dependency, None)
            logical_id = self._id(expert_id)
            snapshot = self._cache.snapshot(logical_id)
            if snapshot is None or snapshot.state != self._gec.LifecycleState.LOADING:
                continue
            for tp_rank in range(self._tp_size):
                try:
                    ready_dependency = (
                        dependencies[tp_rank] if tp_rank < len(dependencies) else 0
                    )
                    self._cache.mark_shard_ready(
                        logical_id, tp_rank, now_us, ready_dependency
                    )
                except RuntimeError:
                    # A concurrent model-scope planner may have evicted the
                    # logical expert; the next round will re-admit it.
                    break
            self._cache.record_use(logical_id, now_us)

    def dependencies_for(
        self, expert_ids: Iterable[int]
    ) -> List[torch.cuda.Event]:
        """Return pending device-local H2D events for resident experts."""
        with self._inflight_lock:
            events: List[torch.cuda.Event] = []
            seen_events = set()
            for expert_id in expert_ids:
                for dependency in self._ready_dependencies(int(expert_id)):
                    event = self._dependency_events.get(dependency)
                    if event is None or id(event) in seen_events:
                        continue
                    seen_events.add(id(event))
                    events.append(event)
            return events

    def abort_h2d(self, expert_ids: Iterable[int]) -> None:
        with self._inflight_lock:
            self._abort_h2d_locked(expert_ids)

    def _abort_h2d_locked(self, expert_ids: Iterable[int]) -> None:
        """Rollback admissions whose copy operation raised before enqueue."""
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            dependencies = self._ready_dependencies(expert_id)
            shards = self._physical_shards(expert_id)
            for shard in shards:
                self._inflight.complete(shard)
            for dependency in dependencies:
                self._dependency_events.pop(dependency, None)
            logical_id = self._id(expert_id)
            snapshot = self._cache.snapshot(logical_id)
            if snapshot is None or snapshot.state not in (
                self._gec.LifecycleState.SCHEDULED,
                self._gec.LifecycleState.LOADING,
            ):
                continue
            try:
                self._cache.cancel_loading(logical_id)
            except (AttributeError, RuntimeError):
                # Older kt_kernel builds do not expose rollback; leave the
                # state untouched rather than masking the original copy error.
                logger.exception("[kt-gec] failed to rollback %s", logical_id)

    @property
    def rounds(self) -> int:
        return self._rounds

    def record_copy(
        self,
        num_experts: int,
        duration_us: int,
        queue_us: int = 0,
        event_wait_us: int = 0,
    ) -> None:
        """Record one physical H2D copy batch (doc section 23 timeline)."""
        for reason, duration in (
            (self._gec.WaitReason.H2D_QUEUED, queue_us),
            (self._gec.WaitReason.H2D_RUNNING, duration_us),
            (self._gec.WaitReason.H2D_EVENT_WAIT, event_wait_us),
        ):
            record = self._gec.WaitRecord()
            record.layer_id = self.layer_idx
            record.expert_id = -1  # batch record: covers num_experts experts
            record.tp_rank = -1
            record.reason = reason
            record.wait_duration_us = max(0, int(duration))
            self._telemetry.record_wait(record)
        self._last_copy_experts = num_experts  # informational, for logs
        logger.debug(
            "[kt-gec] layer=%d H2D_RUNNING experts=%d duration_us=%d",
            self.layer_idx,
            num_experts,
            max(0, duration_us),
        )

    def record_wait(
        self,
        reason,
        duration_us: int,
        expert_id: int = -1,
        tp_rank: int = -1,
    ) -> None:
        record = self._gec.WaitRecord()
        record.layer_id = self.layer_idx
        record.expert_id = int(expert_id)
        record.tp_rank = int(tp_rank)
        record.reason = reason
        record.wait_duration_us = max(0, int(duration_us))
        self._telemetry.record_wait(record)

    def record_router_dependency(self, duration_us: int) -> None:
        self.record_wait(self._gec.WaitReason.ROUTER_DEPENDENCY, duration_us)

    def record_cache_wait(self, expert_id: int, duration_us: int) -> None:
        self.record_wait(self._gec.WaitReason.CACHE_WAIT, duration_us, expert_id)

    def record_cpu_fallback(self, duration_us: int) -> None:
        self.record_wait(self._gec.WaitReason.CPU_FALLBACK, duration_us)

    def summary(self) -> str:
        """Cache stats + wait decomposition, for periodic DEBUG logging."""
        stats = self._cache.stats()
        return (
            f"GEC layer {self.layer_idx}: rounds={self._rounds} "
            f"hits={stats.hits} misses={stats.misses} "
            f"hit_rate={stats.hit_rate():.3f} admissions={stats.admissions} "
            f"evictions={stats.evictions} reh2d={stats.reh2d}\n"
            + self._telemetry.summary()
            + (
                "\n" + self._coordinator.summary()
                if self._coordinator is not None
                else ""
            )
        )


def build_gec_planner(kt_config, num_experts: int, num_gpu_experts: int, tp_rank: int):
    """Create a layer view over the model-scoped planner.

    Raises:
        RuntimeError: GEC parameters are set but kt_kernel has no gec module
            (kt_kernel too old); fail fast at startup instead of silently
            falling back to the legacy frequency policy.
    """
    coordinator = getattr(kt_config, "kt_gec_coordinator", None)
    if not gec_enabled(kt_config) or (tp_rank != 0 and coordinator is None):
        return None
    if num_gpu_experts < 1 and coordinator is None:
        logger.debug(
            "[kt-gec] layer %d has no physical GPU rows; keeping GEC global "
            "capacity for layers with resident rows",
            getattr(kt_config, "layer_idx", -1),
        )
        return None
    gec_module = _load_gec_module()
    if gec_module is None:
        raise RuntimeError(
            "--kt-expert-gpu-slots requires kt_kernel with the gec module; "
            "rebuild kt-kernel from this tree (the gec bindings are new)."
        )
    capacity = num_gpu_experts
    # The default is the current layer's physical capacity, as defined by
    # kt-layer-h2d-slots. Using the model-global capacity here turns a 256-row
    # layer in a 2560-slot model into a 2560-expert H2D round and inflates
    # prefill/TTFT. A zero-row layer may still use the global value so the
    # coordinator can borrow rows from a colder layer.
    default_budget = (
        capacity
        if capacity > 0 or coordinator is None
        else coordinator.capacity
    )
    budget = (
        kt_config.kt_layer_h2d_slots
        if kt_config.kt_layer_h2d_slots is not None
        else default_budget
    )
    planner = GecResidencyPlanner(
        layer_idx=kt_config.layer_idx,
        num_experts=num_experts,
        capacity=capacity,
        budget=budget,
        gec_module=gec_module,
        coordinator=coordinator,
    )
    if planner.capacity > 0 and planner.budget >= planner.capacity:
        logger.warning(
            "[kt-gec] layer %d: --kt-layer-h2d-slots (%d) >= per-layer "
            "capacity (%d); admission throttling is disabled and every update "
            "round may replace the entire resident set (H2D churn on routing "
            "shifts). Use a smaller budget (e.g. %d) to bound per-round "
            "admissions.",
            kt_config.layer_idx,
            planner.budget,
            planner.capacity,
            max(1, planner.capacity // 4),
        )
    logger.info(
        "[kt-gec] layer %d: capacity=%d budget=%d batch=%s (global slots=%d, shared=%s)",
        kt_config.layer_idx,
        planner.capacity,
        planner.budget,
        kt_config.kt_layer_h2d_batch_size,
        kt_config.kt_expert_gpu_slots,
        coordinator is not None,
    )
    return planner
