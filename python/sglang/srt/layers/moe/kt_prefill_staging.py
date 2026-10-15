# SPDX-License-Identifier: Apache-2.0
"""Cross-layer staging prefetch planning for the KT prefill streaming path.

Design baseline: ``doc/KT-GEC-PREFETCH-STREAMING-PLAN.md``.

The MXFP4 layerwise prefill manager runs layer L on the GPU while the second
staging bank prefetches layer L+1 (doc sections 7 and 9).  This module owns the
decision layer of that pipeline:

- :class:`PrefillHistoryTable` keeps one hit history per MoE layer and scores
  experts with ``decayed + 0.25 * recent + 0.5 * last_router_demand``
  (section 8.3).  Layer L's router only ever updates ``history[L]``.
- :class:`StagingPrefetchPlanner` turns ``--kt-layer-h2d-slots`` into the
  per-bank staging window, filters out the target layer's resident experts and
  selects the deterministic top-S prefetch candidates (sections 3.3, 8.1 and
  8.5).  ``S > N`` clamps to the whole layer.
- Cold start rotates one cursor over the expert ids in chunks instead of
  disabling staging or picking randomly (section 8.4).
- :func:`staging_requested_for_batch` encodes the section 13.1 entry
  conditions: positive window, threshold configured and reached, prefill-like
  forward and a runtime that can host the MXFP4 layerwise slots.
- :class:`StagingTelemetry` accumulates the per-layer counters of section 15.1
  so the manager can log one round summary every 16 layers.

Forward-mode publication (section 14.6) also lives here: ``ModelRunner``
publishes ``forward_batch.forward_mode`` once per forward through
:func:`publish_forward_mode`, and the KT wrapper classifies it with
:func:`is_prefill_like_mode`.  ``runtime_context.ForwardFlags`` deliberately
has no ``forward_mode`` slot, so ``is_extend_in_batch`` stays only as the
compatibility fallback when nothing was published.

The module imports nothing from SGLang, so the whole decision layer stays
unit-testable on a CPU-only host.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Section 8.3 score weights.  The decayed hit count is the dominant term; the
# recent window tracks distribution changes quickly; the last router demand is
# a short-lived boost that must never outrank long-term history.
SCORE_RECENT_WEIGHT = 0.25
SCORE_LAST_DEMAND_WEIGHT = 0.5
# ``recent`` follows the distribution with a fast exponential memory.
RECENT_RETAIN = 0.5
# Periodic sweep over every layer so a stale hot expert cannot pin a slot
# forever (section 8.3).
DECAY_FACTOR = 0.9
DECAY_INTERVAL_LAYERS = 64


# ---------------------------------------------------------------------------
# Forward-mode publication (section 14.6)
# ---------------------------------------------------------------------------

_PUBLISHED_FORWARD_MODE = None


def publish_forward_mode(forward_mode) -> None:
    """Publish the current ``forward_batch.forward_mode`` for KT readers."""

    global _PUBLISHED_FORWARD_MODE
    _PUBLISHED_FORWARD_MODE = forward_mode


def published_forward_mode():
    """Return the most recently published forward mode, if any."""

    return _PUBLISHED_FORWARD_MODE


def is_prefill_like_mode(forward_mode) -> bool:
    """Whether ``forward_mode`` must run the prefill staging pipeline.

    TARGET_VERIFY forwards route a full draft window through the MoE layers and
    are treated exactly like extend batches; decode and idle batches must never
    trigger weight migration (section 13.1).
    """

    if forward_mode is None:
        return False
    for name in ("is_extend", "is_target_verify"):
        probe = getattr(forward_mode, name, None)
        if not callable(probe):
            continue
        try:
            if probe():
                return True
        except Exception:  # pragma: no cover - defensive against mode stubs
            continue
    return False


# ---------------------------------------------------------------------------
# Staging window
# ---------------------------------------------------------------------------


def clamp_staging_window(window, num_experts: int) -> Optional[int]:
    """Translate ``--kt-layer-h2d-slots`` into the logical staging window.

    ``None`` or a non-positive value leaves the staging feature unselected
    (section 3.3: ``S > 0`` is one half of the joint gate).  ``S > N`` clamps
    to the whole layer, matching section 3.3.1: a window wider than the layer
    yields no extra GPU compute coverage.
    """

    if window is None:
        return None
    window = int(window)
    if window <= 0:
        return None
    return min(window, int(num_experts))


def effective_experts(
    resident_ids: Iterable[int],
    staged_ids: Iterable[int],
    num_experts: int,
) -> Tuple[List[int], bool]:
    """Return ``resident(L) + staged(L)`` and whether it covers the layer.

    ``effective_gpu(L) = resident(L) + staged(L)`` and every other expert runs
    the CPU fallback (doc section 3.5).  ``partial`` is False exactly when the
    staging window covered the whole layer -- the whole-layer prefetch case
    whose acceptance criteria require a near-zero CPU expert fraction (section
    16.2).
    """

    effective = sorted(
        {int(expert_id) for expert_id in resident_ids}
        | {int(expert_id) for expert_id in staged_ids}
    )
    return effective, len(effective) < int(num_experts)


def host_write_batches(
    staged_ids: Sequence[int], batch_size: int
) -> List[List[int]]:
    """Split one layer's staged experts into host-write batches.

    This is the lightweight batching of doc section 20 item 8.
    The layerwise manager submits every expert of a batch, drains the KT writer
    once and only then enqueues that batch's H2D copies.  A batch therefore has
    to fit the pinned host ring exactly: batch ``b`` overwrites the same host
    slots batch ``b - 1`` used, which is what keeps the slot count at
    ``--kt-layer-h2d-batch-size`` instead of growing with the staging window
    (section 7.5).  A window that is not a multiple of the batch size simply
    ends on a shorter batch; no batch is ever empty, so the drain/TP fence count
    stays deterministic for every rank.
    """

    step = max(1, int(batch_size or 1))
    ids = list(staged_ids)
    return [ids[start : start + step] for start in range(0, len(ids), step)]


def host_ring_depth(batch_size: int) -> int:
    """Pinned host slots for one layerwise context (doc section 7.5).

    The ring always holds two rolling generations of ``batch_size`` slots, so
    the host write of batch ``b + 1`` targets the generation batch ``b``'s H2D
    is not reading and the two overlap.  A batch size of one degenerates to the
    original single-expert double buffer.
    """

    return 2 * max(1, int(batch_size or 1))


def host_generation_slots(generation: int, batch_size: int, count: int) -> List[int]:
    """Physical host slots of one rolling generation (doc sections 7.5, 20.8).

    Consecutive batches alternate generations, so ``count`` never exceeds
    ``batch_size`` and every returned slot stays inside the ``2 * batch_size``
    ring.  The caller keeps the generation counter running across layers: the
    first batch of layer L+1 then writes the generation the last batch of layer
    L is not reading, instead of stalling on its H2D.
    """

    step = max(1, int(batch_size or 1))
    start = (int(generation) % 2) * step
    return [start + offset for offset in range(int(count))]


def staging_requested_for_batch(
    window: Optional[int],
    threshold: Optional[int],
    num_tokens: int,
    *,
    prefill_like: bool,
    runtime_supported: bool = True,
) -> bool:
    """Section 13.1 entry conditions for the prefill streaming mode.

    ``--kt-layer-h2d-slots`` selects the feature and sizes the staging window,
    ``--kt-gpu-prefill-token-threshold`` decides when this batch may move
    weight, the forward must be prefill-like (never decode or idle, section
    13.2), and the runtime must be able to build the MXFP4 layerwise slots.
    All four are required; ``window`` is expected to be the pre-clamped value
    from :func:`clamp_staging_window`.
    """

    return bool(
        window is not None
        and threshold is not None
        and threshold > 0
        and num_tokens >= threshold
        and prefill_like
        and runtime_supported
    )


# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------


class LayerHistory:
    """``history[layer_id][expert_id]`` counters for one MoE layer."""

    __slots__ = ("decayed", "recent", "last_demand", "updates")

    def __init__(self, num_experts: int) -> None:
        self.decayed = torch.zeros(int(num_experts), dtype=torch.float32)
        self.recent = torch.zeros(int(num_experts), dtype=torch.float32)
        self.last_demand = torch.zeros(int(num_experts), dtype=torch.float32)
        self.updates = 0

    def record(self, counts: torch.Tensor) -> None:
        """Fold one router demand histogram into this layer's history."""

        self.decayed.add_(counts)
        self.recent.mul_(RECENT_RETAIN).add_(counts)
        self.last_demand.copy_(counts)
        self.updates += 1

    def decay(self, factor: float = DECAY_FACTOR) -> None:
        self.decayed.mul_(factor)
        self.recent.mul_(factor)

    def scores(self) -> torch.Tensor:
        """Section 8.3 score; the decayed term stays the primary signal."""

        return (
            self.decayed
            + SCORE_RECENT_WEIGHT * self.recent
            + SCORE_LAST_DEMAND_WEIGHT * self.last_demand
        )

    @property
    def is_cold(self) -> bool:
        return self.updates == 0


class PrefillHistoryTable:
    """Per-layer router histories shared by every prefetch decision."""

    def __init__(
        self,
        num_experts: int,
        *,
        decay_factor: float = DECAY_FACTOR,
        decay_interval: int = DECAY_INTERVAL_LAYERS,
    ) -> None:
        self.num_experts = int(num_experts)
        self.decay_factor = float(decay_factor)
        self.decay_interval = max(1, int(decay_interval))
        self._layers: Dict[int, LayerHistory] = {}
        self._visits = 0

    def entry(self, layer_idx: int) -> LayerHistory:
        entry = self._layers.get(layer_idx)
        if entry is None:
            entry = LayerHistory(self.num_experts)
            self._layers[layer_idx] = entry
        return entry

    def record_router_counts(self, layer_idx: int, counts) -> torch.Tensor:
        """Record one layer's demand; ``counts`` may be a list or CPU tensor."""

        entry = self.entry(layer_idx)
        if not torch.is_tensor(counts):
            counts = torch.tensor(counts, dtype=torch.float32)
        counts = counts.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if counts.numel() != self.num_experts:
            raise ValueError(
                "staging history expects "
                f"{self.num_experts} expert counts, got {counts.numel()}"
            )
        entry.record(counts)
        return counts

    def note_layer_visit(self, layer_idx: int) -> bool:
        """Count one layer execution; returns True when a decay sweep ran."""

        self.entry(layer_idx)
        self._visits += 1
        if self._visits % self.decay_interval == 0:
            for entry in self._layers.values():
                entry.decay(self.decay_factor)
            return True
        return False

    def scores(self, layer_idx: int) -> torch.Tensor:
        return self.entry(layer_idx).scores()

    def is_cold(self, layer_idx: int) -> bool:
        return self.entry(layer_idx).is_cold


# ---------------------------------------------------------------------------
# Prefetch planner
# ---------------------------------------------------------------------------


class StagingPrefetchPlanner:
    """Selects the top-S staging candidates for one target layer.

    ``candidates(L+1) = all_experts(L+1) - resident(L+1)`` (section 8.1).
    The bank that is about to be overwritten is not an exclusion set, and
    resident experts are always excluded so the same ``(layer_id, expert_id)``
    can never be owned by the resident pool and a staging bank at once
    (invariants 3 and 5).
    """

    def __init__(
        self,
        num_experts: int,
        window: int,
        *,
        history: Optional[PrefillHistoryTable] = None,
    ) -> None:
        self.num_experts = int(num_experts)
        window = clamp_staging_window(window, self.num_experts)
        if window is None:
            raise ValueError("staging planner requires a positive window")
        self.window = int(window)
        self.history = history or PrefillHistoryTable(self.num_experts)
        # Cold start (section 8.4): deterministic chunked rotation over the
        # candidate ids.  Never random, never a reason to disable staging.
        self._cold_cursor = 0

    def record_router_counts(self, layer_idx: int, counts) -> torch.Tensor:
        return self.history.record_router_counts(layer_idx, counts)

    def note_layer_visit(self, layer_idx: int) -> bool:
        return self.history.note_layer_visit(layer_idx)

    def select(self, layer_idx: int, resident_ids: Iterable[int]) -> List[int]:
        """Return the ascending staging selection for ``layer_idx``."""

        resident = {int(expert_id) for expert_id in resident_ids}
        candidates = [
            expert_id
            for expert_id in range(self.num_experts)
            if expert_id not in resident
        ]
        size = min(self.window, len(candidates))
        if size <= 0:
            return []

        entry = self.history.entry(layer_idx)
        if entry.is_cold:
            return self._cold_select(candidates, size)

        scores = entry.scores().tolist()
        ranked = sorted(
            candidates, key=lambda expert_id: (-scores[expert_id], expert_id)
        )
        return sorted(ranked[:size])

    def _cold_select(self, candidates: Sequence[int], size: int) -> List[int]:
        cursor = self._cold_cursor % len(candidates)
        selected = [
            candidates[(cursor + offset) % len(candidates)]
            for offset in range(size)
        ]
        self._cold_cursor = (cursor + size) % len(candidates)
        return sorted(selected)

    @property
    def cold_cursor(self) -> int:
        return self._cold_cursor


# ---------------------------------------------------------------------------
# Telemetry (section 15)
# ---------------------------------------------------------------------------


class StagingTelemetry:
    """Per-layer counters plus a rolling round summary.

    Only the fields section 15.1 makes available on the staging path are
    tracked: counts are summed per round and millisecond fields are averaged.
    ``take_summary`` resets the window and returns ``None`` until a full round
    of ``interval`` layers has been recorded.  ``cpu_route_count`` also carries
    the host-side counts the hybrid gate hands over (section 14.5), which
    never require a device readback.
    """

    _SUM_FIELDS = (
        "candidate_count",
        "selected_count",
        "resident_count",
        "staging_hit_count",
        "staging_miss_count",
        "cpu_expert_count",
        "gpu_expert_count",
        "cpu_route_count",
    )
    _MEAN_FIELDS = (
        "copy_bytes",
        "copy_ms",
        "postprocess_ms",
        "bank_reuse_wait_ms",
        "ready_wait_ms",
        "compute_ms",
    )

    def __init__(self, interval: int = 16) -> None:
        self.interval = max(1, int(interval))
        self._records: List[Dict[str, float]] = []
        # Route counts the hybrid gate already holds on the host (section
        # 14.5).  They join the next summary instead of paying a device
        # readback per layer.
        self._host_cpu_routes = 0.0

    def record_layer(self, **fields) -> None:
        self._records.append(
            {
                name: float(value)
                for name, value in fields.items()
                if value is not None
            }
        )

    def add_cpu_routes(self, count: int) -> None:
        """Accumulate a CPU route count obtained without a device sync."""

        self._host_cpu_routes += float(count)

    def pending(self) -> int:
        return len(self._records)

    def take_summary(self) -> Optional[Dict[str, float]]:
        if len(self._records) < self.interval:
            return None
        records, self._records = self._records, []
        host_cpu_routes, self._host_cpu_routes = self._host_cpu_routes, 0.0
        summary: Dict[str, float] = {"layers": float(len(records))}
        for name in self._SUM_FIELDS:
            summary[name] = float(
                sum(record.get(name, 0.0) for record in records)
            )
        # The window's ``cpu_route_count`` is the staging path's deferred
        # scalar plus the hybrid gate's host-side counts (section 15.1).
        summary["cpu_route_count"] += host_cpu_routes
        for name in self._MEAN_FIELDS:
            values = [record[name] for record in records if name in record]
            if values:
                summary[name] = sum(values) / len(values)
        return summary


def format_round_summary(summary: Dict[str, float]) -> str:
    """Render one :meth:`StagingTelemetry.take_summary` payload (section 15.3)."""

    return (
        "[kt-prefetch] round layers=%d candidates=%d selected=%d resident=%d "
        "staged_hits=%d staged_misses=%d cpu=%d gpu=%d cpu_routes=%d "
        "copy_ms=%.2f postprocess_ms=%.2f bank_reuse_wait_ms=%.2f "
        "ready_wait_ms=%.2f compute_ms=%.2f"
        % (
            summary.get("layers", 0),
            summary.get("candidate_count", 0),
            summary.get("selected_count", 0),
            summary.get("resident_count", 0),
            summary.get("staging_hit_count", 0),
            summary.get("staging_miss_count", 0),
            summary.get("cpu_expert_count", 0),
            summary.get("gpu_expert_count", 0),
            summary.get("cpu_route_count", 0),
            summary.get("copy_ms", 0.0),
            summary.get("postprocess_ms", 0.0),
            summary.get("bank_reuse_wait_ms", 0.0),
            summary.get("ready_wait_ms", 0.0),
            summary.get("compute_ms", 0.0),
        )
    )
