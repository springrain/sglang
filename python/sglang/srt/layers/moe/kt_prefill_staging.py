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
  8.5).  ``S > N`` clamps to the whole layer.  The window is also the width of
  one host-ring generation, so a layer drains its whole staged set in a single
  pass and the pinned ring is ``2 * S`` slots (section 7.5).
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

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
_PUBLISHED_FORWARD_ID: Optional[int] = None


def publish_forward_mode(forward_mode, forward_id: Optional[int] = None) -> None:
    """Publish the current forward mode and its chunk/request identity.

    ``forward_id`` is the model runner's monotonically increasing pass id.  A
    new id is the only safe point at which the KT staging manager may change
    resident ownership: every MoE layer of the previous chunk has completed
    and the current chunk has not issued a prefetch yet.  Keeping it beside the
    mode avoids a second process-global publication hook.
    """

    global _PUBLISHED_FORWARD_MODE, _PUBLISHED_FORWARD_ID
    _PUBLISHED_FORWARD_MODE = forward_mode
    if forward_id is not None:
        _PUBLISHED_FORWARD_ID = int(forward_id)


def published_forward_mode():
    """Return the most recently published forward mode, if any."""

    return _PUBLISHED_FORWARD_MODE


def published_forward_id() -> Optional[int]:
    """Return the current forward/chunk identity, if ModelRunner published it."""

    return _PUBLISHED_FORWARD_ID


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


def host_ring_depth(window: int) -> int:
    """Pinned host slots for one layerwise context (doc section 7.5).

    The ring holds exactly two generations of the staging window ``S``
    (``--kt-layer-h2d-slots``), which is the FreeToken prefill shape: one
    buffer per layer width, ping-ponged by layer parity.  A layer therefore
    drains its whole staged set in a single host-write pass, and layer L+1
    writes the generation layer L's H2D is not reading.  A window below one
    degenerates to the original two-slot double buffer.
    """

    return 2 * max(1, int(window or 1))


def host_generation_range(generation: int, ring_depth: int, count: int) -> List[int]:
    """Physical host slots of one generation of the ring (doc sections 7.5, 20.8).

    One generation is half the ring, i.e. the staging window ``S``, so a whole
    staged set always fits in one pass.  ``count`` above that would index past
    the end of the pinned buffer, so it raises instead of silently reading or
    writing out of bounds.  The caller keeps the generation counter running
    across layers: layer L+1 then writes the generation layer L is not reading,
    which preserves the cross-layer ping-pong now that a layer only ever has a
    single batch.
    """

    depth = max(2, int(ring_depth or 2))
    step = depth // 2
    count = int(count)
    if count > step:
        raise ValueError(
            f"host generation holds {step} slots but {count} experts were "
            "requested; the staging window must bound the staged set"
        )
    start = (int(generation) % 2) * step
    return [start + offset for offset in range(count)]


def uniform_resident_counts(
    num_moe_layers: int, total_slots: int, num_experts: int
) -> List[int]:
    """Spread a global resident budget evenly across all MoE layers.

    The first ``remainder`` layers receive one extra row.  Every count is
    clamped independently to ``num_experts`` and the result is deterministic,
    which keeps the startup layout identical on every TP rank (doc section 5,
    invariant 10).
    """

    layers = max(0, int(num_moe_layers))
    experts = max(0, int(num_experts))
    if layers == 0 or experts == 0:
        return [0] * layers
    slots = max(0, int(total_slots))
    slots = min(slots, layers * experts)
    base, remainder = divmod(slots, layers)
    return [
        min(experts, base + (1 if index < remainder else 0))
        for index in range(layers)
    ]


def fused_shared_expert_ids(
    num_experts: int, num_fused_shared_experts: int
) -> List[int]:
    """Return the trailing logical ids occupied by fused shared experts.

    SGLang appends fused shared experts after the routed ids, so a layer with
    ``num_experts`` total experts and ``S`` fused shared experts owns ids
    ``[num_experts - S, num_experts)``.
    """

    total = max(0, int(num_experts))
    shared = max(0, min(int(num_fused_shared_experts), total))
    return list(range(total - shared, total))


def shared_first_resident_ids(
    num_experts: int, count: int, shared_ids: Iterable[int]
) -> List[int]:
    """Choose ``count`` resident ids with shared experts reserved first.

    Shared ids are deduplicated and sorted before routed ids.  If the resident
    budget is smaller than the shared set, only the lowest shared ids fit; if
    it is larger, the remainder is filled from ascending routed ids.  The
    deterministic order is also the compact-row assignment order.  When every
    expert fits, natural id order is used so the identity layout stays
    contiguous; shared priority is moot because nothing is excluded.
    """

    total = max(0, int(num_experts))
    wanted = max(0, min(int(count), total))
    if wanted == total:
        return list(range(total))
    shared = sorted(
        {
            int(expert_id)
            for expert_id in shared_ids
            if 0 <= int(expert_id) < total
        }
    )
    selected = shared[:wanted]
    if len(selected) < wanted:
        shared_set = set(shared)
        selected.extend(
            expert_id
            for expert_id in range(total)
            if expert_id not in shared_set
        )
        selected = selected[:wanted]
    return selected


def contiguous_copy_runs(
    destination_rows: Sequence[int], host_slots: Sequence[int]
) -> List[Tuple[int, int]]:
    """Collapse per-expert H2D copies into stride-level runs.

    Consecutive pairs that keep BOTH the destination physical row and the host
    slot stepping by one address the same bytes as one strided copy.  Compact
    banks separate logical ids from physical rows, so callers must pass the row
    mapping's destinations here rather than logical expert ids.
    """

    runs: List[Tuple[int, int]] = []
    count = min(len(destination_rows), len(host_slots))
    index = 0
    while index < count:
        length = 1
        while index + length < count:
            prev_dst = int(destination_rows[index + length - 1])
            next_dst = int(destination_rows[index + length])
            prev_host = int(host_slots[index + length - 1])
            next_host = int(host_slots[index + length])
            if next_dst != prev_dst + 1 or next_host != prev_host + 1:
                break
            length += 1
        runs.append((index, length))
        index += length
    return runs


def contiguous_stage_runs(
    expert_ids: Sequence[int], host_slots: Sequence[int]
) -> List[Tuple[int, int]]:
    """Compatibility wrapper for the legacy identity-row H2D loop."""

    return contiguous_copy_runs(expert_ids, host_slots)


def apply_row_owner_promotions(
    row_owner: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
    *,
    num_experts: Optional[int] = None,
) -> List[int]:
    """Apply boundary promotion pairs without changing row ownership geometry.

    ``row_owner`` is ordered by physical row.  A pending promotion replaces the
    victim id at its existing row, so a later load can use the ordinary host
    transfer path without renumbering the resident rows.  The helper validates
    the complete batch first; a partially applied swap would leave the GEC mask
    and the compact bank mapping inconsistent.
    """

    owner = [int(expert_id) for expert_id in row_owner]
    promoted_seen = set()
    for victim_id, promoted_id in pairs:
        victim_id = int(victim_id)
        promoted_id = int(promoted_id)
        if num_experts is not None:
            total = int(num_experts)
            if not 0 <= victim_id < total or not 0 <= promoted_id < total:
                raise ValueError("promotion expert id is out of range")
        if promoted_id in promoted_seen:
            raise ValueError(f"expert {promoted_id} is promoted more than once")
        try:
            row = owner.index(victim_id)
        except ValueError as exc:
            raise ValueError(f"promotion victim {victim_id} is not resident") from exc
        if promoted_id in owner:
            raise ValueError(f"promotion expert {promoted_id} is already resident")
        owner[row] = promoted_id
        promoted_seen.add(promoted_id)
    return owner


def fit_staged_window(
    staged_ids: Sequence[int],
    resident_host_count: int,
    window: int,
) -> List[int]:
    """Fit host-arriving promoted rows and optional staging into one generation.

    A promoted expert whose bytes are not present in the original canonical
    image rides the same pinned host generation as the staged window.  The
    generation has exactly ``S`` slots, so reserve one slot per such promotion
    and take staged candidates only from the remainder.
    """

    staged = [int(expert_id) for expert_id in staged_ids]
    host_rows = max(0, int(resident_host_count))
    capacity = max(0, int(window))
    if host_rows > capacity:
        raise ValueError(
            f"{host_rows} promoted rows exceed the {capacity}-slot staging window"
        )
    return staged[: capacity - host_rows]


class ForwardBoundaryTracker:
    """Detect the first layer visit of a new chunk/request.

    ``ModelRunner`` increments ``forward_pass_id`` once per forward.  The
    resident layout is frozen within one id and may change only when a
    different id is observed.  The first observation is not a boundary: there
    is no completed chunk whose staging hits could justify a promotion.
    """

    def __init__(self) -> None:
        self._forward_id: Optional[int] = None

    def observe(self, forward_id: Optional[int]) -> bool:
        if forward_id is None:
            return False
        current = int(forward_id)
        previous = self._forward_id
        self._forward_id = current
        return previous is not None and previous != current

    @property
    def forward_id(self) -> Optional[int]:
        return self._forward_id


def select_boundary_promotions(
    resident_ids: Sequence[int],
    staged_ids: Sequence[int],
    hit_counts: Mapping[int, int],
    *,
    protected_ids: Iterable[int] = (),
    scores: Optional[Mapping[int, float]] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Pair staged winners with the coldest unprotected resident rows.

    Candidates are staged experts with at least one hit in the completed
    chunk.  They are ordered by ``(-hits, -score, id)``; victims are ordered by
    ``(score, id)``.  Shared/protected experts can be promoted but are never
    evicted.  The fully host-side ordering makes every TP rank derive the same
    swaps from the same history.
    """

    resident = [int(expert_id) for expert_id in resident_ids]
    protected = {int(expert_id) for expert_id in protected_ids}
    score_map = scores or {}

    candidates = sorted(
        {
            int(expert_id)
            for expert_id in staged_ids
            if int(hit_counts.get(int(expert_id), 0)) > 0
        },
        key=lambda expert_id: (
            -int(hit_counts.get(expert_id, 0)),
            -float(score_map.get(expert_id, 0.0)),
            expert_id,
        ),
    )
    victims = sorted(
        (expert_id for expert_id in resident if expert_id not in protected),
        key=lambda expert_id: (
            float(score_map.get(expert_id, 0.0)),
            expert_id,
        ),
    )
    pairs = min(len(candidates), len(victims))
    if limit is not None:
        pairs = min(pairs, max(0, int(limit)))
    return [
        (victims[index], candidates[index])
        for index in range(pairs)
    ]


class StagingRowPool:
    """Unified resident/staging row ownership for compact bank images.

    A bank owns ``resident_capacity + staging_capacity`` physical rows.  Every
    row has one owner and one current role; a layer's mapping is simply the
    owner-to-row table for that layer.  Promotion swaps the roles of the
    victim resident row and the winning staged row, so no bytes move.  Row
    assignments are preserved across ``configure`` calls whenever an expert
    keeps the same role, keeping repeated loads deterministic.
    """

    def __init__(
        self,
        num_experts: int,
        *,
        resident_capacity: int,
        staging_capacity: int,
        protected_ids: Iterable[int] = (),
    ) -> None:
        self.num_experts = max(0, int(num_experts))
        self.resident_capacity = max(0, int(resident_capacity))
        self.staging_capacity = max(0, int(staging_capacity))
        self.physical_rows = self.resident_capacity + self.staging_capacity
        self.protected_ids = frozenset(
            int(expert_id) for expert_id in protected_ids
        )
        self._owners: Dict[int, List[int]] = {}
        self._roles: Dict[int, List[str]] = {}
        self._mappings: Dict[int, List[int]] = {}

    def _state(self, layer_idx: int) -> Tuple[List[int], List[str], List[int]]:
        owner = self._owners.get(layer_idx)
        if owner is None:
            owner = [-1] * self.physical_rows
            role = ["free"] * self.physical_rows
            mapping = [-1] * self.num_experts
            self._owners[layer_idx] = owner
            self._roles[layer_idx] = role
            self._mappings[layer_idx] = mapping
        else:
            role = self._roles[layer_idx]
            mapping = self._mappings[layer_idx]
        return owner, role, mapping

    def configure(
        self,
        layer_idx: int,
        resident_ids: Iterable[int],
        staged_ids: Iterable[int],
    ) -> List[int]:
        """Assign this layer's current roles and return ``logical -> row``."""

        residents = list(
            dict.fromkeys(int(expert_id) for expert_id in resident_ids)
        )
        staged = list(dict.fromkeys(int(expert_id) for expert_id in staged_ids))
        if len(residents) > self.resident_capacity:
            raise ValueError(
                f"layer {layer_idx} has {len(residents)} residents but the "
                f"bank reserves {self.resident_capacity} rows"
            )
        if len(staged) > self.staging_capacity:
            raise ValueError(
                f"layer {layer_idx} has {len(staged)} staged experts but the "
                f"bank reserves {self.staging_capacity} rows"
            )
        for expert_id in residents + staged:
            if not 0 <= expert_id < self.num_experts:
                raise ValueError(
                    f"layer {layer_idx} expert id {expert_id} is out of range"
                )
        if set(residents) & set(staged):
            raise ValueError(
                f"layer {layer_idx} cannot own an expert as both resident and staged"
            )

        old_owner, old_role, _ = self._state(layer_idx)
        new_owner = [-1] * self.physical_rows
        new_role = ["free"] * self.physical_rows
        resident_set = set(residents)
        staged_set = set(staged)

        for row, expert_id in enumerate(old_owner):
            if expert_id < 0:
                continue
            # A row keeps its identity when the expert merely changes roles.
            # This is what makes a promotion metadata-only: a staged row that
            # becomes resident is not reassigned to a different physical row.
            if expert_id in resident_set:
                new_owner[row] = expert_id
                new_role[row] = "resident"
            elif expert_id in staged_set:
                new_owner[row] = expert_id
                new_role[row] = "staging"

        assigned = {expert_id for expert_id in new_owner if expert_id >= 0}

        def assign(expert_ids: Iterable[int], role: str) -> None:
            for expert_id in expert_ids:
                if expert_id in assigned:
                    continue
                try:
                    row = new_owner.index(-1)
                except ValueError as exc:
                    raise RuntimeError(
                        f"layer {layer_idx} exhausted its compact row pool"
                    ) from exc
                new_owner[row] = expert_id
                new_role[row] = role
                assigned.add(expert_id)

        assign((expert_id for expert_id in residents if expert_id not in assigned), "resident")
        assign((expert_id for expert_id in staged if expert_id not in assigned), "staging")

        mapping = [-1] * self.num_experts
        for row, expert_id in enumerate(new_owner):
            if expert_id >= 0:
                mapping[expert_id] = row
        self._owners[layer_idx] = new_owner
        self._roles[layer_idx] = new_role
        self._mappings[layer_idx] = mapping
        return list(mapping)

    def clear(self) -> None:
        """Forget every layer assignment for a bank that is about to be reused."""

        self._owners.clear()
        self._roles.clear()
        self._mappings.clear()

    def mapping(self, layer_idx: int) -> List[int]:
        return list(self._state(layer_idx)[2])

    def row_for(self, layer_idx: int, expert_id: int) -> int:
        mapping = self._state(layer_idx)[2]
        expert_id = int(expert_id)
        if not 0 <= expert_id < len(mapping):
            raise ValueError(f"expert id {expert_id} is out of range")
        row = mapping[expert_id]
        if row < 0:
            raise KeyError(
                f"layer {layer_idx} expert {expert_id} is not loaded in the bank"
            )
        return int(row)

    def resident_ids(self, layer_idx: int) -> List[int]:
        owner, role, _ = self._state(layer_idx)
        return [
            expert_id
            for expert_id, owner_role in zip(owner, role)
            if expert_id >= 0 and owner_role == "resident"
        ]

    def staged_ids(self, layer_idx: int) -> List[int]:
        owner, role, _ = self._state(layer_idx)
        return [
            expert_id
            for expert_id, owner_role in zip(owner, role)
            if expert_id >= 0 and owner_role == "staging"
        ]

    def row_roles(self, layer_idx: int) -> List[str]:
        return list(self._state(layer_idx)[1])

    def promote(self, layer_idx: int, victim_id: int, promoted_id: int) -> None:
        """Swap a staged winner into the resident role without moving bytes."""

        victim_id = int(victim_id)
        promoted_id = int(promoted_id)
        if victim_id in self.protected_ids:
            raise ValueError("shared/protected experts cannot be evicted")
        owner, role, mapping = self._state(layer_idx)
        if not 0 <= victim_id < len(mapping) or not 0 <= promoted_id < len(mapping):
            raise ValueError("promotion expert id is out of range")
        victim_row = mapping[victim_id]
        promoted_row = mapping[promoted_id]
        if victim_row < 0 or role[victim_row] != "resident":
            raise ValueError(f"expert {victim_id} is not a resident of layer {layer_idx}")
        if promoted_row < 0 or role[promoted_row] != "staging":
            raise ValueError(f"expert {promoted_id} is not staged in layer {layer_idx}")

        owner[victim_row] = -1
        role[victim_row] = "free"
        role[promoted_row] = "resident"
        mapping[victim_id] = -1


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
        "copy_submits",
        "copy_pairs",
    )
    _MEAN_FIELDS = (
        "copy_bytes",
        "drain_count",
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
    """Render one :meth:`StagingTelemetry.take_summary` payload (section 15.3).

    ``copy_bytes_mean`` and ``drains`` are per-layer means (``_MEAN_FIELDS``),
    unlike the summed counts beside them; ``copy_bytes_mean`` is a shape-derived
    logical payload estimate, not a measured transfer size.  ``drains`` is 1
    for a (re)loaded layer and 0 when the slot was reused warm, so its window
    mean is the reload ratio rather than a per-batch count.  ``copy_submits``
    and ``copy_pairs`` are window sums: the former counts the actual ``copy_``
    calls the collapsed H2D loop issued, the latter the (expert, tensor) pairs
    it moved, so ``copy_pairs / copy_submits`` reads as the merge factor of
    :func:`contiguous_stage_runs`.
    """

    return (
        "[kt-prefetch] round layers=%d candidates=%d selected=%d resident=%d "
        "staged_hits=%d staged_misses=%d cpu=%d gpu=%d cpu_routes=%d "
        "copy_bytes_mean=%d drains=%.2f "
        "copy_submits=%d copy_pairs=%d "
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
            summary.get("copy_bytes", 0.0),
            summary.get("drain_count", 0.0),
            summary.get("copy_submits", 0.0),
            summary.get("copy_pairs", 0.0),
            summary.get("copy_ms", 0.0),
            summary.get("postprocess_ms", 0.0),
            summary.get("bank_reuse_wait_ms", 0.0),
            summary.get("ready_wait_ms", 0.0),
            summary.get("compute_ms", 0.0),
        )
    )
