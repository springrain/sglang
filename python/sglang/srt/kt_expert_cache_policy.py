# SPDX-License-Identifier: Apache-2.0
"""Pure policy helpers for KT's per-layer expert cache.

This module deliberately has no Torch, CUDA, or backend dependencies.  The
runtime owns tensors and asynchronous work; these helpers only turn immutable
snapshots into deterministic policy decisions that can be tested offline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DECAYED_LFU_STRATEGY = "decayed-lfu"
DEFAULT_PREFILL_STREAM_TOP_N = 4
READY_SLOT_STATE = "READY"


@dataclass(frozen=True)
class ResidentSlot:
    """Policy-visible metadata for one persistent GPU expert slot."""

    slot_id: int
    expert_id: int
    reuse_signal: float
    last_access_epoch: int
    resident_since_epoch: int
    state: str = READY_SLOT_STATE


@dataclass(frozen=True)
class Replacement:
    """A deterministic candidate-to-victim persistent replacement."""

    candidate_expert_id: int
    victim_expert_id: int
    slot_id: int


def resolve_prefill_stream_top_n(
    strategy: str,
    capacity: int | None,
    configured: int | None,
) -> int | None:
    """Resolve ``--kt-prefill-stream-top-n`` without silently clamping it.

    Static placement strategies do not create a stream controller, so their
    effective value stays ``None``.  An explicitly supplied value with a
    static strategy is rejected instead of being silently ignored.
    """

    if strategy != DECAYED_LFU_STRATEGY:
        if configured is not None:
            raise ValueError(
                "--kt-prefill-stream-top-n is only valid with "
                "--kt-expert-placement-strategy decayed-lfu."
            )
        return None

    if capacity is None or capacity <= 0:
        raise ValueError(
            "--kt-expert-placement-strategy decayed-lfu requires a positive "
            "--kt-num-gpu-experts."
        )

    stream_top_n = min(DEFAULT_PREFILL_STREAM_TOP_N, capacity)
    if configured is not None:
        stream_top_n = configured
    if stream_top_n < 0 or stream_top_n > capacity:
        raise ValueError(
            "--kt-prefill-stream-top-n must satisfy 0 <= N <= "
            f"--kt-num-gpu-experts ({capacity}), got {stream_top_n}."
        )
    return stream_top_n


def is_cache_managed_layer(layer_idx: int, num_gpu_layers: int | None) -> bool:
    """Return whether a global layer index is behind the full-GPU prefix."""

    full_gpu_layers = num_gpu_layers or 0
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative.")
    if full_gpu_layers < 0:
        raise ValueError("num_gpu_layers must be non-negative.")
    return layer_idx >= full_gpu_layers


def count_route_assignments(
    topk_ids: Iterable[int], num_experts: int
) -> tuple[int, ...]:
    """Count a flat logical-expert-id snapshot, ignoring invalid IDs."""

    if num_experts < 0:
        raise ValueError("num_experts must be non-negative.")
    counts = [0] * num_experts
    for raw_expert_id in topk_ids:
        expert_id = int(raw_expert_id)
        if 0 <= expert_id < num_experts:
            counts[expert_id] += 1
    return tuple(counts)


def update_decayed_lfu(
    previous_reuse: Sequence[float],
    window_counts: Sequence[int],
    *,
    decay: float = 0.5,
    reference_assignments: float,
) -> tuple[float, ...]:
    """Apply the design's evidence-weighted Decayed-LFU update.

    ``window_counts`` is an absolute per-expert assignment count.  The
    observation is normalized before it is blended into the historical
    signal, and a short window receives proportionally less evidence weight.
    """

    previous = tuple(float(value) for value in previous_reuse)
    counts = tuple(float(value) for value in window_counts)
    if len(previous) != len(counts):
        raise ValueError("previous_reuse and window_counts must have equal length.")
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be finite and between 0 and 1.")
    if not math.isfinite(reference_assignments) or reference_assignments <= 0:
        raise ValueError("reference_assignments must be finite and positive.")
    if any(not math.isfinite(value) or value < 0 for value in previous):
        raise ValueError("previous_reuse values must be finite and non-negative.")
    if any(not math.isfinite(value) or value < 0 for value in counts):
        raise ValueError("window_counts values must be finite and non-negative.")

    total = sum(counts)
    evidence_weight = min(total / reference_assignments, 1.0)
    observation_scale = (1.0 - decay) * evidence_weight / total if total > 0 else 0.0
    return tuple(
        decay * old_signal + observation_scale * count
        for old_signal, count in zip(previous, counts)
    )


def stable_stream_top_n(
    window_counts: Sequence[int],
    reuse_signal: Sequence[float],
    top_n: int,
) -> tuple[int, ...]:
    """Rank active experts by count, reuse signal, then logical expert ID."""

    counts = tuple(float(value) for value in window_counts)
    reuse = tuple(float(value) for value in reuse_signal)
    if len(counts) != len(reuse):
        raise ValueError("window_counts and reuse_signal must have equal length.")
    if top_n < 0:
        raise ValueError("top_n must be non-negative.")
    if any(not math.isfinite(value) or value < 0 for value in counts):
        raise ValueError("window_counts values must be finite and non-negative.")
    if any(not math.isfinite(value) or value < 0 for value in reuse):
        raise ValueError("reuse_signal values must be finite and non-negative.")

    active_experts = (expert_id for expert_id, count in enumerate(counts) if count > 0)
    ranked = sorted(
        active_experts,
        key=lambda expert_id: (
            -counts[expert_id],
            -reuse[expert_id],
            expert_id,
        ),
    )
    return tuple(ranked[:top_n])


def select_stream_candidates(
    stream_hotset: Sequence[int],
    resident_expert_ids: Iterable[int],
) -> tuple[int, ...]:
    """Filter resident hits from the hotset while preserving hotset order.

    Deliberately do not scan lower-ranked experts to refill the result.  Thus a
    Top-4 hotset with three resident hits produces exactly one candidate.
    """

    resident = frozenset(int(expert_id) for expert_id in resident_expert_ids)
    return tuple(
        int(expert_id) for expert_id in stream_hotset if int(expert_id) not in resident
    )


def plan_victim_replacements(
    candidate_expert_ids: Sequence[int],
    resident_slots: Sequence[ResidentSlot],
    *,
    current_active_expert_ids: Iterable[int],
    stream_hotset: Iterable[int],
    current_epoch: int,
    min_residency_windows: int = 2,
    idle_expire_windows: int = 4,
    max_replacements: int | None = None,
) -> tuple[Replacement, ...]:
    """Pair ordered stream candidates with deterministic eligible victims.

    Only READY slots that were not activated in the current window, are outside
    the current hotset, and are past minimum residency are eligible.  Inactive
    expired residents are preferred over other inactive residents.  Ties use
    lower reuse signal, older access epoch, and finally physical slot ID.
    """

    if current_epoch < 0:
        raise ValueError("current_epoch must be non-negative.")
    if min_residency_windows < 0:
        raise ValueError("min_residency_windows must be non-negative.")
    if idle_expire_windows < 0:
        raise ValueError("idle_expire_windows must be non-negative.")
    if max_replacements is not None and max_replacements < 0:
        raise ValueError("max_replacements must be non-negative.")

    candidates = tuple(int(expert_id) for expert_id in candidate_expert_ids)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_expert_ids must be unique.")

    active = frozenset(int(expert_id) for expert_id in current_active_expert_ids)
    protected = frozenset(int(expert_id) for expert_id in stream_hotset)
    resident_experts = [slot.expert_id for slot in resident_slots]
    resident_slot_ids = [slot.slot_id for slot in resident_slots]
    if len(set(resident_experts)) != len(resident_experts):
        raise ValueError("resident expert IDs must be unique.")
    if len(set(resident_slot_ids)) != len(resident_slot_ids):
        raise ValueError("resident slot IDs must be unique.")
    if set(candidates).intersection(resident_experts):
        raise ValueError("replacement candidates must not already be resident.")

    eligible: list[ResidentSlot] = []
    for slot in resident_slots:
        if (
            slot.state != READY_SLOT_STATE
            or slot.expert_id in protected
            or slot.expert_id in active
        ):
            continue
        if not math.isfinite(slot.reuse_signal) or slot.reuse_signal < 0:
            raise ValueError("resident reuse_signal must be finite and non-negative.")
        if slot.resident_since_epoch > current_epoch:
            raise ValueError("resident_since_epoch cannot be in the future.")
        if slot.last_access_epoch > current_epoch:
            raise ValueError("last_access_epoch cannot be in the future.")
        if current_epoch - slot.resident_since_epoch < min_residency_windows:
            continue
        eligible.append(slot)

    def victim_key(slot: ResidentSlot) -> tuple[float, float, int, int]:
        is_expired = (
            current_epoch - slot.last_access_epoch >= idle_expire_windows
        )
        priority = 0 if is_expired else 1
        return (priority, slot.reuse_signal, slot.last_access_epoch, slot.slot_id)

    eligible.sort(key=victim_key)
    replacement_limit = len(candidates)
    if max_replacements is not None:
        replacement_limit = min(replacement_limit, max_replacements)
    replacement_limit = min(replacement_limit, len(eligible))

    return tuple(
        Replacement(
            candidate_expert_id=candidate,
            victim_expert_id=victim.expert_id,
            slot_id=victim.slot_id,
        )
        for candidate, victim in zip(
            candidates[:replacement_limit], eligible[:replacement_limit]
        )
    )
