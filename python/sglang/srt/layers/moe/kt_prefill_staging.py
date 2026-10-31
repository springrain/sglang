# SPDX-License-Identifier: Apache-2.0
"""Forward-mode publication and deterministic expert-layout helpers.

``ModelRunner`` publishes ``forward_batch.forward_mode`` once per forward
through :func:`publish_forward_mode`, and the KT wrapper classifies it with
:func:`is_prefill_like_mode`.  ``runtime_context.ForwardFlags`` deliberately
has no ``forward_mode`` slot, so ``is_extend_in_batch`` stays only as the
compatibility fallback when nothing was published.

The remaining layout helpers keep resident selection and host-copy run
collapsing identical on every TP rank.  They import nothing from SGLang, so
the module stays unit-testable on a CPU-only host.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Forward-mode publication
# ---------------------------------------------------------------------------

_PUBLISHED_FORWARD_MODE = None
_PUBLISHED_FORWARD_ID: Optional[int] = None


def publish_forward_mode(forward_mode, forward_id: Optional[int] = None) -> None:
    """Publish the current forward mode and its chunk/request identity.

    ``forward_id`` is the model runner's monotonically increasing pass id.
    A new id is the safe point at which the KT prefill manager may advance
    its layer order: every MoE layer of the previous chunk has completed.
    Keeping it beside the mode avoids a second process-global hook.
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
    """Whether ``forward_mode`` must run the prefill weight pipeline.

    TARGET_VERIFY forwards route a full draft window through the MoE layers and
    are treated exactly like extend batches; decode and idle batches must never
    trigger weight migration.
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
# Deterministic resident layout
# ---------------------------------------------------------------------------


def uniform_resident_counts(
    num_moe_layers: int, total_slots: int, num_experts: int
) -> List[int]:
    """Spread a global resident budget evenly across all MoE layers.

    The first ``remainder`` layers receive one extra row.  Every count is
    clamped independently to ``num_experts`` and the result is deterministic,
    which keeps the startup layout identical on every TP rank.
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


def host_generation_range(
    generation: int, ring_depth: int, count: int
) -> List[int]:
    """Return the physical host slots of one ring generation.

    The full-layer path keeps two generations in the pinned host ring.  A
    layer writes the generation the previous layer's H2D is not reading, so
    consecutive layers alternate between the two halves of the ring.
    ``count`` above one generation would index past the pinned buffer, so it
    raises instead of silently reading or writing out of bounds.
    """

    depth = max(2, int(ring_depth or 2))
    step = depth // 2
    count = int(count)
    if count > step:
        raise ValueError(
            f"host generation holds {step} slots but {count} experts were "
            "requested; the host ring generation must bound the batch"
        )
    start = (int(generation) % 2) * step
    return [start + offset for offset in range(count)]


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
