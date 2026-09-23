# SPDX-License-Identifier: Apache-2.0

import pytest
from sglang.srt.kt_expert_cache_policy import (
    ResidentSlot,
    count_route_assignments,
    is_cache_managed_layer,
    plan_victim_replacements,
    resolve_prefill_stream_top_n,
    select_stream_candidates,
    stable_stream_top_n,
    update_decayed_lfu,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_stream_width_resolution_is_bounded_and_strategy_specific():
    assert resolve_prefill_stream_top_n("decayed-lfu", 32, None) == 4
    assert resolve_prefill_stream_top_n("decayed-lfu", 2, None) == 2
    assert resolve_prefill_stream_top_n("decayed-lfu", 32, 0) == 0
    assert resolve_prefill_stream_top_n("decayed-lfu", 32, 32) == 32
    assert resolve_prefill_stream_top_n("uniform", 32, None) is None

    with pytest.raises(ValueError, match="only valid with"):
        resolve_prefill_stream_top_n("uniform", 32, 4)
    with pytest.raises(ValueError, match="0 <= N <="):
        resolve_prefill_stream_top_n("decayed-lfu", 4, 5)
    with pytest.raises(ValueError, match="0 <= N <="):
        resolve_prefill_stream_top_n("decayed-lfu", 4, -1)


def test_full_gpu_prefix_is_never_cache_managed():
    assert not is_cache_managed_layer(0, 2)
    assert not is_cache_managed_layer(1, 2)
    assert is_cache_managed_layer(2, 2)
    assert is_cache_managed_layer(9, 2)
    assert is_cache_managed_layer(0, None)


def test_route_counts_ignore_invalid_logical_ids():
    assert count_route_assignments([0, 2, -1, 2, 4, 99], 5) == (1, 0, 2, 0, 1)


def test_decayed_lfu_uses_normalized_evidence_weight():
    updated = update_decayed_lfu(
        [0.4, 0.2],
        [3, 1],
        decay=0.5,
        reference_assignments=8,
    )
    assert updated == pytest.approx((0.3875, 0.1625))

    no_observation = update_decayed_lfu(
        updated,
        [0, 0],
        decay=0.5,
        reference_assignments=8,
    )
    assert no_observation == pytest.approx(tuple(value * 0.5 for value in updated))


def test_stream_top_n_is_stable_and_only_contains_active_experts():
    # Count is primary, reuse signal is the first tie-break, and logical ID is
    # the final deterministic tie-break. Expert 4 is inactive despite heat.
    hotset = stable_stream_top_n(
        [7, 7, 9, 7, 0],
        [0.2, 0.8, 0.1, 0.8, 100.0],
        4,
    )
    assert hotset == (2, 1, 3, 0)


def test_resident_hits_reduce_candidates_without_backfill():
    hotset = (8, 3, 5, 1)
    assert select_stream_candidates(hotset, {8, 5, 1}) == (3,)
    assert select_stream_candidates(hotset, set(hotset)) == ()


def test_victim_plan_is_safe_bounded_and_deterministic():
    residents = (
        # Expired and inactive: selected first despite a higher reuse signal.
        ResidentSlot(0, 10, 0.8, last_access_epoch=2, resident_since_epoch=0),
        # Inactive but not expired.
        ResidentSlot(1, 11, 0.1, last_access_epoch=9, resident_since_epoch=0),
        # Active non-hotset resident: protected for the entire current window.
        ResidentSlot(2, 12, 0.01, last_access_epoch=10, resident_since_epoch=0),
        # Protected by the current hotset.
        ResidentSlot(3, 13, 0.0, last_access_epoch=0, resident_since_epoch=0),
        # Too new to evict.
        ResidentSlot(4, 14, 0.0, last_access_epoch=0, resident_since_epoch=9),
        # Non-READY slots are never victims.
        ResidentSlot(
            5,
            15,
            0.0,
            last_access_epoch=0,
            resident_since_epoch=0,
            state="IN_USE",
        ),
    )

    plan = plan_victim_replacements(
        [20, 21, 22, 23],
        residents,
        current_active_expert_ids={12, 13},
        stream_hotset={13, 20, 21, 22, 23},
        current_epoch=10,
        min_residency_windows=2,
        idle_expire_windows=4,
        max_replacements=3,
    )

    assert [
        (item.candidate_expert_id, item.victim_expert_id, item.slot_id) for item in plan
    ] == [
        (20, 10, 0),
        (21, 11, 1),
    ]


def test_victim_plan_never_evicts_an_expert_active_in_current_window():
    residents = (
        ResidentSlot(0, 10, 0.01, last_access_epoch=12, resident_since_epoch=0),
        ResidentSlot(1, 11, 0.02, last_access_epoch=12, resident_since_epoch=0),
    )

    plan = plan_victim_replacements(
        [20, 21],
        residents,
        current_active_expert_ids={10, 11, 20, 21},
        stream_hotset={20, 21},
        current_epoch=12,
        min_residency_windows=2,
        idle_expire_windows=4,
        max_replacements=2,
    )

    assert plan == ()
