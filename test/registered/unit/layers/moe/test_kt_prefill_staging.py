"""Unit coverage for the KT prefill-staging decision layer.

Mirrors doc/KT-GEC-PREFETCH-STREAMING-PLAN.md section 18.1: bank selection
invariants, staging-window clamping, resident/staging dedup, score decay,
cold-start rotation, TP-consistent determinism and the prefill-like gate.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

import pytest

torch = pytest.importorskip("torch")

try:  # pragma: no cover - registration is a no-op off the CI image
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # pragma: no cover - CPU-only hosts without SGLang deps
    register_cpu_ci = None


def _load_staging_module():
    """Import the planner module, tolerating a host without SGLang imports.

    ``kt_prefill_staging`` deliberately imports nothing from SGLang, so on a
    host where the full package cannot be imported (missing ``resource``,
    ``fcntl``, kernels, ...) the single file still loads directly.  CI uses the
    normal package import.
    """

    try:
        from sglang.srt.layers.moe import kt_prefill_staging as module

        return module
    except Exception:
        path = (
            pathlib.Path(__file__).resolve().parents[5]
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "moe"
            / "kt_prefill_staging.py"
        )
        spec = importlib.util.spec_from_file_location(
            "kt_prefill_staging_under_test", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


staging = _load_staging_module()

if register_cpu_ci is not None:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Mode:
    """Minimal ForwardMode stand-in for gate tests."""

    def __init__(self, extend=False, target_verify=False, decode=False):
        self._extend = extend
        self._target_verify = target_verify
        self._decode = decode

    def is_extend(self):
        return self._extend

    def is_target_verify(self):
        return self._target_verify

    def is_decode(self):
        return self._decode


class TestStagingWindow(unittest.TestCase):
    """``--kt-layer-h2d-slots`` semantics (doc section 3.3)."""

    def test_unset_or_non_positive_leaves_staging_unselected(self):
        self.assertIsNone(staging.clamp_staging_window(None, 256))
        self.assertIsNone(staging.clamp_staging_window(0, 256))
        self.assertIsNone(staging.clamp_staging_window(-4, 256))

    def test_window_wider_than_the_layer_clamps_to_the_layer(self):
        self.assertEqual(staging.clamp_staging_window(385, 256), 256)
        self.assertEqual(staging.clamp_staging_window(256, 256), 256)
        self.assertEqual(staging.clamp_staging_window(64, 256), 64)

    def test_planner_clamps_its_own_window(self):
        planner = staging.StagingPrefetchPlanner(num_experts=8, window=100)
        self.assertEqual(planner.window, 8)

    def test_planner_rejects_a_disabled_window(self):
        with self.assertRaises(ValueError):
            staging.StagingPrefetchPlanner(num_experts=8, window=0)


class TestEffectiveSet(unittest.TestCase):
    """``effective_gpu(L) = resident(L) + staged(L)`` (doc section 3.5)."""

    def test_full_coverage_is_not_partial(self):
        effective, partial = staging.effective_experts([0, 1], [2, 3], 4)
        self.assertEqual(effective, [0, 1, 2, 3])
        self.assertFalse(partial)

    def test_uncovered_experts_mark_partial(self):
        effective, partial = staging.effective_experts([0], [2], 4)
        self.assertEqual(effective, [0, 2])
        self.assertTrue(partial)

    def test_resident_and_staged_sets_are_deduplicated(self):
        effective, partial = staging.effective_experts([0, 1], [1, 2], 3)
        self.assertEqual(effective, [0, 1, 2])
        self.assertFalse(partial)


class TestHostRing(unittest.TestCase):
    """One whole staged set per layer rolls through the pinned ring.

    Doc section 20 item 8: the layerwise manager submits the layer's whole
    staging window as a single pass, pays one KT writer drain plus one TP fence
    for it, and then hands the *other* generation of host slots to the next
    layer, so the write of ``L + 1`` overlaps the H2D of ``L`` while the H2D of
    ``L + 1`` overlaps the write of ``L``.  A generation is therefore the
    staging window ``S`` itself (``--kt-layer-h2d-slots``) and the ring is
    ``2 * S`` slots, the FreeToken prefill shape.
    """

    def test_ring_always_holds_two_generations(self):
        for window in (1, 2, 8, 16, 64, 384):
            self.assertEqual(staging.host_ring_depth(window), 2 * window)
        for degenerate in (0, -4, None):
            self.assertEqual(staging.host_ring_depth(degenerate), 2)

    def test_generation_holds_the_whole_staging_window(self):
        for window in (1, 2, 8, 64, 384):
            ring = staging.host_ring_depth(window)
            for generation in (0, 1, 2, 3):
                slots = staging.host_generation_range(generation, ring, window)
                self.assertEqual(len(slots), window)
                start = (generation % 2) * window
                self.assertEqual(slots, list(range(start, start + window)))
                self.assertTrue(all(slot < ring for slot in slots))

    def test_consecutive_generations_do_not_alias(self):
        window = 8
        ring = staging.host_ring_depth(window)
        first = staging.host_generation_range(0, ring, window)
        second = staging.host_generation_range(1, ring, window)
        self.assertEqual(first, list(range(0, 8)))
        self.assertEqual(second, list(range(8, 16)))
        self.assertEqual(set(first) & set(second), set())

    def test_short_staging_set_stays_inside_its_generation(self):
        ring = staging.host_ring_depth(8)
        for generation in (0, 1, 2, 3):
            for count in (1, 3, 7, 8):
                slots = staging.host_generation_range(generation, ring, count)
                expected = (generation % 2) * 8
                self.assertEqual(slots, list(range(expected, expected + count)))

    def test_oversized_staging_set_is_rejected(self):
        """A generation cannot hold more than the window, so refuse to index past it."""

        window = 8
        ring = staging.host_ring_depth(window)
        with self.assertRaises(ValueError):
            staging.host_generation_range(0, ring, window + 1)

    def test_degenerate_window_rolls_on_a_pair(self):
        self.assertEqual(staging.host_ring_depth(1), 2)
        self.assertEqual(staging.host_generation_range(0, 2, 1), [0])
        self.assertEqual(staging.host_generation_range(1, 2, 1), [1])


class TestContiguousStageRuns(unittest.TestCase):
    """The per-expert H2D loop collapses to stride-level runs.

    Doc section 15.3: consecutive ``(expert, host slot)`` pairs that both step
    by one address the same bytes as a single strided copy, so the loop may
    issue one ``copy_`` over an ``(n, chunk)`` view instead of ``n`` per-expert
    submissions.  The *selection* is never widened or reordered: only exactly
    adjacent pairs merge, and anything sparse degrades to single-expert runs,
    which is the historical loop's behaviour.
    """

    def test_dense_run_merges_into_one_slice(self):
        runs = staging.contiguous_stage_runs([5, 6, 7, 8, 20, 21], [0, 1, 2, 3, 8, 9])
        self.assertEqual(runs, [(0, 4), (4, 2)])

    def test_runs_always_partition_the_selection(self):
        """Merging must not drop, duplicate or reorder a single pair."""

        for ids, slots in (
            ([5, 6, 7, 8, 20, 21], [0, 1, 2, 3, 8, 9]),
            ([1, 5, 9], [0, 1, 2]),
            ([0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]),
            ([7], [3]),
        ):
            runs = staging.contiguous_stage_runs(ids, slots)
            covered = []
            for start, length in runs:
                self.assertGreaterEqual(length, 1)
                covered.extend(range(start, start + length))
            self.assertEqual(covered, list(range(len(ids))))

    def test_sparse_selection_degrades_to_singletons(self):
        runs = staging.contiguous_stage_runs([1, 5, 9], [0, 1, 2])
        self.assertEqual(runs, [(0, 1), (1, 1), (2, 1)])

    def test_descending_experts_never_merge(self):
        runs = staging.contiguous_stage_runs([3, 2, 1], [0, 1, 2])
        self.assertEqual(runs, [(0, 1), (1, 1), (2, 1)])

    def test_a_single_non_adjacent_side_breaks_the_run(self):
        """Stride merging requires BOTH sides to step by one."""

        self.assertEqual(
            staging.contiguous_stage_runs([5, 6, 7], [0, 1, 5]),
            [(0, 2), (2, 1)],
        )
        self.assertEqual(
            staging.contiguous_stage_runs([5, 6, 9], [0, 1, 2]),
            [(0, 2), (2, 1)],
        )

    def test_empty_and_mismatched_inputs_are_safe(self):
        self.assertEqual(staging.contiguous_stage_runs([], []), [])
        # The shorter side bounds the walk; nothing past it is touched.
        self.assertEqual(staging.contiguous_stage_runs([1, 2, 3], [0, 1]), [(0, 2)])
        self.assertEqual(staging.contiguous_stage_runs([1, 2], [0, 1, 2]), [(0, 2)])

    def test_selection_collapses_when_slots_follow_the_generation(self):
        """The hot path: a contiguous expert run over a generation's slots."""

        window = 64
        ring = staging.host_ring_depth(window)
        slots = staging.host_generation_range(0, ring, window)
        runs = staging.contiguous_stage_runs(list(range(0, window)), slots)
        self.assertEqual(runs, [(0, window)])


class TestStagingEntryConditions(unittest.TestCase):
    """Joint gate of ``--kt-layer-h2d-slots`` and the prefill threshold.

    Doc section 13.1: positive window, threshold configured and reached,
    prefill-like forward, and a runtime that can host the MXFP4 layerwise
    slots.  Any missing condition keeps the resident/CPU path.
    """

    def _requested(self, **overrides):
        kwargs = {
            "window": 256,
            "threshold": 2048,
            "num_tokens": 2048,
            "prefill_like": True,
            "runtime_supported": True,
        }
        kwargs.update(overrides)
        return staging.staging_requested_for_batch(**kwargs)

    def test_all_conditions_met_enables_staging(self):
        self.assertTrue(self._requested())
        self.assertTrue(self._requested(num_tokens=4096))

    def test_window_is_required(self):
        self.assertFalse(self._requested(window=None))

    def test_threshold_is_required_and_must_be_reached(self):
        self.assertFalse(self._requested(threshold=None))
        self.assertFalse(self._requested(threshold=0))
        self.assertFalse(self._requested(num_tokens=2047))

    def test_decode_like_forwards_stay_on_the_resident_path(self):
        self.assertFalse(self._requested(prefill_like=False))

    def test_unsupported_runtime_falls_back(self):
        self.assertFalse(self._requested(runtime_supported=False))


class TestPrefetchSelection(unittest.TestCase):
    """Candidate filtering, dedup and determinism (doc sections 8.1 and 8.5)."""

    def setUp(self):
        self.num_experts = 16

    def test_candidates_exclude_resident_experts(self):
        planner = staging.StagingPrefetchPlanner(self.num_experts, window=4)
        selected = planner.select(3, [0, 1, 2])
        self.assertEqual(selected, [3, 4, 5, 6])
        self.assertFalse(set(selected) & {0, 1, 2})

    def test_window_covers_only_the_available_candidates(self):
        planner = staging.StagingPrefetchPlanner(self.num_experts, window=8)
        selected = planner.select(3, list(range(14)))
        self.assertEqual(selected, [14, 15])

    def test_full_window_selects_the_whole_layer(self):
        planner = staging.StagingPrefetchPlanner(self.num_experts, window=16)
        self.assertEqual(planner.select(7, []), list(range(self.num_experts)))

    def test_partial_window_leaves_a_cpu_fallback_set(self):
        planner = staging.StagingPrefetchPlanner(self.num_experts, window=6)
        resident = [0, 1, 2, 3]
        selected = planner.select(4, resident)
        self.assertEqual(len(selected), 6)
        effective, partial = staging.effective_experts(
            resident, selected, self.num_experts
        )
        self.assertTrue(partial)
        self.assertEqual(len(effective), 10)
        self.assertEqual(
            sorted(set(range(self.num_experts)) - set(effective)),
            [10, 11, 12, 13, 14, 15],
        )

    def test_layer_covering_window_needs_no_cpu_fallback(self):
        """``kt-layer-h2d-slots=256`` must give a 100 percent GPU set."""

        planner = staging.StagingPrefetchPlanner(self.num_experts, window=16)
        resident = [0, 1, 2, 3]
        selected = planner.select(9, resident)
        self.assertEqual(len(selected), 12)
        effective, partial = staging.effective_experts(
            resident, selected, self.num_experts
        )
        self.assertFalse(partial)
        self.assertEqual(effective, list(range(self.num_experts)))

    def test_history_orders_the_selection(self):
        planner = staging.StagingPrefetchPlanner(self.num_experts, window=2)
        counts = [0] * self.num_experts
        counts[9] = 40
        counts[4] = 11
        counts[15] = 3
        planner.record_router_counts(5, counts)
        self.assertEqual(planner.select(5, [0]), [4, 9])
        self.assertEqual(planner.select(5, [4]), [9, 15])

    def test_selection_is_deterministic_for_tp_ranks(self):
        """Every rank derives the same selection from the same state."""

        planners = [
            staging.StagingPrefetchPlanner(self.num_experts, window=3)
            for _ in range(2)
        ]
        counts = [1] * self.num_experts
        for planner in planners:
            planner.record_router_counts(11, counts)
        resident = [0, 5, 7]
        self.assertEqual(
            planners[0].select(11, resident),
            planners[1].select(11, resident),
        )
        # Cold start is deterministic too.
        cold = [
            staging.StagingPrefetchPlanner(self.num_experts, window=3)
            for _ in range(2)
        ]
        self.assertEqual(cold[0].select(2, []), cold[1].select(2, []))

    def test_other_layers_history_never_leaks_into_the_target_layer(self):
        planner = staging.StagingPrefetchPlanner(self.num_experts, window=2)
        counts = [0] * self.num_experts
        counts[1] = 100
        planner.record_router_counts(4, counts)
        # Layer 5 has no history yet, so it must cold-start on its own ids.
        self.assertEqual(planner.select(5, []), [0, 1])


class TestColdStart(unittest.TestCase):
    """Deterministic chunked rotation (doc section 8.4)."""

    def test_cursor_rotates_in_window_chunks(self):
        planner = staging.StagingPrefetchPlanner(num_experts=8, window=3)
        self.assertEqual(planner.select(0, []), [0, 1, 2])
        self.assertEqual(planner.select(0, []), [3, 4, 5])
        # The window wraps around and the result stays in ascending id order.
        self.assertEqual(planner.select(0, []), [0, 6, 7])
        self.assertEqual(planner.cold_cursor, 1)

    def test_rotation_touches_every_candidate(self):
        planner = staging.StagingPrefetchPlanner(num_experts=10, window=4)
        seen = set()
        for _ in range(3):
            seen.update(planner.select(0, []))
        self.assertEqual(seen, set(range(10)))


class TestHistoryScores(unittest.TestCase):
    """Score weights and periodic decay (doc section 8.3)."""

    def test_score_matches_the_documented_weights(self):
        table = staging.PrefillHistoryTable(num_experts=4)
        table.record_router_counts(2, [0, 0, 0, 0])
        table.record_router_counts(2, [0, 0, 2, 0])
        scores = table.scores(2)
        # decayed=2, recent=2 (it retains 0.5 before each update and then the
        # newest demand is added), last_demand=2.
        self.assertAlmostEqual(float(scores[2]), 2.0 + 0.25 * 2.0 + 0.5 * 2.0)
        self.assertAlmostEqual(float(scores[0]), 0.0)

    def test_recent_window_follows_distribution_changes(self):
        table = staging.PrefillHistoryTable(num_experts=4)
        table.record_router_counts(1, [8, 0, 0, 0])
        table.record_router_counts(1, [0, 8, 0, 0])
        scores = table.scores(1)
        # The latest demand plus the fast recent term lift the new hot expert
        # above the older one even though the decayed sums tie.
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_periodic_decay_fades_unvisited_layers(self):
        table = staging.PrefillHistoryTable(
            num_experts=4, decay_factor=0.5, decay_interval=2
        )
        table.record_router_counts(1, [4, 0, 0, 0])
        before = float(table.scores(1)[0])
        self.assertFalse(table.note_layer_visit(1))
        self.assertTrue(table.note_layer_visit(1))
        after = float(table.scores(1)[0])
        self.assertLess(after, before)

    def test_history_rejects_a_size_mismatch(self):
        table = staging.PrefillHistoryTable(num_experts=4)
        with self.assertRaises(ValueError):
            table.record_router_counts(1, [1, 2])


class TestForwardModeGate(unittest.TestCase):
    """Prefill-like classification and publication (doc section 14.6)."""

    def test_extend_and_target_verify_are_prefill_like(self):
        self.assertTrue(staging.is_prefill_like_mode(_Mode(extend=True)))
        self.assertTrue(staging.is_prefill_like_mode(_Mode(target_verify=True)))

    def test_decode_and_missing_modes_are_not_prefill_like(self):
        self.assertFalse(staging.is_prefill_like_mode(_Mode(decode=True)))
        self.assertFalse(staging.is_prefill_like_mode(None))
        self.assertFalse(staging.is_prefill_like_mode(object()))

    def test_publication_round_trips(self):
        staging.publish_forward_mode(None, forward_id=1)
        self.assertIsNone(staging.published_forward_mode())
        self.assertEqual(staging.published_forward_id(), 1)
        mode = _Mode(target_verify=True)
        staging.publish_forward_mode(mode, forward_id=2)
        self.assertIs(staging.published_forward_mode(), mode)
        self.assertEqual(staging.published_forward_id(), 2)
        staging.publish_forward_mode(None, forward_id=3)
        self.assertEqual(staging.published_forward_id(), 3)


class TestUniformResidency(unittest.TestCase):
    """Global GEC budget is spread over every MoE layer, shared ids first."""

    def test_budget_is_split_evenly_with_remainder_on_early_layers(self):
        self.assertEqual(
            staging.uniform_resident_counts(4, 10, 8), [3, 3, 2, 2]
        )

    def test_each_layer_is_clamped_to_its_expert_count(self):
        self.assertEqual(
            staging.uniform_resident_counts(2, 100, 8), [8, 8]
        )

    def test_zero_dimensions_are_safe(self):
        self.assertEqual(staging.uniform_resident_counts(0, 8, 4), [])
        self.assertEqual(staging.uniform_resident_counts(3, 8, 0), [0, 0, 0])

    def test_fused_shared_experts_are_the_trailing_ids(self):
        self.assertEqual(staging.fused_shared_expert_ids(8, 2), [6, 7])
        self.assertEqual(staging.fused_shared_expert_ids(8, 20), list(range(8)))

    def test_shared_ids_are_selected_before_routed_ids(self):
        self.assertEqual(
            staging.shared_first_resident_ids(8, 4, [6, 7]), [6, 7, 0, 1]
        )
        self.assertEqual(
            staging.shared_first_resident_ids(8, 1, [6, 7]), [6]
        )
        self.assertEqual(
            staging.shared_first_resident_ids(8, 8, [6, 7]), list(range(8))
        )

    def test_compact_destinations_drive_copy_run_collapsing(self):
        self.assertEqual(
            staging.contiguous_copy_runs([5, 6, 7, 2], [0, 1, 2, 3]),
            [(0, 3), (3, 1)],
        )


class TestForwardBoundary(unittest.TestCase):
    """A new pass/chunk id is the only resident-ownership update point."""

    def test_first_observation_never_reports_a_boundary(self):
        tracker = staging.ForwardBoundaryTracker()
        self.assertFalse(tracker.observe(None))
        self.assertFalse(tracker.observe(7))
        self.assertFalse(tracker.observe(7))
        self.assertTrue(tracker.observe(8))
        self.assertFalse(tracker.observe(8))
        self.assertEqual(tracker.forward_id, 8)


class TestBoundaryPromotion(unittest.TestCase):
    """Staged winners replace the coldest unprotected resident rows."""

    def test_winners_and_victims_are_deterministic(self):
        pairs = staging.select_boundary_promotions(
            resident_ids=[0, 1, 2],
            staged_ids=[3, 4],
            hit_counts={3: 2, 4: 9},
            scores={0: 0.0, 1: 4.0, 2: 0.0, 3: 1.0, 4: 0.0},
        )
        self.assertEqual(pairs, [(0, 4), (2, 3)])

    def test_staged_experts_without_hits_are_not_promoted(self):
        self.assertEqual(
            staging.select_boundary_promotions(
                [0, 1], [2, 3], {2: 0, 3: 0}
            ),
            [],
        )

    def test_protected_experts_are_never_evicted_but_may_promote(self):
        pairs = staging.select_boundary_promotions(
            [0, 1],
            [4],
            {4: 3},
            protected_ids=[0, 4],
            scores={0: 0.0, 1: 1.0, 4: 2.0},
        )
        self.assertEqual(pairs, [(1, 4)])


class TestPendingPromotionCommit(unittest.TestCase):
    """A deferred promotion keeps row identity and shares the host window."""

    def test_row_owner_promotions_replace_victims_in_place(self):
        self.assertEqual(
            staging.apply_row_owner_promotions(
                [4, 5, 6], [(5, 9), (4, 8)], num_experts=10
            ),
            [8, 9, 6],
        )

    def test_row_owner_promotions_reject_duplicate_winners(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            staging.apply_row_owner_promotions(
                [1, 2, 3], [(1, 4), (2, 4)], num_experts=8
            )

    def test_row_owner_promotions_reject_a_missing_victim(self):
        with self.assertRaisesRegex(ValueError, "not resident"):
            staging.apply_row_owner_promotions(
                [1, 2], [(7, 4)], num_experts=8
            )

    def test_promoted_rows_are_reserved_before_the_staged_window(self):
        self.assertEqual(
            staging.fit_staged_window([3, 4, 5, 6], 2, 4), [3, 4]
        )
        self.assertEqual(staging.fit_staged_window([3, 4], 0, 2), [3, 4])

    def test_promotions_wider_than_the_host_generation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            staging.fit_staged_window([3], 3, 2)


class TestStagingRowPool(unittest.TestCase):
    """Resident and staging share one physical row pool per bank."""

    def test_physical_geometry_is_resident_plus_staging(self):
        pool = staging.StagingRowPool(
            8, resident_capacity=2, staging_capacity=2, protected_ids=[7]
        )
        self.assertEqual(pool.physical_rows, 4)
        mapping = pool.configure(3, [6, 7], [0, 1])
        self.assertEqual(sorted(row for row in mapping if row >= 0), [0, 1, 2, 3])

    def test_promotion_swaps_roles_without_reassigning_rows(self):
        pool = staging.StagingRowPool(
            8, resident_capacity=2, staging_capacity=2, protected_ids=[7]
        )
        before = pool.configure(3, [6, 7], [0, 1])
        promoted_row = before[0]
        victim_row = before[6]
        pool.promote(3, victim_id=6, promoted_id=0)
        after = pool.mapping(3)
        self.assertEqual(after[0], promoted_row)
        self.assertEqual(after[6], -1)
        roles = pool.row_roles(3)
        self.assertEqual(roles[promoted_row], "resident")
        self.assertEqual(roles[victim_row], "free")

    def test_configure_preserves_a_role_transition_in_place(self):
        pool = staging.StagingRowPool(
            8, resident_capacity=2, staging_capacity=2
        )
        before = pool.configure(1, [6, 7], [0, 1])
        pool.promote(1, victim_id=6, promoted_id=0)
        after = pool.configure(1, [7, 0], [1, 2])
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[7], before[7])
        self.assertNotEqual(after[1], -1)
        self.assertNotEqual(after[2], -1)

    def test_protected_rows_cannot_be_evicted(self):
        pool = staging.StagingRowPool(
            8, resident_capacity=2, staging_capacity=1, protected_ids=[7]
        )
        pool.configure(2, [6, 7], [0])
        with self.assertRaises(ValueError):
            pool.promote(2, victim_id=7, promoted_id=0)

    def test_clear_discards_stale_bank_ownership(self):
        pool = staging.StagingRowPool(
            8, resident_capacity=2, staging_capacity=1
        )
        pool.configure(1, [6, 7], [0])
        pool.clear()
        self.assertEqual(pool.mapping(1), [-1] * 8)
        self.assertEqual(pool.row_roles(1), ["free"] * 3)


class TestTelemetry(unittest.TestCase):
    """Per-round counters (doc section 15)."""

    def test_summary_waits_for_a_full_round(self):
        telemetry = staging.StagingTelemetry(interval=2)
        telemetry.record_layer(candidate_count=4, selected_count=3, compute_ms=1.0)
        self.assertIsNone(telemetry.take_summary())
        telemetry.record_layer(candidate_count=6, selected_count=5, compute_ms=3.0)
        summary = telemetry.take_summary()
        self.assertEqual(summary["candidate_count"], 10)
        self.assertEqual(summary["selected_count"], 8)
        self.assertEqual(summary["compute_ms"], 2.0)
        self.assertEqual(telemetry.pending(), 0)

    def test_round_summary_renders_the_documented_fields(self):
        telemetry = staging.StagingTelemetry(interval=1)
        telemetry.record_layer(
            candidate_count=192,
            selected_count=192,
            resident_count=64,
            staging_hit_count=180,
            staging_miss_count=12,
            cpu_expert_count=0,
            gpu_expert_count=256,
            cpu_route_count=0,
            copy_ms=4.5,
            ready_wait_ms=0.0,
            compute_ms=9.25,
        )
        text = staging.format_round_summary(telemetry.take_summary())
        self.assertTrue(text.startswith("[kt-prefetch] round"))
        self.assertIn("candidates=192", text)
        self.assertIn("staged_hits=180", text)
        self.assertIn("cpu=0", text)
        self.assertIn("cpu_routes=0", text)

    def test_cpu_route_counts_are_summed_per_round(self):
        telemetry = staging.StagingTelemetry(interval=2)
        telemetry.record_layer(cpu_route_count=12, compute_ms=1.0)
        telemetry.record_layer(cpu_route_count=8, compute_ms=1.0)
        summary = telemetry.take_summary()
        self.assertEqual(summary["cpu_route_count"], 20)
        self.assertIn("cpu_routes=20", staging.format_round_summary(summary))

    def test_copy_submissions_are_summed_and_rendered(self):
        """Section 15.3: the counters make the run merge visible in a round.

        ``copy_submits`` counts real ``copy_`` calls, ``copy_pairs`` the pairs
        moved, so a collapsed window shows ``copy_pairs > copy_submits``.
        """

        telemetry = staging.StagingTelemetry(interval=2)
        telemetry.record_layer(copy_submits=40, copy_pairs=256, compute_ms=1.0)
        telemetry.record_layer(copy_submits=40, copy_pairs=256, compute_ms=1.0)
        summary = telemetry.take_summary()
        self.assertEqual(summary["copy_submits"], 80)
        self.assertEqual(summary["copy_pairs"], 512)
        text = staging.format_round_summary(summary)
        self.assertIn("copy_submits=80", text)
        self.assertIn("copy_pairs=512", text)

    def test_uncollapsed_loop_reads_one_submit_per_pair(self):
        """A fully sparse selection must render the merge factor as 1.0."""

        telemetry = staging.StagingTelemetry(interval=1)
        telemetry.record_layer(copy_submits=12, copy_pairs=12, compute_ms=1.0)
        summary = telemetry.take_summary()
        self.assertEqual(
            summary["copy_pairs"] / summary["copy_submits"],
            1.0,
        )

    def test_host_route_counts_join_the_next_summary_window(self):
        telemetry = staging.StagingTelemetry(interval=2)
        telemetry.add_cpu_routes(30)
        telemetry.record_layer(cpu_route_count=12, compute_ms=1.0)
        # An incomplete window keeps the folded counts for the next summary.
        self.assertIsNone(telemetry.take_summary())
        telemetry.record_layer(cpu_route_count=8, compute_ms=1.0)
        summary = telemetry.take_summary()
        self.assertEqual(summary["cpu_route_count"], 50)
        self.assertIn("cpu_routes=50", staging.format_round_summary(summary))
        # Every folded count belongs to exactly one window.
        telemetry.record_layer(cpu_route_count=0, compute_ms=1.0)
        telemetry.record_layer(cpu_route_count=0, compute_ms=1.0)
        self.assertEqual(telemetry.take_summary()["cpu_route_count"], 0)


if __name__ == "__main__":
    unittest.main()
