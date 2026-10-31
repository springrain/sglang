"""Unit coverage for the KT prefill helper module.

The staging planner, row pool and telemetry were removed with
``--kt-layer-h2d-slots``.  What remains is the forward-mode publication used
to classify prefill forwards plus the deterministic resident and copy-layout
helpers shared by every TP rank.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

try:  # pragma: no cover - registration is a no-op off the CI image
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # pragma: no cover - CPU-only hosts without SGLang deps
    register_cpu_ci = None


def _load_staging_module():
    """Import the helper module without requiring the full SGLang package."""

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
    """Minimal ForwardMode stand-in for publication tests."""

    def __init__(self, extend=False, target_verify=False):
        self._extend = extend
        self._target_verify = target_verify

    def is_extend(self):
        return self._extend

    def is_target_verify(self):
        return self._target_verify


class TestForwardModePublication(unittest.TestCase):
    def setUp(self):
        staging.publish_forward_mode(None, 0)

    def test_published_mode_and_forward_id_are_visible(self):
        mode = _Mode(extend=True)
        staging.publish_forward_mode(mode, 17)

        self.assertIs(staging.published_forward_mode(), mode)
        self.assertEqual(staging.published_forward_id(), 17)

    def test_omitted_forward_id_keeps_the_last_published_value(self):
        staging.publish_forward_mode(_Mode(), 23)
        staging.publish_forward_mode(_Mode())

        self.assertEqual(staging.published_forward_id(), 23)

    def test_prefill_like_modes_include_extend_and_target_verify(self):
        self.assertTrue(staging.is_prefill_like_mode(_Mode(extend=True)))
        self.assertTrue(staging.is_prefill_like_mode(_Mode(target_verify=True)))
        self.assertFalse(staging.is_prefill_like_mode(_Mode()))
        self.assertFalse(staging.is_prefill_like_mode(None))
        self.assertFalse(staging.is_prefill_like_mode(object()))


class TestResidentLayout(unittest.TestCase):
    def test_uniform_residency_spreads_the_global_budget(self):
        self.assertEqual(
            staging.uniform_resident_counts(3, 8, 4), [3, 3, 2]
        )
        self.assertEqual(
            staging.uniform_resident_counts(2, 100, 8), [8, 8]
        )
        self.assertEqual(
            staging.uniform_resident_counts(4, 0, 8), [0, 0, 0, 0]
        )
        self.assertEqual(staging.uniform_resident_counts(0, 8, 4), [])
        self.assertEqual(staging.uniform_resident_counts(2, -1, 4), [0, 0])

    def test_fused_shared_experts_are_the_trailing_ids(self):
        self.assertEqual(staging.fused_shared_expert_ids(8, 2), [6, 7])
        self.assertEqual(staging.fused_shared_expert_ids(4, 8), [0, 1, 2, 3])
        self.assertEqual(staging.fused_shared_expert_ids(4, 0), [])

    def test_shared_experts_are_selected_before_routed_ids(self):
        self.assertEqual(
            staging.shared_first_resident_ids(8, 4, [7, 6]),
            [6, 7, 0, 1],
        )
        self.assertEqual(
            staging.shared_first_resident_ids(8, 1, [7, 6]),
            [6],
        )
        self.assertEqual(
            staging.shared_first_resident_ids(8, 3, [6, 6, -1, 9]),
            [6, 0, 1],
        )

    def test_a_full_budget_keeps_natural_id_order(self):
        self.assertEqual(
            staging.shared_first_resident_ids(4, 4, [3, 2]),
            [0, 1, 2, 3],
        )


class TestHostGenerationRange(unittest.TestCase):
    def test_generations_alternate_without_overlap(self):
        first = staging.host_generation_range(0, 4, 2)
        second = staging.host_generation_range(1, 4, 2)

        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [2, 3])
        self.assertFalse(set(first) & set(second))
        self.assertEqual(staging.host_generation_range(2, 4, 2), first)

    def test_negative_generation_uses_the_other_half(self):
        self.assertEqual(staging.host_generation_range(-1, 4, 1), [2])

    def test_degenerate_ring_still_has_two_slots(self):
        self.assertEqual(staging.host_generation_range(0, None, 1), [0])
        self.assertEqual(staging.host_generation_range(1, 0, 1), [1])

    def test_count_must_fit_inside_one_generation(self):
        with self.assertRaisesRegex(ValueError, "host generation holds 2 slots"):
            staging.host_generation_range(0, 4, 3)


class TestContiguousCopyRuns(unittest.TestCase):
    def test_consecutive_destinations_and_slots_collapse(self):
        self.assertEqual(
            staging.contiguous_copy_runs([4, 5, 6], [0, 1, 2]),
            [(0, 3)],
        )

    def test_either_side_breaking_starts_a_new_run(self):
        self.assertEqual(
            staging.contiguous_copy_runs([4, 5, 9, 10], [0, 1, 4, 5]),
            [(0, 2), (2, 2)],
        )
        self.assertEqual(
            staging.contiguous_copy_runs([4, 5, 6], [0, 1, 9]),
            [(0, 2), (2, 1)],
        )
        self.assertEqual(
            staging.contiguous_copy_runs([4, 8, 9], [0, 1, 2]),
            [(0, 1), (1, 2)],
        )

    def test_mismatched_lengths_use_the_shorter_sequence(self):
        self.assertEqual(
            staging.contiguous_copy_runs([4, 5, 6], [0, 1]),
            [(0, 2)],
        )
        self.assertEqual(staging.contiguous_copy_runs([], []), [])


if __name__ == "__main__":
    unittest.main()
