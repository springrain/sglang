"""GEC residency planner wiring (doc/KT-CPU-PICE-GPU.md)."""

import types
import unittest
from unittest.mock import patch
from pathlib import Path

from sglang.srt.layers.moe import kt_gec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**overrides):
    cfg = types.SimpleNamespace(
        layer_idx=3,
        kt_expert_gpu_slots=None,
        kt_layer_h2d_slots=None,
        kt_layer_h2d_batch_size=None,
        kt_enable_dynamic_expert_update=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _counts(num_experts, *hot):
    counts = [0] * num_experts
    for expert_id in hot:
        counts[expert_id] = 10
    return counts


class TestStableRowUpdate(unittest.TestCase):
    """Stable row assignment diffing (kt_gec.plan_stable_row_update)."""

    def test_unchanged_set_is_noop(self):
        import torch

        owner = torch.tensor([2, 5, 7], dtype=torch.int64)
        arrived, freed, new_owner = kt_gec.plan_stable_row_update(
            owner, torch.tensor([2, 5, 7], dtype=torch.int64)
        )
        self.assertEqual(arrived.numel(), 0)
        self.assertEqual(freed.numel(), 0)
        self.assertTrue(torch.equal(new_owner, owner))
        # A no-op round must not alias the old owner for in-place mutation.
        self.assertIsNot(new_owner, owner)

    def test_partial_replacement_keeps_rows_stable(self):
        import torch

        owner = torch.tensor([2, 5, 7], dtype=torch.int64)
        # Resident 2 leaves, 9 arrives; rows 1 and 2 must stay put.
        arrived, freed, new_owner = kt_gec.plan_stable_row_update(
            owner, torch.tensor([5, 7, 9], dtype=torch.int64)
        )
        self.assertEqual(arrived.tolist(), [9])
        self.assertEqual(freed.tolist(), [0])
        self.assertEqual(new_owner.tolist(), [9, 5, 7])

    def test_full_replacement_and_determinism(self):
        import torch

        owner = torch.tensor([0, 1, 2], dtype=torch.int64)
        residents = torch.tensor([5, 6, 7], dtype=torch.int64)
        first = kt_gec.plan_stable_row_update(owner, residents)
        second = kt_gec.plan_stable_row_update(owner, residents)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(first[0].tolist(), [5, 6, 7])
        self.assertEqual(first[1].tolist(), [0, 1, 2])
        self.assertEqual(first[2].tolist(), [5, 6, 7])

    def test_resident_set_invariant(self):
        import torch

        owner = torch.tensor([3, 1, 4], dtype=torch.int64)
        residents = torch.tensor([1, 4, 8], dtype=torch.int64)
        _, _, new_owner = kt_gec.plan_stable_row_update(owner, residents)
        self.assertEqual(sorted(new_owner.tolist()), sorted(residents.tolist()))

    def test_size_mismatch_fails_fast(self):
        import torch

        with self.assertRaisesRegex(RuntimeError, "row diff mismatch"):
            kt_gec.plan_stable_row_update(
                torch.tensor([1, 2, 3], dtype=torch.int64),
                torch.tensor([1, 2], dtype=torch.int64),
            )


class TestGecGating(unittest.TestCase):
    def test_disabled_without_slots(self):
        cfg = _config(kt_enable_dynamic_expert_update=True)
        self.assertFalse(kt_gec.gec_enabled(cfg))
        self.assertIsNone(
            kt_gec.build_gec_planner(cfg, num_experts=8, num_gpu_experts=3, tp_rank=0)
        )

    def test_enabled_without_legacy_dynamic_update(self):
        cfg = _config(kt_expert_gpu_slots=2560)
        self.assertTrue(kt_gec.gec_enabled(cfg))

    def test_disabled_for_zero_slots(self):
        cfg = _config(kt_expert_gpu_slots=0)
        self.assertFalse(kt_gec.gec_enabled(cfg))

    def test_disabled_on_nonzero_rank(self):
        cfg = _config(
            kt_expert_gpu_slots=2560, kt_enable_dynamic_expert_update=True
        )
        self.assertTrue(kt_gec.gec_enabled(cfg))
        self.assertIsNone(
            kt_gec.build_gec_planner(cfg, num_experts=8, num_gpu_experts=3, tp_rank=1)
        )

    def test_missing_gec_module_fails_fast(self):
        cfg = _config(kt_expert_gpu_slots=2560)
        with patch.object(kt_gec, "_load_gec_module", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "gec"):
                kt_gec.build_gec_planner(
                    cfg, num_experts=8, num_gpu_experts=3, tp_rank=0
                )


@unittest.skipIf(kt_gec._load_gec_module() is None, "kt_kernel.gec unavailable")
class TestGecDependencyLookup(unittest.TestCase):
    def test_unknown_logical_expert_has_no_pending_dependency(self):
        planner = object.__new__(kt_gec.GecResidencyPlanner)
        planner.layer_idx = 2
        planner.num_experts = 8
        planner._tp_size = 2
        planner._gec = types.SimpleNamespace(
            LogicalExpertId=lambda layer_id, expert_id: (layer_id, expert_id)
        )
        planner._cache = types.SimpleNamespace(snapshot=lambda logical_id: None)

        self.assertEqual(planner._ready_dependencies(3), (0, 0))
        self.assertEqual(planner._ready_dependencies(-1), (0, 0))
        self.assertEqual(planner._ready_dependencies(8), (0, 0))

@unittest.skipIf(kt_gec._load_gec_module() is None, 'kt_kernel.gec unavailable')
class TestGecResidencyPlanner(unittest.TestCase):
    def _planner(self, budget=1):
        return kt_gec.GecResidencyPlanner(
            layer_idx=2,
            num_experts=8,
            capacity=3,
            budget=budget,
            gec_module=kt_gec._load_gec_module(),
        )

    def test_initialize_residents_registers_ready_entries(self):
        planner = self._planner()
        planner.initialize_residents([2, 4])
        for expert_id in (2, 4):
            snapshot = planner._cache.snapshot(planner._id(expert_id))
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.state, planner._gec.LifecycleState.READY)
        self.assertEqual([item.expert_id for item in planner._resident_ids()], [2, 4])
        self.assertTrue(planner._prefilled)

    def test_startup_prefill_fills_capacity(self):
        planner = self._planner()
        # Budget-exempt startup prefill, deterministic layer/expert order.
        self.assertEqual(planner.plan(_counts(8, 5, 6, 7)), [0, 1, 2])

    def test_budget_limits_admissions_and_lru_evicts(self):
        planner = self._planner(budget=1)
        self.assertEqual(planner.plan(_counts(8, 5, 6, 7)), [0, 1, 2])
        # One miss admitted within budget; probationary resident 0 remains
        # protected and the least-recently-used cached resident is evicted.
        self.assertEqual(planner.plan(_counts(8, 7, 1)), [0, 1, 7])
        self.assertEqual(planner.plan(_counts(8, 1, 6, 2)), [0, 1, 2])

    def test_activation_counts_roll_resident_rows(self):
        planner = self._planner(budget=1)
        self.assertEqual(planner.plan(_counts(8, 0, 1, 2)), [0, 1, 2])
        # A newly hot miss enters within the one-expert H2D budget; the
        # previous resident set is retained until the normal LRU decision.
        rolled = planner.plan(_counts(8, 3))
        self.assertEqual(len(rolled), 3)
        self.assertIn(3, rolled)

    def test_gpu_hit_is_not_reconsidered_as_h2d_candidate(self):
        planner = self._planner(budget=1)
        self.assertEqual(planner.plan(_counts(8, 0, 1, 2)), [0, 1, 2])

        # Expert 0 is already resident. The single admission slot must be
        # spent on the CPU miss 3, rather than reloading the GPU hit 0.
        rolled = planner.plan(_counts(8, 0, 3))
        self.assertIn(0, rolled)
        self.assertIn(3, rolled)
        self.assertEqual(len(rolled), 3)

    def test_resident_set_size_invariant(self):
        planner = self._planner(budget=1)
        planner.plan(_counts(8, 5, 6, 7))
        for hot in ((3, 4, 5), (1,), (2, 7)):
            self.assertEqual(len(planner.plan(_counts(8, *hot))), 3)

    def test_record_copy_and_summary(self):
        planner = self._planner()
        planner.plan(_counts(8, 5, 6, 7))
        planner.record_copy(2, 1234)
        summary = planner.summary()
        self.assertIn("GEC layer 2", summary)
        self.assertIn("H2D_RUNNING", summary)


@unittest.skipIf(kt_gec._load_gec_module() is None, "kt_kernel.gec unavailable")
class TestGlobalGecCoordinator(unittest.TestCase):
    def _planner(self, coordinator, layer_idx):
        return kt_gec.GecResidencyPlanner(
            layer_idx=layer_idx,
            num_experts=4,
            capacity=1,
            budget=1,
            gec_module=kt_gec._load_gec_module(),
            coordinator=coordinator,
        )

    def test_layers_share_cache_and_compete_for_global_slot(self):
        module = kt_gec._load_gec_module()
        coordinator = kt_gec.GlobalGecCoordinator(module, capacity=1)
        layer0 = self._planner(coordinator, 0)
        layer1 = self._planner(coordinator, 1)

        self.assertEqual(layer0.plan(_counts(4, 2)), [2])
        self.assertIs(layer0._cache, layer1._cache)
        layer0.complete_h2d([2])
        # The first expert was used during startup and is no longer probationary,
        # so a request from another layer may evict it from the global slot.
        self.assertEqual(layer1.plan(_counts(4, 3)), [3])
        self.assertEqual(coordinator.cache.resident_size(), 1)
        self.assertEqual(coordinator.resident_ids(0), [])
        self.assertEqual([item.expert_id for item in coordinator.resident_ids(1)], [3])

    def test_cpu_execution_lease_releases_without_device_event(self):
        module = kt_gec._load_gec_module()
        import torch

        if torch.cuda.is_available():
            self.skipTest("CPU-only lease behavior is covered without CUDA")
        planner = kt_gec.GecResidencyPlanner(
            layer_idx=0,
            num_experts=4,
            capacity=1,
            budget=1,
            gec_module=module,
        )
        self.assertEqual(planner.plan(_counts(4, 1)), [1])
        lease = planner.begin_gpu_use([1])
        if lease is not None:
            planner.poll_gpu_completions()
        self.assertEqual(planner._cache.refcount(planner._id(1)), 0)

    def test_global_capacity_migrates_between_layers(self):
        module = kt_gec._load_gec_module()
        coordinator = kt_gec.GlobalGecCoordinator(module, capacity=1)
        layer0 = kt_gec.GecResidencyPlanner(
            layer_idx=0,
            num_experts=4,
            capacity=1,
            budget=1,
            gec_module=module,
            coordinator=coordinator,
        )
        layer1 = kt_gec.GecResidencyPlanner(
            layer_idx=1,
            num_experts=4,
            capacity=0,
            budget=1,
            gec_module=module,
            coordinator=coordinator,
        )
        resized = []
        coordinator.register_physical_layer(
            0, layer0, lambda capacity, owner: resized.append((0, capacity, owner))
        )
        coordinator.register_physical_layer(
            1, layer1, lambda capacity, owner: resized.append((1, capacity, owner))
        )
        self.assertEqual(layer0.plan(_counts(4, 2)), [2])
        layer0.complete_h2d([2])
        self.assertEqual(layer1.plan(_counts(4, 3)), [3])
        self.assertEqual(coordinator.layer_capacity(0), 0)
        self.assertEqual(coordinator.layer_capacity(1), 1)
        self.assertIn((0, 0, []), resized)
        self.assertIn((1, 1, []), resized)

    def test_request_concurrency_derives_copy_pool_depth(self):
        module = kt_gec._load_gec_module()
        coordinator = kt_gec.GlobalGecCoordinator(
            module, capacity=4, max_running_requests=17, h2d_batch_size=4
        )
        self.assertEqual(coordinator.buffer_pool_depth, 17)

    def test_default_h2d_budget_uses_layer_capacity(self):
        wrapper = Path(kt_gec.__file__).with_name("kt_gec.py")
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn("default_budget = (", text)
        self.assertIn("if capacity > 0 or coordinator is None", text)


class TestDeviceLocalPipeline(unittest.TestCase):
    def test_buffer_pool_has_bounded_reusable_slots(self):
        pool = kt_gec.GecBufferPool(2)
        self.assertEqual(pool.free_count, 2)
        slot = pool.acquire()
        self.assertEqual(slot.state, "H2D")
        pool.mark_ready(slot, None)
        pool.release_unstarted(slot)
        self.assertEqual(pool.free_count, 2)

    def test_buffer_pool_waits_for_gpu_completion_before_reuse(self):
        class _Event:
            def __init__(self, complete=False):
                self.complete = complete

            def query(self):
                return self.complete

        pool = kt_gec.GecBufferPool(1)
        slot = pool.acquire()
        pool.mark_ready(slot, None)
        pool.begin_gpu_use(slot)
        completion = _Event(False)
        pool.gpu_complete(slot, completion)

        with self.assertRaises(RuntimeError):
            pool.acquire()

        completion.complete = True
        pool.poll()
        recycled = pool.acquire()
        self.assertIs(recycled, slot)
        self.assertEqual(recycled.state, "H2D")

    def test_cuda_pipeline_does_not_drop_unconsumed_slots(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CPU fallback lifecycle is covered without CUDA")
        pipeline = kt_gec.GecCudaPipeline(torch.device("cpu"), depth=1)
        pipeline.submit(lambda: None, wait_on_current=False)
        with self.assertRaises(RuntimeError):
            pipeline.submit(lambda: None, wait_on_current=False)
        pipeline.release_ready()
        pipeline.submit(lambda: None, wait_on_current=False)

    def test_cpu_copy_failure_releases_slot(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CPU fallback lifecycle is covered without CUDA")
        pipeline = kt_gec.GecCudaPipeline(torch.device("cpu"), depth=1)
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            pipeline.submit(
                lambda: (_ for _ in ()).throw(RuntimeError("copy failed")),
                wait_on_current=False,
            )
        self.assertEqual(pipeline.pool.free_count, 1)

    def test_cpu_fallback_executes_operation(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CPU fallback path is only meaningful without CUDA")
        pipeline = kt_gec.GecCudaPipeline(torch.device("cpu"), depth=2)
        observed = []
        self.assertIsNone(
            pipeline.submit(lambda: observed.append("copy"), wait_on_current=False)
        )
        self.assertEqual(observed, ["copy"])

class TestRuntimeContract(unittest.TestCase):
    def test_mxfp4_gec_uses_persistent_resident_rows(self):
        wrapper = Path(kt_gec.__file__).with_name("kt_ep_wrapper.py")
        text = wrapper.read_text(encoding="utf-8")
        apply_start = text.index(
            "    def apply(\n", text.index("class KTEPWrapperMethod")
        )
        apply_end = text.index("    def _select_experts_with_gec(", apply_start)
        apply_text = text[apply_start:apply_end]
        self.assertIn("_gec_persistent_mxfp4", apply_text)
        self.assertIn("_full_gpu_gate", apply_text)
        self.assertIn("ctx.copy_selected_cpu_weights", text)
        self.assertIn("prepare_v4_mxfp4_marlin", text)

    def test_gec_streaming_has_batch_events_and_batched_cpu_drain(self):
        wrapper = Path(kt_gec.__file__).with_name("kt_ep_wrapper.py")
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn("cpu_buffer_depth", text)
        self.assertIn("wrapper.sync_write_weight_scale_to_buffer()", text)
        self.assertIn("stream_batches", text)
        self.assertIn("wait_on_current=not stream_batches", text)
        self.assertIn("stream_batches=bool(self._gec_enabled)", text)
        self.assertIn('configured_batch = getattr(self.kt_config, "kt_layer_h2d_batch_size", None)', text)
        self.assertIn("copy_batch = (", text)
        self.assertIn("stream_batch_events", text)

    def test_int4_gec_uses_selected_row_cpu_copy(self):
        wrapper = Path(kt_gec.__file__).with_name("kt_ep_wrapper.py")
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn("def _copy_selected_cpu_int4_weights(", text)
        self.assertIn("if getattr(self, \"_is_int4_quant\", False):", text)
        self.assertIn("refusing full-shadow fallback", text)
        self.assertIn("_SHARED_GEC_CPU_CONTEXT", text)
        self.assertIn("global_num_experts=max(1, self.num_gpu_experts)", text)

    def test_gec_residency_update_precedes_cpu_submit(self):
        wrapper = Path(kt_gec.__file__).with_name("kt_ep_wrapper.py")
        text = wrapper.read_text(encoding="utf-8")
        apply_start = text.index(
            "    def apply(\n", text.index("class KTEPWrapperMethod")
        )
        hybrid_start = text.index(
            "        # GEC must update the resident set", apply_start
        )
        update = text.index("self._update_gpu_experts_from_batch(", hybrid_start)
        submit = text.index("self._submit_with_staged_input(", hybrid_start)
        self.assertLess(update, submit)
        self.assertIn("mask_gpu_expert_ids_for_cpu", text)

    def test_dynamic_update_has_no_process_wide_synchronize(self):
        wrapper = Path(kt_gec.__file__).with_name("kt_ep_wrapper.py")
        text = wrapper.read_text(encoding="utf-8")
        start = text.index("    def _update_gpu_experts_from_batch(")
        end = text.index("    def __getattr__", start)
        self.assertNotIn("torch.cuda.synchronize", text[start:end])
        self.assertIn("GecCudaPipeline", text[start:end])


if __name__ == "__main__":
    unittest.main()
