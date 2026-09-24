from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    collect_kt_cpu_only_routed_expert_layers,
    routed_expert_layer_index,
    safetensors_weights_iterator,
    should_materialize_routed_expert_tensor,
)


class _FakeModel:
    def __init__(self, quant_methods):
        self._modules = [
            SimpleNamespace(quant_method=method) for method in quant_methods
        ]

    def modules(self):
        yield self
        yield from self._modules


def _kt_method(layer_idx, num_gpu_experts):
    return SimpleNamespace(
        _quant_wrapper_id="kt_ep",
        kt_config=SimpleNamespace(layer_idx=layer_idx),
        num_gpu_experts=num_gpu_experts,
    )


class _FakeSafeOpen:
    def __init__(self, names, materialized):
        self._names = names
        self._materialized = materialized

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def keys(self):
        return list(self._names)

    def get_tensor(self, name):
        self._materialized.append(name)
        return torch.tensor([len(name)], dtype=torch.int64)


def test_routed_expert_name_predicate_is_conservative():
    skipped = frozenset({10})

    routed_layer_10 = [
        "language_model.model.layers.10.block_sparse_moe.experts.7.w1.weight_packed",
        "model.layers.10.mlp.experts.8.gate_proj.weight_scale",
        "layers.10.ffn.experts.9.w2.scale",
    ]
    for name in routed_layer_10:
        assert routed_expert_layer_index(name) == 10
        assert not should_materialize_routed_expert_tensor(name, skipped)

    preserved = [
        # GPU-pinned / resident layer: not in the explicit skip set.
        "language_model.model.layers.9.block_sparse_moe.experts.7.w1.weight_packed",
        # Shared experts and routers are not routed per-expert tensors.
        "model.layers.10.mlp.shared_experts.gate_proj.weight",
        "model.layers.10.mlp.gate.e_score_correction_bias",
        # Dense and non-layer weights remain untouched.
        "model.layers.10.mlp.gate_proj.weight",
        "model.embed_tokens.weight",
        # Same numeric layer in MTP / another tower must not be conflated with
        # the K3 main-language layer.
        "model.mtp.layers.10.mlp.experts.7.w1.weight_packed",
        "vision_model.layers.10.mlp.experts.7.w1.weight_packed",
        "encoder.layers.10.mlp.experts.7.w1.weight_packed",
    ]
    for name in preserved:
        assert should_materialize_routed_expert_tensor(name, skipped)

    assert should_materialize_routed_expert_tensor(routed_layer_10[0], frozenset())


def test_collects_only_actual_zero_resident_kt_layers():
    ordinary_method = SimpleNamespace(
        _quant_wrapper_id="other",
        kt_config=SimpleNamespace(layer_idx=12),
        num_gpu_experts=0,
    )
    model = _FakeModel(
        [
            _kt_method(layer_idx=10, num_gpu_experts=0),
            _kt_method(layer_idx=11, num_gpu_experts=1),
            _kt_method(layer_idx=12, num_gpu_experts=8),
            ordinary_method,
            None,
        ]
    )

    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"]
        )
    )

    assert collect_kt_cpu_only_routed_expert_layers(
        model, model_config
    ) == frozenset({10})


def test_other_models_do_not_enable_global_expert_filtering():
    model = _FakeModel([_kt_method(layer_idx=10, num_gpu_experts=0)])
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LagunaForCausalLM"])
    )

    assert not collect_kt_cpu_only_routed_expert_layers(model, model_config)


def _exercise_iterator(iterator_factory):
    names = [
        "language_model.model.layers.9.block_sparse_moe.experts.0.w1.weight_packed",
        "language_model.model.layers.10.block_sparse_moe.experts.0.w1.weight_packed",
        "language_model.model.layers.10.block_sparse_moe.experts.0.w1.weight_scale",
        "language_model.model.layers.10.block_sparse_moe.gate.weight",
    ]
    materialized = []

    def fake_safe_open(*args, **kwargs):
        return _FakeSafeOpen(names, materialized)

    def tensor_filter(name):
        return should_materialize_routed_expert_tensor(name, frozenset({10}))

    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch(
            "sglang.srt.model_loader.weight_utils.safetensors.safe_open",
            side_effect=fake_safe_open,
        ),
    ):
        loaded = dict(iterator_factory(tensor_filter))

    expected = {names[0], names[3]}
    assert set(loaded) == expected
    # The skipped K3 expert weight and scale must never reach get_tensor().
    assert set(materialized) == expected


def test_single_thread_filter_runs_before_get_tensor():
    _exercise_iterator(
        lambda tensor_filter: safetensors_weights_iterator(
            ["model.safetensors"], tensor_filter=tensor_filter
        )
    )


def test_buffered_filter_runs_before_get_tensor():
    _exercise_iterator(
        lambda tensor_filter: buffered_multi_thread_safetensors_weights_iterator(
            ["model.safetensors"], max_workers=1, tensor_filter=tensor_filter
        )
    )


def test_default_model_source_captures_actual_kt_placement():
    model = _FakeModel(
        [
            _kt_method(layer_idx=10, num_gpu_experts=0),
            _kt_method(layer_idx=11, num_gpu_experts=1),
        ]
    )
    model.fall_back_to_pt_during_load = True
    model.allow_patterns_overrides = None
    model_config = SimpleNamespace(
        model_path="/model",
        revision=None,
        hf_config=SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"]
        ),
    )

    source = DefaultModelLoader.Source.init_new(model_config, model)

    assert source.kt_cpu_only_routed_expert_layers == frozenset({10})


def test_default_loader_wires_filter_only_for_cpu_only_kt_layers():
    loader = DefaultModelLoader(
        LoadConfig(
            load_format=LoadFormat.SAFETENSORS,
            model_loader_extra_config={"enable_multithread_load": False},
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path="/model",
        revision=None,
        fall_back_to_pt=False,
        kt_cpu_only_routed_expert_layers=frozenset({10}),
    )
    runtime_model_args = SimpleNamespace(
        weight_loader_disable_mmap=False,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=4,
        weight_loader_drop_cache_after_load=False,
    )

    with (
        patch.object(
            DefaultModelLoader,
            "_prepare_weights",
            return_value=("/model", ["model.safetensors"], True),
        ),
        patch(
            "sglang.srt.model_loader.loader.get_server_args",
            return_value=SimpleNamespace(kt_weight_path="/model"),
        ),
        patch(
            "sglang.srt.model_loader.loader.get_model",
            return_value=runtime_model_args,
        ),
        patch(
            "sglang.srt.model_loader.loader.safetensors_weights_iterator",
            return_value=iter(()),
        ) as iterator,
    ):
        list(loader._get_weights_iterator(source))

    tensor_filter = iterator.call_args.kwargs["tensor_filter"]
    assert not tensor_filter(
        "language_model.model.layers.10.block_sparse_moe.experts.0.w1.weight_packed"
    )
    assert tensor_filter(
        "language_model.model.layers.11.block_sparse_moe.experts.0.w1.weight_packed"
    )


def test_default_loader_non_kt_call_has_no_filter_kwarg():
    loader = DefaultModelLoader(
        LoadConfig(
            load_format=LoadFormat.SAFETENSORS,
            model_loader_extra_config={"enable_multithread_load": False},
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path="/model",
        revision=None,
        fall_back_to_pt=False,
    )
    runtime_model_args = SimpleNamespace(
        weight_loader_disable_mmap=False,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=4,
        weight_loader_drop_cache_after_load=False,
    )

    with (
        patch.object(
            DefaultModelLoader,
            "_prepare_weights",
            return_value=("/model", ["model.safetensors"], True),
        ),
        patch(
            "sglang.srt.model_loader.loader.get_server_args",
            return_value=SimpleNamespace(kt_weight_path=None),
        ),
        patch(
            "sglang.srt.model_loader.loader.get_model",
            return_value=runtime_model_args,
        ),
        patch(
            "sglang.srt.model_loader.loader.safetensors_weights_iterator",
            return_value=iter(()),
        ) as iterator,
    ):
        list(loader._get_weights_iterator(source))

    assert "tensor_filter" not in iterator.call_args.kwargs


def test_fast_safetensors_fails_before_copying_cpu_only_expert_shards():
    loader = DefaultModelLoader(
        LoadConfig(load_format=LoadFormat.FASTSAFETENSORS)
    )
    source = DefaultModelLoader.Source(
        model_or_path="/model",
        revision=None,
        fall_back_to_pt=False,
        kt_cpu_only_routed_expert_layers=frozenset({10}),
    )

    with patch.object(
        DefaultModelLoader,
        "_prepare_weights",
        return_value=("/model", ["model.safetensors"], True),
    ):
        try:
            list(loader._get_weights_iterator(source))
        except ValueError as exc:
            assert "fastsafetensors" in str(exc)
            assert "standard safetensors loader" in str(exc)
        else:
            raise AssertionError("expected KT fastsafetensors compatibility error")
