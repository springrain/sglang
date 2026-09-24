# SPDX-License-Identifier: Apache-2.0
"""CPU-visible contracts for KT's V4 MXFP4 streamed-expert backend."""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


SOURCE_PATH = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/quantization/v4_marlin_moe.py"
)


def _load_contract_symbols():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    wanted_functions = {
        "get_v4_mxfp4_marlin_streaming_capability",
        "normalize_v4_mxfp4_activation",
        "require_v4_mxfp4_marlin_streaming_support",
        "remap_v4_mxfp4_streamed_assignments",
    }
    wanted_assignments = {
        "_KT_V4_MARLIN_STREAMING_COMPUTE_CAPABILITIES",
        "_V4_MARLIN_ACTIVATION_TYPES",
    }
    selected = []
    for node in tree.body:
        is_target = (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in wanted_assignments
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.ClassDef)
            and node.name == "V4MarlinStreamingCapability"
        ) or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted_functions
        )
        if is_target:
            selected.append(copy.deepcopy(node))

    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        "Sequence": Sequence,
        "dataclass": dataclass,
        "math": math,
        "torch": torch,
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"activation": "silu", "swiglu_limit": 10.0},
            {
                "activation_type": 0,
                "swiglu_limit": 10.0,
                "swiglu_alpha": 0.0,
                "situ_beta": 0.0,
                "situ_linear_beta": 0.0,
            },
        ),
        (
            {
                "activation": "swiglu_oai",
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
            },
            {
                "activation_type": 1,
                "swiglu_limit": 7.0,
                "swiglu_alpha": 1.702,
                "situ_beta": 0.0,
                "situ_linear_beta": 0.0,
            },
        ),
        (
            {
                "activation": "situ",
                "situ_beta": 4.0,
                "situ_linear_beta": 25.0,
            },
            {
                "activation_type": 2,
                "swiglu_limit": 0.0,
                "swiglu_alpha": 0.0,
                "situ_beta": 4.0,
                "situ_linear_beta": 25.0,
            },
        ),
    ],
)
def test_v4_activation_contract_normalizes_all_kt_modes(kwargs, expected):
    normalize = _load_contract_symbols()["normalize_v4_mxfp4_activation"]
    assert normalize(**kwargs) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"activation": "silu", "swiglu_alpha": 1.0},
        {"activation": "swiglu_oai", "swiglu_alpha": 0.0},
        {"activation": "situ", "situ_beta": 0.0},
        {"activation": "situ", "situ_beta": 4.0, "swiglu_limit": 1.0},
    ],
)
def test_v4_activation_contract_rejects_ambiguous_combinations(kwargs):
    normalize = _load_contract_symbols()["normalize_v4_mxfp4_activation"]
    with pytest.raises(ValueError):
        normalize(**kwargs)


def test_streamed_assignment_remap_is_exact_once_on_cpu():
    remap = _load_contract_symbols()["remap_v4_mxfp4_streamed_assignments"]
    ids = torch.tensor(
        [[3, 0], [1, 2], [3, 1], [-1, 1]],
        dtype=torch.int32,
    )
    weights = torch.tensor(
        [[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.0, 1.0]],
        dtype=torch.float32,
    )

    local_ids, streamed_weights = remap(
        topk_ids=ids,
        topk_weights=weights,
        logical_expert_ids=(3, 1),
    )

    assert torch.equal(
        local_ids,
        torch.tensor([[0, -1], [1, -1], [0, 1], [-1, 1]], dtype=torch.int32),
    )
    assert torch.equal(
        streamed_weights,
        torch.tensor(
            [[0.9, 0.0], [0.6, 0.0], [0.7, 0.3], [0.0, 1.0]],
            dtype=torch.float32,
        ),
    )


def test_streamed_assignment_remap_rejects_duplicate_candidates():
    remap = _load_contract_symbols()["remap_v4_mxfp4_streamed_assignments"]
    ids = torch.tensor([[1, 2]], dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    with pytest.raises(ValueError, match="unique"):
        remap(
            topk_ids=ids,
            topk_weights=weights,
            logical_expert_ids=(1, 1),
        )


def test_v4_streaming_capability_rejects_cpu_device():
    symbols = _load_contract_symbols()
    get_capability = symbols["get_v4_mxfp4_marlin_streaming_capability"]
    require_support = symbols["require_v4_mxfp4_marlin_streaming_support"]

    capability = get_capability(torch.device("cpu"))
    assert not capability.available
    assert capability.single_expert_prepare
    assert capability.caller_owned_output
    assert capability.streamed_ready_group

    with pytest.raises(RuntimeError, match="CUDA"):
        require_support(torch.device("cpu"))
