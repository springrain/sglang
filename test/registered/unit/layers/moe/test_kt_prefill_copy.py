"""Unit coverage for the optional KT prefill copy backends."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

kt_copy = pytest.importorskip("sglang.srt.layers.moe.kt_prefill_copy")


@pytest.fixture(autouse=True)
def _reset_fast_path_state():
    kt_copy.reset_fast_path_cache()
    yield
    kt_copy.reset_fast_path_cache()


def test_row_nbytes_requires_a_contiguous_payload():
    tensor = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)

    assert kt_copy.row_nbytes(tensor) == 4 * 5 * 4
    assert kt_copy.row_nbytes(tensor.transpose(0, 1)) is None
    assert kt_copy.row_nbytes(torch.empty(0, 4)) is None


def test_row_copy_plan_uses_identity_rows_for_the_full_shadow():
    named = [
        ("w13_weight", torch.zeros(4, 2, 2), torch.zeros(3, 2, 2)),
        ("w2_weight", torch.zeros(4, 2, 2), torch.zeros(3, 2, 2)),
    ]

    plan = kt_copy.build_row_copy_plan(
        named,
        [(3, 0), (1, 2)],
        num_experts=4,
        old_row_owner=(3, 1),
        w13_up_first=True,
    )

    assert plan is not None
    assert plan.source_rows == (3, 1)
    assert plan.destination_rows == (0, 2)
    assert plan.banks[0].swap_halves
    assert not plan.banks[1].swap_halves


def test_row_copy_plan_resolves_compact_canonical_rows():
    named = [
        ("w13_weight", torch.zeros(2, 2, 2), torch.zeros(2, 2, 2)),
        ("w2_weight", torch.zeros(2, 2, 2), torch.zeros(2, 2, 2)),
    ]

    plan = kt_copy.build_row_copy_plan(
        named,
        [(3, 0), (1, 1)],
        num_experts=4,
        old_row_owner=(3, 1),
        w13_up_first=False,
    )

    assert plan is not None
    assert plan.source_rows == (0, 1)
    assert plan.destination_rows == (0, 1)


def test_row_copy_plan_rejects_mixed_source_row_conventions():
    named = [
        ("w13_weight", torch.zeros(4, 2, 2), torch.zeros(2, 2, 2)),
        ("w2_weight", torch.zeros(2, 2, 2), torch.zeros(2, 2, 2)),
    ]

    plan = kt_copy.build_row_copy_plan(
        named,
        [(3, 0), (1, 1)],
        num_experts=4,
        old_row_owner=(3, 1),
        w13_up_first=False,
    )

    assert plan is None


def test_fast_paths_reject_cpu_tensors_without_loading_a_kernel(monkeypatch):
    def _unexpected_load():
        raise AssertionError("CPU fallback must not load the CUDA JIT module")

    monkeypatch.setattr(kt_copy, "_load_module", _unexpected_load)
    source = torch.zeros(2, 2)
    destination = torch.zeros(2, 2)

    assert not kt_copy.try_multi_row_copy(
        [kt_copy.RowCopyBank(source=source, destination=destination)],
        [0],
        [0],
        torch.device("cpu"),
    )
    assert not kt_copy.try_batch_memcpy(
        [
            kt_copy.BatchCopy(
                source=source,
                source_row=0,
                destination=destination,
                destination_row=0,
                row_bytes=16,
                rows=1,
            )
        ],
        SimpleNamespace(cuda_stream=1),
    )


def test_environment_kill_switch_short_circuits_both_paths(monkeypatch):
    monkeypatch.setenv("SGLANG_KT_DISABLE_PREFILL_FAST_COPY", "1")
    source = torch.zeros(2, 2)
    destination = torch.zeros(2, 2)

    assert not kt_copy.try_multi_row_copy(
        [kt_copy.RowCopyBank(source=source, destination=destination)],
        [0],
        [0],
        torch.device("cpu"),
    )
    assert not kt_copy.try_batch_memcpy([], SimpleNamespace(cuda_stream=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_multi_row_copy_matches_torch():
    device = torch.device("cuda")
    source = torch.arange(4 * 64, dtype=torch.uint8, device=device).reshape(4, 64)
    destination = torch.zeros(2, 64, dtype=torch.uint8, device=device)

    if not kt_copy.try_multi_row_copy(
        [kt_copy.RowCopyBank(source=source, destination=destination)],
        [3, 1],
        [0, 1],
        device,
    ):
        pytest.skip("fused row copy backend is unavailable")
    torch.cuda.synchronize(device)

    assert torch.equal(destination[0], source[3])
    assert torch.equal(destination[1], source[1])


@pytest.mark.skipif(
    not torch.cuda.is_available() or (torch.version.cuda or "").split(".")[0] != "13",
    reason="needs CUDA 13 batch memcpy",
)
def test_cuda_batch_memcpy_matches_torch():
    device = torch.device("cuda")
    source = torch.arange(2 * 32, dtype=torch.uint8).pin_memory().reshape(2, 32)
    destination = torch.zeros(3, 32, dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        accepted = kt_copy.try_batch_memcpy(
            [
                kt_copy.BatchCopy(
                    source=source,
                    source_row=1,
                    destination=destination,
                    destination_row=2,
                    row_bytes=32,
                    rows=1,
                )
            ],
            stream,
        )
    if not accepted:
        pytest.skip("batch H2D backend is unavailable")
    stream.synchronize()

    assert torch.equal(destination[2], source[1])
