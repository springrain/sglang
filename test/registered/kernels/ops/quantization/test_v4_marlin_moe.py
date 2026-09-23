import pytest
import torch
from sglang.kernels.ops.quantization.gptq_marlin_repack import (
    gptq_marlin_repack,
    mxfp4_marlin_repack,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_v4_direct_repack_matches_gptq_transpose_bitwise():
    for k, n in ((128, 128), (256, 192)):
        raw = torch.randint(0, 256, (3, n, k // 2), dtype=torch.uint8, device="cuda")
        direct = mxfp4_marlin_repack(raw, k, n)
        perm = torch.empty(0, dtype=torch.int32, device="cuda")
        reference = torch.stack(
            [
                gptq_marlin_repack(
                    raw[expert].view(torch.int32).view(n, k // 8).T.contiguous(),
                    perm,
                    k,
                    n,
                    4,
                )
                for expert in range(raw.shape[0])
            ]
        )
        torch.testing.assert_close(direct, reference, rtol=0, atol=0)


def _reference_scales(scales, k, n):
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales

    result = []
    for expert in range(scales.shape[0]):
        value = scales[expert].to(torch.float32).T.contiguous()
        value = marlin_permute_scales(value, k, n, 32)
        value = value.view(-1, 4)[:, [0, 2, 1, 3]].reshape(k // 32, n)
        result.append(value.to(torch.float8_e8m0fnu))
    return torch.stack(result)


@pytest.mark.parametrize("input_is_e8m0", [False, True])
def test_v4_scale_swizzle_matches_reference_bitwise(input_is_e8m0):
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        _swizzle_e8m0_scales,
    )

    e, k, n = 3, 256, 192
    exponents = torch.randint(-8, 8, (e, n, k // 32), device="cuda")
    scales = torch.exp2(exponents.float())
    if input_is_e8m0:
        scales = scales.to(torch.float8_e8m0fnu)
    actual = _swizzle_e8m0_scales(scales, size_k=k, size_n=n)
    reference = _reference_scales(scales, k, n)
    torch.testing.assert_close(
        actual.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
    )


def _dequantize(raw, scale):
    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

    return MXFP4QuantizeUtil.dequantize(raw, torch.bfloat16, scale, [32])


def _make_v4_raw_weights(experts, hidden_size, intermediate_size):
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    s13 = torch.randint(
        120,
        132,
        (experts, 2 * intermediate_size, hidden_size // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    s2 = torch.randint(
        120,
        132,
        (experts, hidden_size, intermediate_size // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    return w13, s13, w2, s2


def _assert_prepared_equal(actual, expected):
    torch.testing.assert_close(actual.w13, expected.w13, rtol=0, atol=0)
    torch.testing.assert_close(actual.w2, expected.w2, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.w13_scale.view(torch.uint8),
        expected.w13_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.w2_scale.view(torch.uint8),
        expected.w2_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_v4_single_expert_prepare_only_writes_target_slot():
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        allocate_v4_mxfp4_marlin,
        prepare_v4_mxfp4_marlin,
        prepare_v4_mxfp4_marlin_expert,
    )

    torch.manual_seed(11)
    experts, k, n = 4, 128, 128
    w13, s13, w2, s2 = _make_v4_raw_weights(experts, k, n)
    prepared = allocate_v4_mxfp4_marlin(
        num_experts=experts,
        hidden_size=k,
        intermediate_size=n,
        device=w13.device,
    )
    prepared.w13.fill_(0x13579B)
    prepared.w2.fill_(-0x2468A)
    prepared.w13_scale.view(torch.uint8).fill_(73)
    prepared.w2_scale.view(torch.uint8).fill_(91)
    before = tuple(
        tensor.clone()
        for tensor in (
            prepared.w13,
            prepared.w13_scale.view(torch.uint8),
            prepared.w2,
            prepared.w2_scale.view(torch.uint8),
        )
    )

    source_expert = 3
    target_slot = 1
    prepared_slot = prepare_v4_mxfp4_marlin_expert(
        w13[source_expert],
        s13[source_expert],
        w2[source_expert],
        s2[source_expert],
        out=prepared,
        slot_index=target_slot,
    )
    reference = prepare_v4_mxfp4_marlin(
        w13[source_expert : source_expert + 1],
        s13[source_expert : source_expert + 1],
        w2[source_expert : source_expert + 1],
        s2[source_expert : source_expert + 1],
    )
    _assert_prepared_equal(prepared_slot, reference)

    assert prepared_slot.num_experts == 1
    assert prepared_slot.w13.data_ptr() == prepared.w13[target_slot].data_ptr()
    after = (
        prepared.w13,
        prepared.w13_scale.view(torch.uint8),
        prepared.w2,
        prepared.w2_scale.view(torch.uint8),
    )
    for before_tensor, after_tensor in zip(before, after):
        for slot in range(experts):
            if slot != target_slot:
                torch.testing.assert_close(
                    after_tensor[slot], before_tensor[slot], rtol=0, atol=0
                )


def test_v4_streamed_ready_group_matches_filtered_full_layer():
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        allocate_v4_mxfp4_marlin,
        apply_v4_marlin_moe,
        apply_v4_mxfp4_marlin_streamed_experts,
        prepare_v4_mxfp4_marlin,
        prepare_v4_mxfp4_marlin_expert,
    )

    torch.manual_seed(13)
    experts, m, k, n, topk = 4, 7, 128, 128, 2
    w13, s13, w2, s2 = _make_v4_raw_weights(experts, k, n)
    full = prepare_v4_mxfp4_marlin(w13, s13, w2, s2)
    candidates = (3, 1)
    staging = allocate_v4_mxfp4_marlin(
        num_experts=len(candidates),
        hidden_size=k,
        intermediate_size=n,
        device=w13.device,
    )
    for staging_slot, logical_expert in enumerate(candidates):
        prepare_v4_mxfp4_marlin_expert(
            w13[logical_expert],
            s13[logical_expert],
            w2[logical_expert],
            s2[logical_expert],
            out=staging,
            slot_index=staging_slot,
        )

    hidden_states = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * 0.01
    topk_ids = torch.tensor(
        [[3, 0], [1, 2], [0, 2], [3, 1], [2, 3], [-1, 1], [0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.rand((m, topk), dtype=torch.float32, device="cuda")
    topk_weights[topk_ids < 0] = 0

    candidate_mask = torch.zeros_like(topk_ids, dtype=torch.bool)
    for logical_expert in candidates:
        candidate_mask |= topk_ids.eq(logical_expert)
    filtered_ids = topk_ids.masked_fill(~candidate_mask, -1)
    filtered_weights = topk_weights.masked_fill(~candidate_mask, 0)
    reference = apply_v4_marlin_moe(
        hidden_states=hidden_states,
        prepared=full,
        topk_weights=filtered_weights,
        topk_ids=filtered_ids,
        routed_scaling_factor=0.75,
    ).clone()

    caller_owned_output = torch.empty_like(hidden_states)
    actual = apply_v4_mxfp4_marlin_streamed_experts(
        hidden_states=hidden_states,
        prepared_staging=staging,
        logical_expert_ids=candidates,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        routed_scaling_factor=0.75,
        out=caller_owned_output,
    )
    assert actual.data_ptr() == caller_owned_output.data_ptr()
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.05)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.float().flatten(), dim=0
    )
    assert cosine > 0.999


def test_v4_streamed_contract_rejects_invalid_candidate_mapping():
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        remap_v4_mxfp4_streamed_assignments,
    )

    ids = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
    weights = torch.ones_like(ids, dtype=torch.float32)
    with pytest.raises(ValueError, match="unique"):
        remap_v4_mxfp4_streamed_assignments(
            topk_ids=ids,
            topk_weights=weights,
            logical_expert_ids=(1, 1),
        )


def test_v4_streaming_capability_fails_fast_for_non_cuda_device():
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        get_v4_mxfp4_marlin_streaming_capability,
        require_v4_mxfp4_marlin_streaming_support,
    )

    capability = get_v4_mxfp4_marlin_streaming_capability(torch.device("cpu"))
    assert not capability.available
    assert capability.single_expert_prepare
    assert capability.caller_owned_output
    assert capability.streamed_ready_group
    with pytest.raises(RuntimeError, match="CUDA device"):
        require_v4_mxfp4_marlin_streaming_support(torch.device("cpu"))


@pytest.mark.parametrize("swiglu_limit", [None, 2.0])
def test_v4_marlin_moe_reference_determinism_and_graph(swiglu_limit):
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        apply_v4_marlin_moe,
        prepare_v4_mxfp4_marlin,
    )

    torch.manual_seed(7)
    e, m, k, n, topk = 4, 5, 128, 128, 2
    w13 = torch.randint(0, 256, (e, 2 * n, k // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((e, 2 * n, k // 32), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((e, k, n // 32), 127, dtype=torch.uint8, device="cuda")
    prepared = prepare_v4_mxfp4_marlin(w13, s13, w2, s2)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            prepared.w13,
            prepared.w13_scale,
            prepared.w2,
            prepared.w2_scale,
        )
    )
    prepare_v4_mxfp4_marlin(w13, s13, w2, s2, out=prepared)
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            prepared.w13,
            prepared.w13_scale,
            prepared.w2,
            prepared.w2_scale,
        )
    )

    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * 0.01
    ids = torch.tensor(
        [[0, 1], [2, 3], [1, -1], [3, 0], [2, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    gates = torch.rand((m, topk), dtype=torch.float32, device="cuda")
    gates[ids < 0] = 0

    dw13 = _dequantize(w13, s13)
    dw2 = _dequantize(w2, s2)
    reference_fp32 = torch.zeros_like(x, dtype=torch.float32)
    for token in range(m):
        for slot in range(topk):
            expert = int(ids[token, slot])
            if expert < 0:
                continue
            first = (
                (x[token].float() @ dw13[expert].float().T).to(torch.bfloat16).float()
            )
            gate, up = first[:n], first[n:]
            if swiglu_limit is not None:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            expert_output = (
                (activated.float() @ dw2[expert].float().T) * gates[token, slot]
            ).to(torch.bfloat16)
            reference_fp32[token] += expert_output.float()

    kwargs = {
        "hidden_states": x,
        "prepared": prepared,
        "topk_weights": gates,
        "topk_ids": ids,
        "routed_scaling_factor": 0.75,
        "swiglu_limit": swiglu_limit,
    }
    reference = (reference_fp32 * 0.75).to(torch.bfloat16)
    actual = apply_v4_marlin_moe(**kwargs).clone()
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.05)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.float().flatten(), dim=0
    )
    assert cosine > 0.999

    for _ in range(20):
        repeated = apply_v4_marlin_moe(**kwargs).clone()
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_v4_marlin_moe(**kwargs)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
