from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn import Module

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0, round_up, set_weight_attrs
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


def build_marlin_moe_quant_info(layer: Module) -> MarlinMoeQuantInfo:
    """Build the Marlin quant_info for an MXFP4 MoE layer.

    Single source for the runner inputs shared by the marlin path of
    ``Mxfp4MoEMethod.apply`` and :class:`Mxfp4MarlinMoEMethod`, including
    the dispatcher's EP mapping (global -> local expert ids) when EP is on.
    """
    expert_map = getattr(layer.dispatcher, "local_expert_mapping", None)
    global_num_experts = layer.dispatcher.num_experts if expert_map is not None else -1
    return MarlinMoeQuantInfo(
        w13_qweight=layer.w13_weight,
        w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale,
        w2_scales=layer.w2_weight_scale,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
        w13_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        # KT's layerwise prefill manager uses a stable, caller-owned Marlin
        # image while retaining the native checkpoint tensors for the next
        # layer.  The flag is set by kt_ep_wrapper after it selects this
        # backend; ordinary SGLang Marlin keeps its existing in-place path.
        self._kt_layerwise_enabled = False

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        # KT's prepared-weight layerwise path calls ``apply_v4_marlin_moe``
        # directly.  Constructing the generic MoeRunner here is unnecessary
        # and can fail when the process uses an A2A backend without a MARLIN
        # fused-op registration.  Keep the normal runner for non-KT Marlin.
        if self._kt_layerwise_enabled:
            self.runner = None
            return

        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        layer._dsv4_mxfp4_backend = None  # set in process_weights_after_loading
        fp4_block_k = 32
        intermediate_size_per_partition = round_up(intermediate_size_per_partition, 128)
        hidden_size = round_up(hidden_size, 256)
        self.hidden_pad = hidden_size - layer.hidden_size

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # The upstream Marlin path stores E8M0 scales as float8 bytes.  The
        # KT layerwise shadow keeps the checkpoint-native numeric scale in
        # float32, matching the old KT loader and allowing BF16 shared-memory
        # staging to be copied without depending on BF16 -> float8 copy
        # support.  `prepare_v4_mxfp4_marlin` converts the shadow scales to
        # the compact Marlin float8 layout before execution.
        def _scale_ones(*shape: int) -> torch.Tensor:
            if self._kt_layerwise_enabled:
                return torch.ones(shape, dtype=torch.float32)
            return torch.full(shape, 127, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            )

        w13_weight_scale = torch.nn.Parameter(
            _scale_ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            _scale_ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            deinterleave_moe_mxfp4_w13_for_marlin,
            prepare_moe_mxfp4_layer_for_marlin,
        )

        if self._kt_layerwise_enabled:
            from sglang.srt.layers.quantization.v4_marlin_moe import (
                prepare_v4_mxfp4_marlin,
            )

            capability = torch.cuda.get_device_capability(layer.w13_weight.device)
            if capability not in ((8, 9), (12, 0)):
                raise RuntimeError(
                    "KT MXFP4 layerwise Marlin requires SM89 or SM120, got "
                    f"SM{capability[0]}{capability[1]}."
                )
            raw_names = (
                "w13_weight",
                "w13_weight_scale_inv",
                "w2_weight",
                "w2_weight_scale_inv",
            )
            if not all(hasattr(layer, name) for name in raw_names):
                raise RuntimeError(
                    "KT MXFP4 layerwise requires native raw w13/w2 weights "
                    "and E8M0 scales before Marlin preparation."
                )
            layer._v4_marlin_weights = prepare_v4_mxfp4_marlin(
                layer.w13_weight.data,
                layer.w13_weight_scale_inv.data,
                layer.w2_weight.data,
                layer.w2_weight_scale_inv.data,
            )
            layer._v4_marlin_path = True
            layer._dsv4_mxfp4_backend = "kt_marlin"
            # Keep the raw attributes alive.  KTEP uses them as the resident
            # expert source while the layerwise manager transports CPU experts
            # into its alternate slot.
            return

        # Let the FP8 base method handle ROCm normalization, etc.  The KT
        # branch above intentionally consumes the native checkpoint layout
        # before any backend-specific in-place shuffle or scale conversion.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        if not is_sm90_supported() and not is_sm120_supported():
            raise RuntimeError("MXFP4 Marlin requires SM90 or SM120.")

        if not check_moe_marlin_supports_layer(layer, 32, allow_tile_padding=True):
            raise RuntimeError(
                "Current MXFP4 MoE layer does not satisfy Marlin constraints."
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]``
        # (first half) and ``up = intermediate[:, N:]`` (second half).
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]),
        # we must *not* call ``reorder_w1w3_to_w3w1`` here.

        log_info_on_rank0(
            logger,
            f"Preparing MXFP4 experts for Marlin backend " f"(layer: {self.prefix})...",
        )
        if self.runner.config.gemm1_alpha is not None:
            deinterleave_moe_mxfp4_w13_for_marlin(layer)
        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._dsv4_mxfp4_backend = "marlin"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        # The current SGLang top-k carrier exposes only the standard checker
        # (hash/top-k carriers are represented by the same ``topk_ids`` /
        # ``topk_weights`` fields in this path).  Do not call the removed
        # ``format_is_hash`` helper from the old KT branch.
        if not TopKOutputChecker.format_is_standard(topk_output):
            if not hasattr(topk_output, "topk_ids") or not hasattr(
                topk_output, "topk_weights"
            ):
                raise ValueError(f"Unsupported topk output format: {topk_output.format}")
        hidden_states = dispatch_output.hidden_states

        if getattr(layer, "_v4_marlin_path", False) and self._kt_layerwise_enabled:
            from sglang.srt.layers.quantization.v4_marlin_moe import (
                apply_v4_marlin_moe,
            )

            topk_ids = topk_output.topk_ids
            topk_weights = topk_output.topk_weights
            runner_config = getattr(self, "runner", None)
            runner_config = getattr(runner_config, "config", None)
            if runner_config is None:
                runner_config = getattr(self, "moe_runner_config", None)
            routed_scale = getattr(runner_config, "routed_scaling_factor", None)
            if routed_scale is None:
                routed_scale = 1.0
            target_hidden_size = layer.w13_weight.shape[2] * 2
            if hidden_states.shape[-1] > target_hidden_size:
                raise ValueError(
                    "KT MXFP4 hidden size exceeds the padded weight size: "
                    f"{hidden_states.shape[-1]} > {target_hidden_size}"
                )
            hidden_states_padded = hidden_states
            hidden_pad = target_hidden_size - hidden_states.shape[-1]
            if hidden_pad:
                hidden_states_padded = F.pad(hidden_states, (0, hidden_pad))

            output = apply_v4_marlin_moe(
                hidden_states=hidden_states_padded,
                prepared=layer._v4_marlin_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                routed_scaling_factor=routed_scale,
                swiglu_limit=getattr(
                    getattr(layer, "moe_runner_config", None), "swiglu_limit", None
                ),
            )
            if hidden_pad:
                output = output[..., : hidden_states.shape[-1]]
            return StandardCombineInput(hidden_states=output)

        target_hidden_size = layer.w13_weight.shape[1] * 16
        if hidden_states.shape[-1] == target_hidden_size:
            hidden_states_padded = hidden_states
        else:
            hidden_states_padded = torch.nn.functional.pad(
                hidden_states,
                (0, target_hidden_size - hidden_states.shape[-1]),
                mode="constant",
                value=0.0,
            )

        quant_info = build_marlin_moe_quant_info(layer)
        runner_output = self.runner.run(
            dispatch_output._replace(hidden_states=hidden_states_padded),
            quant_info=quant_info,
        )

        return StandardCombineInput(hidden_states=runner_output.hidden_states)
