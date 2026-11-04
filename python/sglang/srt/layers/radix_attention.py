# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.layers.attention.graph_utils import (
    _attention_on_real_rows,
    _zero_padded_tokens,
    allocate_attention_outputs,
    attention_input_scope,
    attention_output_tokens,
    attention_query_tokens,
    padded_extend_real_tokens,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    get_prefill_raw_num_tokens,
    is_in_full_prefill_graph,
    is_in_prefill_graph_scope,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """String-valued attention types, compatible with torch.compile."""

    DECODER = "decoder"
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
        use_prefill_attention_wrapper: bool = True,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.use_prefill_attention_wrapper = use_prefill_attention_wrapper
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.q_scale_float = None
        self.idx_q_scale_float = None
        self.idx_k_scale_float = None
        self.idx_v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        key_value_num_tokens: Optional[int] = None,
        **kwargs,
    ):
        trace_state = None
        if self.layer_id == 0 and logger.isEnabledFor(logging.DEBUG):
            started_ns = time.perf_counter_ns()
            start_event = None
            stream_id = "cpu"
            capture_active = False
            if q is not None and q.device.type == "cuda":
                try:
                    stream = torch.cuda.current_stream(q.device)
                    stream_id = getattr(stream, "cuda_stream", "unknown")
                    capture_active = torch.cuda.is_current_stream_capturing()
                    if not capture_active:
                        start_event = torch.cuda.Event(enable_timing=True)
                        start_event.record(stream)
                except Exception:
                    start_event = None
            trace_state = (started_ns, start_event, stream_id)
            try:
                backend_name = type(get_attn_backend()).__name__
            except Exception:
                backend_name = "unknown"
            attn_kind = "mqa" if self.tp_k_head_num == 1 else "mha"
            out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
            logger.debug(
                "[kt-attn] mono_ms=%.3f phase=enter pass=%s layer=0 mode=%s "
                "tokens=%s q_shape=%s k_shape=%s v_shape=%s save_kv_cache=%s "
                "key_value_num_tokens=%s out_cache_loc=%s positions=%s "
                "stream=%s capture=%s backend=%s kind=%s",
                started_ns / 1e6,
                getattr(forward_batch, "_kt_debug_pass_id", -1),
                getattr(forward_batch, "forward_mode", "unknown"),
                getattr(forward_batch, "_forward_num_tokens", lambda: -1)(),
                tuple(q.shape) if torch.is_tensor(q) else None,
                tuple(k.shape) if torch.is_tensor(k) else None,
                tuple(v.shape) if torch.is_tensor(v) else None,
                save_kv_cache,
                key_value_num_tokens,
                tuple(out_cache_loc.shape) if torch.is_tensor(out_cache_loc) else None,
                tuple(forward_batch.positions.shape)
                if torch.is_tensor(getattr(forward_batch, "positions", None))
                else None,
                stream_id,
                capture_active,
                backend_name,
                attn_kind,
            )
        try:
            return self._forward_impl(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=save_kv_cache,
                key_value_num_tokens=key_value_num_tokens,
                **kwargs,
            )
        finally:
            if trace_state is not None:
                started_ns, start_event, stream_id = trace_state
                cuda_ms = None
                if start_event is not None:
                    try:
                        end_event = torch.cuda.Event(enable_timing=True)
                        end_event.record(torch.cuda.current_stream(q.device))
                        if end_event.query():
                            cuda_ms = start_event.elapsed_time(end_event)
                    except Exception:
                        pass
                logger.debug(
                    "[kt-attn] mono_ms=%.3f phase=exit pass=%s layer=0 wall_ms=%.3f "
                    "cuda_ms=%s stream=%s",
                    time.perf_counter_ns() / 1e6,
                    getattr(forward_batch, "_kt_debug_pass_id", -1),
                    (time.perf_counter_ns() - started_ns) / 1e6,
                    f"{cuda_ms:.3f}" if cuda_ms is not None else "na",
                    stream_id,
                )

    def _forward_impl(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        key_value_num_tokens: Optional[int] = None,
        **kwargs,
    ):
        if k is not None:
            assert v is not None
            k = k.view(
                -1,
                self.tp_k_head_num,
                self.v_head_dim if "k_rope" in kwargs else self.qk_head_dim,
            )
            if "k_rope" not in kwargs:
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        breakable_cg = is_in_breakable_cuda_graph()
        full_cg = is_in_full_prefill_graph()
        use_eager_attention = (
            self.use_prefill_attention_wrapper
            and is_in_prefill_graph_scope()
            and forward_batch.forward_mode.is_extend()
            and (breakable_cg or full_cg)
        )
        if use_eager_attention:
            is_sparse = kwargs.get("idx_q") is not None
            output, idx_output = allocate_attention_outputs(
                self, q, v, kwargs.get("idx_q")
            )
            return_lse = not is_sparse and bool(
                kwargs.get("return_lse") or forward_batch.mha_return_lse
            )
            lse = (
                q.new_empty((q.shape[0], self.tp_q_head_num), dtype=torch.float32)
                if return_lse
                else None
            )
            self._eager_attention(
                q,
                k,
                v,
                output,
                save_kv_cache,
                key_value_num_tokens,
                idx_output,
                lse,
                **kwargs,
            )
            if is_sparse:
                return idx_output, output
            if return_lse:
                return output.view(-1, self.tp_q_head_num, self.v_head_dim), lse
            return output
        real_num_tokens = padded_extend_real_tokens(q, forward_batch)
        if real_num_tokens is not None:
            return _attention_on_real_rows(
                self,
                real_num_tokens,
                q,
                k,
                v,
                forward_batch,
                save_kv_cache,
                key_value_num_tokens=key_value_num_tokens,
                **kwargs,
            )
        return get_attn_backend().forward(
            q, k, v, self, forward_batch, save_kv_cache, **kwargs
        )

    @eager_on_graph
    def _eager_attention(
        self,
        q,
        k,
        v,
        output,
        save_kv_cache: bool = True,
        key_value_num_tokens: Optional[int] = None,
        idx_output=None,
        lse_output=None,
        **kwargs,
    ) -> None:
        """Run real tokens, retaining padded outputs for the next graph segment."""
        forward_batch = get_forward_batch()
        n = forward_batch.global_num_token_non_padded_cpu
        kv_n = n if key_value_num_tokens is None else key_value_num_tokens
        cache_n = n
        n = attention_query_tokens(forward_batch, n)
        return_lse = bool(kwargs.get("return_lse") or forward_batch.mha_return_lse)
        if not return_lse:
            # A buffer from a capture that wanted an LSE keeps what it holds.
            lse_output = None
        if n == 0:
            output.zero_()
            if idx_output is not None:
                idx_output.zero_()
            if lse_output is not None:
                lse_output.zero_()
        else:
            with attention_input_scope(
                forward_batch, output, n, kv_n, kwargs, cache_tokens=cache_n
            ) as kwargs:
                result = get_attn_backend().forward(
                    q[:n],
                    k[:kv_n] if k is not None else None,
                    v[:kv_n] if v is not None else None,
                    self,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
            if idx_output is not None:
                idx_result, result = result
                if idx_result is not None:
                    idx_output[:n].view(idx_result.shape).copy_(idx_result)
            elif return_lse:
                result, lse, *_ = result
            if result.data_ptr() != output.data_ptr():
                output[:n].view(result.shape).copy_(result)
            raw = get_prefill_raw_num_tokens()
            raw = attention_output_tokens(forward_batch, raw)
            _zero_padded_tokens(output, raw)
            if idx_output is not None:
                _zero_padded_tokens(idx_output, raw)
            if lse_output is not None:
                lse_output[: lse.shape[0]].copy_(lse)
                lse_output[lse.shape[0] :].zero_()
