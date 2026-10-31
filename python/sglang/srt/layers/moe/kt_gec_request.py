"""Request-level admission state for the Global Expert Cache (GEC).

The prefill-token threshold is a property of a request, not of a forward
batch.  A mixed batch can therefore contain one long prompt that qualifies
for GEC and several short requests that must keep the ordinary execution path
without updating LFU or starting background residency changes.

This module intentionally has no SGLang imports.  ``ModelRunner`` publishes a
lightweight layout once per forward and the MoE planner consumes it from any
rank/thread of that forward.  Keeping the state separate from the cache also
lets the threshold rule be tested without CUDA or kt_kernel.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


def _mode_predicate(mode, name: str) -> bool:
    probe = getattr(mode, name, None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _is_prefill_forward_mode(mode) -> bool:
    if mode is None:
        return False
    # Target verification is decode-like for admission even though SGLang's
    # general prefill predicate accepts it for attention/staging purposes.
    if _mode_predicate(mode, "is_decode") or _mode_predicate(mode, "is_target_verify"):
        return False
    return _mode_predicate(mode, "is_extend")


@dataclass(frozen=True)
class RequestForwardLayout:
    """Host-side request layout of one forward.

    ``token_counts[i]`` is the number of rows ``request_ids[i]`` contributes
    to the flattened token axis.  ``prefill_lengths[i]`` is the prompt length
    observed for that request (cached prefix plus the current extend chunk).
    Trailing rows not covered by ``sum(token_counts)`` are padding and are
    never eligible for LFU accounting.
    """

    forward_id: int
    is_prefill: bool
    request_ids: Tuple[str, ...]
    token_counts: Tuple[int, ...]
    prefill_lengths: Tuple[int, ...]
    is_decode: Tuple[bool, ...]
    num_tokens: int

    @property
    def covered_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def padding_tokens(self) -> int:
        return max(0, self.num_tokens - self.covered_tokens)

    def token_span(self, request_index: int) -> Tuple[int, int]:
        start = sum(self.token_counts[:request_index])
        return start, start + self.token_counts[request_index]


def build_request_layout(
    *,
    forward_id: int,
    forward_mode,
    request_ids: Optional[Sequence[object]],
    num_tokens: int,
    extend_seq_lens_cpu: Optional[Sequence[int]] = None,
    extend_prefix_lens_cpu: Optional[Sequence[int]] = None,
    request_is_decode: Optional[Sequence[bool]] = None,
) -> Optional[RequestForwardLayout]:
    """Build a layout, or return ``None`` when request ownership is ambiguous.

    A forward with padding may cover fewer rows than ``num_tokens``; the
    uncovered tail is deliberately absent from every request span.  If the
    supplied counts exceed the actual token axis, the layout is rejected so
    GEC can fail closed instead of attributing another request's routes.
    """

    if request_ids is None:
        return None
    ids = tuple(str(rid) for rid in request_ids)
    if not ids:
        return None
    total = max(0, int(num_tokens))
    request_count = len(ids)
    is_prefill = _is_prefill_forward_mode(forward_mode)

    counts = None
    if (
        is_prefill
        and extend_seq_lens_cpu is not None
        and len(extend_seq_lens_cpu) == request_count
    ):
        counts = tuple(max(0, int(value)) for value in extend_seq_lens_cpu)
    if counts is None:
        if _mode_predicate(forward_mode, "is_decode"):
            # A decode row contributes exactly one token per request. Rows
            # after the request count are CUDA-graph padding and stay outside
            # every request span.
            if total < request_count:
                return None
            counts = (1,) * request_count
        elif is_prefill:
            # Prefill chunks require per-request extend lengths. A batch total
            # cannot prove that any individual prompt crossed the threshold,
            # and even a divisible total could hide a short request.
            return None
        elif total == request_count:
            counts = (1,) * request_count
        else:
            return None
    if sum(counts) > total:
        return None

    if request_is_decode is not None and len(request_is_decode) == request_count:
        decode = tuple(bool(value) for value in request_is_decode)
    elif is_prefill:
        decode = (False,) * request_count
    else:
        decode = (True,) * request_count

    if (
        extend_prefix_lens_cpu is not None
        and len(extend_prefix_lens_cpu) == request_count
    ):
        prefill_lengths = tuple(
            max(0, int(prefix)) + count
            for prefix, count in zip(extend_prefix_lens_cpu, counts)
        )
    else:
        prefill_lengths = counts

    return RequestForwardLayout(
        forward_id=int(forward_id),
        is_prefill=is_prefill,
        request_ids=ids,
        token_counts=counts,
        prefill_lengths=prefill_lengths,
        is_decode=decode,
        num_tokens=total,
    )


@dataclass(frozen=True)
class RequestAdmissionWindow:
    """The admission decision shared by every MoE layer of one forward."""

    layout: Optional[RequestForwardLayout]
    qualified: Tuple[bool, ...]
    manage: bool
    select_all: bool
    forward_id: int = 0

    @property
    def covered_tokens(self) -> Optional[int]:
        return None if self.layout is None else self.layout.covered_tokens

    def selected_request_indices(self) -> Tuple[int, ...]:
        return tuple(index for index, allowed in enumerate(self.qualified) if allowed)


class RequestAdmissionTracker:
    """Persist qualification across prefill chunks and the decode lifetime."""

    def __init__(self, threshold: Optional[int] = 0) -> None:
        self.threshold = max(0, int(threshold or 0))
        self._qualified: Dict[str, int] = {}
        self._prefill_seen: Dict[str, int] = {}
        self._last_observed_forward: Dict[str, int] = {}
        self._cached_layout: Optional[RequestForwardLayout] = None
        self._cached_window: Optional[RequestAdmissionWindow] = None
        self._cached_forward_id: Optional[int] = None
        self._lock = threading.RLock()
        _TRACKERS.add(self)

    @property
    def qualified_request_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._qualified)

    @property
    def cached_forward_id(self) -> Optional[int]:
        with self._lock:
            return self._cached_forward_id

    def begin_forward(
        self,
        layout: Optional[RequestForwardLayout],
        forward_id: Optional[int] = None,
    ) -> RequestAdmissionWindow:
        if forward_id is None:
            forward_id = layout.forward_id if layout is not None else 0
        forward_id = int(forward_id)
        with self._lock:
            if (
                self._cached_window is not None
                and layout is self._cached_layout
                and forward_id == self._cached_forward_id
            ):
                return self._cached_window

            if self.threshold <= 0:
                request_count = 0 if layout is None else len(layout.request_ids)
                window = RequestAdmissionWindow(
                    layout=layout,
                    qualified=(True,) * request_count,
                    manage=True,
                    select_all=True,
                    forward_id=forward_id,
                )
            elif layout is None:
                window = RequestAdmissionWindow(
                    layout=None,
                    qualified=(),
                    manage=False,
                    select_all=False,
                    forward_id=forward_id,
                )
            else:
                for index, request_id in enumerate(layout.request_ids):
                    if layout.is_decode[index]:
                        continue
                    if self._last_observed_forward.get(request_id) == forward_id:
                        continue
                    self._last_observed_forward[request_id] = forward_id
                    previous = self._prefill_seen.get(request_id, 0)
                    measured = max(
                        layout.prefill_lengths[index],
                        previous + layout.token_counts[index],
                    )
                    self._prefill_seen[request_id] = max(previous, measured)
                    if measured >= self.threshold:
                        self._qualified[request_id] = forward_id
                qualified = tuple(
                    request_id in self._qualified for request_id in layout.request_ids
                )
                window = RequestAdmissionWindow(
                    layout=layout,
                    qualified=qualified,
                    manage=any(qualified),
                    select_all=bool(qualified) and all(qualified),
                    forward_id=forward_id,
                )

            self._cached_layout = layout
            self._cached_forward_id = forward_id
            self._cached_window = window
            return window

    def forget_request(self, request_id: object) -> None:
        rid = str(request_id)
        with self._lock:
            self._qualified.pop(rid, None)
            self._prefill_seen.pop(rid, None)
            self._last_observed_forward.pop(rid, None)
            self._cached_layout = None
            self._cached_forward_id = None
            self._cached_window = None

    def forget_requests(self, request_ids: Iterable[object]) -> None:
        for request_id in request_ids:
            self.forget_request(request_id)


_TRACKERS: "weakref.WeakSet[RequestAdmissionTracker]" = weakref.WeakSet()


def forget_request_contexts(request_ids: Iterable[object]) -> int:
    """Forget finished/aborted state in every process-local request tracker.

    The scheduler owns request completion while GEC trackers live beside the
    model runners. This registry keeps the two decoupled: callers only pass
    request ids, and a tracker leaves the registry with its coordinator.
    """

    rids = tuple(dict.fromkeys(str(item) for item in request_ids))
    if not rids:
        return 0
    trackers = tuple(_TRACKERS)
    for tracker in trackers:
        tracker.forget_requests(rids)
    return len(trackers)


_PUBLISHED_REQUEST_LAYOUT: Optional[RequestForwardLayout] = None
_PUBLISHED_FORWARD_ID = 0


def publish_request_layout(
    layout: Optional[RequestForwardLayout], forward_id: Optional[int] = None
) -> None:
    """Publish the request layout of the forward currently being executed."""

    global _PUBLISHED_FORWARD_ID, _PUBLISHED_REQUEST_LAYOUT
    if forward_id is None:
        forward_id = layout.forward_id if layout is not None else _PUBLISHED_FORWARD_ID
    _PUBLISHED_FORWARD_ID = int(forward_id)
    _PUBLISHED_REQUEST_LAYOUT = layout


def published_request_layout() -> Optional[RequestForwardLayout]:
    return _PUBLISHED_REQUEST_LAYOUT


def published_forward_id() -> int:
    return _PUBLISHED_FORWARD_ID
