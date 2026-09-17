"""Per-rank pinned weight bank for the MXFP4 layerwise prefill (cpu-off-datapath).

The bank eliminates the TP0 host-write relay: each rank keeps a process-local
cudaHostRegister'd copy of its own pre-sharded expert rows, produced offline
by ``bench/bench_fp4_moe.py pack-bank`` through the golden
``write_weight_scale_to_buffer_task`` path.  Tile bytes equal the production
SHM image bit-for-bit because the same C++ writer lays them out.

Layout
------
``<weight_path>/bank/tile_layer{L}_rank{R}.bin`` concatenates four field
blocks in RAW_NAMES order::

    [w13_weight | w13_weight_scale_inv | w2_weight | w2_weight_scale_inv]

Each block is ``num_experts`` rows of the field's per-rank row bytes:
    w13_weight           = 2 * ((inter/tp) * hidden // 2)
    w13_weight_scale_inv = 2 * ((inter/tp) * (hidden // group)) * 2
    w2_weight            = hidden * (inter/tp) // 2
    w2_weight_scale_inv  = (hidden * (inter/tp) // group) * 2
(scale tables are widened ue8m0 -> bf16 inside the writer's convert_or_copy,
so every bank scale row is bfloat16).  All row byte counts are multiples of
16; packing fails early otherwise.

Manifest (``bank/manifest.json``, every object dumped with sort_keys=True)::
    layout_version = 1
    topo   = {gpu_tp_count}
    dims   = {num_layers, num_experts, hidden_size, intermediate_size,
              group_size}
    dtype  = {field: "uint8"|"bfloat16"}
    map    = {fields, rows_per_tile, tile_naming}
    keys_sha256      = {group: sha256(canonical-json(group))}
    weights_sha256   = pack-time hash of the quantized SOURCE weights.
                       Startup never recomputes it (documented deviation);
                       it is a manifest-integrity key only.
    tiles  = {name: {layer, rank, fields: [{field, rows, nbytes, sha256}]}}

Validation is fail-closed: four key groups mismatch under their own reason
literals (D1), tile field shapes/sha256 are checked while the fill pipeline
already has the bytes resident, so the check costs no extra IO.

CLI parameters (read by the wrapper; zero sglang deps in this module)
---------------------------------------------------------------------
--kt-direct-bank-dma 0|1  default 1 (on).  The bank is
    discovered at ``<weight_path>/bank/manifest.json``; any validation or
    capacity failure forces the flag back to off with exactly one
    [KT-DEGRADE] warning and the legacy dual-slot path runs bit-identical.
--kt-dump-slot-bytes 0|1  default 0 (off) -- dumps raw slot bytes for
    debugging; the readback must synchronize the raw_ready event first.
--kt-bank-dma-batch 0|1   reserved for batch-DMA composition; inert
    here.
--kt-bank-dma-lean 0|1    reserved for the phase-2 no-op consensus
    trim; inert here.

Hot-update clause
-----------------
Bank rows are read-only and cover ALL experts (GPU-resident ones included).
Hot expert swaps only move GPU-resident weights (copy_experts_weights_mxfp4
+ mxfp4_raw_source) and never touch the bank, so enabling the bank is a
zero-diff change to the hot-update path (R14).  An expert evicted back to
CPU is served from its immutable bank row on the next prefill load.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import msgspec
import torch

# Field order in every tile file; mirrors RAW_NAMES on the SHM path.
BANK_FIELDS: Tuple[str, ...] = (
    "w13_weight",
    "w13_weight_scale_inv",
    "w2_weight",
    "w2_weight_scale_inv",
)
BANK_KEY_GROUPS: Tuple[str, ...] = ("topo", "dims", "dtype", "map")
BANK_LAYOUT_VERSION = 1
BANK_TILE_NAMING = "tile_layer{layer}_rank{rank}.bin"


class _BankFieldEntry(msgspec.Struct, frozen=True):
    field: str
    rows: int
    nbytes: int
    sha256: str


class _BankTileEntry(msgspec.Struct, frozen=True):
    layer: int
    rank: int
    fields: Tuple[_BankFieldEntry, ...]


class _BankGeometry(msgspec.Struct, frozen=True):
    gpu_tp_count: int
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    group_size: int


class _BankManifest(msgspec.Struct, frozen=True):
    layout_version: int
    backend: str
    topo: Dict[str, int]
    dims: Dict[str, int]
    dtype: Dict[str, str]
    map: Dict[str, object]
    keys_sha256: Dict[str, str]
    weights_sha256: str
    tiles: Dict[str, _BankTileEntry]


class _RankBank(msgspec.Struct, frozen=True):
    """Process-local pinned arena; rows are read-only for the process life."""

    root: str
    rank: int
    geometry: _BankGeometry
    row_nbytes: Dict[str, int]
    # (layer, field) -> pinned tensor reshaped to (num_experts, *expert_shape)
    # with the SHM bank dtype (uint8 weights, bf16 scales), so a row view
    # feeds torch.Tensor.copy_ with the exact legacy source dtype.
    tensors: Dict[Tuple[int, str], torch.Tensor]

    def row(self, layer: int, field: str, expert: int) -> torch.Tensor:
        # Expert-shaped view; data_ptr stays 16B aligned because every row
        # stride is a multiple of 16 (asserted at pack and at load).
        return self.tensors[(layer, field)][expert]


class _BankDegradeError(RuntimeError):
    """Raised with the exact degrade reason literal as its message."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# geometry + manifest validation (pure functions)
# ---------------------------------------------------------------------------


def bank_field_row_bytes(
    hidden: int, inter: int, group_size: int, tp_count: int
) -> Dict[str, int]:
    """One expert's per-rank bank row bytes under the MXFP4 writer geometry.

    Deliberately duplicated with bench/bench_fp4_moe.py: the two sites
    cross-validate each other in the D3 golden-tuple test; sglang must not
    import bench code, so the formula is restated, not shared.
    """
    if inter % tp_count != 0:
        raise ValueError(f"inter={inter} not divisible by tp={tp_count}")
    if hidden % group_size != 0:
        raise ValueError(f"hidden={hidden} not divisible by group={group_size}")
    if (inter // tp_count) % group_size != 0:
        raise ValueError(
            f"inter/tp={inter // tp_count} not divisible by group={group_size}"
        )
    # W13 splits global-N: two mats (gate, up) of (inter/tp) x hidden nibbles.
    # W2 splits global-K: hidden rows of (inter/tp) nibbles.
    row_bytes = {
        "w13_weight": 2 * ((inter // tp_count) * hidden // 2),
        "w13_weight_scale_inv": 2 * ((inter // tp_count) * (hidden // group_size)) * 2,
        "w2_weight": hidden * (inter // tp_count) // 2,
        "w2_weight_scale_inv": (hidden * (inter // tp_count) // group_size) * 2,
    }
    misaligned = [f for f, n in row_bytes.items() if n % 16 != 0]
    if misaligned:
        raise ValueError(f"bank rows not 16B aligned: {misaligned}")
    return row_bytes


def _canonical_json_sha256(payload) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _bank_expected_groups(geometry: _BankGeometry) -> Dict[str, object]:
    topo = {"gpu_tp_count": geometry.gpu_tp_count}
    dims = {
        "num_layers": geometry.num_layers,
        "num_experts": geometry.num_experts,
        "hidden_size": geometry.hidden_size,
        "intermediate_size": geometry.intermediate_size,
        "group_size": geometry.group_size,
    }
    dtype = {
        field: ("bfloat16" if field.endswith("scale_inv") else "uint8")
        for field in BANK_FIELDS
    }
    tile_map = {
        "fields": list(BANK_FIELDS),
        "rows_per_tile": geometry.num_experts,
        "tile_naming": BANK_TILE_NAMING,
    }
    return {"topo": topo, "dims": dims, "dtype": dtype, "map": tile_map}


def _bank_manifest_from_dict(payload: dict) -> _BankManifest:
    tiles = {}
    raw_tiles = payload.get("tiles", {})
    for name, tile in raw_tiles.items():
        fields = tuple(
            _BankFieldEntry(
                field=entry["field"],
                rows=int(entry["rows"]),
                nbytes=int(entry["nbytes"]),
                sha256=str(entry["sha256"]),
            )
            for entry in tile["fields"]
        )
        tiles[name] = _BankTileEntry(
            layer=int(tile["layer"]), rank=int(tile["rank"]), fields=fields
        )
    return _BankManifest(
        layout_version=int(payload.get("layout_version", -1)),
        backend=str(payload.get("backend", "")),
        topo=dict(payload.get("topo", {})),
        dims=dict(payload.get("dims", {})),
        dtype=dict(payload.get("dtype", {})),
        map=dict(payload.get("map", {})),
        keys_sha256=dict(payload.get("keys_sha256", {})),
        weights_sha256=str(payload.get("weights_sha256", "")),
        tiles=tiles,
    )


def bank_manifest_path(weight_path: str) -> str:
    return os.path.join(weight_path, "bank", "manifest.json")


def _bank_manifest_validate(
    manifest: _BankManifest, geometry: _BankGeometry, rank: int
) -> Optional[str]:
    """Return None on pass, else the degrade reason literal (D1).

    Only this rank's tiles are checked; cross-rank consistency is carried by
    the all-gathered manifest digest at freeze time.
    """
    if manifest.layout_version != BANK_LAYOUT_VERSION:
        return "layout_version"
    expected = _bank_expected_groups(geometry=geometry)
    for group in BANK_KEY_GROUPS:
        stored = manifest.keys_sha256.get(group)
        if stored is None or stored != _canonical_json_sha256(expected[group]):
            return group
        carried = {"topo": manifest.topo, "dims": manifest.dims,
                   "dtype": manifest.dtype, "map": manifest.map}[group]
        if _canonical_json_sha256(carried) != stored:
            return group
    weights_sha = manifest.weights_sha256
    if len(weights_sha) != 64 or any(c not in "0123456789abcdef" for c in weights_sha):
        return "weights_sha256_malformed"
    row_bytes = bank_field_row_bytes(
        hidden=geometry.hidden_size,
        inter=geometry.intermediate_size,
        group_size=geometry.group_size,
        tp_count=geometry.gpu_tp_count,
    )
    for layer in range(geometry.num_layers):
        name = BANK_TILE_NAMING.format(layer=layer, rank=rank)
        tile = manifest.tiles.get(name)
        if tile is None:
            return "tile_missing"
        if tile.layer != layer or len(tile.fields) != len(BANK_FIELDS):
            return "tile_field_shape"
        for entry, field in zip(tile.fields, BANK_FIELDS):
            if entry.field != field:
                return "tile_field_shape"
            if entry.rows != geometry.num_experts:
                return "tile_field_shape"
            if entry.nbytes != row_bytes[field]:
                return "tile_field_shape"
    return None


# ---------------------------------------------------------------------------
# capacity precheck (process-local RAM; no /dev/shm leg)
# ---------------------------------------------------------------------------


def _bank_capacity_precheck(
    total_nbytes: int,
    mem_available: int,
    memlock_limit: int,
    numa_free: Optional[Tuple[int, ...]],
) -> bool:
    # Pure predicate mirroring _staging_window_capacity_ok minus the shm
    # leg: the bank is process-local, so only RAM, RLIMIT_MEMLOCK and the
    # NUMA first-touch budget gate it.  The NUMA leg compares against the
    # SUM of per-node free bytes: the default Linux policy spills pages
    # across nodes when the local node runs short, so the aggregate
    # budget is the real constraint (a single-node layout degenerates to
    # the exact comparison).
    if total_nbytes > mem_available or total_nbytes > memlock_limit:
        return False
    if numa_free is not None and sum(numa_free) < total_nbytes:
        return False
    return True


def _bank_capacity_sources() -> Optional[Tuple[int, int]]:
    # Best-effort (MemAvailable, RLIMIT_MEMLOCK) in bytes; any probe failure
    # fails closed (mirrors _staging_capacity_sources).
    try:
        mem_available = None
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) * 1024
                    break
        if mem_available is None:
            return None
        import resource

        memlock_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        if memlock_limit == resource.RLIM_INFINITY:
            memlock_limit = sys.maxsize
        return mem_available, memlock_limit
    except Exception:
        return None


def _bank_numa_free_bytes() -> Optional[Tuple[int, ...]]:
    # Per-node MemFree from /sys; missing sysfs fails closed like the other
    # probes.  Returns nodes in node0..nodeN order.
    try:
        frees: List[int] = []
        node_idx = 0
        while True:
            path = f"/sys/devices/system/node/node{node_idx}/meminfo"
            if not os.path.exists(path):
                break
            free_kb = None
            with open(path, "r") as handle:
                for line in handle:
                    if "MemFree:" in line:
                        free_kb = int(line.split()[3])
                        break
            if free_kb is None:
                return None
            free_after = free_kb * 1024
            frees.append(free_after)
            node_idx += 1
        if not frees:
            return None
        return tuple(frees)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# tile loading helpers (bytes layer; NUMA/register stay with the caller)
# ---------------------------------------------------------------------------


def _bank_load_layer_bytes(
    root: str, manifest: _BankManifest, layer: int, rank: int
) -> Dict[str, bytes]:
    """Read one rank's tile for a layer and verify every field's sha256."""
    name = BANK_TILE_NAMING.format(layer=layer, rank=rank)
    tile = manifest.tiles.get(name)
    if tile is None:
        raise _BankDegradeError(reason="tile_missing", detail=name)
    path = os.path.join(root, "bank", name)
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        raise _BankDegradeError(reason="tile_unreadable", detail=str(exc))
    fields: Dict[str, bytes] = {}
    offset = 0
    for entry in tile.fields:
        end = offset + entry.rows * entry.nbytes
        if end > len(blob):
            raise _BankDegradeError(reason="tile_field_shape", detail=name)
        payload = blob[offset:end]
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise _BankDegradeError(reason="tile_sha256", detail=name)
        fields[entry.field] = payload
        offset = end
    if offset != len(blob):
        raise _BankDegradeError(reason="tile_field_shape", detail=name)
    return fields
