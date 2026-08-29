#!/usr/bin/env python3
"""Decode LONGSPEAR_ROUTE_TRACE records and summarize per-layer locality."""

from __future__ import annotations

import argparse
import math
import struct
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Mapping, Optional, Tuple


MAGIC = "LONGSPEAR_ROUTE_TRACE"
CACHE_SIZES = (64, 96, 128, 160, 224)
RECORD_HEADER = struct.Struct("<HH")


class TraceError(ValueError):
    pass


@dataclass
class LruState:
    capacity: int
    entries: OrderedDict[int, None] = field(default_factory=OrderedDict)
    hits: int = 0

    def access(self, expert: int) -> None:
        if expert in self.entries:
            self.hits += 1
            self.entries.move_to_end(expert)
            return
        self.entries[expert] = None
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


@dataclass
class LayerStats:
    n_experts: int
    passes: int = 0
    rows: int = 0
    selections: int = 0
    unique_per_pass: Counter[int] = field(default_factory=Counter)
    expert_frequency: List[int] = field(init=False)
    last_use: Dict[int, int] = field(default_factory=dict)
    reuse_distance: Counter[int] = field(default_factory=Counter)
    reuse_distance_sum: int = 0
    cold_uses: int = 0
    lru: Dict[int, LruState] = field(init=False)

    def __post_init__(self) -> None:
        self.expert_frequency = [0] * self.n_experts
        self.lru = {size: LruState(size) for size in CACHE_SIZES}

    def add_pass(self, n_rows: int, experts: Iterable[int]) -> None:
        ids = list(experts)
        self.passes += 1
        self.rows += n_rows
        self.unique_per_pass[len(set(ids))] += 1

        for expert in ids:
            if not 0 <= expert < self.n_experts:
                raise TraceError(f"expert ID {expert} is outside [0, {self.n_experts})")
            position = self.selections
            self.selections += 1
            self.expert_frequency[expert] += 1

            previous = self.last_use.get(expert)
            if previous is None:
                self.cold_uses += 1
            else:
                distance = position - previous
                self.reuse_distance[distance] += 1
                self.reuse_distance_sum += distance
            self.last_use[expert] = position

            for cache in self.lru.values():
                cache.access(expert)


def parse_header(line: bytes) -> Dict[str, str]:
    try:
        text = line.decode("ascii").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise TraceError("trace header is not ASCII") from exc

    fields = text.split()
    if len(fields) < 3 or fields[0] != MAGIC or fields[1] != "v1":
        raise TraceError("unsupported or missing LONGSPEAR_ROUTE_TRACE v1 header")

    metadata: Dict[str, str] = {"version": fields[1]}
    for field in fields[2:]:
        if "=" not in field:
            raise TraceError(f"malformed header field: {field!r}")
        key, value = field.split("=", 1)
        metadata[key] = value

    required = {"model", "experts", "top_k", "layers", "main_layers", "endian", "record"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise TraceError(f"header is missing fields: {', '.join(missing)}")
    if metadata["endian"] != "little":
        raise TraceError(f"unsupported endian: {metadata['endian']}")
    return metadata


def read_exact(stream: BinaryIO, size: int, record_index: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise TraceError(
            f"truncated record {record_index}: expected {size} payload bytes, got {len(data)}"
        )
    return data


def decode_trace(stream: BinaryIO) -> Tuple[Dict[str, str], Dict[int, LayerStats]]:
    header = parse_header(stream.readline())
    try:
        n_experts = int(header["experts"])
        top_k = int(header["top_k"])
        n_layers = int(header["layers"])
    except ValueError as exc:
        raise TraceError("experts, top_k, and layers must be integers") from exc
    if n_experts <= 0 or top_k <= 0 or n_layers <= 0:
        raise TraceError("experts, top_k, and layers must be positive")

    stats: Dict[int, LayerStats] = {}
    record_index = 0
    while True:
        raw_header = stream.read(RECORD_HEADER.size)
        if not raw_header:
            break
        if len(raw_header) != RECORD_HEADER.size:
            raise TraceError(f"truncated record header at record {record_index}")

        layer, n_rows = RECORD_HEADER.unpack(raw_header)
        if layer >= n_layers:
            raise TraceError(f"record {record_index} has layer {layer}, but header declares {n_layers} layers")
        n_ids = n_rows * top_k
        payload = read_exact(stream, n_ids * 2, record_index)
        experts = struct.unpack(f"<{n_ids}H", payload) if n_ids else ()

        layer_stats = stats.setdefault(layer, LayerStats(n_experts))
        layer_stats.add_pass(n_rows, experts)
        record_index += 1

    return header, stats


def histogram_text(histogram: Mapping[int, int]) -> str:
    return ",".join(f"{value}:{count}" for value, count in sorted(histogram.items())) or "-"


def percentile(histogram: Mapping[int, int], quantile: float) -> Optional[int]:
    total = sum(histogram.values())
    if total == 0:
        return None
    target = math.ceil(total * quantile)
    seen = 0
    for value, count in sorted(histogram.items()):
        seen += count
        if seen >= target:
            return value
    raise AssertionError("unreachable percentile")


def top_frequency_text(stats: LayerStats, count: int) -> str:
    ranked = sorted(enumerate(stats.expert_frequency), key=lambda item: (-item[1], item[0]))
    selected = [(expert, frequency) for expert, frequency in ranked if frequency > 0][:count]
    return ",".join(f"{expert}:{frequency}" for expert, frequency in selected) or "-"


def print_summary(
    header: Mapping[str, str],
    stats: Mapping[int, LayerStats],
    top_experts: int,
    all_experts: bool,
) -> None:
    total_records = sum(item.passes for item in stats.values())
    total_selections = sum(item.selections for item in stats.values())
    print(
        f"model={header['model']} experts={header['experts']} top_k={header['top_k']} "
        f"layers_seen={len(stats)}/{header['layers']} records={total_records} "
        f"route_selections={total_selections}"
    )

    cache_headers = [f"LRU{size}" for size in CACHE_SIZES]
    columns = [
        "layer",
        "kind",
        "passes",
        "rows",
        "routes",
        "unique/pass mean [hist]",
        "reuse mean/p50/p95",
        *cache_headers,
        f"top{top_experts} expert:count",
    ]
    print(" | ".join(columns))
    print(" | ".join("-" * len(column) for column in columns))

    main_layers = int(header["main_layers"])
    for layer, item in sorted(stats.items()):
        unique_sum = sum(value * count for value, count in item.unique_per_pass.items())
        unique_mean = unique_sum / item.passes if item.passes else 0.0
        reuse_count = sum(item.reuse_distance.values())
        reuse_mean = item.reuse_distance_sum / reuse_count if reuse_count else 0.0
        p50 = percentile(item.reuse_distance, 0.50)
        p95 = percentile(item.reuse_distance, 0.95)
        reuse = f"{reuse_mean:.1f}/{p50 if p50 is not None else '-'}/{p95 if p95 is not None else '-'}"
        cache_rates = [
            f"{100.0 * item.lru[size].hits / item.selections:.2f}%" if item.selections else "-"
            for size in CACHE_SIZES
        ]
        row = [
            str(layer),
            "main" if layer < main_layers else "mtp",
            str(item.passes),
            str(item.rows),
            str(item.selections),
            f"{unique_mean:.1f} [{histogram_text(item.unique_per_pass)}]",
            reuse,
            *cache_rates,
            top_frequency_text(item, top_experts),
        ]
        print(" | ".join(row))

    if all_experts:
        print("\nper-expert frequency (layer | expert=count,...):")
        for layer, item in sorted(stats.items()):
            frequencies = ",".join(
                f"{expert}={count}" for expert, count in enumerate(item.expert_frequency)
            )
            print(f"{layer} | {frequencies}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode a qwen4exp LONGSPEAR_ROUTE_TRACE binary stream."
    )
    parser.add_argument("trace", type=Path, help="route trace written by llama-server")
    parser.add_argument(
        "--top-experts",
        type=int,
        default=10,
        metavar="N",
        help="show the N most frequent experts per layer in the compact table (default: 10)",
    )
    parser.add_argument(
        "--all-experts",
        action="store_true",
        help="also print all per-expert frequency counts for every observed layer",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_experts <= 0:
        print("error: --top-experts must be positive", file=sys.stderr)
        return 2
    try:
        with args.trace.open("rb") as stream:
            header, stats = decode_trace(stream)
        print_summary(header, stats, args.top_experts, args.all_experts)
    except (OSError, TraceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
