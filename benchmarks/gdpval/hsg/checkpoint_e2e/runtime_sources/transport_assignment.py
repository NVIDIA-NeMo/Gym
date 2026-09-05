# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic provider-free repair for transport-incompatible GDPVal pairs."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


_BINARY_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".wav",
    ".wave",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".aif",
    ".aiff",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}
_AV_EXTENSIONS = {
    ".wav",
    ".wave",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".aif",
    ".aiff",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}
_IGNORED_NAMES = {
    "finish_params.json",
    "last_responses.jsonl",
    "last_responses.jsonl.lock",
    "last_responses.jsonl.offset",
}


@dataclass(frozen=True)
class Footprint:
    raw_bytes: int
    max_file_bytes: int
    has_av: bool
    file_count: int


@dataclass(frozen=True)
class PairCost:
    compatible: bool
    wire_bytes: int
    raw_bytes: int
    max_file_bytes: int
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class _BinaryArtifact:
    path: Path
    member: str | None
    suffix: str
    size: int


def _artifact_sha256(artifact: _BinaryArtifact) -> str:
    digest = hashlib.sha256()
    if artifact.member is None:
        with artifact.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        with zipfile.ZipFile(artifact.path, "r") as archive, archive.open(artifact.member, "r") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _find_gdpval_config(global_config: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = []
    for value in global_config.values():
        if not isinstance(value, Mapping):
            continue
        resources = value.get("resources_servers")
        if not isinstance(resources, Mapping):
            continue
        gdpval = resources.get("gdpval")
        if isinstance(gdpval, Mapping) and gdpval.get("reference_models"):
            matches.append(gdpval)
    if len(matches) != 1:
        raise ValueError(f"expected one GDPVal reference config, found {len(matches)}")
    return matches[0]


def _repeat_dirs(root: Path, task_id: str) -> list[Path]:
    task = root / f"task_{task_id}"
    if not task.is_dir():
        return []
    repeats = sorted(path for path in task.iterdir() if path.is_dir() and path.name.startswith("repeat_"))
    return repeats or [task]


def _has_valid_finish_marker(repeat_dirs: Sequence[Path]) -> bool:
    """Whether a reference task has at least one persisted completion marker."""
    for repeat in repeat_dirs:
        marker = repeat / "finish_params.json"
        try:
            marker_stat = marker.stat()  # Follow the immutable transport-view symlink.
            if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_size > 16 * 1024 * 1024:
                continue
            document = json.loads(marker.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        # Stirrup writes null for a normal max-turn completion without explicit
        # finish arguments. Atomic persistence is the completion signal; reject
        # other JSON shapes and corrupt files.
        if document is None or isinstance(document, dict):
            return True
    return False


def _footprint(directory: Path, *, include_reference_files: bool = True) -> Footprint:
    artifacts: list[_BinaryArtifact] = []
    # Match app.py exactly: the submission directory and its optional
    # reference_files directory are built as two semantic sections. Reference
    # assets recurse below reference_files; candidate run-state subdirectories
    # are not evidence and must not affect routing.
    paths = list(directory.iterdir())
    reference_files = directory / "reference_files"
    if include_reference_files and reference_files.is_dir():
        paths.extend(reference_files.rglob("*"))
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in _IGNORED_NAMES:
            continue
        suffix = path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                for info in archive.infolist():
                    member = Path(info.filename.replace("\\", "/"))
                    if (
                        info.is_dir()
                        or member.is_absolute()
                        or ".." in member.parts
                        or stat.S_ISLNK(info.external_attr >> 16)
                    ):
                        continue
                    member_suffix = member.suffix.lower()
                    if member_suffix not in _BINARY_EXTENSIONS:
                        continue
                    artifacts.append(
                        _BinaryArtifact(
                            path=path,
                            member=info.filename,
                            suffix=member_suffix,
                            size=info.file_size,
                        )
                    )
            continue
        if suffix not in _BINARY_EXTENSIONS:
            continue
        artifacts.append(_BinaryArtifact(path=path, member=None, suffix=suffix, size=path.stat().st_size))

    # A common deliverable layout includes both loose stems and a convenience
    # ZIP containing those exact stems. The judge renderer retains the bytes
    # once and labels byte-identical duplicates, so the provider-cap planner
    # must use the same lossless semantic footprint. Only exact AV payloads are
    # deduplicated; same-size but different files remain distinct evidence.
    av_size_counts = Counter(artifact.size for artifact in artifacts if artifact.suffix in _AV_EXTENSIONS)
    seen_av: set[tuple[int, str]] = set()
    retained: list[_BinaryArtifact] = []
    for artifact in artifacts:
        if artifact.suffix in _AV_EXTENSIONS and av_size_counts[artifact.size] > 1:
            identity = (artifact.size, _artifact_sha256(artifact))
            if identity in seen_av:
                continue
            seen_av.add(identity)
        retained.append(artifact)

    raw = sum(artifact.size for artifact in retained)
    maximum = max((artifact.size for artifact in retained), default=0)
    count = len(retained)
    has_av = any(artifact.suffix in _AV_EXTENSIONS for artifact in retained)
    return Footprint(raw_bytes=raw, max_file_bytes=maximum, has_av=has_av, file_count=count)


def _pair_cost(
    candidate: Footprint,
    reference: Footprint,
    *,
    max_file_bytes: int,
    max_raw_bytes: int,
    max_wire_bytes: int,
    framing_reserve_bytes: int,
) -> PairCost:
    raw = candidate.raw_bytes + reference.raw_bytes
    maximum = max(candidate.max_file_bytes, reference.max_file_bytes)
    # Audio/video is routed only to Gemini, so its native base64 request is the
    # binding case. PDF/image-only tasks retain GPT raster and native-PDF paths;
    # their exact provider request is gated later by the resource server.
    if not (candidate.has_av or reference.has_av):
        reasons = ("file_over_cap",) if maximum > max_file_bytes else ()
        return PairCost(not reasons, raw, raw, maximum, reasons)
    wire = 4 * math.ceil(raw / 3) + framing_reserve_bytes
    reasons_list = []
    if maximum > max_file_bytes:
        reasons_list.append("file_over_cap")
    if raw > max_raw_bytes:
        reasons_list.append("aggregate_raw_over_cap")
    if wire >= max_wire_bytes:
        reasons_list.append("serialized_request_over_cap")
    return PairCost(not reasons_list, wire, raw, maximum, tuple(reasons_list))


class _Edge:
    __slots__ = ("to", "rev", "capacity", "cost")

    def __init__(self, to: int, rev: int, capacity: int, cost: int) -> None:
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost


def _add_edge(graph: list[list[_Edge]], source: int, target: int, capacity: int, cost: int) -> None:
    graph[source].append(_Edge(target, len(graph[target]), capacity, cost))
    graph[target].append(_Edge(source, len(graph[source]) - 1, 0, -cost))


def _solve_capacity_assignment(
    task_ids: Sequence[str],
    reference_ids: Sequence[str],
    original: Mapping[str, str],
    costs: Mapping[tuple[str, str], PairCost],
) -> Dict[str, str]:
    """Minimum-change min-cost flow with exact original reference capacities."""
    tasks = sorted(task_ids)
    references = sorted(reference_ids)
    capacities = Counter(original.values())
    maximum_pair_cost = max((pair.wire_bytes for pair in costs.values() if pair.compatible), default=0)
    change_penalty = (maximum_pair_cost + len(references) + 1) * (len(tasks) + 1)

    source = 0
    task_base = 1
    reference_base = task_base + len(tasks)
    sink = reference_base + len(references)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for task_index, task_id in enumerate(tasks):
        _add_edge(graph, source, task_base + task_index, 1, 0)
        for reference_index, reference_id in enumerate(references):
            pair = costs[(task_id, reference_id)]
            if not pair.compatible:
                continue
            changed = reference_id != original[task_id]
            cost = (change_penalty if changed else 0) + pair.wire_bytes + reference_index
            _add_edge(graph, task_base + task_index, reference_base + reference_index, 1, cost)
    for reference_index, reference_id in enumerate(references):
        _add_edge(graph, reference_base + reference_index, sink, capacities[reference_id], 0)

    potential = [0] * len(graph)
    flow = 0
    while flow < len(tasks):
        infinity = 10**40
        distance = [infinity] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        distance[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate = current_distance + edge.cost + potential[node] - potential[edge.to]
                if candidate < distance[edge.to]:
                    distance[edge.to] = candidate
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (candidate, edge.to))
        if previous_node[sink] < 0:
            blocked = sorted(
                task_id
                for task_id in tasks
                if not any(costs[(task_id, reference_id)].compatible for reference_id in references)
            )
            raise ValueError(f"no count-preserving transport-compatible assignment; tasks_without_any_route={blocked}")
        for node, value in enumerate(distance):
            if value < infinity:
                potential[node] += value
        node = sink
        while node != source:
            parent = previous_node[node]
            edge = graph[parent][previous_edge[node]]
            edge.capacity -= 1
            graph[node][edge.rev].capacity += 1
            node = parent
        flow += 1

    result: Dict[str, str] = {}
    for task_index, task_id in enumerate(tasks):
        node = task_base + task_index
        for edge in graph[node]:
            if reference_base <= edge.to < sink and edge.capacity == 0:
                result[task_id] = references[edge.to - reference_base]
                break
    if len(result) != len(tasks) or Counter(result.values()) != capacities:
        raise RuntimeError("transport assignment solver violated its capacity contract")
    return result


def make_assignment_repair(
    global_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Callable[[int, Sequence[str], Mapping[str, str]], tuple[Dict[str, str], Dict[str, Any]]]:
    """Bind filesystem roots and return the multistage plan-repair callback."""
    gdpval = _find_gdpval_config(global_config)
    candidate_root = Path(
        str(gdpval.get("persist_deliverables_dir") or os.environ.get("PERSIST_DELIVERABLES_DIR", ""))
    )
    if not candidate_root.is_dir():
        raise ValueError(f"candidate transport view is unreadable: {candidate_root}")
    all_references = {
        str(reference_id): Path(str(reference_config["deliverables_dir"]))
        for reference_id, reference_config in (gdpval.get("reference_models") or {}).items()
        if isinstance(reference_config, Mapping)
    }
    if not all_references or any(not path.is_dir() for path in all_references.values()):
        raise ValueError("one or more reference transport views are unreadable")

    max_file_bytes = int(config.get("max_file_bytes", 320 * 1024 * 1024))
    max_raw_bytes = int(config.get("max_raw_bytes", 315 * 1024 * 1024))
    max_wire_bytes = int(config.get("max_wire_bytes", 420 * 1024 * 1024))
    framing_reserve_bytes = int(config.get("framing_reserve_bytes", 4 * 1024 * 1024))
    footprint_cache: dict[tuple[str, str, bool, bool], list[Footprint]] = {}

    def footprints(
        root: Path,
        task_id: str,
        *,
        require_completed_reference: bool = False,
        include_reference_files: bool = False,
    ) -> list[Footprint]:
        key = (str(root), task_id, require_completed_reference, include_reference_files)
        if key not in footprint_cache:
            directories = _repeat_dirs(root, task_id)
            if not directories:
                if require_completed_reference:
                    return []
                raise ValueError(f"missing transport deliverable: {root}/task_{task_id}")
            if require_completed_reference and not _has_valid_finish_marker(directories):
                return []
            footprint_cache[key] = [
                _footprint(directory, include_reference_files=include_reference_files) for directory in directories
            ]
        return footprint_cache[key]

    def repair(
        stage_index: int,
        reference_ids: Sequence[str],
        original: Mapping[str, str],
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        references = sorted(str(reference_id) for reference_id in reference_ids)
        missing = sorted(set(references) - set(all_references))
        if missing:
            raise ValueError(f"stage selected unknown reference transport views: {missing}")
        costs: dict[tuple[str, str], PairCost] = {}
        for task_id in sorted(original):
            candidates = footprints(candidate_root, task_id)
            for reference_id in references:
                references_for_task = footprints(
                    all_references[reference_id],
                    task_id,
                    require_completed_reference=True,
                    include_reference_files=True,
                )
                if not references_for_task:
                    costs[(task_id, reference_id)] = PairCost(
                        compatible=False,
                        wire_bytes=0,
                        raw_bytes=0,
                        max_file_bytes=0,
                        reasons=("reference_incomplete",),
                    )
                    continue
                combinations = [
                    _pair_cost(
                        candidate,
                        reference,
                        max_file_bytes=max_file_bytes,
                        max_raw_bytes=max_raw_bytes,
                        max_wire_bytes=max_wire_bytes,
                        framing_reserve_bytes=framing_reserve_bytes,
                    )
                    for candidate in candidates
                    for reference in references_for_task
                ]
                # One assignment must work for every candidate/reference repeat.
                incompatible = [pair for pair in combinations if not pair.compatible]
                if incompatible:
                    worst = max(incompatible, key=lambda pair: (pair.wire_bytes, pair.raw_bytes))
                    costs[(task_id, reference_id)] = worst
                else:
                    costs[(task_id, reference_id)] = max(
                        combinations,
                        key=lambda pair: (pair.wire_bytes, pair.raw_bytes),
                    )

        repaired = _solve_capacity_assignment(list(original), references, original, costs)
        changes = []
        for task_id in sorted(original):
            before = original[task_id]
            after = repaired[task_id]
            if before != after:
                before_cost = costs[(task_id, before)]
                after_cost = costs[(task_id, after)]
                changes.append(
                    {
                        "task_id": task_id,
                        "before": before,
                        "after": after,
                        "before_compatible": before_cost.compatible,
                        "before_reasons": list(before_cost.reasons),
                        "before_wire_bytes": before_cost.wire_bytes,
                        "after_wire_bytes": after_cost.wire_bytes,
                    }
                )
        initially_incompatible = [
            {
                "task_id": task_id,
                "reference_id": original[task_id],
                "reasons": list(costs[(task_id, original[task_id])].reasons),
            }
            for task_id in sorted(original)
            if not costs[(task_id, original[task_id])].compatible
        ]
        receipt = {
            "schema": "gdpval.transport-assignment-repair.v1",
            "stage_index": stage_index,
            "policy": "minimum changed tasks; exact selected-reference counts; provider-free",
            "limits": {
                "max_file_bytes": max_file_bytes,
                "max_raw_bytes": max_raw_bytes,
                "max_wire_bytes": max_wire_bytes,
                "framing_reserve_bytes": framing_reserve_bytes,
            },
            "reference_counts": dict(sorted(Counter(original.values()).items())),
            "initially_incompatible": initially_incompatible,
            "changes": changes,
        }
        return repaired, receipt

    return repair
