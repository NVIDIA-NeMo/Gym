# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency graph over harness-side spans: longest (critical) path with waits, and parallel execution time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ng_latency.intervals import intersect, total


WORK_CATEGORIES = ("model", "tool")


@dataclass
class Node:
    id: str
    category: str
    start: float
    end: float
    session: str
    span_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Graph:
    agent_start: float
    agent_end: float
    nodes: dict[str, Node] = field(default_factory=dict)
    preds: dict[str, list[str]] = field(default_factory=dict)  # node id -> predecessor node ids ("" = agent_start)


@dataclass
class CriticalPath:
    duration: float
    by_category: dict[str, float]
    node_ids: list[str]


def _interval(span: dict[str, Any]) -> Optional[tuple[float, float]]:
    start = span.get("start_wall_s")
    end = span.get("end_wall_s")
    if start is None or end is None or end < start:
        return None
    return float(start), float(end)


def _node_id(span: dict[str, Any]) -> str:
    kind = "model" if span["name"] == "model_call" else "tool"
    key = (span.get("ids") or {}).get("tool_call_id") if kind == "tool" else None
    return f"{kind}:{key or span['span_id']}"


def build_graph(
    spans: list[dict[str, Any]], *, agent_start: float, agent_end: float, main_session: str, tool_span_name: str
) -> Graph:
    graph = Graph(agent_start=agent_start, agent_end=agent_end)
    domain = next(
        (
            s["clock"]["domain"]
            for s in spans
            if s["name"] == "agent" and (s.get("ids") or {}).get("invocation_id") == main_session
        ),
        None,
    )
    if domain is None:  # fall back to the most common model_call domain, ties by first appearance
        counts: dict[str, int] = {}
        for s in spans:
            if s["name"] == "model_call":
                counts[s["clock"]["domain"]] = counts.get(s["clock"]["domain"], 0) + 1
        domain = max(counts, key=lambda d: counts[d]) if counts else None
    spawn_tool_to_child: dict[str, str] = {}
    for span in spans:
        if span["name"] == "subagent" and span.get("attrs", {}).get("spawned_by_tool_call_id"):
            child = (span.get("ids") or {}).get("invocation_id") or ""
            spawn_tool_to_child[span["attrs"]["spawned_by_tool_call_id"]] = child
    sessions_with_nodes: set[str] = set()
    for span in spans:
        if span["name"] not in ("model_call", tool_span_name) or span["clock"]["domain"] != domain:
            continue
        interval = _interval(span)
        if interval is None or interval[1] <= agent_start or interval[0] >= agent_end:
            continue
        session = (span.get("ids") or {}).get("invocation_id") or main_session
        category = "model" if span["name"] == "model_call" else "tool"
        node = Node(_node_id(span), category, *interval, session, span["span_id"])
        graph.nodes[node.id] = node
        sessions_with_nodes.add(session)
    # Expand a spawning tool into its child's chain when the child produced nodes.
    for tool_call_id, child in spawn_tool_to_child.items():
        node_id = f"tool:{tool_call_id}"
        if node_id in graph.nodes and child in sessions_with_nodes:
            expanded = graph.nodes.pop(node_id)
            for node in graph.nodes.values():
                if node.session == child:
                    node.session = expanded.session  # the child's work sits on the parent's chain
    # Predecessor: the latest-ending node of the same chain that ended at or before this node's start.
    for node in sorted(graph.nodes.values(), key=lambda n: (n.start, n.end, n.id)):
        candidates = [
            other
            for other in graph.nodes.values()
            if other.session == node.session and other.end <= node.start and other.id != node.id
        ]
        if candidates:
            best_end = max(c.end for c in candidates)
            graph.preds[node.id] = sorted(c.id for c in candidates if c.end == best_end)
        else:
            graph.preds[node.id] = [""]
    return graph


def _rank(option: tuple[float, float, list[str]]) -> tuple[float, float, list[int]]:
    duration, work, path = option
    return round(duration, 9), round(work, 9), [-ord(c) for c in "/".join(path)]


def critical_path(graph: Graph) -> CriticalPath:
    """Longest start-to-end path by duration; ties go to more work time, then lexicographic node ids."""
    order = sorted(graph.nodes.values(), key=lambda n: (n.end, n.start, n.id))
    best: dict[str, tuple[float, float, list[str]]] = {"": (0.0, 0.0, [])}  # id -> (duration, work, path)
    for node in order:
        options = []
        for pred in graph.preds[node.id]:
            p_duration, p_work, p_path = best[pred]
            pred_end = graph.agent_start if pred == "" else graph.nodes[pred].end
            wait = max(0.0, node.start - pred_end)
            options.append((p_duration + wait + node.duration, p_work + node.duration, p_path + [node.id]))
        best[node.id] = max(options, key=_rank)
    finals = []
    for node_id, (duration, work, path) in best.items():
        end = graph.agent_start if node_id == "" else graph.nodes[node_id].end
        finals.append((duration + max(0.0, graph.agent_end - end), work, path))
    duration, _work, path = max(finals, key=_rank)
    by_category = {"model": 0.0, "tool": 0.0}
    for node_id in path:
        node = graph.nodes[node_id]
        by_category[node.category] += node.duration
    by_category["other"] = max(0.0, duration - by_category["model"] - by_category["tool"])
    return CriticalPath(duration=duration, by_category=by_category, node_ids=path)


def _ancestors(graph: Graph, node_id: str) -> set[str]:
    seen: set[str] = set()
    stack = [p for p in graph.preds.get(node_id, []) if p]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(p for p in graph.preds.get(current, []) if p)
    return seen


def parallel_execution_time(graph: Graph) -> float:
    """Time with two or more independent (neither ancestor nor descendant) work nodes active."""
    nodes = sorted(graph.nodes.values(), key=lambda n: (n.start, n.end, n.id))
    ancestors = {n.id: _ancestors(graph, n.id) for n in nodes}
    overlaps: list[tuple[float, float]] = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if b.start >= a.end:
                break
            if a.id in ancestors[b.id] or b.id in ancestors[a.id]:
                continue
            overlaps.extend(intersect([(a.start, a.end)], [(b.start, b.end)]))
    return total(overlaps)
