# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind original task instructions, graders and workspace assets without executing them."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml


PIN_PATH = Path(__file__).resolve().parents[2] / "benchmarks/wildclawbench/source-pin.json"


def digest(path: Path) -> str:
    """Hash large media in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def verify_source(root: Path) -> None:
    """Allow Git checkouts and exported archives, but reject modified execution code."""
    pinned = source_pin()["files"]
    for directory in ("src", "tasks", "skills"):
        actual = {
            path.relative_to(root).as_posix()
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        if actual - pinned.keys():
            raise ValueError(f"Unpinned WildClawBench source files: {sorted(actual - pinned.keys())}")
    for name, expected in pinned.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"WildClawBench source pin mismatch: {name}")


def task_spec(root: Path, task_id: str) -> dict:
    """Read declarative sections; never import or execute a task's grading code."""
    pin = source_pin()
    path = root / pin["tasks"][task_id]
    text = path.read_text()
    frontmatter = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if frontmatter is None:
        raise ValueError(f"Missing task metadata: {path}")
    metadata = yaml.safe_load(frontmatter[1])
    parts = re.split(r"^##\s+([^\n]+)\n", text[frontmatter.end() :], flags=re.MULTILINE)
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    workspace = sections["Workspace Path"].strip().strip("`").strip()
    relative = Path(workspace)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "workspace":
        raise ValueError(f"Task workspace must be inside workspace/: {task_id}")
    if metadata["id"] != task_id or not sections["Prompt"].strip():
        raise ValueError(f"Task ID or prompt mismatch: {task_id}")
    return dict(
        task_id=task_id,
        task_path=pin["tasks"][task_id],
        task_sha256=digest(path),
        source_revision=pin["revision"],
        category=path.parent.name,
        modality=metadata["modality"],
        timeout_seconds=int(metadata["timeout_seconds"]),
        prompt=sections["Prompt"].strip(),
        workspace=workspace,
    )


def workspace_digest(root: Path, workspace: str) -> str:
    """Include exec, tmp and ground-truth assets; expose only the aggregate hash."""
    directory = root / workspace
    if not (directory / "exec").is_dir():
        raise FileNotFoundError(f"Missing prepared task workspace: {directory}/exec")
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() and not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Workspace symlink escapes task directory: {path}")
        if path.is_file():
            entries.append((path.relative_to(directory).as_posix(), digest(path)))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def selected_tasks(selector: str, *, shard_index: int = 0, shard_count: int = 1) -> list[str]:
    """Select exact task IDs and deterministically partition without overlap."""
    available = source_pin()["tasks"]
    chosen = sorted(available) if selector == "all" else selector.split(",")
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - available.keys():
        raise ValueError("Select unique, existing task IDs, or 'all'")
    if not 0 <= shard_index < shard_count <= len(chosen):
        raise ValueError("Require 0 <= shard_index < shard_count <= selected tasks")
    return sorted(chosen)[shard_index::shard_count]


def prepare_rows(root: Path, task_ids: list[str]) -> list[dict]:
    """Keep all grader text and ground truth out of actor-visible inputs."""
    verify_source(root)
    rows = []
    for task_id in task_ids:
        spec = task_spec(root, task_id)
        prompt = spec.pop("prompt")
        spec["workspace_sha256"] = workspace_digest(root, spec["workspace"])
        rows.append(
            dict(
                responses_create_params=dict(input=[dict(role="user", content=prompt)]),
                verifier_metadata=spec,
            )
        )
    return rows
