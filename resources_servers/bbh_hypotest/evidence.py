# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic artifact-to-notebook adapter for terminal BBH harnesses."""

import json
import re
from pathlib import Path
from typing import Any


EVIDENCE_CANDIDATES = ("analysis.py", "analysis.ipynb", "notebook.ipynb")
ARTIFACT_INSTRUCTION = """\
Use your terminal tools to inspect the files in the current workspace and solve the hypothesis task.
Create a reproducible evidence artifact at ./analysis.py. Use paths relative to the current directory
for provided data; do not hardcode a sandbox mount path. The script must load the provided data,
perform the analysis, and print every numerical result needed to support your conclusion. Run and
fix it before finishing. Separate major analysis stages with `# %%`; the grader executes each section
as a notebook code cell with state preserved. Keep outputs concise. If no markers are present, the
entire script runs as one cell. You may instead create ./analysis.ipynb, but ./analysis.py is preferred.
Your final assistant message must be the scientific conclusion that should be graded. Do not put
credentials in the artifact and do not modify the source data.
"""
PYTHON_CELL_MARKER = re.compile(r"^[ \t]*#[ \t]*%%(?:[ \t].*)?$", re.MULTILINE)


class EvidenceError(ValueError):
    """The harness failed to produce a usable evidence artifact."""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or part.get("content") or "")
            if isinstance(part, dict)
            else str(getattr(part, "text", part))
            for part in content
        )
    return "" if content is None else str(content)


def build_hypotest_prompt(observations: list[Any]) -> str:
    rendered = []
    for observation in observations:
        if hasattr(observation, "model_dump"):
            observation = observation.model_dump()
        if isinstance(observation, dict):
            role = str(observation.get("role") or "user").upper()
            text = _content_text(observation.get("content"))
        else:
            role = str(getattr(observation, "role", "user")).upper()
            text = _content_text(getattr(observation, "content", observation))
        if text:
            rendered.append(f"[{role}]\n{text}")
    rendered.append(f"Submission requirements:\n{ARTIFACT_INSTRUCTION.strip()}")
    return "\n\n".join(rendered)


def safe_artifact_path(workspace: Path, artifact_path: str) -> Path:
    root = workspace.resolve()
    candidate = (root / artifact_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EvidenceError(f"Evidence path escapes workspace: {artifact_path}") from exc
    return candidate


def find_evidence_artifact(workspace: Path, artifact_path: str | None = None) -> Path:
    if artifact_path:
        candidate = safe_artifact_path(workspace, artifact_path)
        if not candidate.is_file():
            raise EvidenceError(f"Evidence artifact does not exist: {artifact_path}")
        return candidate
    for relative_path in EVIDENCE_CANDIDATES:
        candidate = safe_artifact_path(workspace, relative_path)
        if candidate.is_file():
            return candidate
    raise EvidenceError("No evidence artifact found; expected analysis.py or analysis.ipynb")


def load_evidence_cells(workspace: Path, artifact_path: str | None = None) -> tuple[Path, list[str]]:
    artifact = find_evidence_artifact(workspace, artifact_path)
    if artifact.suffix == ".py":
        source = artifact.read_text(encoding="utf-8")
        if not source.strip():
            raise EvidenceError(f"Evidence artifact is empty: {artifact.name}")
        if not PYTHON_CELL_MARKER.search(source):
            return artifact, [source]
        return artifact, [cell.strip() for cell in PYTHON_CELL_MARKER.split(source) if cell.strip()]
    if artifact.suffix != ".ipynb":
        raise EvidenceError(f"Unsupported evidence artifact: {artifact.name}")
    try:
        notebook = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Invalid notebook artifact: {artifact.name}") from exc
    cells = []
    for cell in notebook.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source") or ""
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        if str(source).strip():
            cells.append(str(source))
    if not cells:
        raise EvidenceError(f"Notebook has no executable code cells: {artifact.name}")
    return artifact, cells
