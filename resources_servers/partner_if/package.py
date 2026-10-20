# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task packages: the row contract, execution profiles and checksummed files.

A package is a directory whose `snapshot-sha256.json` records every other file.
`expected.jsonl` holds the full membership, `profiles.json` assigns each task to one
execution profile, and `<profile>.jsonl` repeats that profile's rows for the servers
that load them. The curation repo builds packages with these checks; benchmark
prepare scripts repeat them after download.
"""

import hashlib
import json
import re
from collections.abc import Collection
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.partner_if.grading_contract import GradingContract, StrictModel
from resources_servers.partner_if.schema import VerifierMetadata, validate_input


CHECKSUMS = "snapshot-sha256.json"


def resolve_repo_path(path: Path) -> Path:
    """Servers start inside their own directories, so resolve relative package paths like Gym's other
    read paths: under NEMO_GYM_EXTRA_ROOTS (a benchmark plugin root), the cwd, then the Gym root."""
    return _resolve_under_cwd_or_install(path)


RepoPath = Annotated[Path, AfterValidator(resolve_repo_path)]


class TaskBundle(StrictModel):
    """A native task's packaged contract and the documents staged for its actor."""

    contract: GradingContract
    documents: dict[str, str] = Field(default_factory=dict)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def validate_rows(rows: list[dict], *, allow_synthetic: bool = False) -> list[dict]:
    if not rows:
        raise ValueError("Empty benchmark")
    seen = set()
    for row in rows:
        if not {"responses_create_params", "verifier_metadata"}.issubset(row) or set(row) - {
            "responses_create_params",
            "verifier_metadata",
            "task_name",
        }:
            raise ValueError(
                "Catalogue rows must contain only actor params and verifier_metadata; routing belongs in config"
            )
        params = row["responses_create_params"]
        NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
        contract = VerifierMetadata.model_validate(row["verifier_metadata"]).partner_if
        if contract.evidence_mode == "live_agent":
            # Native harnesses may furnish tools; HTTP profiles declare them in the
            # row. In either case initial messages must be self-contained text.
            names = set()
            for tool in params.get("tools") or []:
                name = tool.get("name", "")
                if tool.get("type") != "function" or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
                    raise ValueError("Live tools require unique safe named function definitions")
                names.add(name)
            validate_input(
                {key: value for key, value in params.items() if key != "tools"}, evidence_mode="text_response"
            )
            if "task_name" in row and (not isinstance(row["task_name"], str) or not row["task_name"]):
                raise ValueError("Native task_name must be a nonempty string")
        else:
            if "task_name" in row:
                raise ValueError("Text-only rows cannot route to a native task package")
            validate_input(params, evidence_mode=contract.evidence_mode)
        if contract.data_kind == "synthetic_smoke" and not allow_synthetic:
            raise ValueError("Do not include synthetic smoke rows in a benchmark")
        if contract.task_id in seen:
            raise ValueError(f"Duplicate task ID: {contract.task_id}")
        seen.add(contract.task_id)
    return rows  # Do not round-trip Pydantic defaults into the original actor prompt.


def partition(rows: list[dict], profiles: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Every expected task belongs to exactly one namespaced execution profile."""
    validate_rows(rows)
    by_id = {row["verifier_metadata"]["partner_if"]["task_id"]: row for row in rows}
    membership = [identifier for ids in profiles.values() for identifier in ids]
    if "expected" in profiles:
        raise ValueError("The expected profile name is reserved for full suite membership")
    if len(membership) != len(set(membership)) or set(membership) != set(by_id):
        raise ValueError("Execution profiles must cover every expected task exactly once")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or not ids for name, ids in profiles.items()):
        raise ValueError("Profiles require safe names and nonempty membership")
    return {name: [by_id[identifier] for identifier in ids] for name, ids in profiles.items()}


def verify_package(root: Path) -> Path:
    """Reject changed, missing, unrecorded, linked or escaping package files."""
    root = root.resolve()
    hashes = json.loads((root / CHECKSUMS).read_text())
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Package requires a nonempty path/checksum manifest")
    for relative, expected in hashes.items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe package path: {relative}")
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package checksum mismatch: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != set(hashes) | {CHECKSUMS}:
        raise ValueError("Package has unrecorded or missing files")
    return root


def select_rows(root: Path, profiles: Collection[str], task_ids: list[str] | None = None) -> list[dict]:
    """Unchanged rows of the named execution profiles in package order, optionally narrowed to task IDs.

    IDs that belong to the package but not to these profiles are skipped, so one selection can
    serve benchmark configs that run different profiles.
    """
    rows = read_jsonl(root / "expected.jsonl")
    membership = json.loads((root / "profiles.json").read_text())
    for name, group in partition(rows, membership).items():
        if read_jsonl(root / f"{name}.jsonl") != group:
            raise ValueError(f"Profile differs from expected membership: {name}")
    if unknown := set(profiles) - set(membership):
        raise ValueError(f"Unknown execution profiles: {', '.join(sorted(unknown))}")
    selected = {identifier for name in profiles for identifier in membership[name]}
    if task_ids is not None:
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("Task selection must contain nonempty, unique task IDs")
        if unknown := set(task_ids) - {identifier for ids in membership.values() for identifier in ids}:
            raise ValueError(f"Unknown task IDs: {', '.join(sorted(unknown))}")
        selected &= set(task_ids)
    result = [row for row in rows if row["verifier_metadata"]["partner_if"]["task_id"] in selected]
    if not result:
        raise ValueError(f"No selected tasks in execution profiles: {', '.join(sorted(profiles))}")
    return result
