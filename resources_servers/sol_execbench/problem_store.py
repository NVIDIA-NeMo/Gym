# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-owned, content-addressed native problems; no dataset-row paths are trusted."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


NATIVE_REVISION = "a9fa0804c793d438e70850c33fe34426e66d53dd"  # pragma: allowlist secret -- public upstream commit
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


def canonical_json(value: object) -> bytes:
    """Preserve definition argument order, which is part of the native ABI."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def problem_digest(task_id: str, definition: dict, workloads: list[dict], assets: list[dict]) -> str:
    """Hash the ordered problem payload shared with dataset preparation."""
    return hashlib.sha256(
        canonical_json(dict(task_id=task_id, definition=definition, workloads=workloads, assets=assets))
    ).hexdigest()


def safe_relative_path(value: str) -> str:
    """Accept one canonical POSIX relative path without traversal."""
    path = PurePosixPath(value)
    if (
        not value
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: Sha256


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1)
    problem_digest: Sha256
    definition: dict[str, JsonValue]
    workloads: list[dict[str, JsonValue]] = Field(min_length=1)
    assets: list[Asset]

    @model_validator(mode="after")
    def validate_identity(self) -> "Problem":
        assets = [asset.model_dump() for asset in self.assets]
        if problem_digest(self.task_id, self.definition, self.workloads, assets) != self.problem_digest:
            raise ValueError(f"Problem digest mismatch: {self.task_id}")
        uuids = [workload.get("uuid") for workload in self.workloads]
        if any(not isinstance(uid, str) or not uid for uid in uuids) or len(set(uuids)) != len(uuids):
            raise ValueError("Workload UUIDs must be nonempty and unique")
        paths = [safe_relative_path(asset.path) for asset in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate asset paths")
        for workload in self.workloads:
            inputs = workload.get("inputs", {})
            if not isinstance(inputs, dict):
                raise ValueError("Workload inputs must be an object")
            for spec in inputs.values():
                if not isinstance(spec, dict):
                    raise ValueError("Each workload input must be an object")
                if "shards" in spec:
                    raise ValueError("The pinned native evaluator does not support sharded inputs")
                if spec.get("type") == "safetensors" and spec.get("path") not in paths:
                    raise ValueError("Every safetensors input must identify a pinned manifest asset")
        return self


class ProblemManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    source: dict[str, str]
    native_revision: Literal[NATIVE_REVISION]
    asset_source: dict[str, str] | None = None
    problems: list[Problem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tasks(self) -> "ProblemManifest":
        ids = [problem.task_id for problem in self.problems]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task IDs")
        return self


def load_manifest(path: Path, sha256: str) -> ProblemManifest:
    """Verify exact manifest bytes before accepting trusted definitions and paths."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Problem manifest SHA256 mismatch")
    return ProblemManifest.model_validate_json(data)


def checked_asset(root: Path, asset: Asset) -> Path:
    """Resolve and hash an asset without allowing a symlink to escape its root."""
    path = (root / safe_relative_path(asset.path)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Asset must be a regular file inside the manifest directory")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != asset.sha256:
        raise ValueError(f"Asset SHA256 mismatch: {asset.path}")
    return path
