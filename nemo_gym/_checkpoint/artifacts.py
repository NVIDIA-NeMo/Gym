# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Versioned checkpoint artifacts shared by Gym participants."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN, capture_key_for


CHECKPOINT_ARTIFACT_SCHEMA_VERSION = 1
AGENT_CONTINUATION_INDEX_FEATURE = "agent_continuation_index_v1"
EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE = "external_storage_reference_index_v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class CheckpointArtifactError(ValueError):
    """A checkpoint artifact is missing, corrupt, or structurally invalid."""


class CheckpointArtifactReference(BaseModel):
    """Small, digest-bound coordinate for a potentially large artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    records: int = Field(ge=0)
    bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_relative_path(self) -> "CheckpointArtifactReference":
        path = Path(self.relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("checkpoint artifact path must be a safe relative path")
        return self


class AgentContinuationRoot(BaseModel):
    """The last committed model call from which one parked agent can resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    capture_key: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    last_committed_model_call_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_capture_key(self) -> "AgentContinuationRoot":
        expected = capture_key_for(self.rollout_id, self.attempt_index)
        if self.capture_key != expected:
            raise ValueError(
                "continuation capture_key does not match rollout identity: "
                f"expected={expected!r}, actual={self.capture_key!r}"
            )
        return self


class ExternalStorageReference(BaseModel):
    """Opaque external token-storage row needed by a parked continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    capture_key: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    boundary_model_call_id: str = Field(min_length=1)
    kind: Literal["token_capture_staging"] = "token_capture_staging"
    key: str = Field(min_length=1)


_ArtifactRecord = TypeVar("_ArtifactRecord", bound=BaseModel)


def _artifact_path(checkpoint_root: Path, reference: CheckpointArtifactReference) -> Path:
    root = Path(checkpoint_root).resolve()
    path = (root / reference.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CheckpointArtifactError(f"checkpoint artifact escapes its root: {reference.relative_path!r}") from error
    return path


def write_jsonl_artifact(
    checkpoint_root: Path,
    relative_path: str | Path,
    records: Iterable[BaseModel],
) -> CheckpointArtifactReference:
    """Atomically write one fsynced JSONL artifact and return its coordinate."""
    root = Path(checkpoint_root).resolve()
    requested = Path(relative_path)
    placeholder = CheckpointArtifactReference(
        relative_path=str(requested),
        sha256=hashlib.sha256(b"").hexdigest(),
        records=0,
        bytes=0,
    )
    target = _artifact_path(root, placeholder)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    record_count = 0
    byte_count = 0
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".artifact-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for record in records:
                payload = record.model_dump_json().encode() + b"\n"
                handle.write(payload)
                digest.update(payload)
                record_count += 1
                byte_count += len(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)
    return CheckpointArtifactReference(
        relative_path=str(target.relative_to(root)),
        sha256=digest.hexdigest(),
        records=record_count,
        bytes=byte_count,
    )


def read_jsonl_artifact(
    checkpoint_root: Path,
    reference: CheckpointArtifactReference,
    record_type: type[_ArtifactRecord],
) -> list[_ArtifactRecord]:
    """Validate and parse an artifact without trusting its containing manifest."""
    path = _artifact_path(checkpoint_root, reference)
    if not path.is_file():
        raise CheckpointArtifactError(f"checkpoint artifact is missing: {reference.relative_path!r}")

    digest = hashlib.sha256()
    records: list[_ArtifactRecord] = []
    byte_count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            payload = line.strip()
            if not payload:
                continue
            try:
                raw: Any = json.loads(payload)
                records.append(record_type.model_validate(raw))
            except Exception as error:
                raise CheckpointArtifactError(
                    f"invalid checkpoint artifact row at {reference.relative_path}:{line_number}"
                ) from error

    if digest.hexdigest() != reference.sha256:
        raise CheckpointArtifactError(f"checkpoint artifact digest mismatch: {reference.relative_path!r}")
    if byte_count != reference.bytes:
        raise CheckpointArtifactError(f"checkpoint artifact byte count mismatch: {reference.relative_path!r}")
    if len(records) != reference.records:
        raise CheckpointArtifactError(f"checkpoint artifact record count mismatch: {reference.relative_path!r}")
    return records
