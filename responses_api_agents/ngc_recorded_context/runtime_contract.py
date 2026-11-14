# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated data contract for a frozen replay, independent of its private case."""

import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OrderingContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    task_names: list[str]
    rubric_version: str
    criterion_id: str
    probe_namespace: str


class RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["recorded-context-runtime-v1"]
    input_hashes: dict[str, str]
    source_root: str
    source_schemas: dict[str, str]
    probe_script: str
    path_aliases: dict[str, str] = Field(default_factory=dict)
    answer_paths: list[str]
    workspace_prefix: str
    status_script: str
    session_id_path: str
    transcript_template: str
    optional_runtime_paths: list[str] = Field(default_factory=list)
    get_wrappers: dict[str, str] = Field(default_factory=dict)
    ordering: OrderingContract

    @field_validator("input_hashes")
    @classmethod
    def checked_files(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if len(PurePosixPath(name).parts) != 1 or name in {".", ".."} or not name:
                raise ValueError("Replay asset names must be simple filenames")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Replay assets require SHA-256 identities")
        if not {"case.json", "bundle.json", "actor-params.json"} <= value.keys():
            raise ValueError("Replay contract omits required actor assets")
        return value

    @model_validator(mode="after")
    def checked_paths(self) -> "RuntimeContract":
        paths = [
            self.source_root,
            self.probe_script,
            self.workspace_prefix,
            self.status_script,
            self.session_id_path,
            self.transcript_template,
            *self.answer_paths,
            *self.optional_runtime_paths,
            *self.get_wrappers,
        ]
        if any(not p.startswith("/") or ".." in PurePosixPath(p).parts for p in paths):
            raise ValueError("Replay paths must be absolute and cannot traverse parents")
        if not self.answer_paths or any(not p.startswith(self.workspace_prefix) for p in self.answer_paths):
            raise ValueError("Answer artifacts must stay inside the modeled workspace")
        if self.transcript_template.count("{session_id}") != 1:
            raise ValueError("Transcript path requires exactly one session_id placeholder")
        if set(self.source_schemas) != {"coverage", "inventory", "extra"}:
            raise ValueError("Replay source schemas must be declared explicitly")
        return self


def read_runtime_contract(path: Path, *, expected_sha256: str | None = None) -> RuntimeContract:
    """Read data only; externally supplied contracts must have an explicit pin."""
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Replay assets require runtime.json. For a version-1 package, configure an external "
            "runtime_contract_path with runtime_contract_sha256, or prepare a pinned version-2 candidate."
        ) from error
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Replay runtime contract checksum mismatch")
    return RuntimeContract.model_validate_json(data)
