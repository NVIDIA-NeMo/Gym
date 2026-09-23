# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable, script-graded SWE tasks. Assets never become model input."""

import base64
import hashlib
import zlib
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TASK_BYTES = 128 * 1024 * 1024
VERIFY_FIELD = {"consumed_by": ["verify"], "legacy_location": "verifier_metadata"}


class TaskFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    content_b64: str = Field(max_length=2 * MAX_FILE_BYTES)
    encoding: Literal["base64", "gzip+base64"] = "base64"
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mode: int = Field(default=0o644, ge=0, le=0o777)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))
            or "\\" in value
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError("asset paths must be normalized relative POSIX paths")
        return value

    def decoded(self) -> bytes:
        data = base64.b64decode(self.content_b64, validate=True)
        if self.encoding == "gzip+base64":
            decoder = zlib.decompressobj(31)
            data = decoder.decompress(data, MAX_FILE_BYTES + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("invalid, oversized, or concatenated gzip asset")
        if len(data) > MAX_FILE_BYTES or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("asset size or checksum mismatch")
        return data


class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1, max_length=256, json_schema_extra=VERIFY_FIELD)
    image_ref: str = Field(min_length=1, json_schema_extra=VERIFY_FIELD)
    workdir: str = Field(json_schema_extra=VERIFY_FIELD)
    test_files: list[TaskFile] = Field(min_length=1, max_length=10000, json_schema_extra=VERIFY_FIELD)
    solution_files: list[TaskFile] = Field(default_factory=list, max_length=10000, json_schema_extra=VERIFY_FIELD)
    setup_script: str = Field(default="", json_schema_extra=VERIFY_FIELD)
    verifier_timeout_s: int = Field(default=300, gt=0, json_schema_extra=VERIFY_FIELD)
    solution_timeout_s: int = Field(default=1800, gt=0, json_schema_extra=VERIFY_FIELD)
    agent_timeout_s: int = Field(default=1800, gt=0, json_schema_extra=VERIFY_FIELD)
    cpu: float = Field(default=2, gt=0, allow_inf_nan=False, json_schema_extra=VERIFY_FIELD)
    memory_mib: int = Field(default=8192, gt=0, json_schema_extra=VERIFY_FIELD)
    disk_gib: int = Field(default=10, gt=0, json_schema_extra=VERIFY_FIELD)

    @field_validator("workdir")
    @classmethod
    def absolute_workdir(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or any(ord(char) < 32 for char in value):
            raise ValueError("workdir must be an absolute POSIX path without traversal")
        return value

    @model_validator(mode="after")
    def validate_assets(self):
        total = 0
        for files in (self.test_files, self.solution_files):
            names = {asset.path for asset in files}
            if len(names) != len(files):
                raise ValueError("duplicate asset paths")
            for asset in files:
                if any(str(parent) in names for parent in PurePosixPath(asset.path).parents):
                    raise ValueError("asset path conflicts with a directory")
                total += len(asset.decoded())
        if total > MAX_TASK_BYTES:
            raise ValueError("task assets exceed the size limit")
        if "test.sh" not in {asset.path for asset in self.test_files}:
            raise ValueError("test_files must contain test.sh")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class TaskRow(BaseModel):
    """Current wire shape: the server reads task settings from verifier_metadata."""

    model_config = ConfigDict(extra="allow")
    verifier_metadata: TaskMetadata


class TaskData(TaskMetadata):
    """Gym's schema tooling flattens verifier_metadata before validating a row."""

    model_config = ConfigDict(extra="allow")
    provenance: dict[str, Any] | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    public_source: dict[str, Any] | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
