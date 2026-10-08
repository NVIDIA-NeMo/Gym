# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained task rows; grading assets stay outside the model input."""

from base64 import b64encode
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PhaseLimits(BaseModel):
    cpus: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    storage_mb: int = Field(gt=0)
    timeout_sec: float = Field(gt=0, allow_inf_nan=False)
    env: dict[str, str] = Field(default_factory=dict)


class TaskFiles(BaseModel):
    """Original file text, including trailing newlines; destinations are fixed by the server."""

    test_script: str
    test_patch: str
    grader: str
    grader_config: str
    solve_script: str
    solution_patch: str

    def verification_files(self, model_patch: bytes) -> dict[str, str]:
        return {
            "/tests/test.sh": self.test_script,
            "/tests/test.patch": self.test_patch,
            "/tests/grader.py": self.grader,
            "/tests/config.json": self.grader_config,
            "/logs/artifacts/model.patch.b64": b64encode(model_patch).decode("ascii"),
        }

    def solution_files(self) -> dict[str, str]:
        return {"/solution/solve.sh": self.solve_script, "/solution/solution.patch": self.solution_patch}


VERIFY = {"consumed_by": ["verify"]}


class TaskData(BaseModel):
    """Task-owned JSONL fields, separate from responses_create_params.input."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$", json_schema_extra=VERIFY)
    image: str = Field(min_length=1, json_schema_extra=VERIFY)
    verifier_image: str = Field(min_length=1, json_schema_extra=VERIFY)
    workdir: Literal["/app", "/workspace/repo"] = Field(default="/app", json_schema_extra=VERIFY)
    base_commit: str = Field(pattern=r"^[a-f0-9]{40}$", json_schema_extra=VERIFY)
    agent: PhaseLimits = Field(json_schema_extra=VERIFY)
    verifier: PhaseLimits = Field(json_schema_extra=VERIFY)
    solution_timeout_sec: float = Field(default=1800, gt=0, allow_inf_nan=False, json_schema_extra=VERIFY)
    collect_timeout_sec: float = Field(default=300, gt=0, allow_inf_nan=False, json_schema_extra=VERIFY)
    grader_applies_model_patch: bool = Field(
        default=True,
        description=(
            "DeepSWE v1.0 graders apply /logs/artifacts/model.patch themselves; v1.1 graders grade the tree as-is, "
            "so when this is False the server applies the candidate patch in the verifier workdir before test.sh."
        ),
        json_schema_extra=VERIFY,
    )
    files: TaskFiles = Field(json_schema_extra=VERIFY)
    public_source: dict[str, str] | None = Field(
        default=None,
        description="Optional public-example provenance; not used by the verifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
