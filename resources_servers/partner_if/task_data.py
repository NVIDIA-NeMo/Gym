# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light task contract shared by Gym data tooling and the IF verifier."""

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Dimension = Literal["if", "completion"]
# An exact `name==version` pin, with optional extras.
PinnedRequirement = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9]([\w.-]*[A-Za-z0-9])?(\[[\w.,-]+\])?==[A-Za-z0-9][\w.+!-]*$")
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifierSpec(StrictModel):
    id: Text
    params: dict[str, JsonValue] = Field(default_factory=dict)


class GradingItem(StrictModel):
    id: Text
    constraint: Text
    verifier: VerifierSpec
    dimension: Dimension = "if"
    category: Literal["always", "task_context"] = "always"
    depends_on: list[Text] = Field(default_factory=list)
    guidance: str = ""
    # Source grounding and analogue data are preserved but are not extra scoring rules.
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    taxonomy: list[Text] = Field(default_factory=list)
    analogues: list[dict[str, JsonValue]] = Field(default_factory=list)


class GradingContract(StrictModel):
    schema_version: Literal["partner-if-v4"] = "partner-if-v4"
    task_id: Text
    origin: Text
    source: Text
    rubric_version: Text
    audit_signoff: StrictBool
    data_kind: Literal["reported", "reconstruction", "public_task", "synthetic_smoke"]
    evidence_mode: Literal["text_response", "frozen_tool_history", "tool_call_response", "live_agent"] = (
        "text_response"
    )
    rubrics: list[GradingItem] = Field(min_length=1)
    # Gates use the same verifier registry but never earn points. For example,
    # citation presence must not be confused with full citation coverage.
    prerequisites: list[GradingItem] = Field(default_factory=list)

    @property
    def checks(self) -> list[GradingItem]:
        """All registered checks, including explicitly unscored prerequisite gates."""
        return self.prerequisites + self.rubrics

    @model_validator(mode="after")
    def validate_graph(self):
        by_id = {r.id: r for r in self.checks}
        if len(by_id) != len(self.checks):
            raise ValueError("Duplicate rubric IDs")
        if not any(r.dimension == "if" for r in self.rubrics):
            raise ValueError("An IF benchmark task must have an IF dimension")
        visited, visiting = set(), set()

        def visit(identifier: str) -> None:
            if identifier not in by_id:
                raise ValueError(f"Unknown prerequisite rubric: {identifier}")
            if identifier in visiting:
                raise ValueError("Cyclic rubric dependencies")
            if identifier in visited:
                return
            visiting.add(identifier)
            for dependency in by_id[identifier].depends_on:
                visit(dependency)
                if by_id[identifier].dimension == "if" and by_id[dependency].dimension == "completion":
                    raise ValueError("An IF requirement cannot depend on a completion requirement")
            visiting.remove(identifier)
            visited.add(identifier)

        for rubric in self.checks:
            if len(set(rubric.depends_on)) != len(rubric.depends_on):
                raise ValueError("Duplicate prerequisites")
            visit(rubric.id)
        return self


class VerifierMetadata(StrictModel):
    """What the verify wire reads from a row's `verifier_metadata`."""

    partner_if: GradingContract = Field(
        json_schema_extra={
            "legacy_location": "verifier_metadata",
            "consumed_by": ["verify", "metrics", "provenance"],
        }
    )
    # A benchmark's own annotations, which it validates. Never read by grading.
    curation: dict[str, JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)


class ToolBackendBundle(StrictModel):
    """A task's own `ToolBackend` (tool_backend.py) and its fixture, which a sandboxed mock-tool server runs in a
    fresh sandbox for each rollout. The actor never sees it, and the sandbox gets these files and the row's
    `responses_create_params`, never its contract."""

    # The backend class, as module:Class; the module is one of `files`.
    entrypoint: str = Field(pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    # Text files by path relative to the bundle's root, which is first on the backend's import path.
    files: dict[str, str] = Field(min_length=1)
    # The file in `files` that the backend class is constructed with.
    fixture: str
    # Packages the backend needs beyond the standard library, installed beside it.
    requirements: list[PinnedRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_layout(self):
        for path in self.files:
            parts = PurePosixPath(path).parts
            if str(PurePosixPath(path)) != path or path.startswith("/") or {".", ".."} & set(parts) or "\\" in path:
                raise ValueError(f"Bundle file paths must be normalized relative paths, not {path!r}")
        if self.fixture not in self.files:
            raise ValueError(f"The fixture {self.fixture!r} is not one of the bundle's files")
        module = self.entrypoint.partition(":")[0].replace(".", "/")
        if f"{module}.py" not in self.files and f"{module}/__init__.py" not in self.files:
            raise ValueError(f"The entrypoint module {module!r} is not one of the bundle's files")
        return self


class TaskData(VerifierMetadata):
    """A row's task-owned fields, flat, as Gym's data tooling validates them.

    Native rows add the top-level fields their harness reads: the Harbor agent's `task_name` (`HarborRunRequest`)
    and, on SWE rows, a Scale-SWE instance (`SWEIFVerifyRequest`), which that wire requires to have `instance_id`,
    `workdir` and `image_url`. Rows for a sandboxed mock-tool server add their `tool_backend`. Other rows omit
    them, so they are optional here.
    """

    task_name: str | None = Field(default=None, json_schema_extra={"consumed_by": ["run", "provenance"]})
    # The task's own tool backend, for a sandboxed mock-tool server.
    tool_backend: ToolBackendBundle | None = Field(default=None, json_schema_extra={"consumed_by": ["run"]})
    # The fields of a Scale-SWE instance (resources_servers/scale_swe/task_data.py).
    instance_id: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo: str | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    language: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify", "provenance"]})
    workdir: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    image_url: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    patch: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    pre_commands: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str | None = Field(default=None, json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    f2p_patch: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    f2p_script: str | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    FAIL_TO_PASS: str | list[str] | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: str | list[str] | None = Field(default=None, json_schema_extra={"consumed_by": ["verify"]})
