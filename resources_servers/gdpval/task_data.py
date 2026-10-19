# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gdpval server.

GDPVal economically-valuable-work rows, all fields top-level (no ``verifier_metadata``).
Required-ness mirrors ``GDPValVerifyRequest`` (app.py:236): ``task_id`` is the only required
field. ``deliverables_dir`` and ``reference_ids`` on that request are injected per verify call by
the agent/driver (deliverable collection and comparison-mode reference filtering) and are NOT
dataset columns. ``reference_files`` is the reverse case: present in the committed data but
undeclared on the wire model, so today it is silently dropped at the verify boundary (default
Pydantic ``extra="ignore"``); it is declared here so tooling sees the column.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description=(
            "GDPVal task UUID; wire-required. Seeds the per-task rng (make_rng), locates the reference "
            "deliverable directory (task_{task_id}) in comparison mode, and tags logs."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    sector: Optional[str] = Field(
        default=None,
        description="Economic sector label (e.g. 'Professional, Scientific, and Technical Services'); "
        "declared on the wire but pass-through only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    occupation: Optional[str] = Field(
        default=None,
        description=(
            "Occupation label (e.g. 'Accountants and Auditors'); not read by verify() but used for "
            "stratified task sampling by multistage_orchestrator.py:164."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Full task prompt; read as `body.prompt or ''` and interpolated into the judge "
        "prompt as the task description.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric_json: Optional[Any] = Field(
        default=None,
        description=(
            "Rubric for rubric-mode scoring. Typed Any on the wire; in the committed data it is a "
            "JSON-encoded string (~12KB) of an array of {score: int, criterion: str, required, "
            "rubric_item_id, ...} objects — per-criterion 'score'/'weight' entries weight structured-mode "
            "scoring. At least one of rubric_json/rubric_pretty must be present for rubric mode "
            "(app.py:368)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric_pretty: Optional[str] = Field(
        default=None,
        description="Human-readable rubric text ('[+2] ...' lines); read as `body.rubric_pretty or ''` "
        "and shown to the judge alongside/instead of rubric_json.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    reference_file_urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Public URLs of task input files (huggingface openai/gdpval reference_files); declared on the "
            "wire but unread by verify() — the agent downloads them when building the workspace."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    reference_files: Optional[List[str]] = Field(
        default=None,
        description=(
            "Repo-relative local paths of the same task input files. Present in the committed data but "
            "UNDECLARED on GDPValVerifyRequest, so it is silently dropped at the verify boundary today."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )


WORKDIR = "/workspace"
INPUT_DIR = f"{WORKDIR}/input"
OUTPUT_DIR = f"{WORKDIR}/output"


def relative_file(value: str) -> str:
    """Accept only unambiguous, relative POSIX file paths."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe reference path: {value!r}")
    if path.as_posix() != value or value == ".":
        raise ValueError(f"Noncanonical reference path: {value!r}")
    return value


class GDPFileTask(BaseModel):
    """Validate task inputs without exposing verifier metadata to the harness."""

    model_config = ConfigDict(extra="ignore")
    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference_files: list[str] = Field(default_factory=list)
    reference_file_urls: list[str] = Field(default_factory=list)

    @field_validator("reference_files", "reference_file_urls", mode="before")
    @classmethod
    def parse_lists(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if len(self.reference_files) != len(self.reference_file_urls):
            raise ValueError("Every reference file must have one download URL")
        if len(set(self.reference_files)) != len(self.reference_files):
            raise ValueError("Duplicate reference paths")
        for name in self.reference_files:
            relative_file(name)
        for url in self.reference_file_urls:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Reference URLs must be HTTPS URLs without credentials")
        return self


def prepare_row(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Adapt a GDP row for single_agent_turn, retaining verifier-only metadata."""
    task = GDPFileTask.model_validate(row)
    references = "\n".join(f"- {INPUT_DIR}/{name}" for name in task.reference_files) or "None"
    prompt = (
        f"Complete the following task using your sandbox tools. Work in {WORKDIR}.\n"
        f"Reference files (do not modify):\n{references}\n\n"
        f"Save all final deliverables directly in {OUTPUT_DIR}, with no subdirectories. "
        "Only files in that directory will be submitted. Keep scripts, logs and scratch files elsewhere. "
        "When finished, reply with a brief summary and the names of your deliverables.\n\n"
        f"Task:\n{task.prompt}"
    )
    result = dict(row)
    # Seeding accepts JSON-encoded lists, but the verifier requires actual lists.
    result["reference_files"] = task.reference_files
    result["reference_file_urls"] = task.reference_file_urls
    params = dict(row.get("responses_create_params") or {})
    params["input"] = [{"role": "user", "content": prompt}]
    result["responses_create_params"] = params
    return result


def prepare_task(row: dict[str, JsonValue], *, taskset: str) -> MaterializedTask[SingleAgentTurnTaskInput]:
    """Prepare a single_agent_turn task while keeping verifier metadata outside the model prompt."""
    prepared = prepare_row(row)
    params = prepared.pop("responses_create_params")
    return MaterializedTask[SingleAgentTurnTaskInput](
        task_id=TaskId(taskset=taskset, task_id=prepared["task_id"]),
        task_input=SingleAgentTurnTaskInput(responses_create_params=params, task_data=prepared),
    )


def main() -> None:
    """Prepare a separate JSONL; never overwrite the source dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taskset", help="Emit tasks for this taskset; use 'gdpval' with the sandbox recipes")
    args = parser.parse_args()
    source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [
        prepare_task(row, taskset=args.taskset).model_dump(mode="json")
        if args.taskset is not None
        else prepare_row(row)
        for row in source
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
