# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDP file-task inputs for agents borrowing a Resources-owned sandbox."""

import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self


WORKDIR = "/workspace"
INPUT_DIR = f"{WORKDIR}/input"
OUTPUT_DIR = f"{WORKDIR}/output"
_PROMPT = Path(__file__).parent / "prompts" / "nooa_user_prompt.txt"


def relative_file(value: str) -> str:
    """Validate one canonical relative POSIX path before using it on either host."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe GDP file path: {value!r}")
    if path.as_posix() != value or value == ".":
        raise ValueError(f"Noncanonical GDP file path: {value!r}")
    return value


class GDPFileTask(BaseModel):
    """Only public task inputs; rubrics and comparison references stay with Resources."""

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
        normalized = []
        for name in self.reference_files:
            relative_file(name)
            # The canonical dataset prefixes paths with reference_files/. The
            # grader reads that directory nonrecursively, as Stirrup exports it.
            normalized.append(name.removeprefix("reference_files/"))
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate reference paths after reference_files/ normalization")
        for url in self.reference_file_urls:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Reference URLs must be HTTPS URLs without embedded credentials")
        return self


def prepare_row(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Make an explicit native task while preserving verifier-only task metadata."""
    task = GDPFileTask.model_validate(row)
    references = "\n".join(f"- {INPUT_DIR}/{name}" for name in task.reference_files) or "None"
    prompt = _PROMPT.read_text().format(reference_files=references, task=task.prompt)
    task_data = dict(row)
    original = task_data.pop("responses_create_params", None) or {}
    if not isinstance(original, dict):
        raise ValueError("responses_create_params must be an object")
    params = dict(original)
    params["input"] = [{"role": "user", "content": prompt}]
    params.update(max_output_tokens=32768, temperature=1.0, top_p=1.0)
    # Normalize the legacy JSON-encoded arrays for the existing verify model.
    task_data["reference_files"] = task.reference_files
    task_data["reference_file_urls"] = task.reference_file_urls
    return {
        "task_id": {"taskset": "gdpval-nooa", "task_id": task.task_id},
        "task_input": {"responses_create_params": params, "task_data": task_data},
    }
