# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters supply bounded, visible data; grading never follows host filesystem paths."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from resources_servers.partner_if.grading_contract import StrictModel


# A captured artifact over this many characters is refused, never truncated. The judge's input has its own
# limit (judge_max_input_chars).
MAX_EVIDENCE_CHARS = 1_000_000


class EvidenceUnavailable(ValueError):
    """An infrastructure capture gap, not an absent model deliverable."""


class EvidenceTooLarge(EvidenceUnavailable):
    """An artifact over MAX_EVIDENCE_CHARS, or a judge input over its limit."""


class Artifact(StrictModel):
    status: Literal["present", "absent", "error"]
    text: str = ""
    error: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.status != "present" and self.text:
            raise ValueError("Absent/error artifact must not contain surrogate text")
        if self.status == "error" and not self.error:
            raise ValueError("Capture error requires an explanation")
        if self.status != "error" and self.error:
            raise ValueError("Only an error artifact carries a capture error")
        return self


class VisibleEvent(StrictModel):
    source_index: int = Field(ge=0)
    kind: Literal["assistant", "tool_call", "tool_result", "context"]
    role: Literal["assistant", "system", "developer", "user", "tool"] | None = None
    text: str = ""
    tool_name: str | None = None
    call_id: str | None = None
    arguments: JsonValue = None


class EvidenceBundle(StrictModel):
    context: list[dict[str, JsonValue]] = Field(default_factory=list)
    instructions: str | None = None
    tools: list[dict[str, JsonValue]] = Field(default_factory=list)
    documents: dict[str, str] = Field(default_factory=dict)
    answer: str = ""
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    trajectory: list[VisibleEvent] = Field(default_factory=list)
    trajectory_error: str | None = None

    def artifact_text(self, name: str) -> str | None:
        artifact = self.artifacts.get(name)
        if artifact is None:
            raise EvidenceUnavailable(f"No capture receipt for artifact {name}")
        if artifact.status == "error":
            raise EvidenceUnavailable(f"Artifact {name}: {artifact.error}")
        if len(artifact.text) > MAX_EVIDENCE_CHARS:
            raise EvidenceTooLarge(f"Artifact {name} exceeds {MAX_EVIDENCE_CHARS:,} characters")
        return artifact.text if artifact.status == "present" else None

    def source_text(self, source: str) -> str | None:
        if source == "answer":
            return self.answer
        if source.startswith("artifact:") and source.removeprefix("artifact:"):
            return self.artifact_text(source.removeprefix("artifact:"))
        raise ValueError(f"Unsupported evidence source: {source}")
