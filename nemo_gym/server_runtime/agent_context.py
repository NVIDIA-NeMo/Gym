# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-task data an agent session receives beyond its initial model input."""

import json
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


# Reserved dataset-row key, a sibling of verifier_metadata. It maps each participant role to the AgentContext that
# role's agent session receives; the Environment Server names its roles and refuses any other.
AGENT_CONTEXT_KEY_NAME = "agent_context"
# One session's whole context stays small: the seed is kept for the whole session and compared on every retry.
AGENT_CONTEXT_MAX_BYTES = 64 * 1024


class AgentContextArtifact(BaseModel):
    """A file or directory the agent reads, named by a path relative to a root the agent configures.

    The content never travels in the seed. The agent resolves the path when it seeds the session, reads or stages the
    content when its turn runs, and releases anything it staged when the session closes or retires for a checkpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def require_relative_path(cls, path: str) -> str:
        if "\\" in path or "\x00" in path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError(
                f"artifact path must be relative, use '/' separators and contain no '..' or NUL byte: {path!r}"
            )
        return path


class AgentContext(BaseModel):
    """Task data the agent's harness needs that is not part of the initial model input.

    Small typed fields go inline in ``values``; images, documents, files and skill directories go by reference in
    ``artifacts``. A key appears in one of the two. Each agent declares the keys it accepts and refuses a seed that
    carries any other key or misses a required one. A task row holds one context per participant role, and the
    Environment Server sends each role's context to that role's session.
    """

    # Frozen like AgentContextArtifact: the seed that holds it is stored for the session and compared on retries.
    model_config = ConfigDict(extra="forbid", frozen=True)

    values: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: dict[str, AgentContextArtifact] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_keys_and_size(self) -> "AgentContext":
        if shared := sorted(self.values.keys() & self.artifacts.keys()):
            raise ValueError(f"agent_context keys must appear in values or artifacts, not both: {shared}")
        size = len(json.dumps(self.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False).encode())
        if size > AGENT_CONTEXT_MAX_BYTES:
            raise ValueError(
                f"agent_context takes {size} bytes as UTF-8 JSON, over the {AGENT_CONTEXT_MAX_BYTES}-byte "
                "limit; pass large content as an artifact"
            )
        return self

    def fields(self) -> dict[str, JsonValue | AgentContextArtifact]:
        """Values and artifacts in one namespace, as an agent's context model validates them."""
        return {**self.values, **self.artifacts}
