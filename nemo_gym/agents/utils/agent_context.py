# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the files an agent session's agent_context names."""

from pathlib import Path
from typing import Literal

from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.server_runtime.agent_context import AgentContextArtifact


def resolve_agent_context_artifact(
    artifact: AgentContextArtifact, root: Path, *, kind: Literal["file", "directory"]
) -> Path:
    """The artifact's path under ``root``, checked to stay inside it, through symlinks too, and to be a ``kind``.

    Raises AgentSeedRejectedError otherwise, so an agent that resolves its artifacts while seeding refuses the
    session with a final 422 instead of failing its first activation.
    """
    base = root.resolve()
    path = (base / artifact.path).resolve()
    if not path.is_relative_to(base):
        raise AgentSeedRejectedError(f"agent_context artifact {artifact.path!r} resolves outside {base}")
    if not (path.is_file() if kind == "file" else path.is_dir()):
        raise AgentSeedRejectedError(f"agent_context artifact {artifact.path!r} is not a {kind} under {base}")
    return path
