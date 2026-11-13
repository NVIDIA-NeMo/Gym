# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the files an agent session's agent_context names."""

import errno
from pathlib import Path
from typing import Literal

from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.server_runtime.agent_context import AgentContextArtifact


# Errors the path itself causes, which a retry meets again. Any other OSError (an I/O error on shared storage)
# propagates, so the seed fails as a transient error.
_PATH_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.ENAMETOOLONG, errno.ELOOP, errno.ENOTDIR, errno.ENOENT})


def resolve_agent_context_artifact(
    artifact: AgentContextArtifact, root: Path, *, kind: Literal["file", "directory"]
) -> Path:
    """The artifact's path under ``root``, checked to stay inside it, through symlinks too, and to be a ``kind``.

    Raises AgentSeedRejectedError otherwise, or when the path cannot be looked up (too long, no permission), so an
    agent that resolves its artifacts while seeding refuses the session with a terminal 422 instead of failing its
    first activation. The check runs on resolved paths: a directory under ``root`` that is a symlink to other
    storage counts as outside, so configure the real directory as the root.

    For a ``directory``, only the directory itself is checked. Files under it can be symlinks to anywhere, so an
    agent that reads them resolves each one and checks that it stays under the returned directory.
    """
    try:
        base = root.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        # The agent's configuration, not the task, is wrong; say so instead of blaming the artifact.
        raise AgentSeedRejectedError(f"the agent's agent_context root {root} does not exist") from None
    try:
        path = (base / artifact.path).resolve(strict=True)
        inside = path.is_relative_to(base)
        is_kind = inside and (path.is_file() if kind == "file" else path.is_dir())
    except (FileNotFoundError, NotADirectoryError):
        raise AgentSeedRejectedError(
            f"agent_context artifact {artifact.path!r} is not a {kind} under {root}"
        ) from None
    except ValueError as error:
        # An embedded NUL byte, in an artifact built without validation.
        raise AgentSeedRejectedError(f"agent_context artifact {artifact.path!r} cannot be resolved: {error}") from None
    except OSError as error:
        if error.errno not in _PATH_ERRNOS:
            raise
        raise AgentSeedRejectedError(
            f"agent_context artifact {artifact.path!r} cannot be resolved under {root}: {error.strerror}"
        ) from None
    if not inside:
        raise AgentSeedRejectedError(
            f"agent_context artifact {artifact.path!r} resolves outside {base}; when the content lives on other "
            "storage through a symlink, configure the real directory as the root, not the symlink"
        )
    if not is_kind:
        raise AgentSeedRejectedError(f"agent_context artifact {artifact.path!r} is not a {kind} under {base}")
    return path
