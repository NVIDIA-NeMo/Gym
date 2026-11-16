# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Where servers and agents put the files they write while they run.

A server starts with its own source folder as the working directory.
A relative path, or one built from ``Path.cwd()``, therefore writes into the Gym checkout.
On a cluster that checkout is on a shared filesystem, and every such write loads its metadata servers.
Each kind of runtime write has one place instead:

- Scratch, needed only while a request or session runs: :func:`scratch_dir` or :func:`async_scratch_dir`.
  The folder is under the node's temporary directory.
  It is under ``shared_scratch_dir`` instead when another process or node reads it.
  It is removed when the block exits.
- Debug output, such as verification logs, screenshots, trajectories and failure artifacts: :func:`debug_artifact_dir`.
  It is off unless ``debug_artifacts_dir`` or the server's own setting names a folder.
- Caches: the global ``cache_dir``.
- Run output that a run reports or reads back to resume: ``<results_dir>/<server name>/``.
"""

import asyncio
import logging
import re
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from nemo_gym.config.global_config import (
    DEBUG_ARTIFACTS_DIR_KEY_NAME,
    SHARED_SCRATCH_DIR_KEY_NAME,
    maybe_get_global_config_dict,
)
from nemo_gym.core.episode_context import current_rollout_id


logger = logging.getLogger(__name__)

_KIND = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Keeps the leaf folder name well under the 255-byte limit after the timestamp and random suffix.
_MAX_LABEL_LENGTH = 100
NO_ROLLOUT_SEGMENT = "no-rollout"
UNNAMED_AGENT_INSTANCE_SEGMENT = "unnamed"


def debug_artifact_dir(
    *, server_name: str, kind: str, label: str | None = None, override: Path | None = None
) -> Path | None:
    """Return a new folder for one unit of work's debug output, or ``None`` when debug output is off.

    The root is ``override`` when the server's own setting names a folder.
    Otherwise it is ``<debug_artifacts_dir>/<server_name>`` when the global setting names one.
    Otherwise there is none.
    Below the root the folder is ``<kind>/<capture key>/<agent instance>/<UTC time>-[<label>-]<random>``.
    The capture key and agent instance come from the request context.
    So every server separates rollouts and agent instances the same way.
    Outside a rollout those segments are ``no-rollout`` and ``unnamed``.
    The random part keeps two calls from sharing a folder.
    That covers two calls in one rollout, and two runs that reuse task and rollout indices.

    Call this once per unit of work that owns the files.
    That is once per request, or once when a session opens, keeping the folder on the session.
    Nothing is created here; the folder exists only if the caller writes into it.
    """
    if not _KIND.fullmatch(kind):
        raise ValueError(f"kind must be lowercase letters, digits, '-' and '_', got {kind!r}")
    if override is not None:
        if not override.is_absolute():
            raise ValueError(
                f"A debug output folder must be an absolute path, got {str(override)!r}: a server resolves a "
                "relative path against its own source folder."
            )
        root = override
    else:
        global_root = (maybe_get_global_config_dict() or {}).get(DEBUG_ARTIFACTS_DIR_KEY_NAME)
        if not global_root:
            return None
        root = Path(global_root) / quote(server_name, safe="")

    leaf = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if label:
        leaf += "-" + quote(label, safe="")[:_MAX_LABEL_LENGTH]
    leaf += "-" + uuid4().hex[:12]
    rollout = current_rollout_id() or NO_ROLLOUT_SEGMENT
    return root / kind / rollout / (_current_agent_instance() or UNNAMED_AGENT_INSTANCE_SEGMENT) / leaf


def _current_agent_instance() -> str | None:
    """The agent instance the current request runs as, or ``None`` when it is unnamed.

    Requests carry no agent instance name in context yet.
    Once named agent sessions set one, this function reads it, and callers do not change.
    """
    return None


def _scratch_parent(shared: bool) -> Path | None:
    if not shared:
        return None
    configured = (maybe_get_global_config_dict() or {}).get(SHARED_SCRATCH_DIR_KEY_NAME)
    if not configured:
        # A process without Gym's configuration, such as a unit test, runs on one node.
        return None
    parent = Path(configured)
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _create_scratch(kind: str, shared: bool) -> Path:
    if not _KIND.fullmatch(kind):
        raise ValueError(f"kind must be lowercase letters, digits, '-' and '_', got {kind!r}")
    return Path(tempfile.mkdtemp(prefix=f"nemo-gym-{kind}-", dir=_scratch_parent(shared)))


def _finish_scratch(path: Path, keep_as: Path | None) -> None:
    if keep_as is not None:
        # Debug output must never fail the request it describes.
        try:
            keep_as.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, keep_as)
        except Exception:
            logger.warning("Could not keep scratch folder %s as debug output at %s", path, keep_as, exc_info=True)
    shutil.rmtree(path, ignore_errors=True)


@contextmanager
def scratch_dir(*, kind: str, shared: bool = False, keep_as: Path | None = None) -> Iterator[Path]:
    """Create a scratch folder for one request or session and remove it when the block exits.

    The folder is under the node's temporary directory (``TMPDIR``).
    Pass ``shared=True`` when another process or node reads the folder by path.
    The folder is then under ``shared_scratch_dir``, which every node of a run sees.
    Prefer passing such data in the request when the reader can take it.

    ``keep_as`` is a folder from :func:`debug_artifact_dir`.
    When it is set, the scratch folder is moved there on exit instead of being deleted.
    A server then writes to one place whether debug output is on or off.

    This blocks on filesystem calls; use :func:`async_scratch_dir` in a coroutine.
    """
    path = _create_scratch(kind, shared)
    try:
        yield path
    finally:
        _finish_scratch(path, keep_as)


@asynccontextmanager
async def async_scratch_dir(*, kind: str, shared: bool = False, keep_as: Path | None = None) -> AsyncIterator[Path]:
    """:func:`scratch_dir` for a coroutine: creating and removing the folder run in a worker thread."""
    path = await asyncio.to_thread(_create_scratch, kind, shared)
    try:
        yield path
    finally:
        # Shielded, so a second cancellation while the folder is being removed does not leave it behind.
        await asyncio.shield(asyncio.to_thread(_finish_scratch, path, keep_as))
