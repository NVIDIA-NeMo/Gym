# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automatic job attribution (team / user / workload / run) for sandbox metadata.

Sandbox providers merge these keys into every sandbox's metadata so cluster
operators can attribute running sandboxes to the team, user, and workload that
created them (on OpenSandbox, metadata becomes queryable Kubernetes labels on
the sandbox). Each field resolves from explicit configuration first, then
``NEMO_GYM_*`` environment variables, then Slurm job environment variables,
then (for ``user`` only) the OS login name and (for ``workload``) the server
instance's config path. Fields that cannot be resolved are omitted rather than
guessed.

``run`` identifies one launch of the creating process so a run's sandboxes can
be listed or garbage-collected exactly, even when the same user runs the same
workload twice. It resolves from explicit configuration, then ``NEMO_GYM_RUN_ID``,
then a per-process generated id.
"""

import getpass
import logging
import os
import uuid
from collections.abc import Mapping


LOGGER = logging.getLogger(__name__)

TEAM_KEY = "team"
USER_KEY = "user"
WORKLOAD_KEY = "workload"
RUN_KEY = "run"

TEAM_ENV_VARS = ("NEMO_GYM_TEAM", "SLURM_JOB_ACCOUNT")
USER_ENV_VARS = ("NEMO_GYM_USER", "SLURM_JOB_USER")
# NEMO_GYM_CONFIG_PATH is set by the gym CLI on every server process it spawns
# (see nemo_gym.global_config.NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, not imported
# here to keep this module dependency-free) and names the server instance,
# which is the best automatic description of the workload.
WORKLOAD_ENV_VARS = ("NEMO_GYM_WORKLOAD", "SLURM_JOB_NAME", "NEMO_GYM_CONFIG_PATH")
RUN_ENV_VARS = ("NEMO_GYM_RUN_ID",)

# Container images default to running as root, so a "root" login attributes the
# image, not a person. Treat it as unresolved instead of emitting a false identity.
IGNORED_LOGIN_NAMES = frozenset({"root"})

_process_run_id: str | None = None
_logged_attribution = False


def _first_env(environ: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = (environ.get(name) or "").strip()
        if value:
            return value
    return None


def _detect_user(environ: Mapping[str, str]) -> str | None:
    value = _first_env(environ, USER_ENV_VARS)
    if value:
        return value
    try:
        login = getpass.getuser().strip()
    except Exception:
        # getpass raises on hosts with neither login env vars nor a passwd entry for the uid.
        return None
    if not login or login.lower() in IGNORED_LOGIN_NAMES:
        return None
    return login


def resolve_attribution(
    *,
    team: str | None = None,
    user: str | None = None,
    workload: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve ``team`` / ``user`` / ``workload`` attribution metadata.

    Args:
        team: Explicit team; falls back to ``NEMO_GYM_TEAM``, then ``SLURM_JOB_ACCOUNT``.
        user: Explicit user; falls back to ``NEMO_GYM_USER``, then ``SLURM_JOB_USER``,
            then the OS login name (``root`` is ignored — containers run as root by
            default, so it attributes the image, not a person).
        workload: Explicit workload; falls back to ``NEMO_GYM_WORKLOAD``, then
            ``SLURM_JOB_NAME``, then ``NEMO_GYM_CONFIG_PATH`` (the server instance
            name the gym CLI sets on every server process it spawns).
        environ: Environment mapping override, for testing. Defaults to ``os.environ``.

    Returns:
        A dict with only the resolved keys among ``team``, ``user``, and ``workload``.
        Values are returned raw; providers apply their own metadata sanitization.
    """
    environ = os.environ if environ is None else environ
    resolved: dict[str, str] = {}
    team = team or _first_env(environ, TEAM_ENV_VARS)
    if team:
        resolved[TEAM_KEY] = str(team)
    user = user or _detect_user(environ)
    if user:
        resolved[USER_KEY] = str(user)
    workload = workload or _first_env(environ, WORKLOAD_ENV_VARS)
    if workload:
        resolved[WORKLOAD_KEY] = str(workload)
    return resolved


def resolve_run_id(run: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Resolve the ``run`` attribution id: explicit value, then ``NEMO_GYM_RUN_ID``,
    then an id generated once per process.

    Unlike :func:`resolve_attribution` fields, ``run`` is always resolvable. It scopes
    sandboxes to one launch of the creating process, so an interrupted run's sandboxes
    can be listed and cleaned up exactly (``team`` / ``user`` / ``workload`` cannot
    distinguish two runs of the same workload by the same user).
    """
    global _process_run_id
    resolved = (run or "").strip() or _first_env(os.environ if environ is None else environ, RUN_ENV_VARS)
    if resolved:
        return resolved
    if _process_run_id is None:
        _process_run_id = uuid.uuid4().hex[:12]
    return _process_run_id


def log_attribution_once(metadata: Mapping[str, str]) -> None:
    """Log the resolved attribution metadata once per process.

    The generated ``run`` id only exists in this process, so surfacing it in the logs
    is what lets operators later filter or garbage-collect this run's sandboxes.
    """
    global _logged_attribution
    if _logged_attribution or not metadata:
        return
    _logged_attribution = True
    LOGGER.info("Sandbox attribution metadata: %s", dict(metadata))
