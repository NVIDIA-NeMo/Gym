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

"""Automatic job attribution (team / user / workload) for sandbox metadata.

Sandbox providers merge these keys into every sandbox's metadata so cluster
operators can attribute running sandboxes to the team, user, and workload that
created them (on OpenSandbox, metadata becomes queryable Kubernetes labels on
the sandbox). Each field resolves from explicit configuration first, then
``NEMO_GYM_*`` environment variables, then Slurm job environment variables,
then (for ``user`` only) the OS login name. Fields that cannot be resolved are
omitted rather than guessed.
"""

import getpass
import logging
import os
from collections.abc import Mapping


LOGGER = logging.getLogger(__name__)

TEAM_KEY = "team"
USER_KEY = "user"
WORKLOAD_KEY = "workload"

TEAM_ENV_VARS = ("NEMO_GYM_TEAM", "SLURM_JOB_ACCOUNT")
USER_ENV_VARS = ("NEMO_GYM_USER", "SLURM_JOB_USER")
WORKLOAD_ENV_VARS = ("NEMO_GYM_WORKLOAD", "SLURM_JOB_NAME")


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
        return getpass.getuser().strip() or None
    except Exception:
        # getpass raises on hosts with neither login env vars nor a passwd entry for the uid.
        return None


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
            then the OS login name.
        workload: Explicit workload; falls back to ``NEMO_GYM_WORKLOAD``, then
            ``SLURM_JOB_NAME``.
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
