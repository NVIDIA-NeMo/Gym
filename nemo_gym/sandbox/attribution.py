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
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)

TEAM_KEY = "team"
USER_KEY = "user"
WORKLOAD_KEY = "workload"
RUN_KEY = "run"
PROCESS_KEY = "process"

TEAM_ENV_VARS = ("NEMO_GYM_TEAM", "SLURM_JOB_ACCOUNT")
USER_ENV_VARS = ("NEMO_GYM_USER", "SLURM_JOB_USER")
# NEMO_GYM_CONFIG_PATH is set by the gym CLI on every server process it spawns
# (see nemo_gym.config.global_config.NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, not imported
# here to keep this module dependency-free) and names the server instance,
# which is the best automatic description of the workload.
WORKLOAD_ENV_VARS = ("NEMO_GYM_WORKLOAD", "SLURM_JOB_NAME", "NEMO_GYM_CONFIG_PATH")
RUN_ENV_VARS = ("NEMO_GYM_RUN_ID",)

# Container images default to running as root, so a "root" login attributes the
# image, not a person. Treat it as unresolved instead of emitting a false identity.
IGNORED_LOGIN_NAMES = frozenset({"root"})

# Kubernetes label values: at most 63 characters of [A-Za-z0-9_.-], starting and ending alphanumeric.
LABEL_VALUE_MAX_LENGTH = 63
METADATA_VALUE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_ATTRIBUTION_KEY_PREFIX = "nemo-gym.nvidia.com/"
ATTRIBUTION_KEY_PREFIX_RE = re.compile(r"(?=.{1,253}$)[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*")

_process_run_id: str | None = None
_process_id_by_pid: dict[int, str] = {}
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


def resolve_process_id() -> str:
    """An id unique to this process (a forked child gets its own).

    ``run`` (see :func:`resolve_run_id`) can be shared by every process of a job and is overridable through
    ``NEMO_GYM_RUN_ID``, so it cannot tell this process's sandboxes from a sibling's. This id can, which is what
    lets a provider clean up only its own process's sandboxes.
    """
    pid = os.getpid()
    if pid not in _process_id_by_pid:
        _process_id_by_pid[pid] = uuid.uuid4().hex[:12]
    return _process_id_by_pid[pid]


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


def sanitize_label_value(value: object) -> str:
    """Make ``value`` a valid Kubernetes label value: invalid characters become ``_``, at most 63 characters.

    An empty value stays empty; a value with no valid character becomes ``metadata``.
    """
    if not str(value):
        return ""  # an empty label value is valid
    normalized = METADATA_VALUE_RE.sub("_", str(value)).strip("._-")
    normalized = normalized[:LABEL_VALUE_MAX_LENGTH].strip("._-")
    return normalized or "metadata"


@dataclass(frozen=True)
class AttributionConfig:
    """Job attribution merged into every sandbox's metadata (Kubernetes labels on OpenSandbox and OpenShell).

    Providers that propagate sandbox metadata as Kubernetes labels make attribution queryable at the
    cluster level (e.g. ``kubectl get pods -l nemo-gym.nvidia.com/team=my-team``). ``key_prefix``
    namespaces the label keys (Kubernetes prefixed-key convention); set it to ``""`` for bare
    ``team`` / ``user`` / ``workload`` / ``run`` keys.

    Unset fields are auto-detected: ``NEMO_GYM_TEAM`` / ``NEMO_GYM_USER`` / ``NEMO_GYM_WORKLOAD``
    environment variables first, then Slurm job env vars (``SLURM_JOB_ACCOUNT`` /
    ``SLURM_JOB_USER`` / ``SLURM_JOB_NAME``), then the OS login name for ``user`` (``root`` is
    ignored) and the gym CLI's ``NEMO_GYM_CONFIG_PATH`` server instance name for ``workload``.
    Fields that cannot be resolved are omitted. ``run`` scopes sandboxes to one launch of the
    creating process (``NEMO_GYM_RUN_ID``, else generated per process and logged) so a run's
    sandboxes can be listed and cleaned up exactly. Explicit ``SandboxSpec.metadata`` keys
    always take precedence over attribution keys.
    """

    enabled: bool = True
    team: str | None = None
    user: str | None = None
    workload: str | None = None
    run: str | None = None
    key_prefix: str = DEFAULT_ATTRIBUTION_KEY_PREFIX

    def __post_init__(self) -> None:
        normalized = self.key_prefix.strip()
        if normalized and not normalized.endswith("/"):
            normalized += "/"
        if normalized and not ATTRIBUTION_KEY_PREFIX_RE.fullmatch(normalized[:-1]):
            # Invalid prefixes would otherwise fail server-side at sandbox create with an
            # opaque error; values are sanitized by the provider but keys pass through.
            raise ValueError(
                f"attribution key_prefix {self.key_prefix!r} must be a lowercase DNS subdomain "
                "(Kubernetes label-key prefix), e.g. 'nemo-gym.nvidia.com/', or '' for bare keys"
            )
        object.__setattr__(self, "key_prefix", normalized)

    def metadata(self) -> dict[str, str]:
        """The prefixed attribution keys for a new sandbox (empty when disabled); values are raw."""
        if not self.enabled:
            return {}
        resolved = resolve_attribution(team=self.team, user=self.user, workload=self.workload)
        resolved[RUN_KEY] = resolve_run_id(self.run)
        metadata = {f"{self.key_prefix}{key}": value for key, value in resolved.items()}
        log_attribution_once(metadata)
        return metadata
