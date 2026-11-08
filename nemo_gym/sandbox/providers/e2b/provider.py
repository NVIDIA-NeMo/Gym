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

"""Sandbox provider backed by the E2B Python SDK.

Works against e2b.dev itself and against any e2b-compatible gateway (point
``connection.api_url``/``connection.sandbox_url`` at it).

Two E2B concepts differ from the provider-neutral :class:`SandboxSpec` and are
handled explicitly rather than silently:

**Templates, not images.** E2B starts sandboxes from a pre-built *template*
name or ID, not from an arbitrary registry reference. Because tagged template
names and OCI image references can both contain ``:``, the direct
``SandboxSpec.image`` shortcut is deliberately limited to unambiguous names in
``[A-Za-z0-9_-]``. Use ``provider_options.template`` or
``create.template_map`` for tagged names, IDs, and image references.

**Resources are fixed at template build time.** ``cpu_count``/``memory_mb`` are
arguments to the template *build*, so a per-sandbox
``SandboxSpec.resources`` cannot be honoured at create time. Requests are
reported once per provider instance (or raise, with ``create.strict_resources``)
instead of being dropped quietly.

**Snapshots keep processes.** A snapshot (``create_snapshot``) saves the whole
VM: memory, running processes and disk. A sandbox created from it resumes those
processes where they were, so the provider sets ``snapshot_keeps_processes``.
The snapshot is usable as soon as the call returns.

**Commands run in login shells.** The SDK starts every command as
``bash -l -c`` and every terminal as ``bash -i -l``. A process does start with
its image's environment, but a Debian image's ``/etc/profile`` then resets
``PATH``. Each new sandbox therefore gets a ``profile.d`` hook
(``create.restore_login_env``) that gives a login shell back the environment it
started with.
"""

import asyncio
import json
import logging
import math
import os
import re
import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable, TypeVar

from nemo_gym.sandbox.attribution import AttributionConfig
from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.e2b._sdk import E2B_SDK_CONSTRAINT, require_e2b_sdk


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Conservative direct-template shortcut. E2B also supports tagged names such
# as ``name:v1`` and template IDs, but those must be explicit because ``:`` and
# other punctuation overlap with OCI image syntax in ``SandboxSpec.image``.
_DIRECT_TEMPLATE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# How a deployment publishes sandbox ports. See ``E2BProvider.endpoint``.
_PORT_ROUTING_MODES = ("auto", "hostname", "gateway")

# Passed straight through to the SDK (``ApiParams``) on every call.
_API_PARAM_KEYS = (
    "api_key",
    "api_url",
    "sandbox_url",
    "domain",
    "debug",
    "validate_api_key",
    "headers",
    "api_headers",
    "proxy",
)

# ``create.network`` keys -> REST API field names.
_NETWORK_API_KEYS = {"allow_out": "allowOut", "deny_out": "denyOut"}


# Provider options a spec may carry. ``resource_requests`` is OpenSandbox's and is reported, not applied: E2B fixes
# resources when the template is built. ``network_policy`` maps onto E2B's ``network`` create option.
_PROVIDER_OPTION_KEYS = frozenset({"template", "resource_requests", "network_policy"})
_NETWORK_POLICY_KEYS = frozenset({"defaultAction", "egress"})

# Every command carries this marker in its environment, so the processes a timed-out command left behind can be
# found and killed. Never by process group: envd runs commands in its own group, and killing it ends the sandbox.
_EXEC_MARKER_ENV = "NEMO_GYM_E2B_EXEC"
_KILL_MARKED_SCRIPT = (
    'm={marker}; for d in /proc/[0-9]*; do p=${{d#/proc/}}; [ "$p" = "$$" ] && continue; '
    'while IFS= read -r -d "" e; do [ "$e" = "$m" ] && {{ kill -KILL "$p" 2>/dev/null; break; }}; '
    'done 2>/dev/null < "$d/environ"; done; true'
)
_KILL_MARKED_TIMEOUT_S = 30.0

# A sandbox process starts with its image's environment, but the SDK runs every command and terminal in a login shell,
# and a Debian image's /etc/profile resets PATH: the directories the image added (/usr/local/go/bin,
# /usr/local/cargo/bin, /opt/conda/bin, ...) are lost, and a test command fails with "go: command not found". Sourced
# last by every login shell, this hook puts back the environment the shell started with: its variables as they were,
# and PATH with the starting entries first, then any the profile added. dash also reads profile.d, so a shell that is
# not bash returns at the first line, before the bash-only syntax.
_LOGIN_ENV_HOOK_PATH = "/etc/profile.d/zz-nemo-gym-env.sh"
_LOGIN_ENV_HOOK = r"""# Written by NeMo Gym's E2B provider: give this login shell back the environment it started with.
[ -n "${BASH_VERSION-}" ] && [ -r "/proc/$$/environ" ] || return 0
__ng_path=
while IFS= read -r -d '' __ng_var; do
  case $__ng_var in
    PATH=*) __ng_path=${__ng_var#PATH=} ;;
    PWD=*|OLDPWD=*|SHLVL=*|_=*) ;;
    *=*) export "$__ng_var" 2>/dev/null ;;
  esac
done <"/proc/$$/environ"
__ng_merged=
while IFS= read -r -d: __ng_dir; do
  case ":$__ng_merged:" in
    *":$__ng_dir:"*) ;;
    *) [ -n "$__ng_dir" ] && __ng_merged=${__ng_merged:+$__ng_merged:}$__ng_dir ;;
  esac
done <<<"$__ng_path:$PATH:"
export PATH=$__ng_merged
unset __ng_path __ng_var __ng_merged __ng_dir
"""

# A command that raised a timeout this close to its deadline ran out its deadline (E2B ends the stream at, or a hair
# before, the configured timeout); an earlier timeout is a transport failure.
_DEADLINE_FRACTION = 0.99
_TIMEOUT_RETURN_CODE = 124

# A kill that finds no sandbox is checked again after this delay: a gateway replica that has not seen the sandbox can
# answer "not found" for one that still runs.
KILL_RECHECK_DELAY_S = 1.0

# A kill of a sandbox in its new-sandbox grace period (``operations.new_sandbox_grace_s``) that finds nothing is
# repeated after this delay until the sandbox is found or the grace period ends.
NEW_SANDBOX_KILL_RETRY_S = 1.0

# AgentENV answers a lease renewal of a sandbox it is killing with 500 "sandbox ... is in invalid state Killing".
_DYING_STATE = re.compile(r"invalid state Kill(?:ing|ed)\b")

# The resource warning is logged once per process; later templates are logged at DEBUG.
_resource_warning_logged = False

# Lease renewal: at most this many renewals at once, each cut off after this long.
LEASE_RENEW_BATCH = 64
LEASE_RENEW_TIMEOUT_S = 10.0

# Waiting for a new sandbox to become reachable (``create.reachable_timeout_s``): each no-op probe is cut off after
# REACHABLE_PROBE_TIMEOUT_S, and a 502 is retried after a delay that doubles up to REACHABLE_RETRY_MAX_DELAY_S.
REACHABLE_PROBE_TIMEOUT_S = 30.0
REACHABLE_RETRY_DELAY_S = 1.0
REACHABLE_RETRY_MAX_DELAY_S = 8.0


class E2BCreateError(SandboxCreateError):
    """Raised when a sandbox cannot be created."""


class E2BSandboxUnreachableError(RuntimeError):
    """Raised when the E2B gateway could not reach a sandbox it reports as running (HTTP 502, or the SDK's
    ``SandboxUnreachableException``); the sandbox may still be alive."""


class E2BSnapshotError(RuntimeError):
    """Raised when the installed e2b SDK or the service cannot snapshot a sandbox."""


def _is_not_found(error: BaseException | None, e2b: Any) -> bool:
    """Whether an SDK error says the resource does not exist, including a 404 the SDK could not parse."""
    if error is None:
        return False
    not_found = tuple(
        exc
        for exc in (getattr(e2b, "NotFoundException", None), getattr(e2b, "SandboxNotFoundException", None))
        if isinstance(exc, type)
    )
    if not_found and isinstance(error, not_found):
        return True
    return getattr(error, "status_code", None) == 404 or str(error).startswith("404:")


def _is_missing_template(error: BaseException | None) -> bool:
    """Whether a create failed because its template does not exist: AgentENV answers HTTP 400 "template ... not found"
    for a template or snapshot id it does not have."""
    if error is None:
        return False
    message = str(error)
    bad_request = getattr(error, "status_code", None) == 400 or message.startswith("400:")
    return bad_request and "not found" in message.lower()


def _is_unreachable(error: BaseException, e2b: Any) -> bool:
    unreachable = getattr(e2b, "SandboxUnreachableException", None)
    if isinstance(unreachable, type) and isinstance(error, unreachable):
        return True
    return getattr(error, "status_code", None) == 502 or str(error).startswith("502:")


def _network_options(network_policy: Any) -> dict[str, list[str]]:
    """Map an OpenSandbox-shaped ``network_policy`` onto E2B's ``network`` create option.

    ``{"defaultAction": "deny", "egress": [{"action": "allow", "target": T}, ...]}`` denies all egress but the allowed
    targets; ``defaultAction: allow`` denies only the targets marked ``deny``. Anything else is refused rather than
    applied loosely.
    """
    if not isinstance(network_policy, Mapping) or not set(network_policy) <= _NETWORK_POLICY_KEYS:
        raise ValueError(
            "E2B provider option 'network_policy' must be a mapping with only defaultAction and egress, "
            f"got {network_policy!r}"
        )
    default = network_policy.get("defaultAction", "allow")
    if default not in ("allow", "deny"):
        raise ValueError(f"network_policy.defaultAction must be 'allow' or 'deny', got {default!r}")
    targets: dict[str, list[str]] = {"allow": [], "deny": []}
    for rule in network_policy.get("egress") or []:
        if (
            not isinstance(rule, Mapping)
            or rule.get("action") not in targets
            or not isinstance(rule.get("target"), str)
        ):
            raise ValueError(f"network_policy.egress rules need action allow|deny and a target string, got {rule!r}")
        targets[rule["action"]].append(rule["target"])
    if default == "deny":
        return {"deny_out": ["0.0.0.0/0"], "allow_out": targets["allow"]}
    return {"deny_out": targets["deny"]} if targets["deny"] else {}


def _require_e2b_sdk() -> Any:
    """Load the optional SDK through the shared helper.

    Keep this small wrapper local so provider unit tests can replace the SDK
    without importing or contacting E2B.
    """
    return require_e2b_sdk("The e2b sandbox provider")


def _config_from_mapping(cls: type[T], value: Any) -> T:
    """Build a config dataclass from a mapping, rejecting unknown keys."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(value).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}. Expected: {', '.join(sorted(allowed))}"
        )
    return cls(**dict(value))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and (
        isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    )


def _validate_optional_number(name: str, value: Any, *, positive: bool) -> None:
    operator = "> 0" if positive else ">= 0"
    if value is not None and (not _is_finite_number(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be {operator}")


def _validate_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_nonnegative_number(name: str, value: Any) -> None:
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class E2BConnectionConfig:
    """Connection settings forwarded to the SDK.

    Any field left ``None`` falls back to the SDK's own environment variables
    (``E2B_API_KEY``, ``E2B_API_URL``, ``E2B_SANDBOX_URL``, ``E2B_DOMAIN``, ...).
    """

    api_key: str | None = None
    api_url: str | None = None
    sandbox_url: str | None = None
    domain: str | None = None
    debug: bool | None = None
    validate_api_key: bool | None = None
    headers: dict[str, str] | None = None
    api_headers: dict[str, str] | None = None
    proxy: str | None = None
    request_timeout_s: float | None = None
    # How ``endpoint()`` publishes a sandbox port. "auto" resolves to "gateway"
    # when a sandbox URL is configured and "hostname" otherwise, mirroring the
    # SDK's own rule for whether one origin fronts every sandbox.
    port_routing: str = "auto"
    # Scheme for "hostname" routing. None mirrors the SDK (http in debug mode,
    # https otherwise); set it for a deployment that serves wildcard DNS over
    # plain HTTP. Ignored by "gateway" routing, which takes the scheme from the
    # gateway URL itself.
    port_scheme: str | None = None

    def __post_init__(self) -> None:
        _validate_optional_number("connection.request_timeout_s", self.request_timeout_s, positive=False)
        if self.port_routing not in _PORT_ROUTING_MODES:
            raise ValueError(f"connection.port_routing must be one of {', '.join(_PORT_ROUTING_MODES)}")
        if self.port_scheme is not None and self.port_scheme not in {"http", "https"}:
            raise ValueError("connection.port_scheme must be 'http' or 'https'")


@dataclass(frozen=True)
class E2BCreateConfig:
    """Sandbox creation settings."""

    # Template name or ID used only when SandboxSpec.image is omitted.
    template: str | None = None
    # Explicit ``SandboxSpec.image`` -> template name or ID mapping. Needed whenever
    # image references are not themselves valid aliases (registry refs contain
    # '/' and ':', which E2B rejects).
    template_map: dict[str, str] = field(default_factory=dict)
    # Sandbox timeout in seconds; the lifecycle policy selects pause or kill.
    # ``SandboxSpec.ttl_s`` overrides it per sandbox.
    timeout_s: float | None = 3600.0
    # True pauses/resumes; False kills at timeout; None sends no lifecycle. E2B 2.42+ then leaves
    # the choice to the server, and AgentENV pauses without resume and never evicts the sandbox.
    auto_resume: bool | None = field(default=False, kw_only=True)
    # A lease in seconds. The sandbox is created with this timeout and the provider that created it renews it every
    # third of a lease, up to the sandbox's own lifetime; a process that dies stops renewing, so its sandboxes end
    # one lease later instead of at their full lifetime. None creates with the full lifetime and does not renew.
    lease_s: float | None = field(default=None, kw_only=True)
    # Seconds to wait, after a create, until the gateway reaches the new sandbox. AgentENV can report a sandbox as
    # created while its gateway still answers 502 "could not be reached"; the provider then runs a no-op command until
    # it succeeds, so callers never get a sandbox whose first command fails. None hands the sandbox over at once.
    reachable_timeout_s: float | None = field(default=None, kw_only=True)
    # Install the login-shell hook (``_LOGIN_ENV_HOOK``) in every new sandbox, so commands keep the PATH their image
    # declares. A sandbox whose hook cannot be written is killed and the create fails. False leaves login shells as
    # the template has them.
    restore_login_env: bool = field(default=True, kw_only=True)
    allow_internet_access: bool = True
    secure: bool = True
    # Outbound traffic rules, passed as the SDK's ``network`` option:
    # ``allow_out``/``deny_out`` lists of CIDRs, IPs or hostnames. ``deny_out:
    # [0.0.0.0/0]`` with an ``allow_out`` list permits only the listed hosts.
    network: dict[str, list[str]] | None = None
    # Deployment-specific create parameters, sent as the API's
    # ``customExtensionParams`` (AgentEnv). The SDK cannot send this field, so
    # setting it creates sandboxes through ``POST /sandboxes`` directly.
    custom_extension_params: dict[str, Any] | None = None
    # Raise instead of warning when a spec requests resources E2B cannot apply
    # per sandbox (they are fixed when the template is built).
    strict_resources: bool = False

    def __post_init__(self) -> None:
        if self.template is not None and (not isinstance(self.template, str) or not self.template.strip()):
            raise ValueError("create.template must be a non-empty string when provided")
        if not isinstance(self.template_map, Mapping):
            raise TypeError("create.template_map must be a mapping")
        for image, template in self.template_map.items():
            if not isinstance(image, str) or not image.strip():
                raise ValueError("create.template_map keys must be non-empty strings")
            if not isinstance(template, str) or not template.strip():
                raise ValueError("create.template_map values must be non-empty template strings")
        _validate_optional_number("create.timeout_s", self.timeout_s, positive=True)
        if self.network is not None:
            if not isinstance(self.network, Mapping):
                raise TypeError("create.network must be a mapping")
            unknown = set(self.network) - set(_NETWORK_API_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown create.network keys: {', '.join(sorted(unknown))}. Expected: allow_out, deny_out"
                )
            for key, rules in self.network.items():
                if not isinstance(rules, (list, tuple)) or not all(isinstance(rule, str) for rule in rules):
                    raise ValueError(f"create.network.{key} must be a list of strings")
        if self.custom_extension_params is not None and not isinstance(self.custom_extension_params, Mapping):
            raise TypeError("create.custom_extension_params must be a mapping")
        if self.auto_resume is not None and not isinstance(self.auto_resume, bool):
            raise ValueError("create.auto_resume must be a boolean or None")
        _validate_optional_number("create.lease_s", self.lease_s, positive=True)
        _validate_optional_number("create.reachable_timeout_s", self.reachable_timeout_s, positive=True)
        if not isinstance(self.restore_login_env, bool):
            raise ValueError("create.restore_login_env must be a boolean")


@dataclass(frozen=True)
class E2BExecConfig:
    """Command execution settings."""

    # Applied when the provider receives ``timeout_s=None``. The public NeMo Gym
    # sandbox facade passes its own 180-second default unless callers override it.
    default_timeout_s: float | None = 180.0
    user: str | None = None
    request_timeout_s: float | None = None
    # Start commands detached and reattach by pid if the output stream drops.
    # The command keeps running inside the sandbox when the stream dies, so its
    # exit code survives a control-plane restart that would otherwise fail the
    # command outright.
    #
    # On by default, matching Harbor's own e2b environment, which dispatches
    # every command with ``background=True``. Set False to have the SDK block
    # on the stream instead; note that reattaching is lossy (see
    # :meth:`_run_background`).
    background: bool = True
    # How many times to reattach before giving up.
    reconnect_attempts: int = 2

    def __post_init__(self) -> None:
        _validate_optional_number("exec.default_timeout_s", self.default_timeout_s, positive=False)
        _validate_optional_number("exec.request_timeout_s", self.request_timeout_s, positive=False)
        _validate_nonnegative_int("exec.reconnect_attempts", self.reconnect_attempts)


@dataclass(frozen=True)
class E2BOperationConfig:
    """Retry policy for transient SDK/transport failures."""

    retries: int = 2
    retry_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0
    # Seconds after a create returns during which "not found" about the new sandbox means "not routable yet": in a
    # sandbox's first seconds a gateway node can briefly lose its route and answer 404 for a sandbox that runs. A kill
    # that finds nothing is repeated, and status() reports UNKNOWN rather than STOPPED. 0 turns this off.
    new_sandbox_grace_s: float = 15.0

    def __post_init__(self) -> None:
        _validate_nonnegative_int("operations.retries", self.retries)
        _validate_nonnegative_number("operations.retry_delay_s", self.retry_delay_s)
        _validate_nonnegative_number("operations.retry_max_delay_s", self.retry_max_delay_s)
        _validate_nonnegative_number("operations.new_sandbox_grace_s", self.new_sandbox_grace_s)


@dataclass(frozen=True)
class E2BSnapshotConfig:
    """Snapshot settings."""

    # Send the checkpointer's snapshot names to the service. E2B files a named snapshot as a template of that name
    # (an alias visible to the whole team), so names are off by default: snapshots are then unnamed, and the
    # checkpointer releases them by the ids in its index; ``list_snapshots`` finds none by name.
    send_names: bool = False


@dataclass
class _Lease:
    sandbox: Any
    # Monotonic time the sandbox's own lifetime ends, or None for no limit.
    lifetime_end: float | None


class E2BProvider:
    """Provider backed by the E2B Python SDK."""

    name = "e2b"
    # A snapshot saves memory and processes, and a sandbox created from it continues them.
    snapshot_keeps_processes = True

    def __init__(
        self,
        *,
        connection: E2BConnectionConfig | Mapping[str, Any] | None = None,
        create: E2BCreateConfig | Mapping[str, Any] | None = None,
        exec: E2BExecConfig | Mapping[str, Any] | None = None,
        operations: E2BOperationConfig | Mapping[str, Any] | None = None,
        attribution: AttributionConfig | Mapping[str, Any] | None = None,
        snapshots: E2BSnapshotConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _config_from_mapping(E2BConnectionConfig, connection)
        self._create = _config_from_mapping(E2BCreateConfig, create)
        self._exec = _config_from_mapping(E2BExecConfig, exec)
        self._operations = _config_from_mapping(E2BOperationConfig, operations)
        self._attribution = _config_from_mapping(AttributionConfig, attribution)
        self._snapshots = _config_from_mapping(E2BSnapshotConfig, snapshots)
        self._warned_resource_specs: set[str] = set()
        self._warned_resource_requests = False
        self._leases: dict[str, _Lease] = {}
        self._lease_task: asyncio.Task[None] | None = None
        # Monotonic end of each new sandbox's grace period (``operations.new_sandbox_grace_s``), by sandbox id.
        self._new_until: dict[str, float] = {}

    # ---------------------------------------------------------------- helpers

    def _api_params(self) -> dict[str, Any]:
        """SDK ``ApiParams`` for connection-scoped calls; omitted keys fall back to env.

        Only ``create``/``connect``/``kill`` open a connection and accept these.
        Everything else runs against an already-connected sandbox -- see
        :meth:`_request_params`.
        """
        params = {key: getattr(self._connection, key) for key in _API_PARAM_KEYS}
        params = {key: value for key, value in params.items() if value is not None}
        if self._connection.request_timeout_s is not None:
            params["request_timeout"] = self._connection.request_timeout_s
        return params

    def _request_params(self) -> dict[str, Any]:
        """Return the request timeout; data-plane methods reject connection parameters."""
        if self._connection.request_timeout_s is None:
            return {}
        return {"request_timeout": self._connection.request_timeout_s}

    def _exec_request_timeout(self) -> float | None:
        """Return the E2B 2.36 stream-open timeout for command requests."""
        if self._exec.request_timeout_s is not None:
            return self._exec.request_timeout_s
        return self._connection.request_timeout_s

    def _resolve_template(self, spec: SandboxSpec) -> str:
        """Map a spec onto an E2B template.

        Precedence: ``provider_options.template`` -> ``create.template_map`` ->
        an unambiguous direct ``spec.image``. ``create.template`` is used only
        when ``spec.image`` is omitted; an unmapped image must not silently
        select an unrelated fallback template.

        Building a template from an image is provisioning, not part of starting
        a sandbox, so it lives in :mod:`nemo_gym.sandbox.providers.e2b.build`
        and never runs on this path.
        """
        options = self._provider_options(spec)
        option = options.get("template")
        if option is not None:
            if not isinstance(option, str) or not option:
                raise ValueError("E2B provider option 'template' must be a non-empty string")
            return option

        if spec.image:
            mapped = self._create.template_map.get(spec.image)
            if mapped:
                return str(mapped)
            if _DIRECT_TEMPLATE_RE.match(spec.image):
                return spec.image
            raise E2BCreateError(
                f"E2B starts sandboxes from a template name or ID, but SandboxSpec.image={spec.image!r} "
                "cannot be treated as an unambiguous direct template name. Map it with create.template_map, "
                "set provider_options.template (including for tagged names or template IDs), or build a "
                "template first with "
                "nemo_gym.sandbox.providers.e2b.build (its output is a ready-made template_map)."
            )

        if self._create.template:
            return self._create.template

        raise E2BCreateError(
            "No E2B template to start from: set SandboxSpec.image, provider_options.template, or create.template."
        )

    def _provider_options(self, spec: SandboxSpec) -> Mapping[str, Any]:
        options = spec.provider_options or {}
        if not isinstance(options, Mapping):
            raise TypeError("E2B provider_options must be a mapping")
        unknown = set(options) - _PROVIDER_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"Unknown E2B provider option(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(sorted(_PROVIDER_OPTION_KEYS))}"
            )
        if "resource_requests" in options and not self._warned_resource_requests:
            self._warned_resource_requests = True
            LOGGER.warning(
                "E2B fixes sandbox resources when the template is built; provider option resource_requests=%r is "
                "not applied",
                options["resource_requests"],
            )
        return options

    def _check_resources(self, spec: SandboxSpec, template: str) -> None:
        """Surface resource requests E2B cannot honour per sandbox."""
        resources = spec.resources
        requested = {
            name: getattr(resources, name)
            for name in ("cpu", "memory_mib", "disk_gib", "gpu", "gpu_type")
            if getattr(resources, name, None) is not None
        }
        if not requested:
            return
        detail = ", ".join(f"{key}={value}" for key, value in sorted(requested.items()))
        message = (
            f"E2B fixes sandbox resources when the template is built, so {detail} requested for template "
            f"{template!r} cannot be applied at create time. The bundled builder can set cpu_count and "
            "memory_mb; disk_gib, gpu, and gpu_type require a suitable pre-built E2B template."
        )
        if self._create.strict_resources:
            raise E2BCreateError(message)
        if template not in self._warned_resource_specs:
            self._warned_resource_specs.add(template)
            global _resource_warning_logged
            if _resource_warning_logged:
                LOGGER.debug("%s", message)
            else:
                _resource_warning_logged = True
                LOGGER.warning("%s Later templates are logged at DEBUG.", message)

    async def _with_retries(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        operation: str,
        retry_timeouts: bool = False,
    ) -> T:
        """Retry transient failures with exponential backoff."""
        e2b = _require_e2b_sdk()
        # Never retry these: they are deterministic and retrying only adds latency.
        non_retryable_candidates = [
            getattr(e2b, "NotFoundException", None),
            getattr(e2b, "SandboxNotFoundException", None),
            getattr(e2b, "AuthenticationException", None),
            getattr(e2b, "InvalidArgumentException", None),
            # Let the caller choose a backoff window for 429s instead of
            # amplifying a deployment-wide limit with fast local retries.
            getattr(e2b, "RateLimitException", None),
        ]
        if not retry_timeouts:
            non_retryable_candidates.append(getattr(e2b, "TimeoutException", None))
        non_retryable = tuple(exc for exc in non_retryable_candidates if isinstance(exc, type))
        attempts = self._operations.retries + 1
        delay = self._operations.retry_delay_s
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return await factory()
            except Exception as exc:  # noqa: BLE001 - transport/5xx errors are provider-specific
                # A sandbox the gateway could not reach is retried: these operations are idempotent. The SDK files it
                # under TimeoutException, which is otherwise final.
                if (isinstance(exc, non_retryable) and not _is_unreachable(exc, e2b)) or _is_not_found(exc, e2b):
                    raise
                last_exc = exc
                if attempt == attempts - 1:
                    break
                LOGGER.debug("e2b %s failed (attempt %d/%d): %s", operation, attempt + 1, attempts, exc)
                await asyncio.sleep(min(delay, self._operations.retry_max_delay_s))
                delay *= 2
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _sandbox(handle: SandboxHandle) -> Any:
        sandbox = handle.raw
        if sandbox is None:
            raise RuntimeError(f"Sandbox handle {handle.sandbox_id} carries no e2b sandbox object")
        return sandbox

    # ------------------------------------------------------------- lifecycle

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.entrypoint:
            raise E2BCreateError(
                "SandboxSpec.entrypoint is not supported by the e2b provider; "
                "the E2B template defines the sandbox entrypoint"
            )
        template = self._resolve_template(spec)
        self._check_resources(spec, template)
        return await self._create_sandbox(spec, template)

    async def create_from_snapshot(self, snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox from a snapshot: E2B starts it from the snapshot's template id.

        The sandbox resumes the snapshot's memory and processes. Raises ``SandboxSnapshotNotFoundError`` when the
        service has no such snapshot.
        """
        self._provider_options(spec)
        try:
            return await self._create_sandbox(spec, snapshot_id)
        except E2BCreateError as error:
            if _is_not_found(error.__cause__, _require_e2b_sdk()) or _is_missing_template(error.__cause__):
                raise SandboxSnapshotNotFoundError(
                    f"E2B has no snapshot {snapshot_id!r}: {error.__cause__}"
                ) from error
            raise

    async def _create_sandbox(self, spec: SandboxSpec, template: str) -> SandboxHandle:
        lifetime_s = spec.ttl_s if spec.ttl_s is not None else self._create.timeout_s
        if lifetime_s is not None and (not _is_finite_number(lifetime_s) or lifetime_s <= 0):
            raise E2BCreateError("E2B sandbox ttl_s must be > 0")
        if spec.ready_timeout_s is not None and (
            not _is_finite_number(spec.ready_timeout_s) or spec.ready_timeout_s <= 0
        ):
            raise E2BCreateError("E2B sandbox ready_timeout_s must be > 0")
        timeout_s = lifetime_s
        if self._create.lease_s is not None:
            timeout_s = self._create.lease_s if lifetime_s is None else min(lifetime_s, self._create.lease_s)
        kwargs: dict[str, Any] = {
            "template": template,
            "timeout": max(1, math.ceil(timeout_s)) if timeout_s is not None else None,
            "allow_internet_access": self._create.allow_internet_access,
            "secure": self._create.secure,
            **self._api_params(),
        }
        if self._create.auto_resume is not None:
            kwargs["lifecycle"] = {
                "on_timeout": "pause" if self._create.auto_resume else "kill",
                "auto_resume": self._create.auto_resume,
            }
        if self._create.network:
            kwargs["network"] = {key: list(rules) for key, rules in self._create.network.items()}
        # A sandbox's own network policy overrides the provider-wide default.
        network_policy = (spec.provider_options or {}).get("network_policy")
        if network_policy is not None:
            kwargs["network"] = _network_options(network_policy)
        if spec.ready_timeout_s is not None:
            # E2B retains this on the returned sandbox connection. Subsequent
            # provider calls explicitly reapply connection.request_timeout_s
            # when configured; otherwise the SDK-retained value remains.
            kwargs["request_timeout"] = float(spec.ready_timeout_s)
        if spec.env:
            kwargs["envs"] = {str(k): str(v) for k, v in spec.env.items()}
        # Job attribution (team / user / workload / run) makes every sandbox findable through the list API's
        # metadata filter. Explicit spec keys win over it.
        metadata = {**self._attribution.metadata(), **spec.metadata}
        if metadata:
            kwargs["metadata"] = {str(k): str(v) for k, v in metadata.items()}

        e2b = _require_e2b_sdk()
        try:
            # Creating a sandbox is not idempotent. Retrying an ambiguous
            # transport failure can leak the first, billable sandbox.
            if self._create.custom_extension_params is None:
                sandbox = await e2b.AsyncSandbox.create(**kwargs)
            else:
                sandbox = await self._create_with_extension_params(e2b, kwargs)
        except Exception as exc:
            raise E2BCreateError(f"Failed to create e2b sandbox from template {template!r}: {exc}") from exc

        # Before the reachability probe, so the probe, a kill after it fails, and a restore's first commands are all
        # covered.
        self._start_grace(str(sandbox.sandbox_id))
        self._hold_lease(sandbox, lifetime_s)
        if self._create.reachable_timeout_s is not None:
            try:
                await self._wait_reachable(sandbox, self._create.reachable_timeout_s)
            except Exception as exc:
                LOGGER.warning(
                    "e2b sandbox %s did not become reachable within %gs: %r",
                    sandbox.sandbox_id,
                    self._create.reachable_timeout_s,
                    exc,
                )
                await self._discard(sandbox, "unreachable")
                raise E2BCreateError(
                    f"e2b sandbox {sandbox.sandbox_id} from template {template!r} was created but could not be "
                    f"reached within {self._create.reachable_timeout_s:g}s: {exc}"
                ) from exc
        if self._create.restore_login_env:
            # Also on a sandbox created from a snapshot, which has the hook already: writing it again is harmless, and
            # it covers snapshots taken without it.
            try:
                await self._with_retries(
                    lambda: sandbox.files.write(
                        _LOGIN_ENV_HOOK_PATH, _LOGIN_ENV_HOOK, user="root", **self._request_params()
                    ),
                    operation="write login-shell hook",
                )
            except Exception as exc:
                await self._discard(sandbox, "login-shell hook not written")
                raise E2BCreateError(
                    f"e2b sandbox {sandbox.sandbox_id} from template {template!r} was created but its login-shell "
                    f"hook {_LOGIN_ENV_HOOK_PATH} could not be written: {exc}"
                ) from exc
        return SandboxHandle(sandbox_id=sandbox.sandbox_id, provider_name=self.name, raw=sandbox)

    async def _discard(self, sandbox: Any, reason: str) -> None:
        """Kill a new sandbox the caller will not get; the lease or its timeout ends one the kill misses."""
        self._leases.pop(str(sandbox.sandbox_id), None)
        try:
            await self._stop(sandbox, str(sandbox.sandbox_id))
        except Exception as kill_error:  # noqa: BLE001 - the lease or timeout still ends it
            LOGGER.warning("could not kill e2b sandbox %s (%s): %r", sandbox.sandbox_id, reason, kill_error)

    async def _create_with_extension_params(self, e2b: Any, kwargs: dict[str, Any]) -> Any:
        """Create through ``POST /sandboxes``, adding ``customExtensionParams``.

        The POST goes through the SDK's own API client, so it gets the URL, auth
        headers, proxy, timeout and HTTP transport of ``AsyncSandbox.create``,
        and like it is never replayed. ``AsyncSandbox.connect`` then attaches;
        it also returns the envd access token a secure sandbox needs.
        """
        config = e2b.ConnectionConfig(**{**self._api_params(), "request_timeout": kwargs.get("request_timeout")})
        body: dict[str, Any] = {
            "templateID": kwargs["template"],
            "timeout": kwargs["timeout"],
            "secure": kwargs["secure"],
            "allow_internet_access": kwargs["allow_internet_access"],
            "envVars": kwargs.get("envs", {}),
            "metadata": kwargs.get("metadata", {}),
            "customExtensionParams": self._create.custom_extension_params,
        }
        if "network" in kwargs:
            body["network"] = {_NETWORK_API_KEYS[key]: rules for key, rules in kwargs["network"].items()}
        if "lifecycle" in kwargs:
            # The SDK's wire form of the lifecycle, so both create paths time out alike.
            body["autoPause"] = kwargs["lifecycle"]["auto_resume"]
            body["autoResume"] = {"enabled": kwargs["lifecycle"]["auto_resume"]}
        client = e2b.api.client_async.get_api_client(config).get_async_httpx_client()
        response = await client.post("/sandboxes", json=body)
        if response.status_code >= 300:
            raise E2BCreateError(f"POST /sandboxes returned HTTP {response.status_code}: {response.text[:500]}")
        sandbox_id = response.json()["sandboxID"]
        try:
            # Pass the create timeout: connect otherwise applies the SDK default.
            return await self._with_retries(
                lambda: e2b.AsyncSandbox.connect(sandbox_id, timeout=kwargs["timeout"], **self._api_params()),
                operation="connect",
            )
        except BaseException:
            # The caller gets no handle to close this sandbox, so stop it here.
            try:
                await e2b.AsyncSandbox.kill(sandbox_id, **self._api_params())
            except Exception as kill_exc:  # noqa: BLE001 - best effort; the sandbox TTL is the backstop
                LOGGER.warning("Could not kill e2b sandbox %s after a failed connect: %s", sandbox_id, kill_exc)
            raise

    async def _wait_reachable(self, sandbox: Any, timeout_s: float) -> None:
        """Run a no-op command until the gateway reaches ``sandbox``.

        "Could not be reached" (502) and "not found" (404, including one whose body the SDK cannot parse) are waited
        out: in a new sandbox's first seconds a gateway node can answer either for a sandbox that runs. The command
        does nothing, so running it again is always safe, unlike the caller's own commands.
        """
        e2b = _require_e2b_sdk()
        deadline = monotonic() + timeout_s
        delay = REACHABLE_RETRY_DELAY_S
        attempts = 0
        while True:
            attempts += 1
            try:
                await sandbox.commands.run(
                    cmd="true",
                    timeout=REACHABLE_PROBE_TIMEOUT_S,
                    request_timeout=self._exec_request_timeout(),
                )
            except Exception as exc:
                not_found = _is_not_found(exc, e2b) or isinstance(exc, json.JSONDecodeError)
                if not (_is_unreachable(exc, e2b) or not_found) or monotonic() + delay > deadline:
                    raise
                if not_found:
                    LOGGER.info("e2b sandbox %s not routable yet (%r); probing again", sandbox.sandbox_id, exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, REACHABLE_RETRY_MAX_DELAY_S)
                continue
            if attempts > 1:
                LOGGER.info("e2b sandbox %s became reachable after %d attempts", sandbox.sandbox_id, attempts)
            return

    # ------------------------------------------------------------------ lease

    def _hold_lease(self, sandbox: Any, lifetime_s: float | None) -> None:
        if self._create.lease_s is None:
            return
        lifetime_end = None if lifetime_s is None else monotonic() + lifetime_s
        self._leases[str(sandbox.sandbox_id)] = _Lease(sandbox=sandbox, lifetime_end=lifetime_end)
        if self._lease_task is None or self._lease_task.done():
            self._lease_task = asyncio.create_task(self._renew_leases(), name="e2b-lease-renewal")

    async def _renew_leases(self) -> None:
        """Renew every held lease each third of a lease, until none is held."""
        assert self._create.lease_s is not None
        while self._leases:
            await asyncio.sleep(self._create.lease_s / 3)
            await self.renew_leases()

    async def renew_leases(self) -> None:
        """Renew every held lease once, ``LEASE_RENEW_BATCH`` at a time."""
        held = list(self._leases.items())
        for start in range(0, len(held), LEASE_RENEW_BATCH):
            await asyncio.gather(
                *(
                    self._renew_lease(sandbox_id, lease)
                    for sandbox_id, lease in held[start : start + LEASE_RENEW_BATCH]
                )
            )

    async def _renew_lease(self, sandbox_id: str, lease: _Lease) -> None:
        assert self._create.lease_s is not None
        if self._leases.get(sandbox_id) is not lease:
            return  # closed or dropped since this round began
        timeout_s = self._create.lease_s
        if lease.lifetime_end is not None:
            remaining = lease.lifetime_end - monotonic()
            if remaining <= 0:
                # Its lifetime is over: it ends at its current timeout.
                self._leases.pop(sandbox_id, None)
                return
            timeout_s = min(timeout_s, remaining)
        try:
            await asyncio.wait_for(
                lease.sandbox.set_timeout(max(1, math.ceil(timeout_s)), **self._request_params()),
                LEASE_RENEW_TIMEOUT_S,
            )
        except Exception as error:  # noqa: BLE001 - a failed renewal is retried at the next round
            if _DYING_STATE.search(str(error)):
                LOGGER.info("e2b sandbox %s is being killed; stopping its lease", sandbox_id)
                self._leases.pop(sandbox_id, None)
                return
            gone = _is_not_found(error, _require_e2b_sdk()) or await self._gone_behind_unparsed_reply(
                error, sandbox_id
            )
            if gone and not self._in_grace(sandbox_id):
                LOGGER.info("e2b sandbox %s is gone; stopping its lease", sandbox_id)
                self._leases.pop(sandbox_id, None)
                return
            LOGGER.warning("e2b lease renewal failed for sandbox %s; retrying next round: %r", sandbox_id, error)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a descriptor for attaching to this sandbox from another process."""
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Attach to the sandbox described by ``descriptor``.

        E2B's public connect API applies its default sandbox timeout when none
        is supplied, so attaching may renew a sandbox that is close to expiry.
        """
        e2b = _require_e2b_sdk()
        sandbox_id = str(descriptor["sandbox_id"])
        sandbox = await self._with_retries(
            lambda: e2b.AsyncSandbox.connect(sandbox_id, **self._api_params()),
            operation="connect",
        )
        return SandboxHandle(sandbox_id=str(sandbox.sandbox_id), provider_name=self.name, raw=sandbox)

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a declared service port to a caller-reachable endpoint.

        E2B deployments publish sandbox ports in one of two shapes, and the
        endpoint differs accordingly:

        ``hostname``
            One DNS name per port, ``{port}-{sandbox_id}.{domain}``. The caller
            dials it directly, so no headers are needed.
        ``gateway``
            A single origin fronts every sandbox and every port, and the proxy
            picks the target from request headers. A self-hosted deployment
            without wildcard DNS for sandbox subdomains uses this, and points
            ``sandbox_url`` at that origin. The SDK mints ``X-Access-Token``
            and ``E2b-Sandbox-Id`` per sandbox and sets ``E2b-Sandbox-Port`` to
            envd's own port, so the target port must be re-pointed here.

        ``auto`` selects ``gateway`` exactly when a sandbox URL is configured,
        which is the same signal the SDK uses to decide that one origin serves
        every sandbox.

        The returned headers are the caller's to send verbatim. ``X-Access-Token``
        is minted per sandbox and is not the deployment's API key.
        """
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"Invalid sandbox TCP port: {port!r}")
        if port < 1 or port > 65535:
            raise ValueError(f"Sandbox TCP port must be between 1 and 65535, got {port}")

        sandbox = self._sandbox(handle)
        connection_config = getattr(sandbox, "connection_config", None)
        if connection_config is None:
            raise RuntimeError(f"Sandbox {handle.sandbox_id} exposes no e2b connection config")

        mode = self._connection.port_routing
        if mode == "auto":
            # Mirror ConnectionConfig's own resolution order rather than reading
            # its private attribute: an explicit config value, else the SDK env var.
            configured_sandbox_url = self._connection.sandbox_url or os.environ.get("E2B_SANDBOX_URL")
            mode = "gateway" if configured_sandbox_url else "hostname"

        if mode == "hostname":
            scheme = self._connection.port_scheme or ("http" if connection_config.debug else "https")
            return SandboxEndpoint(endpoint=f"{scheme}://{sandbox.get_host(port)}")

        base_url = connection_config.get_sandbox_url(sandbox.sandbox_id, sandbox.sandbox_domain)
        try:
            headers = dict(connection_config.sandbox_headers)
        except AttributeError as exc:
            raise RuntimeError(
                "Gateway port routing needs ConnectionConfig.sandbox_headers, which this "
                f"e2b SDK does not expose. Install '{E2B_SDK_CONSTRAINT}', "
                "or set connection.port_routing to 'hostname'."
            ) from exc
        headers["E2b-Sandbox-Port"] = str(port)
        return SandboxEndpoint(endpoint=base_url, headers=headers)

    async def pause(self, handle: SandboxHandle) -> None:
        """Pause the sandbox through the E2B SDK without replaying failed requests."""
        if await self._sandbox(handle).pause(**self._request_params()) is False:
            # The SDK maps every 409 to False. AgentENV sends 409 during a snapshot or fork, which status
            # reports as paused although the VM then runs again, so no 409 counts as paused.
            raise RuntimeError(
                f"e2b sandbox {handle.sandbox_id!r} refused pause with 409 Conflict (already paused or busy); "
                "check status()"
            )

    async def resume(self, handle: SandboxHandle) -> None:
        """Resume the sandbox and replace its SDK handle only after success."""
        sandbox = self._sandbox(handle)
        e2b = _require_e2b_sdk()
        params = self._api_params()
        # Keep the request timeout the SDK retained from create (ready_timeout_s).
        params.setdefault("request_timeout", sandbox.connection_config.request_timeout)
        # Resume restarts the sandbox lifetime, which is the SDK's 300 s default without a timeout.
        timeout_s = self._create.timeout_s
        handle.raw = await e2b.AsyncSandbox.connect(
            handle.sandbox_id,
            timeout=max(1, math.ceil(timeout_s)) if timeout_s is not None else None,
            **params,
        )

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        e2b = _require_e2b_sdk()
        sandbox = self._sandbox(handle)
        not_found = tuple(
            exc
            for exc in (getattr(e2b, "SandboxNotFoundException", None), getattr(e2b, "NotFoundException", None))
            if isinstance(exc, type)
        )
        try:
            # Control-plane status avoids waking paused sandboxes with a health check.
            info = await sandbox.get_info(**self._request_params())
        except not_found:
            # A new sandbox the gateway cannot route yet is not known to be stopped.
            return SandboxStatus.UNKNOWN if self._in_grace(handle.sandbox_id) else SandboxStatus.STOPPED
        except Exception:  # noqa: BLE001 - status must not raise for transient issues
            return SandboxStatus.UNKNOWN
        return {"running": SandboxStatus.RUNNING, "paused": SandboxStatus.PAUSED}.get(
            info.state, SandboxStatus.UNKNOWN
        )

    async def close(self, handle: SandboxHandle) -> None:
        sandbox = handle.raw
        if sandbox is None:
            return
        self._leases.pop(handle.sandbox_id, None)
        await self._stop(sandbox, handle.sandbox_id)
        handle.raw = None

    async def _stop(self, sandbox: Any, sandbox_id: str) -> None:
        """Kill a sandbox, making sure a "not found" really means it is gone.

        During the new-sandbox grace period, a kill that finds nothing is repeated until it finds the sandbox or the
        period ends. After it, a kill that finds nothing is checked again once: a gateway replica that has not seen the
        sandbox can answer "not found" for one that still runs.
        """
        e2b = _require_e2b_sdk()
        not_found = tuple(
            exc
            for exc in (getattr(e2b, "SandboxNotFoundException", None), getattr(e2b, "NotFoundException", None))
            if isinstance(exc, type)
        )
        while await self._kill(sandbox, not_found) is False:
            if not self._in_grace(sandbox_id):
                break
            LOGGER.info("e2b kill did not find new sandbox %s yet; retrying", sandbox_id)
            await asyncio.sleep(NEW_SANDBOX_KILL_RETRY_S)
        else:  # killed
            self._new_until.pop(sandbox_id, None)
            return
        # The kill found nothing after the grace period: look once more.
        await asyncio.sleep(KILL_RECHECK_DELAY_S)
        if await self.status(SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox)) is (
            SandboxStatus.RUNNING
        ):
            LOGGER.warning("e2b sandbox %s still runs after a kill that did not find it; killing again", sandbox_id)
            await self._kill(sandbox, not_found)
        else:
            LOGGER.debug("e2b sandbox %s already gone on close", sandbox_id)
        self._new_until.pop(sandbox_id, None)

    async def _kill(self, sandbox: Any, not_found: tuple[type[BaseException], ...]) -> bool:
        """Kill a sandbox; False when the service did not find it.

        A reply the SDK cannot parse (an empty or plain-text 404) also counts as "not found": during the new-sandbox
        grace period without a lookup, so the caller retries, and after it only when a lookup confirms the 404.
        """
        try:
            killed = await self._with_retries(
                lambda: sandbox.kill(**self._api_params()),
                operation="kill",
                retry_timeouts=True,
            )
        except not_found:
            return False
        except Exception as error:
            if isinstance(error, json.JSONDecodeError) and self._in_grace(sandbox.sandbox_id):
                return False
            if await self._gone_behind_unparsed_reply(error, sandbox.sandbox_id):
                return False
            raise
        return killed is not False

    def _start_grace(self, sandbox_id: str) -> None:
        if self._operations.new_sandbox_grace_s > 0:
            self._new_until[sandbox_id] = monotonic() + self._operations.new_sandbox_grace_s

    def _in_grace(self, sandbox_id: str) -> bool:
        """Whether ``sandbox_id`` is still in its new-sandbox grace period; forgets one whose period has ended."""
        until = self._new_until.get(sandbox_id)
        if until is None:
            return False
        if monotonic() < until:
            return True
        del self._new_until[sandbox_id]
        return False

    async def _gone_behind_unparsed_reply(self, error: BaseException, sandbox_id: str) -> bool:
        """Whether ``error`` is a reply the SDK could not parse about a sandbox the service no longer has.

        AgentENV answers a request about a sandbox it no longer has with a 404 whose body is empty, and the SDK parses
        the body before it checks the status, so the not-found arrives as a ``JSONDecodeError`` instead of the SDK's
        not-found exception. Only for such an error is the sandbox looked up; a 404 there settles it. Any other answer,
        or a lookup that fails, leaves the original error to the caller.
        """
        if not isinstance(error, json.JSONDecodeError):
            return False
        e2b = _require_e2b_sdk()
        client = e2b.api.client_async.get_api_client(e2b.ConnectionConfig(**self._api_params()))
        try:
            response = await client.get_async_httpx_client().get(f"/sandboxes/{sandbox_id}")
        except Exception:  # noqa: BLE001 - an unknown answer keeps the original error
            return False
        return response.status_code == 404

    async def aclose(self) -> None:
        """Stop renewing leases; sandboxes own their connections."""
        self._leases.clear()
        if self._lease_task is not None:
            self._lease_task.cancel()
            await asyncio.gather(self._lease_task, return_exceptions=True)
            self._lease_task = None

    # ------------------------------------------------------------- snapshots

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        """Snapshot the whole VM and return the snapshot id; it is usable when this returns.

        The VM pauses for the moment it is captured and then keeps running. Not retried: a repeated request could
        leave a second snapshot nobody knows the id of. ``name`` is sent only with ``snapshots.send_names``.
        """
        sandbox = self._sandbox(handle)
        create_snapshot = getattr(sandbox, "create_snapshot", None)
        if create_snapshot is None:
            raise E2BSnapshotError(
                "The installed e2b SDK has no AsyncSandbox.create_snapshot; install a release that has it "
                f"within '{E2B_SDK_CONSTRAINT}'"
            )
        try:
            info = await create_snapshot(name if self._snapshots.send_names else None, **self._request_params())
        except Exception as error:
            if not self._in_grace(handle.sandbox_id) and await self._gone_behind_unparsed_reply(
                error, handle.sandbox_id
            ):
                raise E2BSnapshotError(
                    f"e2b sandbox {handle.sandbox_id!r} is gone and cannot be snapshotted (HTTP 404)"
                ) from error
            raise
        snapshot_id = getattr(info, "snapshot_id", None)
        if not snapshot_id:
            raise E2BSnapshotError(f"e2b create_snapshot returned no snapshot id for sandbox {handle.sandbox_id!r}")
        return str(snapshot_id)

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        """Nothing to wait for: an E2B snapshot is usable once ``snapshot`` returns."""
        return None

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot (its template); one that is already gone is not an error."""
        e2b = _require_e2b_sdk()
        try:
            await self._with_retries(
                lambda: e2b.AsyncSandbox.delete_snapshot(snapshot_id, **self._api_params()),
                operation="delete_snapshot",
            )
        except Exception as error:
            if not _is_not_found(error, e2b):
                raise

    async def list_snapshots(self, prefix: str) -> list[dict[str, Any]]:
        """Every snapshot whose name starts with ``prefix``, filtered here: the service matches whole names only.

        A snapshot's name is its template alias without the team namespace and tag. Unnamed snapshots (the default,
        see ``snapshots.send_names``) never match a non-empty prefix.
        """
        e2b = _require_e2b_sdk()
        paginator = e2b.AsyncSandbox.list_snapshots(**self._api_params())
        found: list[dict[str, Any]] = []
        while paginator.has_next:
            for info in await paginator.next_items():
                names = list(getattr(info, "names", None) or [])
                name = names[0].rsplit("/", 1)[-1].split(":", 1)[0] if names else ""
                if name.startswith(prefix):
                    found.append({"id": str(info.snapshot_id), "name": name})
        return found

    # -------------------------------------------------------------- commands

    async def _run_background(self, sandbox: Any, kwargs: dict[str, Any]) -> Any:
        """Run a command detached, reattaching by pid if the stream drops.

        ``commands.run(background=True)`` returns as soon as the process has
        started, handing back its pid. The command then keeps running inside
        the sandbox independently of the stream carrying its output, so losing
        that stream -- a gateway rollout, a proxy restart, a network blip --
        no longer destroys the command: reattach with ``commands.connect(pid)``
        and the real exit code still arrives.

        Reattaching has two inherent limits:

        * **Output emitted while disconnected is lost.** Output already
          received by the previous handle is retained and combined with the
          reattached stream, but the stream is live rather than replayed.
        * **The process must still be running.** ``connect`` raises
          not-found once it has exited, so a command that finishes during the
          gap cannot be recovered.
        """
        e2b = _require_e2b_sdk()
        command_timeout = kwargs.get("timeout")
        started = monotonic()
        deadline = (
            started + float(command_timeout) if command_timeout is not None and float(command_timeout) > 0 else None
        )
        handle = await sandbox.commands.run(**kwargs, background=True)
        pid = getattr(handle, "pid", None)
        stdout = ""
        stderr = ""

        # Never swallowed while reattaching: a non-zero exit is a real outcome, and so is a timeout at the command's
        # own deadline. A timeout before it is the stream's (E2B reports a dropped stream as a TimeoutException, and
        # its "could not be reached" is one too), so the command is reattached like after any transport failure.
        exit_exc = getattr(e2b, "CommandExitException", None)
        timeout_exc = getattr(e2b, "TimeoutException", None)

        def at_deadline(exc: BaseException) -> bool:
            timed_out = isinstance(exc, TimeoutError) or (
                isinstance(timeout_exc, type) and isinstance(exc, timeout_exc)
            )
            return (
                timed_out
                and deadline is not None
                and monotonic() - started >= _DEADLINE_FRACTION * float(command_timeout)
            )

        for attempt in range(self._exec.reconnect_attempts + 1):
            try:
                result = await handle.wait()
                result.stdout = stdout + (getattr(result, "stdout", "") or "")
                result.stderr = stderr + (getattr(result, "stderr", "") or "")
                return result
            except Exception as exc:  # noqa: BLE001 - transport failure unless an exit or the deadline; reattach
                if isinstance(exit_exc, type) and isinstance(exc, exit_exc):
                    exc.stdout = stdout + (getattr(exc, "stdout", "") or "")
                    exc.stderr = stderr + (getattr(exc, "stderr", "") or "")
                    raise
                if at_deadline(exc):
                    raise
                stdout += getattr(handle, "stdout", "") or ""
                stderr += getattr(handle, "stderr", "") or ""
                if pid is None or attempt >= self._exec.reconnect_attempts:
                    raise
                LOGGER.warning(
                    "e2b command stream lost (pid=%s, attempt %d/%d): %s; reattaching",
                    pid,
                    attempt + 1,
                    self._exec.reconnect_attempts,
                    exc,
                )
                reconnect_timeout = command_timeout
                reconnect_request_timeout = kwargs.get("request_timeout")
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"e2b command timed out after {command_timeout}s") from exc
                    reconnect_timeout = remaining
                    if reconnect_request_timeout in (None, 0):
                        reconnect_request_timeout = remaining
                    else:
                        reconnect_request_timeout = min(float(reconnect_request_timeout), remaining)
                try:
                    handle = await sandbox.commands.connect(
                        pid,
                        timeout=reconnect_timeout,
                        request_timeout=reconnect_request_timeout,
                    )
                except Exception as reconnect_exc:
                    # Typically not-found: the command finished while we were
                    # disconnected, so its result is gone. Report the original
                    # transport failure, which explains what actually happened.
                    raise exc from reconnect_exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        e2b = _require_e2b_sdk()
        sandbox = self._sandbox(handle)

        effective_timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        if effective_timeout is not None and (not _is_finite_number(effective_timeout) or effective_timeout < 0):
            raise ValueError("e2b command timeout_s must be >= 0")
        effective_user = user if user is not None else self._exec.user
        kwargs: dict[str, Any] = {"cmd": command}
        if cwd is not None:
            kwargs["cwd"] = cwd
        exec_id = uuid.uuid4().hex
        kwargs["envs"] = {**{str(k): str(v) for k, v in (env or {}).items()}, _EXEC_MARKER_ENV: exec_id}
        if effective_user is not None:
            kwargs["user"] = str(effective_user)
        # E2B treats ``timeout`` as "no timeout" when falsy; keep None explicit.
        kwargs["timeout"] = float(effective_timeout) if effective_timeout is not None else None
        kwargs["request_timeout"] = self._exec_request_timeout()

        timeout_exc = getattr(e2b, "TimeoutException", None)
        exit_exc = getattr(e2b, "CommandExitException", None)
        started = monotonic()
        try:
            if self._exec.background:
                result = await self._run_background(sandbox, kwargs)
            else:
                result = await sandbox.commands.run(**kwargs)
        except Exception as exc:
            # A non-zero exit is a normal outcome, not a provider failure.
            if isinstance(exit_exc, type) and isinstance(exc, exit_exc):
                return SandboxExecResult(
                    stdout=getattr(exc, "stdout", None),
                    stderr=getattr(exc, "stderr", None),
                    return_code=int(getattr(exc, "exit_code", 1) or 1),
                )
            if _is_unreachable(exc, e2b):
                raise E2BSandboxUnreachableError(
                    f"the e2b gateway could not reach sandbox {handle.sandbox_id}: {exc}"
                ) from exc
            # E2B uses one TimeoutException for stream-open timeouts, the
            # running stream deadline, sandbox expiry and server cancellation.
            # A timeout at the configured deadline is the command's own, and is
            # reported like every provider's: exit 124, error_type "timeout".
            # Any earlier one keeps the SDK detail.
            timed_out = isinstance(exc, TimeoutError) or (
                isinstance(timeout_exc, type) and isinstance(exc, timeout_exc)
            )
            if timed_out and effective_timeout and monotonic() - started >= _DEADLINE_FRACTION * effective_timeout:
                await self._kill_marked(sandbox, f"{_EXEC_MARKER_ENV}={exec_id}")
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"Command timed out after {effective_timeout:g}s",
                    return_code=_TIMEOUT_RETURN_CODE,
                    error_type="timeout",
                )
            if timed_out:
                raise TimeoutError(
                    f"e2b command did not complete: {exc} (configured wait/stream budget={effective_timeout}s)"
                ) from exc
            raise

        return SandboxExecResult(
            stdout=getattr(result, "stdout", None),
            stderr=getattr(result, "stderr", None),
            return_code=int(getattr(result, "exit_code", 0) or 0),
        )

    async def _kill_marked(self, sandbox: Any, marker: str) -> None:
        """Kill what a timed-out command left running, found by its environment marker; best effort."""
        script = _KILL_MARKED_SCRIPT.format(marker=shlex.quote(marker))
        try:
            await sandbox.commands.run(
                cmd=f"bash -c {shlex.quote(script)}",
                user="root",
                timeout=_KILL_MARKED_TIMEOUT_S,
                request_timeout=self._exec_request_timeout(),
            )
        except Exception as error:  # noqa: BLE001 - the timeout is reported either way
            LOGGER.warning("could not end the processes a timed-out e2b command left running: %r", error)

    # ----------------------------------------------------------------- files

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        sandbox = self._sandbox(handle)
        await self._with_retries(
            lambda: sandbox.files.write(target_path, data, **self._request_params()),
            operation="write_file",
        )

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        sandbox = self._sandbox(handle)
        data = await self._with_retries(
            lambda: sandbox.files.read(source_path, format="bytes", **self._request_params()),
            operation="read_file",
        )
        return data if isinstance(data, bytes) else bytes(data)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        await self.write_file(handle, target_path, source.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self.read_file(handle, source_path))
