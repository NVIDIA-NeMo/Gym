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
"""Process-global nemo-lens telemetry lifecycle for NeMo Gym.

Gym's process model is the thing this module exists to handle. Megatron-LM is one
process tree and NeMo-RL is a Ray driver plus actors, but a Gym run is **N independent
FastAPI processes** — a resources server, a model server, an agent server, sometimes more
— spawned by a CLI orchestrator via ``Popen``, each with its own interpreter. There is no
shared memory and no parent handle to inherit: the only thing that crosses the boundary
is the environment.

So there are two entry points:

* :func:`configure_telemetry_env` — called once in the **orchestrator** (``gym env
  start`` / ``gym env test``) before any server is spawned. It translates the
  ``telemetry:`` config block into ``NEMO_GYM_OTEL_*`` env vars with ``setdefault``, so
  every server process it spawns resolves the same settings, and any env var the user set
  by hand still wins.
* :func:`init_telemetry` — called once inside **each** server process, from
  ``SimpleServer.run_webserver``. It reads that propagated environment and builds this
  process's providers.

``export_strategy`` defaults to ``all_ranks`` here rather than ``single_rank``. Each Gym
server is rank 0 of its own world of 1, so a rank-based filter would either silence every
process or none; and silencing any one of them puts a hole in the middle of the
distributed trace this integration exists to produce.

Importing this module never requires nemo-lens: every lens import is function-local and
guarded. With lens absent, or ``enabled: false``, the init functions return ``None`` and
every instrumentation site stays a no-op through :mod:`nemo_gym.telemetry._fallbacks`.
"""

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from nemo_gym.telemetry.config import TelemetryConfig


if TYPE_CHECKING:
    from nemo.lens import TelemetryHandle


logger = logging.getLogger(__name__)

#: Process-global handle. One per process; ``None`` when lens is absent or telemetry is
#: disabled. Guarded by ``_INITIALISED`` rather than by a ``None`` check so that a
#: disabled run does not retry the whole setup on every call. ``_INIT_LOCK`` makes the
#: check-and-set atomic: without it, two threads can both observe ``_INITIALISED is
#: False`` before either sets it and both call ``setup_telemetry``, which nemo-lens
#: raises on for a second call in the same process.
_TELEMETRY_HANDLE: Optional["TelemetryHandle"] = None
_INITIALISED = False
_INIT_LOCK = threading.Lock()

#: ``NemoLensConfig.from_env`` reads ``NEMO_GYM_OTEL_<KEY>`` first, then ``NEMO_LENS_<KEY>``.
#: The fallback is what makes the documented ``NEMO_LENS_ENABLED=1`` work unchanged.
_OTEL_PREFIX = "NEMO_GYM_OTEL"
_OTEL_FALLBACK_PREFIX = "NEMO_LENS"

#: Standard OTel env vars, read unprefixed by lens.
_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"

#: Top-level key of the telemetry block in a Gym config. Registered in
#: ``nemo_gym.global_config.NEMO_GYM_RESERVED_TOP_LEVEL_KEYS`` so the config parser does
#: not mistake it for a server instance definition.
TELEMETRY_KEY_NAME = "telemetry"

#: ``TelemetryConfig`` field -> env var. Consumed by :func:`configure_telemetry_env`, the
#: function that actually connects a ``telemetry:`` YAML block (or any other
#: ``TelemetryConfig``) to these env vars via ``setdefault`` — a user does not set them by
#: hand. ``service_name`` is handled separately because it maps to the standard unprefixed
#: ``OTEL_SERVICE_NAME``, and the Gym-owned flags (``service_name_per_server``,
#: ``instrument_aiohttp``) are not lens config fields.
_ENV_FIELD_MAP = {
    "enabled": f"{_OTEL_PREFIX}_ENABLED",
    "span_groups": f"{_OTEL_PREFIX}_SPAN_GROUPS",
    "export_strategy": f"{_OTEL_PREFIX}_EXPORT_STRATEGY",
    "export_rank": f"{_OTEL_PREFIX}_EXPORT_RANK",
    "traces_enabled": f"{_OTEL_PREFIX}_TRACES_ENABLED",
    "metrics_enabled": f"{_OTEL_PREFIX}_METRICS_ENABLED",
    "logs_enabled": f"{_OTEL_PREFIX}_LOGS_ENABLED",
    "exporter": f"{_OTEL_PREFIX}_EXPORTER",
    "run_id": f"{_OTEL_PREFIX}_RUN_ID",
    # Gym-owned flags, read back by init_telemetry / server_utils rather than by lens.
    "service_name_per_server": f"{_OTEL_PREFIX}_SERVICE_NAME_PER_SERVER",
    "instrument_aiohttp": f"{_OTEL_PREFIX}_INSTRUMENT_AIOHTTP",
}

_TRUTHY = ("1", "true", "yes", "on")


def _env_flag(name: str, default: bool) -> bool:
    """Read a Gym-owned boolean env var, tolerating an unset or blank value."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in _TRUTHY


def is_telemetry_env_enabled() -> bool:
    """True when telemetry is switched on in this process's environment.

    Cheap enough to call before doing setup work, and importantly it does not import
    nemo-lens — a process with telemetry off never pays for the lens import at all.
    """
    for prefix in (_OTEL_PREFIX, _OTEL_FALLBACK_PREFIX):
        raw = os.environ.get(f"{prefix}_ENABLED", "").strip().lower()
        if raw:
            return raw in _TRUTHY
    return False


def telemetry_config_from_global_config(global_config_dict: Any) -> TelemetryConfig:
    """Build a :class:`TelemetryConfig` from Gym's merged global config dict.

    Reads the optional top-level ``telemetry:`` block. A run that never mentions
    telemetry gets the all-defaults config, which is disabled.
    """
    block: Any = None
    if global_config_dict is not None:
        try:
            block = global_config_dict.get(TELEMETRY_KEY_NAME)
        except (AttributeError, TypeError):
            block = None
    if block is None:
        return TelemetryConfig()

    # OmegaConf nodes are Mapping-like but not dicts; resolve them before validating.
    if isinstance(block, DictConfig):
        block = OmegaConf.to_container(block, resolve=True)
    return TelemetryConfig.model_validate(block)


def configure_telemetry_env(telemetry_config: Union[TelemetryConfig, None]) -> Optional[str]:
    """Translate the ``telemetry:`` block into env vars for spawned server processes.

    Call once in the orchestrator, **before** spawning any server. ``os.environ`` is what
    ``Popen`` snapshots into each child, so this is how a YAML setting reaches a server
    process that shares nothing else with its parent.

    Uses ``setdefault`` throughout: a raw ``NEMO_GYM_OTEL_*`` / ``NEMO_LENS_*`` /
    ``OTEL_SERVICE_NAME`` set by the user always wins over YAML.

    Returns the run id shared by every process in this run, or ``None`` when telemetry is
    disabled.
    """
    if telemetry_config is None:
        return None

    # Decide enablement before writing anything. A config with no `telemetry:` block
    # resolves to enabled=False, and exporting that would write
    # NEMO_GYM_OTEL_ENABLED=0 — which then shadows a `NEMO_LENS_ENABLED=1` the user set,
    # because the Gym-prefixed variable deliberately wins over the lens fallback. The
    # documented one-liner would silently do nothing.
    if not (telemetry_config.enabled or is_telemetry_env_enabled()):
        return None

    # `enabled` carries the *effective* value, not the config's. Writing the config's
    # False here is what would shadow the env var that just enabled us; writing True with
    # setdefault still lets an explicit NEMO_GYM_OTEL_ENABLED=0 turn everything off.
    enabled_env = _ENV_FIELD_MAP["enabled"]
    os.environ.setdefault(enabled_env, "1")

    for field, env_name in _ENV_FIELD_MAP.items():
        if field == "enabled":
            continue
        value = getattr(telemetry_config, field, None)
        if value is None:
            continue
        os.environ.setdefault(env_name, "1" if value is True else "0" if value is False else str(value))

    if telemetry_config.service_name:
        os.environ.setdefault(_SERVICE_NAME_ENV, str(telemetry_config.service_name))

    if not is_telemetry_env_enabled():
        return None

    # One run id for the whole fleet, so every server's spans and metrics carry the same
    # nemo.run.id resource attribute and a backend can group them. Generated here because
    # the orchestrator is the only process that sees the whole run; each server would
    # otherwise mint its own.
    run_id = os.environ.get(f"{_OTEL_PREFIX}_RUN_ID", "").strip()
    if not run_id:
        run_id = os.environ.get("SLURM_JOB_ID", "").strip() or uuid4().hex[:12]
        os.environ[f"{_OTEL_PREFIX}_RUN_ID"] = run_id
    return run_id


#: Packages a Gym server process needs in order to export telemetry. `nemo-lens[sdk]`
#: brings the providers and exporters; the two instrumentation packages back
#: `instrument_fastapi` / `instrument_aiohttp_client`, which nemo-lens imports lazily.
_SERVER_TELEMETRY_PACKAGES = (
    ("nemo-lens", "nemo-lens[sdk]"),
    ("opentelemetry-instrumentation-fastapi", "opentelemetry-instrumentation-fastapi"),
    ("opentelemetry-instrumentation-aiohttp-client", "opentelemetry-instrumentation-aiohttp-client"),
)


def _installed_requirement(dist_name: str, requirement_name: str) -> Optional[str]:
    """Build a requirement string pinning *dist_name* to the copy installed right here.

    Prefers the recorded install source over the version number. A git-installed
    nemo-lens reports a local version such as ``0.2.0+b85578f`` that exists on no index,
    so ``==`` would be unsatisfiable; the recorded VCS URL and commit reinstall exactly
    what this process is running.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return None

    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            import json

            direct_url = json.loads(raw)
            vcs_info = direct_url.get("vcs_info")
            if vcs_info and direct_url.get("url"):
                commit = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
                vcs = vcs_info.get("vcs", "git")
                if commit:
                    # No spaces around the '@'. PEP 508 allows either form, but
                    # `setup_env_command` interpolates this into a shell command line and
                    # joins head_server_deps unquoted, so the spaced form would be split
                    # into separate arguments. Quoting at that call site would be the
                    # better fix and is tracked separately.
                    return f"{requirement_name}@{vcs}+{direct_url['url']}@{commit}"
    except Exception:  # pragma: no cover - metadata is best-effort
        logger.debug("nemo-lens: could not read direct_url.json for %s", dist_name, exc_info=True)

    return f"{requirement_name}=={dist.version}"


def server_venv_requirements() -> list:
    """Extra requirements to install into every per-server venv.

    Gym builds an isolated venv per server, and those venvs install ``nemo-gym[dev]`` —
    not ``nemo-gym[telemetry]``. Without this, telemetry would be enabled in the
    orchestrator and simply absent in every server process, which is the one
    configuration that produces a trace with a hole in the middle of it.

    Pinning is derived from what *this* process has installed rather than restated here.
    A ``[tool.uv.sources]`` entry only governs dependencies resolved through the local
    project: a bare ``nemo-lens[sdk]`` passed to ``uv pip install`` ignores it and
    resolves from PyPI, which would put lens 0.1.0 in the servers while the orchestrator
    runs the pinned commit. Reading the installed distribution's ``direct_url.json``
    makes that skew impossible by construction instead of by keeping two pins in sync.

    Returns an empty list when telemetry is off or nemo-lens is not installed, so a
    normal run's venvs are byte-for-byte what they are today.

    Not called anywhere yet in this PR — Gym's venv-building code has no telemetry call
    site until the follow-up PR wires this in alongside the other call sites.
    """
    if not is_telemetry_env_enabled():
        return []
    requirements = []
    for dist_name, requirement_name in _SERVER_TELEMETRY_PACKAGES:
        requirement = _installed_requirement(dist_name, requirement_name)
        if requirement is not None:
            requirements.append(requirement)
    if not any(req.startswith("nemo-lens") for req in requirements):
        logger.warning(
            "telemetry is enabled but nemo-lens is not installed in this process, so the "
            "server venvs will not get it either. Install it with: uv sync --extra telemetry"
        )
        return []
    return requirements


def _build_resource_attributes(server_name: Optional[str], server_type: Optional[str]) -> dict:
    """Process-lifetime resource attributes.

    Only values constant for this process's whole life belong here — anything that varies
    per request is a span attribute instead
    (``kb/knowledge/conventions/telemetry-classification.md``).
    """
    from nemo_gym.package_info import __version__

    attrs: dict[str, Any] = {"nemo.gym.version": __version__}
    if server_name:
        attrs["nemo.gym.server.name"] = server_name
    if server_type:
        attrs["nemo.gym.server.type"] = server_type
    return attrs


def init_telemetry(
    server_name: Optional[str] = None,
    server_type: Optional[str] = None,
    resource_attributes: Optional[dict] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Optional["TelemetryHandle"]:
    """Initialise this process's telemetry. Call once per process; idempotent.

    Reads the ``NEMO_GYM_OTEL_*`` environment that :func:`configure_telemetry_env` put in
    place, so a server process needs no config file access to agree with its siblings.

    Args:
        server_name: This server's Gym config name (e.g. ``example_single_tool_call``).
            Used to disambiguate ``service.name`` across the fleet.
        server_type: ``resources_servers`` | ``responses_api_agents`` |
            ``responses_api_models``, or ``orchestrator`` for the CLI.
        resource_attributes: Extra process-lifetime attributes to merge.
        rank / world_size: Passed to the export strategy. Each Gym server process is rank
            0 of a world of 1 by default, which is why ``all_ranks`` is the default
            strategy.

    Returns:
        The :class:`TelemetryHandle`, or ``None`` when nemo-lens is absent or telemetry is
        disabled. Never raises: a telemetry failure must not take a server down.

    Not called anywhere yet in this PR — setting ``telemetry.enabled: true`` alone does
    nothing today. ``SimpleServer.run_webserver`` gaining a call to this is one of the
    call sites that land in the follow-up PR.
    """
    global _TELEMETRY_HANDLE, _INITIALISED
    # The whole check-and-set plus the actual setup_telemetry() call is one critical
    # section: without the lock, two threads can both observe `_INITIALISED is False`
    # before either sets it and both call setup_telemetry, which nemo-lens raises on for
    # a second call in the same process (or, depending on version, silently registers
    # duplicate OTel providers).
    with _INIT_LOCK:
        if _INITIALISED:
            return _TELEMETRY_HANDLE
        _INITIALISED = True

        # Checked before importing lens so a disabled process never pays the import.
        if not is_telemetry_env_enabled():
            return None

        try:
            from nemo.lens import NemoLensConfig, setup_telemetry
        except ImportError:
            logger.debug("nemo-lens is not installed; telemetry stays disabled")
            return None

        from nemo_gym.telemetry.span_groups import GymSpanGroup

        try:
            config = NemoLensConfig.from_env(
                prefix=_OTEL_PREFIX,
                fallback_prefix=_OTEL_FALLBACK_PREFIX,
                span_group_cls=GymSpanGroup,
            )
        except ValueError:
            logger.warning("nemo-lens: invalid telemetry environment; telemetry disabled", exc_info=True)
            return None

        if not config.enabled:
            return None

        # service.name per server, so a backend's service map shows three named Gym
        # services rather than three copies of one. The unsuffixed name is kept as a
        # resource attribute so the fleet is still groupable.
        service_group = config.service_name
        if server_name and _env_flag(_ENV_FIELD_MAP["service_name_per_server"], True):
            config.service_name = f"{service_group}/{server_name}"

        attrs = _build_resource_attributes(server_name, server_type)
        attrs["nemo.gym.service_group"] = service_group
        if resource_attributes:
            attrs.update(resource_attributes)

        try:
            handle = setup_telemetry(config, rank=rank, world_size=world_size, resource_attributes=attrs)
        except Exception:
            logger.warning("nemo-lens: telemetry setup failed; continuing without it", exc_info=True)
            return None

        _TELEMETRY_HANDLE = handle

        if config.logs_enabled and handle.is_exporting:
            try:
                from nemo.lens.logging_bridge import setup_logging_bridge

                setup_logging_bridge()
            except Exception:
                logger.warning("nemo-lens: failed to set up the logging bridge", exc_info=True)

        logger.info(
            "nemo-lens telemetry initialised (service=%s, exporting=%s, run_id=%s, groups=%s)",
            config.service_name,
            handle.is_exporting,
            config.run_id,
            config.span_groups,
        )
        return handle


def get_telemetry() -> Optional["TelemetryHandle"]:
    """Return this process's telemetry handle, or ``None`` if uninitialised/disabled."""
    return _TELEMETRY_HANDLE


def shutdown_telemetry(timeout_ms: int = 5000) -> None:
    """Flush and shut down this process's telemetry providers.

    Idempotent — ``TelemetryHandle.shutdown`` guards against a second call, and Gym
    reaches this from more than one terminal path. Never raises.
    """
    handle = _TELEMETRY_HANDLE
    if handle is None:
        return
    try:
        handle.shutdown(timeout_ms=timeout_ms)
    except Exception:
        logger.warning("nemo-lens: error during telemetry shutdown", exc_info=True)


def _reset_for_testing() -> None:
    """Drop the process-global handle so a test can initialise again.

    Test-only. Production code has exactly one init per process, which is what
    ``_INITIALISED`` enforces.
    """
    global _TELEMETRY_HANDLE, _INITIALISED
    _TELEMETRY_HANDLE = None
    _INITIALISED = False
