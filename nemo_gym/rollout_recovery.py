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
"""Identity checks for resuming saved evaluation results.

The manifest stores digests, not configuration values or credentials. Recovery reuses
completed rollouts; it does not checkpoint an agent's conversation or remote sandbox.
"""

import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import socket
import stat
import warnings
from collections.abc import Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal
from uuid import uuid4

from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.config.config_types import ConfigError
from nemo_gym.utils.path_utils import failures_path_for, resubmit_offset_path_for


def has_saved_generation(row: Mapping[str, Any]) -> bool:
    """Recognize retained response data, including a serialized response with empty output.

    A diagnostic error object or an empty mapping is not generation. A typed response
    has its producer identity and output list even when no assistant text was emitted.
    """
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    stage = failure.get("stage") if isinstance(failure, Mapping) else row.get("_ng_failure_stage")
    for key in ("response", "_ng_failure_response", "_ng_failure_partial_response"):
        answer = row.get(key)
        if not isinstance(answer, Mapping):
            continue
        # The agent outcome policy already rejected this failed response as an answer. Retain its
        # diagnostic output without vetoing RETRY; cleanup/verification responses keep their protection.
        if stage == "agent" and answer.get("status") == "failed":
            continue
        output = answer.get("output")
        if not isinstance(output, list):
            continue
        if (
            answer.get("object") == "response"
            and isinstance(answer.get("id"), str)
            and answer["id"]
            and (output or (answer.get("error") is None and answer.get("status") != "failed"))
        ):
            return True
        if answer.get("status") == "completed" and any(
            isinstance(item, Mapping) and item.get("type") == "message" and item.get("role") == "assistant"
            for item in output
        ):
            return True
    return False


def has_unclassified_execution_failure(row: Mapping[str, Any]) -> bool:
    """A selected route failure names no owned cause that permits running the agent again."""
    if matches_failure_class(row, {"judge_failed", "judge_invalid"}):
        return True  # Verification recovery may reuse an answer; it never reruns the agent.
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    not_sent = isinstance(record, Mapping) and (
        record.get("delivery") == "not_sent" or isinstance(failure, Mapping) and failure.get("stage") == "admission"
    )
    if row.get("_ng_unrouted_run_error") is True and not not_sent:
        # An aborted collection retains this attempt as diagnostic history, not a replacement outcome.
        return True
    if isinstance(record, Mapping) and record.get("source") == "collector":
        if record.get("delivery") == "possibly_delivered" and not not_sent:
            return True
    events = row.get("_ng_failure_events")
    selected = row.get("_ng_failure_id")
    if "_ng_failure_id" not in row:
        return False  # Legacy producer fields remain usable without a selected event report.
    if not isinstance(events, Mapping) or not isinstance(selected, str) or not selected:
        return True
    seen = set()
    while selected is not None:
        if not isinstance(selected, str):
            return True
        event = events.get(selected)
        if not isinstance(event, Mapping) or selected in seen:
            # Store resume and legacy collection also use this gate, before the recovery reader runs.
            return True
        seen.add(selected)
        status = event.get("http_status")
        if (
            event.get("boundary") == "model"
            and event.get("failure_kind") == "model_request_error"
            and event.get("operation") == "model_request"
            and type(status) is int
            and 200 <= status < 300
        ):
            # The provider sent successful headers, but the model operation failed
            # before its body was usable. Running the agent again could regenerate.
            return True
        cause = event.get("cause_id")
        if cause is None:
            return event.get("failure_kind") == "agent_route_failed" or (
                event.get("failure_kind") == "collection_dispatch_failed"
                and row.get("_ng_failure_class") == "agent_run_error"
            )
        selected = cause
    return False


logger = logging.getLogger(__name__)


# Collection/output choices can change between invocations without changing a task.
_COLLECTION_OPTIONS = frozenset(
    {
        "resume_from_cache",
        "resubmit",
        "checkpoint_dir",
        "checkpoint_every_s",
        "checkpoint_timeout_s",
        "dispatch_budget_s",
        "drain_margin_s",
        "dispatch_longest_first",
        "retry_terminal_timeouts",
        "allow_unsafe_resume",
        "output_jsonl_fpath",
        "input_jsonl_fpath",
        "num_samples_in_parallel",
        "max_resident_rollout_tasks",
        "retain_results_in_memory",
        "require_complete",
        "disable_aggregation",
        "disable_health_check",
        "health_check_workers",
        "health_check_ignored_checks",
        "health_check_max_not_final_failure_fraction",
        "health_check_max_final_failure_fraction",
        "upload_rollouts",
        "count_failure_classes_as_zero",
        "count_missing_rollouts_as_zero",
        "route_failures_to_sidecar",
        # Timing/queue reports observe the run; they do not change task execution.
        "dispatch_start_report_within_s",
        "dispatch_start_report_min_fraction",
        "dispatch_all_started_within_s",
        "server_queue_metrics_urls",
        "server_queue_max_waiting_ratio",
        "server_queue_sample_interval_s",
        "long_rollout_s",
        "long_rollout_max_fraction",
        "timing_summary_interval_s",
        "timed_out_max_fraction",
    }
)
# Server addresses and credentials are resolved anew when a job restarts. Keep model
# names, prompts, generation settings and environment configuration in the digest.
_SERVER_RUNTIME_OPTIONS = frozenset(
    {
        "host",
        "port",
        "api_key",
        "openai_api_key",
        "policy_api_key",
        "judge_api_key",
        "tavily_api_key",
        "anthropic_api_key",
        "model_api_key",
        "switchyard_api_key",
        "hf_token",
        "num_workers",
        "base_url",
        "openai_base_url",
        "policy_base_url",
        "judge_base_url",
        "sandbox_model_base_url",
        "model_base_url",
        "vllm_base_url",
        "switchyard_base_url",
        "head_server",
        "disallowed_ports",
        "port_range_low",
        "port_range_high",
        "model_call_capture_dir",
    }
)
_SERVER_KINDS = ("responses_api_agents", "responses_api_models", "resources_servers", "environment_servers")


_DEFAULT_MAX_ROLLOUT_ATTEMPTS = 3


def _get_max_rollout_attempts() -> int:
    """Read ``NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`` (positive int) or default to 3."""
    raw = os.environ.get("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS")
    if raw is None or raw == "":
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS
    try:
        n = int(raw)
        if n < 1:
            raise ValueError(f"must be >= 1, got {n}")
        return n
    except (TypeError, ValueError) as e:
        print(
            f"WARNING: could not parse NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={raw!r} ({e}); "
            f"falling back to default {_DEFAULT_MAX_ROLLOUT_ATTEMPTS}.",
            flush=True,
        )
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS


def load_resubmit_offset(output_fpath: str | Path) -> int:
    """Read the number of sidecar lines excluded from the current failure budget."""
    path = resubmit_offset_path_for(Path(output_fpath))
    if not path.exists():
        return 0
    try:
        offset = int(path.read_text())
        if offset < 0:
            raise ValueError("negative offset")
    except (ValueError, OSError) as error:
        raise ConfigError(f"Cannot read resubmit budget offset {path}: {error}") from error
    return offset


def record_resubmit_offset(output_fpath: str | Path) -> int:
    """Reset the failure budget without altering outcomes or execution identities.

    The caller must hold the run lock and validate/repair history first. The
    line-based format is shared with GDPVal's existing resubmit implementation.
    """
    output = Path(output_fpath)
    failures = failures_path_for(output)
    offset = 0
    if failures.exists():
        with failures.open("rb") as file:
            offset = sum(1 for _ in file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_file(resubmit_offset_path_for(output)) as file:
        file.write(f"{offset}\n".encode())
    return offset


def _plain(value: Any) -> Any:
    """Pydantic's Python dump may still contain nested Hydra containers."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def manifest_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_manifest.json")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _without_runtime_options(value: Any) -> Any:
    # Exclude only known configuration boundaries. Recursively dropping names such
    # as "port" would also erase meaningful tool schemas or environment parameters.
    result = {k: v for k, v in value.items() if k not in _SERVER_RUNTIME_OPTIONS and k not in _COLLECTION_OPTIONS}

    def settings_identity(settings):
        if not isinstance(settings, dict):
            return settings
        settings = {k: v for k, v in settings.items() if k not in _SERVER_RUNTIME_OPTIONS}
        # Only authentication headers at known HTTP settings boundaries are
        # operational. Other headers may change task data or model behavior.
        for name in ("headers", "default_headers", "openai_default_headers", "artifact_request_headers"):
            if isinstance(settings.get(name), dict):
                settings[name] = {
                    key: value
                    for key, value in settings[name].items()
                    if key.lower() not in {"authorization", "proxy-authorization", "x-api-key", "api-key"}
                }
        if isinstance(settings.get("token_id_capture"), dict):
            settings["token_id_capture"] = {k: v for k, v in settings["token_id_capture"].items() if k != "dir"}
        return settings

    for name, block in list(result.items()):
        if not isinstance(block, dict):
            continue
        updated = dict(block)
        for kind in _SERVER_KINDS:
            if kind in block and isinstance(block[kind], dict):
                updated[kind] = {
                    implementation: settings_identity(settings) for implementation, settings in block[kind].items()
                }
        result[name] = updated
    return result


def _normalize_runtime_references(value: Any) -> Any:
    """Keep recognized connection references stable wherever clients nest them.

    Literal nested task data is untouched, including fields named port or api_key.
    Preserve the reference name and any surrounding URL template in the identity.
    """
    if isinstance(value, dict):
        return {key: _normalize_runtime_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_runtime_references(item) for item in value]
    if isinstance(value, str):

        def normalize(match: re.Match[str]) -> str:
            reference = match[1]
            parts = reference.split(".")
            operational = reference in _SERVER_RUNTIME_OPTIONS or (
                len(parts) == 4 and parts[1] in _SERVER_KINDS and parts[3] in _SERVER_RUNTIME_OPTIONS
            )
            return f"<runtime:{reference}>" if operational else match[0]

        return re.sub(r"\$\{([^{}]+)\}", normalize, value)
    return value


def _configuration_identity(config: dict, global_config: dict, rows: list[dict]) -> dict:
    # Resolve only the servers reachable from the materialized agents. Keep the
    # original root as interpolation context, but do not evaluate unused servers
    # (which may require credentials/environment variables unavailable here).
    raw = _normalize_runtime_references(_plain(global_config))
    context = OmegaConf.create(raw)
    filtered = _without_runtime_options(raw)
    servers = {
        name for name, block in raw.items() if isinstance(block, dict) and any(k in block for k in _SERVER_KINDS)
    }
    pending = {(row.get("agent_ref") or {}).get("name") for row in rows} - {None}
    pending.update(row["_ng_environment_server"] for row in rows if row.get("_ng_environment_server"))
    # Agent-routed legacy rows are wrapped by an Environment Server at dispatch.
    # Its orchestration settings are part of the task even without a row stamp.
    agents = {(row.get("agent_ref") or {}).get("name") for row in rows if not row.get("_ng_environment_server")} - {
        None
    }
    if agents:
        for name in servers:
            for settings in context[name].get("environment_servers", {}).values():
                if not OmegaConf.is_dict(settings):
                    continue
                # Match environment_server_agent_refs: any typed top-level agent
                # reference, or agent_server without a type. Resolve mapping/name
                # interpolations, but defer unavailable settings on unused servers.
                # A selected server is resolved in full below, so its errors remain fatal.
                for field in settings:
                    try:
                        reference = settings[field]
                        if not OmegaConf.is_dict(reference):
                            continue
                        kind = reference.get("type")
                        if kind == "responses_api_agents" or (field == "agent_server" and kind is None):
                            if reference.get("name") in agents:
                                pending.add(name)
                    except InterpolationResolutionError:
                        continue
    missing = pending - servers
    if missing:
        raise ConfigError(f"Cannot verify run identity: running configuration is missing servers {sorted(missing)!r}.")
    pending.update(
        row["task_source"] for row in rows if isinstance(row.get("task_source"), str) and row["task_source"] in servers
    )
    if not pending:
        pending = set(servers)
    selected = {}

    def references(value, *, server_reference: bool = False):
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str):
                if (server_reference or value.get("type") in _SERVER_KINDS) and name not in servers:
                    raise ConfigError(f"Cannot verify run identity: running configuration is missing server {name!r}.")
                if name in servers:
                    pending.add(name)
            for key, item in value.items():
                references(
                    item,
                    server_reference=key in {"model_server", "resources_server", "agent_server", "environment_server"},
                )
        elif isinstance(value, list):
            for item in value:
                references(item, server_reference=server_reference)

    while pending:
        name = pending.pop()
        if name in selected or name not in servers:
            continue
        context[name] = filtered[name]
        selected[name] = OmegaConf.to_container(context[name], resolve=True)
        references(selected[name])
    # Collection overrides may themselves contain DictConfig/ListConfig values.
    collection = {key: _plain(value) for key, value in config.items() if key not in _COLLECTION_OPTIONS}
    context["_collection_identity"] = _normalize_runtime_references(collection)
    capture = raw.get("token_id_capture", {})
    context["_capture_identity"] = {key: value for key, value in capture.items() if key != "dir"}
    return {
        "collection": OmegaConf.to_container(context._collection_identity, resolve=True),
        "servers": selected,
        "token_id_capture": OmegaConf.to_container(context._capture_identity, resolve=True),
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_output_file(path: Path) -> Iterator[BinaryIO]:
    """Replace a file after a successful write, retaining its sharing permissions."""
    temporary = None
    try:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        # Exclusive creation honors the caller's umask. Rewrites retain existing
        # sharing permissions instead of inheriting NamedTemporaryFile's 0600.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(descriptor, "wb") as file:
            if path.exists():
                os.fchmod(file.fileno(), stat.S_IMODE(path.stat().st_mode))
            yield file
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict) -> None:
    """Publish JSON metadata without exposing a partial rewrite to readers."""
    with atomic_output_file(path) as file:
        file.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    selection_policy: Literal["latest_allocated"] = "latest_allocated"
    # Controller-owned execution numbers. Reserving a number is not a failure.
    next_attempt: dict[str, Annotated[int, Field(ge=1, strict=True)]] = Field(default_factory=dict)
    max_rollout_attempts: Annotated[int, Field(ge=1, strict=True)] | None = None
    retry_terminal_timeouts: bool = False
    run_id: str
    source_digest: str
    materialized_digest: str
    config_digest: str
    legacy_import: bool = False
    identity_overridden: bool = False

    @classmethod
    def create(cls, source: Path, rows: list[dict], config: dict, global_config: dict) -> "RunManifest":
        return cls(
            max_rollout_attempts=_get_max_rollout_attempts(),
            run_id=uuid4().hex,
            retry_terminal_timeouts=config.get("retry_terminal_timeouts", False),
            source_digest=_file_digest(source),
            materialized_digest=_digest(rows),
            config_digest=_digest(_configuration_identity(config, global_config, rows)),
        )

    def write(self, path: Path) -> None:
        # Publish a complete manifest, or leave the previous one intact.
        atomic_write_json(path, self.model_dump(mode="json"))

    @classmethod
    def import_legacy(cls, rows: list[dict]) -> "RunManifest":
        return cls(
            max_rollout_attempts=_get_max_rollout_attempts(),
            run_id=uuid4().hex,
            source_digest="unverified",
            materialized_digest=_digest(rows),
            config_digest="unverified",
            legacy_import=True,
        )


def validate_resume(
    path: Path,
    current: RunManifest | None,
    materialized_path: Path,
    *,
    allow_unsafe: bool = False,
) -> RunManifest | None:
    """Reject unknown or incompatible saved work before any output is modified.

    ``allow_unsafe`` is an explicit compatibility escape hatch for legacy artifacts or
    deliberate identity overrides. Unsupported/corrupt manifests remain errors.
    """
    if not path.exists():
        problem = "Saved rollouts have no run manifest; their input/configuration identity cannot be verified."
        saved = None
    else:
        try:
            saved = RunManifest.model_validate_json(path.read_bytes())
            with materialized_path.open(encoding="utf-8") as file:
                saved_rows = [json.loads(line) for line in file if line.strip()]
        except (ValidationError, ValueError, OSError) as error:
            raise ConfigError(f"Cannot read recovery manifest {path} or materialized inputs: {error}") from error
        mismatches = []
        if saved.legacy_import:
            mismatches.append("unverified legacy run identity")
        if saved.identity_overridden:
            mismatches.append("previously overridden run identity")
        if saved.materialized_digest != _digest(saved_rows):
            mismatches.append("saved materialized inputs")
        if current is None:
            mismatches.append("current input/configuration identity")
        else:
            for field, description in (
                ("source_digest", "source dataset"),
                ("materialized_digest", "current materialized inputs"),
                ("config_digest", "resolved configuration"),
            ):
                if getattr(saved, field) != getattr(current, field):
                    mismatches.append(description)
        if not mismatches:
            return saved
        problem = "Cannot resume with incompatible " + ", ".join(mismatches) + "."
    if not allow_unsafe:
        raise ConfigError(problem + " Start a fresh run or explicitly set allow_unsafe_resume=true.")
    warnings.warn(problem + " Continuing because allow_unsafe_resume=true; saved inputs will be reused.", stacklevel=2)
    return saved.model_copy(update={"identity_overridden": True}) if saved is not None else None


def observed_elapsed(record: dict[str, Any]) -> float | None:
    """Best-effort per-rollout wallclock from a result/failure row."""
    # Native environments own their result schema; response and metadata need
    # not be mappings. Optional scheduling hints must never invalidate an answer.
    response = record.get("response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    for candidate in (
        record.get("elapsed_seconds"),
        metadata.get("elapsed_seconds") if isinstance(metadata, Mapping) else None,
    ):
        try:
            if candidate is not None:
                value = float(candidate)
                if value > 0:
                    return value
        except (TypeError, ValueError):
            continue
    return None


def matches_failure_class(record: Mapping[str, Any], classes: Collection[str]) -> bool:
    """Match shared failure names and retained producer routing names alike."""
    return record.get("_ng_failure_class") in classes or record.get("_ng_failure_legacy_class") in classes


def is_terminal_failure(record: Mapping[str, Any], *, retry_terminal_timeouts: bool = False) -> bool:
    """Whether a persisted failure must be gated on resume.

    By default a row is terminal iff it is stamped ``_ng_failure_terminal``, so
    an agent that marks its timeouts terminal on purpose keeps them gated.

    ``retry_terminal_timeouts`` is for agents whose older builds incorrectly
    stamped per-attempt timeouts terminal. A timeout reflects the
    load/remaining walltime of that attempt, so it remains retryable (up to the
    normal max-attempt cap).

    ``timeout_exceeded`` is an ``_ng_failure_class`` label that agents already
    write to the failures sidecar (pinchbench, nooa). It predates
    ``nemo_gym.core.failure_kinds`` and is not registered there; ``agent_timeout`` is
    the shared name.

    As ``failure_kinds`` requires, this function decides retryability for the
    occurrence, under an explicit caller opt-in; the name carries no retry meaning.
    """
    if not retry_terminal_timeouts:
        return bool(record.get("_ng_failure_terminal"))
    if matches_failure_class(record, {"timeout_exceeded"}):
        return False
    return bool(record.get("_ng_failure_terminal"))


def run_lock_path_for(output: Path) -> Path:
    """Stable lock file; never unlink it while another process may hold it open."""
    output = output.resolve()
    return output.with_name(output.stem + "_run.lock")


@contextmanager
def run_lock(output: Path, *, checkpoint_dir: Path | None = None) -> Iterator[None]:
    """Own a run before preparing inputs, restoring state, or writing artifacts.

    Collection currently acquires its own lock in run_from_config; do not nest
    it inside this context. Checkpoint integration must extend that acquisition to
    cover restore and supply checkpoint_dir, rather than acquire a second lock.
    When supplied, the directory lock prevents two outputs sharing checkpoints.
    The filesystem must support flock across all writer nodes. Kernel locks are released
    on process death; the presence of an old lock file does not block recovery.
    """
    paths = {run_lock_path_for(output)}
    if checkpoint_dir is not None:
        paths.add(checkpoint_dir.resolve() / ".collector.lock")
    with ExitStack() as files:
        for path in sorted(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            file = files.enter_context(path.open("a+b"))
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                file.seek(0)
                holder = file.read(4096).decode("utf-8", errors="replace").strip() or "owner details unavailable"
                raise ConfigError(
                    f"Another collector owns this run ({path}; holder: {holder}). Wait for it to stop before resuming."
                ) from error
            except OSError as error:
                if error.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK}:
                    raise ConfigError(
                        f"Cannot lock run at {path}: the filesystem must support flock across all writer nodes. "
                        "Use a filesystem with distributed flock support; Lustre noflock disables it and "
                        "localflock only locks within one node."
                    ) from error
                raise
            file.seek(0)
            file.truncate()
            file.write(
                json.dumps(
                    {"pid": os.getpid(), "host": socket.gethostname(), "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID")}
                ).encode()
                + b"\n"
            )
            file.flush()
        yield


def has_only_intentional_gaps(*, expected: int, completed: int, intentionally_omitted: int) -> bool:
    """Accept omissions only alongside completed tasks, without inventing scores."""
    return completed > 0 and intentionally_omitted > 0 and completed + intentionally_omitted == expected


class IncompleteEvaluationError(RuntimeError):
    """Saved partial results are valid, but the evaluation still has unfinished work."""

    exit_code = 75

    def __init__(self, message: str, *, retryable: bool = True, made_progress: bool = True) -> None:
        if retryable and not made_progress:
            message += " No retry progress was made; automatic restarts use the infrastructure retry limit."
        super().__init__(message)
        # 75 continues productive work; an empty invocation must spend a bounded
        # infrastructure retry. 76 stops when another attempt cannot help.
        self.exit_code = (75 if made_progress else 1) if retryable else 76
