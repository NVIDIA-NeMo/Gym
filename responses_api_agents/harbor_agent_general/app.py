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

import asyncio
import copy
import json
import logging
import posixpath
import re
import sys
import time
from pathlib import Path, PurePosixPath

import ray
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trajectories import Trajectory
from harbor.models.trial.config import (
    AgentConfig,
    ArtifactConfig,
    EnvironmentConfig,
    VerifierConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from responses_api_agents.harbor_agent_general.alerts import AlertScheduleConfig
from responses_api_agents.harbor_agent_general.atif import convert_atif_to_gym_responses
from responses_api_agents.harbor_agent_general.custom_envs.nemo_gym_sandbox.environment import (
    SharedWorkspaceConfig,
    validate_sandbox_template_placeholders,
)


logger = logging.getLogger(__name__)

NUM_SAMPLES_IN_PARALLEL_KEY_NAME = "num_samples_in_parallel"
AGENT_TIMEOUT_EXCEPTION_TYPE = "AgentTimeoutError"
AUDITED_OPENCODE_IMPORT_PATH = "responses_api_agents.harbor_agent_general.audited_opencode:AuditedOpenCode"
ALERTED_OPENCODE_IMPORT_PATH = "responses_api_agents.harbor_agent_general.audited_opencode:AlertedOpenCode"
AGENTIC_VERIFIER_IMPORT_PATH = "responses_api_agents.harbor_agent_general.agentic_verifier:AgenticVerifier"
SCORE_INTEGRITY_FILENAME = "score_integrity.json"
_SANDBOX_CLEANUP_TIMEOUT_PREFIX = "Timed out during OpenSandbox kill"
_SANDBOX_LIFECYCLE_RESET_MARKERS = (
    "OpenSandboxLifecycleResetError",
    "OpenSandbox background command state disappeared",
)
_SANDBOX_BACKEND_UNREACHABLE_MARKERS = (
    "Get command status failed: HTTP 502",
    "Could not connect to backend sandbox endpoint",
)
_OPENSANDBOX_API_KEY_ENV_REFERENCE = "${OPENSANDBOX_API_KEY}"

_RAY_WORKER_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


def _policy_agent_timed_out(trial: TrialResult) -> bool:
    return any(
        step.exception_info is not None and step.exception_info.exception_type == AGENT_TIMEOUT_EXCEPTION_TYPE
        for step in trial.step_results or []
    )


def _sandbox_cleanup_failed(trial: TrialResult) -> bool:
    exception_info = trial.exception_info
    return bool(
        exception_info is not None
        and exception_info.exception_type == "TimeoutError"
        and exception_info.exception_message.startswith(_SANDBOX_CLEANUP_TIMEOUT_PREFIX)
    )


def _failure_class_for_error(error: Exception) -> str:
    """Preserve actionable infra failures across Harbor/Ray exception wrappers."""
    rendered = f"{type(error).__name__}: {error}"
    if any(marker in rendered for marker in _SANDBOX_LIFECYCLE_RESET_MARKERS):
        return "sandbox_lifecycle_reset"
    if any(marker in rendered for marker in _SANDBOX_BACKEND_UNREACHABLE_MARKERS):
        return "sandbox_backend_unreachable"
    return "harbor_failed"


def _sandbox_step_infra_error(trial: TrialResult) -> str | None:
    """Return only policy-step failures known to originate in OpenSandbox infra."""
    for step in trial.step_results or []:
        exception_info = step.exception_info
        if exception_info is None:
            continue
        rendered = f"{exception_info.exception_type}: {exception_info.exception_message}"
        if any(
            marker in rendered for marker in (*_SANDBOX_LIFECYCLE_RESET_MARKERS, *_SANDBOX_BACKEND_UNREACHABLE_MARKERS)
        ):
            return rendered
    return None


def _is_opencode_agent(agent: AgentConfig) -> bool:
    return agent.name == "opencode" or agent.import_path in {
        ALERTED_OPENCODE_IMPORT_PATH,
        AUDITED_OPENCODE_IMPORT_PATH,
    }


def _supports_policy_alerts(agent: AgentConfig) -> bool:
    return agent.import_path in {
        ALERTED_OPENCODE_IMPORT_PATH,
        AUDITED_OPENCODE_IMPORT_PATH,
    }


def _policy_alert_metrics(trial: TrialResult) -> dict[str, int | float | bool]:
    contexts = []
    if trial.agent_result is not None:
        contexts.append(trial.agent_result)
    if trial.step_results:
        contexts.extend(step.agent_result for step in trial.step_results if step.agent_result is not None)

    statuses = [
        status
        for context in contexts
        for status in (context.metadata or {}).get("runtime_alerts", [])
        if isinstance(status, dict)
    ]
    if not statuses:
        return {}

    delivered = sum(bool(status.get("delivered")) for status in statuses)
    attempted = sum(int(status.get("attempts", 0)) > 0 for status in statuses)
    metrics: dict[str, int | float | bool] = {
        "policy_alert_count": len(statuses),
        "policy_alert_attempted_count": attempted,
        "policy_alert_delivered_count": delivered,
        "policy_alert_delivery_rate": delivered / len(statuses),
        "policy_alert_all_delivered": delivered == len(statuses),
    }
    session_statuses = [status for status in statuses if "session_user_turn_recorded" in status]
    if session_statuses:
        recorded = sum(bool(status.get("session_user_turn_recorded")) for status in session_statuses)
        responded = sum(bool(status.get("session_alert_processed")) for status in session_statuses)
        zero_token_length = sum(bool(status.get("session_zero_token_length")) for status in session_statuses)
        compacted = sum(bool(status.get("session_compaction_completed")) for status in session_statuses)
        metrics.update(
            {
                "policy_alert_session_recorded_count": recorded,
                "policy_alert_session_record_rate": recorded / len(session_statuses),
                "policy_alert_session_responded_count": responded,
                "policy_alert_session_response_rate": responded / len(session_statuses),
                "policy_alert_zero_token_length_count": zero_token_length,
                "policy_alert_compaction_completed_count": compacted,
            }
        )
    return metrics


class FileAccessAuditConfig(BaseModel):
    """Gym-side classification of an opaque policy filesystem trace."""

    model_config = ConfigDict(extra="forbid")

    honeypot_path: str
    additional_honeypot_paths: list[str] = Field(default_factory=list)
    trace_filename: str = ".policy-fs.trace"

    @staticmethod
    def _normalize_honeypot_path(value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("honeypot paths must be absolute non-root paths without '.' or '..' components")
        return str(path)

    @field_validator("honeypot_path", mode="after")
    @classmethod
    def validate_honeypot_path(cls, value: str) -> str:
        return cls._normalize_honeypot_path(value)

    @field_validator("additional_honeypot_paths", mode="after")
    @classmethod
    def validate_additional_honeypot_paths(cls, values: list[str]) -> list[str]:
        normalized = [cls._normalize_honeypot_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("additional_honeypot_paths must not contain duplicates")
        return normalized

    @field_validator("trace_filename", mode="after")
    @classmethod
    def validate_trace_filename(cls, value: str) -> str:
        if not value or PurePosixPath(value).name != value or value in {".", ".."}:
            raise ValueError("trace_filename must be a filename, not a path")
        return value


class VerifierFileAccessAuditConfig(BaseModel):
    """Gym-side classification of a verifier agent filesystem trace."""

    model_config = ConfigDict(extra="forbid")

    audited_paths: dict[str, str]
    trace_subdir: str = "judge"
    trace_filename: str = ".judge-fs.trace"

    @field_validator("audited_paths", mode="after")
    @classmethod
    def validate_audited_paths(cls, values: dict[str, str]) -> dict[str, str]:
        if not values:
            raise ValueError("audited_paths must not be empty")
        normalized: dict[str, str] = {}
        for name, value in values.items():
            if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
                raise ValueError("audited path names must be lowercase metric identifiers")
            normalized[name] = FileAccessAuditConfig._normalize_honeypot_path(value)
        if len(normalized.values()) != len(set(normalized.values())):
            raise ValueError("audited_paths must not contain duplicate paths")
        return normalized

    @field_validator("trace_subdir", mode="after")
    @classmethod
    def validate_trace_subdir(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("trace_subdir must be a relative path that remains within the verifier directory")
        return value

    @field_validator("trace_filename", mode="after")
    @classmethod
    def validate_trace_filename(cls, value: str) -> str:
        return FileAccessAuditConfig.validate_trace_filename(value)


_SYSCALL_PATTERN = re.compile(r"^(?:\[pid\s+\d+\]\s+|\d+\s+)?(?P<name>[a-zA-Z0-9_]+)\(")
_WRITE_ONLY_SYSCALLS = {
    "chmod",
    "chown",
    "creat",
    "fchmodat",
    "fchownat",
    "link",
    "linkat",
    "mkdir",
    "mkdirat",
    "mknod",
    "mknodat",
    "rename",
    "renameat",
    "renameat2",
    "rmdir",
    "symlink",
    "symlinkat",
    "truncate",
    "unlink",
    "unlinkat",
    "utime",
    "utimensat",
    "utimes",
}
_OPEN_SYSCALLS = {"open", "openat", "openat2"}
_OPEN_WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE")


def _access_modes(line: str) -> tuple[bool, bool]:
    """Return whether one traced file syscall represents a read and/or write."""

    match = _SYSCALL_PATTERN.match(line)
    if match is None:
        return True, False
    syscall = match.group("name")
    if syscall in _WRITE_ONLY_SYSCALLS:
        return False, True
    if syscall not in _OPEN_SYSCALLS:
        return True, False
    writes = any(flag in line for flag in _OPEN_WRITE_FLAGS)
    reads = "O_WRONLY" not in line
    return reads, writes


def _file_access_counts(
    trace_paths: list[Path],
    audited_paths: dict[str, str],
) -> dict[str, dict[str, int]]:
    normalized_paths = [PurePosixPath(path) for path in audited_paths.values()]
    for index, path in enumerate(normalized_paths):
        if any(path in other.parents or other in path.parents for other in normalized_paths[index + 1 :]):
            raise ValueError("audited filesystem paths must not overlap")

    path_patterns = {name: re.compile(re.escape(path) + r'(?=$|[/"<>])') for name, path in audited_paths.items()}
    dirfd_relative_pattern = re.compile(r'<(?P<base>/[^<>]*)>,\s*"(?P<relative>[^"\\]*)"')
    counts = {name: {"read": 0, "write": 0, "total": 0} for name in audited_paths}

    for trace_path in trace_paths:
        for line in trace_path.read_text(errors="replace").splitlines():
            matched_names = {name for name, pattern in path_patterns.items() if pattern.search(line)}
            for match in dirfd_relative_pattern.finditer(line):
                candidate = posixpath.normpath(posixpath.join(match.group("base"), match.group("relative")))
                matched_names.update(
                    name
                    for name, audited_path in audited_paths.items()
                    if candidate == audited_path or candidate.startswith(f"{audited_path}/")
                )
            if not matched_names:
                continue
            reads, writes = _access_modes(line)
            for name in matched_names:
                counts[name]["read"] += int(reads)
                counts[name]["write"] += int(writes)
                counts[name]["total"] += 1
    return counts


def _file_access_audit_metrics(
    trajectory_paths: list[Path],
    config: FileAccessAuditConfig,
) -> dict[str, bool | int]:
    agent_dirs = {path.parent for path in trajectory_paths}
    trace_paths = [agent_dir / config.trace_filename for agent_dir in sorted(agent_dirs)]
    missing = [path for path in trace_paths if not path.is_file()]
    if not trace_paths or missing:
        missing_display = ", ".join(str(path) for path in missing) or "<no agent directories>"
        raise FileNotFoundError(f"Policy filesystem audit trace is missing: {missing_display}")

    honeypot_paths = [config.honeypot_path, *config.additional_honeypot_paths]
    if len(honeypot_paths) != len(set(honeypot_paths)):
        raise ValueError("file access audit honeypot paths must be unique")
    counts = _file_access_counts(
        trace_paths,
        {f"path_{index}": path for index, path in enumerate(honeypot_paths)},
    )
    read_count = sum(value["read"] for value in counts.values())
    write_count = sum(value["write"] for value in counts.values())
    event_count = sum(value["total"] for value in counts.values())
    return {
        "policy_honeypot_accessed": event_count > 0,
        "policy_honeypot_access_event_count": event_count,
        "policy_honeypot_read_accessed": read_count > 0,
        "policy_honeypot_read_access_event_count": read_count,
        "policy_honeypot_write_accessed": write_count > 0,
        "policy_honeypot_write_access_event_count": write_count,
    }


def _verifier_file_access_audit_metrics(
    trajectory_paths: list[Path],
    config: VerifierFileAccessAuditConfig,
) -> dict[str, bool | int]:
    verifier_dirs = {path.parent.parent / "verifier" for path in trajectory_paths}
    trace_paths = [directory / config.trace_subdir / config.trace_filename for directory in sorted(verifier_dirs)]
    missing = [path for path in trace_paths if not path.is_file()]
    if not trace_paths or missing:
        missing_display = ", ".join(str(path) for path in missing) or "<no verifier directories>"
        raise FileNotFoundError(f"Verifier filesystem audit trace is missing: {missing_display}")

    counts = _file_access_counts(trace_paths, config.audited_paths)
    metrics: dict[str, bool | int] = {}
    for name, modes in counts.items():
        for mode in ("read", "write"):
            count = modes[mode]
            metrics[f"verifier_{name}_{mode}_accessed"] = count > 0
            metrics[f"verifier_{name}_{mode}_access_event_count"] = count
    return metrics


def _judge_score_integrity_metrics(trajectory_paths: list[Path]) -> dict[str, bool | int | str]:
    verifier_dirs = {path.parent.parent / "verifier" for path in trajectory_paths}
    integrity_paths = [directory / SCORE_INTEGRITY_FILENAME for directory in sorted(verifier_dirs)]
    missing = [path for path in integrity_paths if not path.is_file()]
    if not integrity_paths or missing:
        missing_display = ", ".join(str(path) for path in missing) or "<no verifier directories>"
        raise FileNotFoundError(f"Judge score integrity result is missing: {missing_display}")

    results = [json.loads(path.read_text()) for path in integrity_paths]
    terminal = all(result.get("terminal") is True for result in results)
    host_fallback_zero = bool(results) and all(result.get("host_fallback_zero") is True for result in results)
    accepted_call_count = sum(int(result.get("accepted_call_count", 0)) for result in results)
    reasons = [str(result.get("reason", "")) for result in results if result.get("reason")]
    return {
        "judge_score_terminal": terminal,
        "judge_score_host_fallback_zero": host_fallback_zero,
        "judge_score_accepted_call_count": accepted_call_count,
        "judge_score_integrity_error": "; ".join(reasons),
    }


def _validated_judge_score_integrity_metrics(
    trial: TrialResult,
    trajectory_paths: list[Path],
) -> dict[str, bool | int | str]:
    verifier_results = (
        [step.verifier_result for step in trial.step_results] if trial.step_results else [trial.verifier_result]
    )
    if not trajectory_paths or len(trajectory_paths) != len(verifier_results):
        raise ValueError("Agentic verifier trajectories and step results must align")

    step_metrics = []
    for path, verifier_result in zip(trajectory_paths, verifier_results, strict=True):
        try:
            metrics = _judge_score_integrity_metrics([path])
        except FileNotFoundError as exc:
            if verifier_result is not None:
                raise
            raise RuntimeError("Agentic verifier did not produce a result or a host score-integrity verdict") from exc

        if not metrics["judge_score_terminal"] and not metrics["judge_score_host_fallback_zero"]:
            reason = metrics["judge_score_integrity_error"] or "judge score was not terminal"
            raise RuntimeError(f"Agentic verifier score failed host integrity validation: {reason}")
        if verifier_result is None or verifier_result.rewards is None:
            raise RuntimeError("Agentic verifier did not produce a result despite a terminal judge score")
        if metrics["judge_score_host_fallback_zero"] and any(
            float(reward) != 0.0 for reward in verifier_result.rewards.values()
        ):
            raise RuntimeError("Agentic verifier host fallback verdict requires zero rewards")
        step_metrics.append(metrics)

    return {
        "judge_score_terminal": all(metrics["judge_score_terminal"] for metrics in step_metrics),
        "judge_score_host_fallback_zero": any(metrics["judge_score_host_fallback_zero"] for metrics in step_metrics),
        "judge_score_accepted_call_count": sum(
            int(metrics["judge_score_accepted_call_count"]) for metrics in step_metrics
        ),
        "judge_score_integrity_error": "; ".join(
            str(metrics["judge_score_integrity_error"])
            for metrics in step_metrics
            if metrics["judge_score_integrity_error"]
        ),
    }


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def harbor_job_worker(job_config_dict: dict, task_name: str) -> str:
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(HarborAgent.run_job(job_config_dict, task_name))


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    harbor_jobs_dir: Path
    harbor_dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    harbor_agent: AgentConfig = Field(default_factory=AgentConfig)
    harbor_environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    harbor_verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)
    model_server: ModelServerRef | None = None
    model_base_url_env_var: str = "OPENAI_BASE_URL"
    model_api_key_env_var: str = "OPENAI_API_KEY"
    model_api_key: str | None = None
    context_window: int = 262144
    max_output_tokens: int = 131072
    reasoning_field: str = "reasoning"
    environment_build_timeout_multiplier: float | None = None
    harbor_ray_task_num_cpus: float = Field(default=0.25, ge=0)
    harbor_debug: bool = False
    harbor_max_retries: int = Field(default=0, ge=0)
    harbor_reward_key: str = Field(default="reward", min_length=1)
    file_access_audit: FileAccessAuditConfig | None = None
    verifier_file_access_audit: VerifierFileAccessAuditConfig | None = None
    policy_alerts: AlertScheduleConfig | None = None

    @field_validator("harbor_jobs_dir", mode="after")
    @classmethod
    def normalize_jobs_dir(cls, jobs_dir: Path) -> Path:
        jobs_dir = jobs_dir.resolve()
        if jobs_dir.suffix.lower() == ".jsonl":
            return jobs_dir.parent / "harbor"
        return jobs_dir

    @model_validator(mode="after")
    def validate_opencode_provider(self) -> "HarborAgentConfig":
        if self.model_server is not None and self.model_api_key is None:
            raise ValueError("model_api_key is required when model_server is configured")
        if self.token_id_capture and self.model_server is None:
            raise ValueError("token_id_capture requires a Gym model_server")
        if (
            self.model_server is not None
            and _is_opencode_agent(self.harbor_agent)
            and not (self.harbor_agent.model_name or "").startswith("nemo/")
        ):
            raise ValueError("OpenCode must use a nemo/<model> name when routed through the Gym model server")
        if self.policy_alerts is not None and not _supports_policy_alerts(self.harbor_agent):
            raise ValueError("policy_alerts requires AlertedOpenCode or AuditedOpenCode")
        environment_kwargs = dict(self.harbor_environment.kwargs or {})
        exec_timeout = environment_kwargs.get("default_exec_timeout_s")
        if (
            self.policy_alerts is not None
            and isinstance(exec_timeout, (int, float))
            and self.policy_alerts.deadline_seconds > exec_timeout
        ):
            raise ValueError("policy_alerts.deadline_seconds cannot exceed environment.kwargs.default_exec_timeout_s")
        for field_name in (
            "sandbox_provider_options",
            "sandbox_path_copies",
            "sandbox_path_symlinks",
            "shared_workspace",
        ):
            validate_sandbox_template_placeholders(
                environment_kwargs.get(field_name),
                context=f"environment.kwargs.{field_name}",
            )
        shared_workspace = environment_kwargs.get("shared_workspace")
        if shared_workspace is not None:
            workspace_config = SharedWorkspaceConfig.model_validate(shared_workspace)
            provider_options = environment_kwargs.get("sandbox_provider_options") or {}
            volumes = provider_options.get("volumes", []) if isinstance(provider_options, dict) else []
            for volume in volumes:
                if not isinstance(volume, dict):
                    continue
                mount_path = volume.get("mountPath", volume.get("mount_path"))
                if volume.get("name") == workspace_config.volume.name:
                    raise ValueError("sandbox_provider_options.volumes duplicates shared workspace volume name")
                if mount_path == workspace_config.volume.mount_path:
                    raise ValueError("sandbox_provider_options.volumes duplicates shared workspace mount path")
        return self

    def agent_for_model_server(
        self,
        base_url: str,
        auxiliary_base_url: str | None = None,
    ) -> AgentConfig:
        if self.model_api_key is None:
            raise ValueError("model_api_key is required to route through the Gym model server")
        env = dict(self.harbor_agent.env)
        env[self.model_base_url_env_var] = base_url
        env[self.model_api_key_env_var] = self.model_api_key

        kwargs = copy.deepcopy(self.harbor_agent.kwargs)
        if _is_opencode_agent(self.harbor_agent):
            model_name = (self.harbor_agent.model_name or "").removeprefix("nemo/")
            opencode_config = kwargs.setdefault("opencode_config", {})
            provider = opencode_config.setdefault("provider", {})
            nemo_provider = provider.setdefault("nemo", {})
            nemo_provider.setdefault("npm", "@ai-sdk/openai-compatible")
            options = nemo_provider.setdefault("options", {})
            options["baseURL"] = base_url
            options.setdefault("apiKey", "EMPTY")  # pragma: allowlist secret
            model = nemo_provider.setdefault("models", {}).setdefault(model_name, {})
            model.setdefault("name", model_name)
            model.setdefault("reasoning", True)
            model.setdefault("tool_call", True)
            model.setdefault("interleaved", {"field": self.reasoning_field})
            limits = model.setdefault("limit", {})
            limits.setdefault("context", self.context_window)
            limits.setdefault("output", self.max_output_tokens)

            # OpenCode generates session titles with its small model. Keep those
            # utility calls on the same Gym model server, but outside the rollout
            # correlation route so they are not trained as policy turns.
            if auxiliary_base_url and auxiliary_base_url != base_url and "small_model" not in opencode_config:
                auxiliary_provider_name = "nemo-auxiliary"
                auxiliary_provider = provider.setdefault(auxiliary_provider_name, {})
                auxiliary_provider.setdefault("npm", "@ai-sdk/openai-compatible")
                auxiliary_options = auxiliary_provider.setdefault("options", {})
                auxiliary_options["baseURL"] = auxiliary_base_url
                auxiliary_options.setdefault("apiKey", "EMPTY")  # pragma: allowlist secret
                auxiliary_model = auxiliary_provider.setdefault("models", {}).setdefault(model_name, {})
                auxiliary_model.setdefault("name", model_name)
                opencode_config["small_model"] = f"{auxiliary_provider_name}/{model_name}"

        updates: dict[str, object] = {"env": env, "kwargs": kwargs}
        if self.policy_alerts is not None:
            kwargs["alert_schedule"] = self.policy_alerts.model_dump(mode="json")
            updates["override_timeout_sec"] = self.policy_alerts.deadline_seconds

        return self.harbor_agent.model_copy(update=updates)

    def build_job_config(self, task_name: str, job_name: str, agent: AgentConfig | None = None) -> JobConfig:
        agent = agent or self.harbor_agent
        if self.policy_alerts is not None:
            agent = agent.model_copy(
                update={
                    "override_timeout_sec": self.policy_alerts.deadline_seconds,
                    "kwargs": {**agent.kwargs, "alert_schedule": self.policy_alerts.model_dump(mode="json")},
                }
            )
        environment_kwargs = copy.deepcopy(self.harbor_environment.kwargs or {})
        sandbox_provider = environment_kwargs.get("sandbox_provider")
        if isinstance(sandbox_provider, dict):
            opensandbox = sandbox_provider.get("opensandbox")
            if isinstance(opensandbox, dict):
                connection = opensandbox.get("connection")
                if isinstance(connection, dict) and connection.get("api_key"):
                    connection["api_key"] = _OPENSANDBOX_API_KEY_ENV_REFERENCE

        return JobConfig(
            job_name=job_name,
            jobs_dir=self.harbor_jobs_dir,
            n_attempts=1,
            n_concurrent_trials=1,
            quiet=True,
            debug=self.harbor_debug,
            retry=RetryConfig(max_retries=self.harbor_max_retries),
            environment_build_timeout_multiplier=self.environment_build_timeout_multiplier,
            environment=self.harbor_environment.model_copy(update={"delete": True, "kwargs": environment_kwargs}),
            verifier=self.harbor_verifier,
            artifacts=self.artifacts,
            agents=[agent],
            datasets=[
                self.harbor_dataset.model_copy(update={"task_names": [task_name]}),
            ],
        )


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_index: int = Field(alias=TASK_INDEX_KEY_NAME)
    rollout_index: int = Field(alias=ROLLOUT_INDEX_KEY_NAME)


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class HarborAgent(SimpleResponsesAPIAgent):
    convert_atif_to_gym_responses = staticmethod(convert_atif_to_gym_responses)

    config: HarborAgentConfig

    _sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context) -> None:
        num_samples_in_parallel = get_global_config_dict().get(NUM_SAMPLES_IN_PARALLEL_KEY_NAME) or 1
        self._sem = asyncio.Semaphore(num_samples_in_parallel)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        # Harbor owns the full run() lifecycle.
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self._sem:
            try:
                rollout_id = self.rollout_id_from_run(body)
                agent = self.config.harbor_agent
                if self.config.model_server is not None:
                    model_base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
                    auxiliary_model_base_url = self.resolve_model_base_url(self.config.model_server.name)
                    agent = self.config.agent_for_model_server(
                        model_base_url,
                        auxiliary_base_url=auxiliary_model_base_url,
                    )
                job_name = f"t{body.task_index}-r{body.rollout_index}"
                if rollout_id is not None:
                    # The routed model URL is part of Harbor's job config. NRL
                    # assigns a new rollout id when it redispatches a failed
                    # sample, so each dispatch needs its own job directory.
                    job_name = f"{job_name}-{rollout_id}"
                job_config = self.config.build_job_config(
                    task_name=body.task_name,
                    job_name=job_name,
                    agent=agent,
                )

                job_ref = harbor_job_worker.options(num_cpus=self.config.harbor_ray_task_num_cpus).remote(
                    job_config.model_dump(mode="json"), body.task_name
                )
                try:
                    trial_dir = Path(await job_ref)
                except asyncio.CancelledError:
                    ray.cancel(job_ref, force=True)
                    raise

                return self.success_response(body, trial_dir)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.exception(
                    "Harbor rollout failed: task_index=%s rollout_index=%s",
                    body.task_index,
                    body.rollout_index,
                )
                return self.failure_response(body, err)

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        trial_paths = TrialPaths(trial_dir)
        trial = TrialResult.model_validate_json(trial_paths.result_path.read_text())
        if trial.step_results:
            trajectory_paths = [
                trial_paths.step_agent_dir(step.step_name) / TaskPaths.TRAJECTORY_FILENAME
                for step in trial.step_results
            ]
        else:
            trajectory_paths = [trial_paths.agent_dir / TaskPaths.TRAJECTORY_FILENAME]

        trajectories = [Trajectory.model_validate_json(path.read_text()) for path in trajectory_paths]
        conversion_warnings: list[str] = []
        converted = [
            item
            for trajectory in trajectories
            for item in self.convert_atif_to_gym_responses(trajectory, conversion_warnings)
        ]
        # The upstream converter retains input turns in ATIF order. Move only
        # the leading prompt to input; subsequent user/tool turns stay in output.
        prompt_end = 0
        for item in converted:
            if item.get("role") not in {"system", "user"}:
                break
            prompt_end += 1
        input_messages, output = converted[:prompt_end], converted[prompt_end:]
        if not input_messages:
            task_paths = TaskPaths(trial.config.task.get_local_path())
            instruction_path = (
                task_paths.step_instruction_path(trial.step_results[0].step_name)
                if trial.step_results
                else task_paths.instruction_path
            )
            input_messages = [{"role": "user", "content": instruction_path.read_text(), "type": "message"}]
        n_input_tokens, n_cache_tokens, n_output_tokens, _ = trial.compute_token_cost_totals()
        response = NeMoGymResponse(
            id=f"harbor-{trial.id}",
            created_at=(trial.finished_at.timestamp() if trial.finished_at is not None else time.time()),
            model=self.config.harbor_agent.model_name,
            object="response",
            output=output,
            parallel_tool_calls=False,
            temperature=body.responses_create_params.temperature,
            tool_choice="auto",
            tools=[],
            top_p=body.responses_create_params.top_p,
            status="completed",
            usage={
                "input_tokens": n_input_tokens or 0,
                "input_tokens_details": {
                    "cached_tokens": n_cache_tokens or 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens": n_output_tokens or 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": (n_input_tokens or 0) + (n_output_tokens or 0),
            },
        )
        verifier_result = trial.verifier_result
        # Harbor selects the configured final-step or aggregate reward.
        rewards = verifier_result.rewards if verifier_result is not None else None
        if rewards and self.config.harbor_reward_key not in rewards:
            raise ValueError(f"Harbor verifier has no reward key {self.config.harbor_reward_key!r}")
        reward = float(rewards[self.config.harbor_reward_key]) if rewards else 0.0
        policy_agent_timed_out = _policy_agent_timed_out(trial)
        sandbox_cleanup_failed = _sandbox_cleanup_failed(trial)
        file_access_metrics = (
            _file_access_audit_metrics(trajectory_paths, self.config.file_access_audit)
            if self.config.file_access_audit is not None
            else {}
        )
        verifier_file_access_metrics = (
            _verifier_file_access_audit_metrics(trajectory_paths, self.config.verifier_file_access_audit)
            if self.config.verifier_file_access_audit is not None
            else {}
        )
        score_integrity_metrics = (
            _validated_judge_score_integrity_metrics(trial, trajectory_paths)
            if self.config.harbor_verifier.import_path == AGENTIC_VERIFIER_IMPORT_PATH
            else {}
        )
        policy_alert_metrics = _policy_alert_metrics(trial)

        return HarborVerifyResponse.model_validate(
            body.model_dump(by_alias=True)
            | {
                "responses_create_params": body.responses_create_params.model_dump(mode="json")
                | {"input": input_messages},
                "response": response,
                "reward": reward,
                "policy_agent_timed_out": policy_agent_timed_out,
                "sandbox_cleanup_failed": sandbox_cleanup_failed,
                "atif_conversion": {
                    "lossless": not conversion_warnings,
                    "warnings": conversion_warnings,
                    "source_trajectory_paths": [str(path) for path in trajectory_paths],
                    "trajectories": [
                        {
                            "schema_version": trajectory.schema_version,
                            "session_id": trajectory.session_id,
                            "trajectory_id": trajectory.trajectory_id,
                            "agent_name": trajectory.agent.name,
                            "agent_version": trajectory.agent.version,
                            "agent_model_name": trajectory.agent.model_name,
                        }
                        for trajectory in trajectories
                    ],
                },
                **file_access_metrics,
                **verifier_file_access_metrics,
                **score_integrity_metrics,
                **policy_alert_metrics,
            }
        )

    def failure_response(self, body: HarborRunRequest, err: Exception) -> HarborVerifyResponse:
        response = NeMoGymResponse(
            id=f"harbor-error-t{body.task_index}-r{body.rollout_index}",
            created_at=time.time(),
            model=self.config.harbor_agent.model_name,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            status="failed",
        )
        return HarborVerifyResponse.model_validate(
            body.model_dump(by_alias=True)
            | {
                "response": response.model_dump(mode="json"),
                "reward": 0.0,
                NG_FAILURE_CLASS_KEY: _failure_class_for_error(err),
                "error": f"{type(err).__name__}: {err}",
            }
        )

    @staticmethod
    async def run_job(job_config_dict: dict, task_name: str) -> str:
        job_config = JobConfig.model_validate(job_config_dict)
        job_err: Exception | None = None

        try:
            job = await Job.create(job_config)
            await job.run()
        except Exception as err:  # noqa: BLE001 - recover Harbor's partial trial artifacts
            job_err = err

        job_dir = job_config.jobs_dir / job_config.job_name
        if job_dir.exists():
            for trial_dir in job_dir.iterdir():
                result_path = TrialPaths(trial_dir).result_path
                if not trial_dir.is_dir() or not result_path.is_file():
                    continue

                trial_result = TrialResult.model_validate_json(result_path.read_text())
                if trial_result.task_name != task_name and Path(trial_result.task_name).name != task_name:
                    continue
                step_infra_error = _sandbox_step_infra_error(trial_result)
                if step_infra_error is not None:
                    # Force Harbor to replace an infra-failed trial on Gym retry.
                    result_path.unlink()
                    raise RuntimeError(f"Harbor agent step failed with {step_infra_error}")
                if trial_result.exception_info is not None:
                    if _sandbox_cleanup_failed(trial_result) and trial_result.verifier_result is not None:
                        trial_paths = TrialPaths(trial_dir)
                        trajectory_paths = [
                            trial_paths.step_agent_dir(step.step_name) / TaskPaths.TRAJECTORY_FILENAME
                            for step in trial_result.step_results or []
                        ]
                        if not trajectory_paths:
                            trajectory_paths = [trial_paths.agent_dir / TaskPaths.TRAJECTORY_FILENAME]
                        if any(path.is_file() for path in trajectory_paths):
                            return str(trial_dir.resolve())
                    exception_info = trial_result.exception_info
                    # Deleting result.json forces Harbor to replace the failed trial on Gym retry.
                    result_path.unlink()
                    raise RuntimeError(
                        f"Harbor trial failed with {exception_info.exception_type}: {exception_info.exception_message}"
                    )

                return str(trial_dir.resolve())

        if job_err is not None:
            raise job_err

        raise FileNotFoundError(f"No Harbor trial result found in {job_dir}")


if __name__ == "__main__":
    HarborAgent.run_webserver()
