# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor verifier that delegates evaluation to an independently configured agent."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.verifier.result import VerifierResult
from harbor.utils.env import resolve_env_vars
from harbor.verifier.base import BaseVerifier
from pydantic import BaseModel, Field, field_validator


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_OPENCODE_API_PACKAGES = {
    "chat_completions": "@ai-sdk/openai-compatible",
    "responses": "@ai-sdk/openai",
}
_PREINSTALLED_OPENCODE_IMPORT_PATH = "responses_api_agents.harbor_agent_general.audited_opencode:PreinstalledOpenCode"
_AUDITED_OPENCODE_IMPORT_PATH = "responses_api_agents.harbor_agent_general.audited_opencode:AuditedOpenCode"
_OPENCODE_IMPORT_PATHS = {_PREINSTALLED_OPENCODE_IMPORT_PATH, _AUDITED_OPENCODE_IMPORT_PATH}
SCORE_INTEGRITY_FILENAME = "score_integrity.json"


@dataclass(frozen=True)
class ScoreIntegrityResult:
    """Host-side verdict over the judge's score tool-call protocol."""

    terminal: bool
    accepted_call_count: int
    reason: str

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "terminal": self.terminal,
                    "accepted_call_count": self.accepted_call_count,
                    "reason": self.reason,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _contains_accepted_result(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("accepted") is True:
            return True
        return any(_contains_accepted_result(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_accepted_result(item) for item in value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return False
        return _contains_accepted_result(decoded)
    return False


def _validate_score_trajectory(trajectory_path: Path) -> ScoreIntegrityResult:
    if not trajectory_path.is_file():
        return ScoreIntegrityResult(False, 0, "judge trajectory is missing")
    try:
        raw = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ScoreIntegrityResult(False, 0, f"judge trajectory is unreadable: {type(exc).__name__}")
    steps = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(steps, list):
        return ScoreIntegrityResult(False, 0, "judge trajectory has no steps list")

    all_tool_call_ids: list[str] = []
    accepted_score_call_ids: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        observation = step.get("observation")
        results = observation.get("results", []) if isinstance(observation, dict) else []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("tool_call_id")
            function_name = tool_call.get("function_name")
            if not isinstance(tool_call_id, str):
                continue
            all_tool_call_ids.append(tool_call_id)
            if not isinstance(function_name, str) or not (
                function_name == "score_solution" or function_name.endswith("_score_solution")
            ):
                continue
            accepted = any(
                isinstance(result, dict)
                and result.get("source_call_id") == tool_call_id
                and _contains_accepted_result(result.get("content"))
                for result in results
            )
            if accepted:
                accepted_score_call_ids.append(tool_call_id)

    if len(accepted_score_call_ids) != 1:
        return ScoreIntegrityResult(
            False,
            len(accepted_score_call_ids),
            f"expected exactly one accepted score_solution call, found {len(accepted_score_call_ids)}",
        )
    if not all_tool_call_ids or all_tool_call_ids[-1] != accepted_score_call_ids[0]:
        return ScoreIntegrityResult(
            False,
            1,
            "accepted score_solution call was not the judge's final tool call",
        )
    return ScoreIntegrityResult(True, 1, "")


class OpenCodeProviderConfig(BaseModel, extra="forbid"):
    """External model provider used by an OpenCode judge."""

    api_mode: Literal["chat_completions", "responses"]
    base_url: str
    provider_name: str = "rubric"
    api_key_env_var: str = "OPENAI_API_KEY"

    @field_validator("base_url", "provider_name", "api_key_env_var")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


class AgenticVerifierConfig(BaseModel, extra="forbid"):
    """Configuration passed through ``JobConfig.verifier.kwargs.config``."""

    judge_agent: AgentConfig
    judge_opencode_provider: OpenCodeProviderConfig | None = None
    judge_env_aliases: dict[str, str] = Field(default_factory=dict)
    instruction_path: str = "judge_instruction.md"
    judge_workdir: str = "/judge"
    reward_path: str = "/logs/verifier/reward.json"
    judge_logs_subdir: str = "judge"
    setup_timeout_sec: float | None = Field(default=None, gt=0)
    run_timeout_sec: float | None = Field(default=None, gt=0)
    client_timeout_grace_sec: float = Field(
        default=60.0,
        ge=0,
        description=("Additional client wait time after the verifier environment's server-enforced command timeout"),
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Total judge attempts, including the initial attempt",
    )

    @field_validator("instruction_path", "judge_logs_subdir")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("path must be non-empty, relative, and remain within its root")
        return value

    @field_validator("judge_workdir", "reward_path")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be absolute and must not contain '..'")
        return value

    @field_validator("reward_path")
    @classmethod
    def validate_reward_path(cls, value: str) -> str:
        verifier_dir = EnvironmentPaths.verifier_dir
        path = PurePosixPath(value)
        if path != verifier_dir and verifier_dir not in path.parents:
            raise ValueError("reward_path must remain under /logs/verifier")
        return value


class _WorkingDirectoryEnvironment:
    """Duck-typed environment view with judge-only cwd and timeout limits."""

    def __init__(
        self,
        environment: BaseEnvironment,
        workdir: str,
        default_timeout_sec: float | None = None,
    ) -> None:
        self._environment = environment
        self._workdir = workdir
        self._default_timeout_sec = default_timeout_sec

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        effective_timeout_sec = timeout_sec
        if self._default_timeout_sec is not None:
            effective_timeout_sec = (
                min(timeout_sec, self._default_timeout_sec) if timeout_sec is not None else self._default_timeout_sec
            )
        return await self._environment.exec(
            command,
            cwd=cwd or self._workdir,
            env=env,
            timeout_sec=effective_timeout_sec,
            user=user,
        )


def _resolve_template(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            return value
        name, default = match.groups()
        if name in variables:
            return variables[name]
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{name}' is required by judge_agent")
    if isinstance(value, dict):
        return {key: _resolve_template(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_template(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_template(item, variables) for item in value)
    return value


class AgenticVerifier(BaseVerifier):
    """Run a Harbor agent as a judge in Harbor's separate verifier environment."""

    def __init__(
        self,
        *,
        task: Task,
        trial_paths: TrialPaths,
        environment: BaseEnvironment,
        config: AgenticVerifierConfig | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task=task,
            trial_paths=trial_paths,
            environment=environment,
            **kwargs,
        )
        self.config = (
            config if isinstance(config, AgenticVerifierConfig) else AgenticVerifierConfig.model_validate(config)
        )

    def _instruction(self) -> str:
        relative = Path(self.config.instruction_path)
        candidates = []
        if self.step_name is not None:
            candidates.append(self.task.paths.step_tests_dir(self.step_name) / relative)
        candidates.append(self.task.paths.tests_dir / relative)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.read_text()
        raise FileNotFoundError(f"Judge instruction {self.config.instruction_path!r} was not found in task tests")

    def _resolved_judge_agent(self) -> AgentConfig:
        verifier_templates = {
            **self.task.config.verifier.env,
            **(self.verifier_env or {}),
            **self.override_env,
        }
        variables = resolve_env_vars(verifier_templates) if verifier_templates else {}
        raw = self.config.judge_agent.model_dump(
            mode="python",
            exclude_none=True,
            context={"redact_sensitive_env": False},
        )
        judge_agent = AgentConfig.model_validate(_resolve_template(raw, variables))
        env = dict(judge_agent.env)
        for target, source in self.config.judge_env_aliases.items():
            if source not in variables:
                raise ValueError(
                    f"Verifier environment variable {source!r} required for "
                    f"judge environment variable {target!r} is missing"
                )
            env[target] = variables[source]
        judge_agent = judge_agent.model_copy(update={"env": env})
        return self._configure_opencode_provider(judge_agent)

    def _configure_opencode_provider(self, judge_agent: AgentConfig) -> AgentConfig:
        provider_config = self.config.judge_opencode_provider
        if provider_config is None:
            return judge_agent
        if judge_agent.name != "opencode" and judge_agent.import_path not in _OPENCODE_IMPORT_PATHS:
            raise ValueError("judge_opencode_provider requires an OpenCode judge agent")
        if not judge_agent.model_name:
            raise ValueError("judge_opencode_provider requires judge_agent.model_name")

        kwargs = copy.deepcopy(judge_agent.kwargs)
        opencode_config = kwargs.setdefault("opencode_config", {})
        providers = opencode_config.setdefault("provider", {})
        provider = providers.setdefault(provider_config.provider_name, {})
        provider["npm"] = _OPENCODE_API_PACKAGES[provider_config.api_mode]
        provider.setdefault("name", "Rubric model")
        options = provider.setdefault("options", {})
        options["baseURL"] = provider_config.base_url
        options.setdefault("apiKey", f"{{env:{provider_config.api_key_env_var}}}")
        model = provider.setdefault("models", {}).setdefault(judge_agent.model_name, {})
        model.setdefault("name", judge_agent.model_name)
        model.setdefault("reasoning", True)
        model.setdefault("tool_call", True)

        selected_model = f"{provider_config.provider_name}/{judge_agent.model_name}"
        opencode_config.setdefault("small_model", selected_model)
        return judge_agent.model_copy(update={"model_name": selected_model, "kwargs": kwargs})

    async def _run_with_timeout(self, awaitable, timeout_sec: float | None) -> None:
        if timeout_sec is None:
            await awaitable
        else:
            await asyncio.wait_for(
                awaitable,
                timeout=timeout_sec + self.config.client_timeout_grace_sec,
            )

    async def _download_judge_logs(self, judge_logs_dir: Path) -> None:
        env_paths = EnvironmentPaths.for_os(self.environment.os)
        try:
            await self.environment.download_dir(
                source_dir=str(env_paths.agent_dir),
                target_dir=judge_logs_dir,
            )
        except Exception:
            self.logger.exception("Failed to download agentic judge logs")

        if not self.environment.capabilities.mounted:
            await self.environment.download_dir(
                source_dir=str(env_paths.verifier_dir),
                target_dir=self.trial_paths.verifier_dir,
            )

    def _parse_rewards(self) -> dict[str, float | int]:
        reward_path = self._local_reward_path()
        if not reward_path.is_file():
            raise FileNotFoundError(f"Agentic judge did not produce {self.config.reward_path}")
        raw = json.loads(reward_path.read_text())
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Agentic judge reward must be a non-empty JSON object")
        rewards: dict[str, float | int] = {}
        for key, value in raw.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Agentic judge rewards must map non-empty strings to finite numbers")
            rewards[key] = value
        return rewards

    def _local_reward_path(self) -> Path:
        reward_relative = PurePosixPath(self.config.reward_path).relative_to(EnvironmentPaths.verifier_dir)
        return self.trial_paths.verifier_dir.joinpath(*reward_relative.parts)

    def _record_host_fallback_zero(self, score_integrity: ScoreIntegrityResult) -> None:
        """Record an auditable zero after all fixed-submission retries fail."""
        reward_path = self._local_reward_path()
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(json.dumps({"reward": 0.0}, indent=2) + "\n")
        integrity_path = self.trial_paths.verifier_dir / SCORE_INTEGRITY_FILENAME
        integrity_path.write_text(
            json.dumps(
                {
                    "terminal": False,
                    "accepted_call_count": score_integrity.accepted_call_count,
                    "reason": score_integrity.reason,
                    "host_fallback_zero": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _score_integrity(self, judge_logs_dir: Path) -> ScoreIntegrityResult:
        result = _validate_score_trajectory(judge_logs_dir / "trajectory.json")
        integrity_path = self.trial_paths.verifier_dir / SCORE_INTEGRITY_FILENAME
        integrity_path.write_text(result.to_json())
        return result

    @staticmethod
    def _retry_instruction(score_integrity: ScoreIntegrityResult) -> str:
        return (
            "Your previous verifier attempt did not produce a terminal score_solution call "
            f"({score_integrity.reason}). The policy submission is unchanged. Continue the "
            "evaluation using any work you already completed. The nonterminal score sink, if "
            "one was written, has been reset; call score_solution exactly once as your final "
            "tool call."
        )

    @override
    async def verify(self) -> VerifierResult:
        if "__verifier__" not in self.environment.session_id:
            raise ValueError("AgenticVerifier requires Harbor verifier.environment_mode='separate'")

        await self.environment.ensure_dirs([self.config.judge_workdir], chmod=True)

        judge_logs_dir = self.trial_paths.verifier_dir / self.config.judge_logs_subdir
        judge_logs_dir.mkdir(parents=True, exist_ok=True)
        judge_config = self._resolved_judge_agent()
        agent_kwargs: dict[str, Any] = {}
        if judge_config.mcp_servers:
            agent_kwargs["mcp_servers"] = judge_config.mcp_servers
        judge = AgentFactory.create_agent_from_config(
            judge_config,
            logs_dir=judge_logs_dir,
            logger=self.logger,
            **agent_kwargs,
        )
        judge.session_id = f"{self.environment.session_id}__judge"
        judge.context_id = self.environment.context_id
        context = AgentContext()
        setup_environment = _WorkingDirectoryEnvironment(
            self.environment,
            self.config.judge_workdir,
            self.config.setup_timeout_sec,
        )
        judge_environment = _WorkingDirectoryEnvironment(
            self.environment,
            self.config.judge_workdir,
            self.config.run_timeout_sec,
        )

        instruction = self._instruction()
        setup_complete = False
        previous_score_integrity: ScoreIntegrityResult | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            run_error: BaseException | None = None
            score_integrity: ScoreIntegrityResult | None = None
            try:
                with self.environment.scoped_exec_env(judge.extra_env):
                    if not setup_complete:
                        await self._run_with_timeout(
                            judge.setup(environment=setup_environment),
                            self.config.setup_timeout_sec,
                        )
                        setup_complete = True

                    retry_instruction: str | None = None
                    if attempt > 1:
                        assert previous_score_integrity is not None
                        retry_instruction = self._retry_instruction(previous_score_integrity)
                    if retry_instruction is not None and getattr(judge, "SUPPORTS_RESUME", False):
                        judge_run = judge.resume(
                            instruction=retry_instruction,
                            environment=judge_environment,
                            context=context,
                        )
                    else:
                        attempt_instruction = instruction
                        if retry_instruction is not None:
                            attempt_instruction += "\n\n" + retry_instruction
                        judge_run = judge.run(
                            instruction=attempt_instruction,
                            environment=judge_environment,
                            context=context,
                        )
                    await self._run_with_timeout(judge_run, self.config.run_timeout_sec)
            except BaseException as exc:
                run_error = exc
            finally:
                try:
                    await self._download_judge_logs(judge_logs_dir)
                    judge.populate_context_post_run(context)
                    (judge_logs_dir / "context.json").write_text(context.model_dump_json(indent=2))
                finally:
                    # A judge command can fail while winding down after score_solution
                    # succeeded. Inspect the recovered trajectory before retrying.
                    score_integrity = self._score_integrity(judge_logs_dir)

            assert score_integrity is not None
            if score_integrity.terminal:
                if run_error is not None:
                    self.logger.warning(
                        "Accepting terminal agentic judge score recovered after %s: %s",
                        type(run_error).__name__,
                        run_error,
                    )
                return VerifierResult(rewards=self._parse_rewards())

            if not setup_complete:
                assert run_error is not None
                raise run_error
            if isinstance(run_error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise run_error
            if attempt == self.config.max_attempts:
                self.logger.error(
                    "Agentic judge produced no terminal score after %d attempts; returning an audited reward 0. "
                    "Last integrity result: %s",
                    attempt,
                    score_integrity.reason,
                )
                self._record_host_fallback_zero(score_integrity)
                return VerifierResult(rewards={"reward": 0.0})

            self.logger.warning(
                "Agentic judge attempt %d/%d produced no terminal score%s; retrying the fixed submission: %s",
                attempt,
                self.config.max_attempts,
                f" after {type(run_error).__name__}: {run_error}" if run_error is not None else "",
                score_integrity.reason,
            )
            # score_solution intentionally rejects a second successful call.
            # OpenCode resume retains its session database but overwrites the
            # per-command trajectory, so removing only the stale sink permits
            # an exact-one-call retry without discarding the judge's analysis.
            if score_integrity.accepted_call_count:
                self._local_reward_path().unlink(missing_ok=True)
            previous_score_integrity = score_integrity

        raise AssertionError("unreachable")
