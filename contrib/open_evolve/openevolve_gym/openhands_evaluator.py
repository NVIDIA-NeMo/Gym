# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEvolve evaluator helpers for candidate instruction files.

The wrapper renders each OpenEvolve candidate into a NeMo Gym prompt overlay,
starts Gym, collects rollouts, and converts the rollout JSONL into the scalar
score dictionary expected by OpenEvolve.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SWE_AGENTS_CONFIG = "responses_api_agents/swe_agents/configs/swebench_multi_tools.yaml"
DEFAULT_MODEL_CONFIG = "responses_api_models/openai_model/configs/openai_model.yaml"
DEFAULT_INPUT_JSONL = "responses_api_agents/swe_agents/data/example.jsonl"
DEFAULT_AGENT_NAME = "swe_agents"
DEFAULT_NEMO_GYM_REPO = Path("../Gym")

SYSTEM_PROMPT_TEMPLATE = """\
You are running as a Codex-style software engineering agent through OpenHands.

Apply the following candidate skill while solving the task. Treat it as operational guidance,
not as task content.

<candidate_skill>
{candidate_skill}
</candidate_skill>
"""

USER_PROMPT_TEMPLATE = """\
Can you implement the necessary changes to the repository so that the requirements specified in
the <github_issue_description> are met?
Your task is to make the changes to non-test files in the {{ workspace_path }} directory.

<github_issue_description>
{{ instance.problem_statement }}
</github_issue_description>

<workspace_path>
{{ workspace_path }}
</workspace_path>

Follow the candidate skill from the system prompt while working. Implement tests first when a
small, local test can demonstrate the requested behavior. Do not modify hidden or evaluation
tests.
"""


@dataclass(frozen=True)
class RenderedOpenHandsCandidate:
    """Paths produced for one OpenEvolve candidate."""

    root: Path
    system_prompt: Path
    user_prompt: Path
    config_overlay: Path


@dataclass(frozen=True)
class OpenHandsEvaluationConfig:
    """Runtime settings for a Gym-backed OpenHands candidate evaluation."""

    nemo_gym_repo: Path = DEFAULT_NEMO_GYM_REPO
    config_overlay: Path | None = None
    output_jsonl: Path = Path("results/openevolve_openhands/rollouts.jsonl")
    input_jsonl_fpath: str = DEFAULT_INPUT_JSONL
    agent_name: str = DEFAULT_AGENT_NAME
    base_config_paths: tuple[str, ...] = (DEFAULT_SWE_AGENTS_CONFIG, DEFAULT_MODEL_CONFIG)
    policy_base_url: str = "http://localhost:8001/v1"
    policy_api_key: str = "dummy"
    policy_model_name: str = "dummy-model"
    limit: int = 1
    num_samples_in_parallel: int = 1
    upload_rollouts_to_wandb: bool = False
    head_server_url: str = "http://127.0.0.1:11000"
    server_ready_timeout_seconds: int = 120
    poll_interval_seconds: float = 1.0
    shutdown_timeout_seconds: int = 10
    ng_run_log: Path | None = None
    extra_ng_run_args: tuple[str, ...] = field(default_factory=tuple)
    extra_collect_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ng_run(self) -> Path:
        """Path to Gym's server startup command."""
        return self.nemo_gym_repo / ".venv" / "bin" / "ng_run"

    @property
    def ng_collect_rollouts(self) -> Path:
        """Path to Gym's rollout collection command."""
        return self.nemo_gym_repo / ".venv" / "bin" / "ng_collect_rollouts"


def render_openhands_candidate(
    candidate_path: str | Path,
    output_dir: str | Path,
) -> RenderedOpenHandsCandidate:
    """Render a candidate instruction file into OpenHands templates and a Gym overlay."""
    candidate_path = Path(candidate_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_skill = candidate_path.read_text()
    system_prompt = output_dir / "system_prompt.j2"
    user_prompt = output_dir / "user_prompt.j2"
    config_overlay = output_dir / "swe_agents_prompt_overlay.yaml"

    system_prompt.write_text(SYSTEM_PROMPT_TEMPLATE.format(candidate_skill=candidate_skill.strip()))
    user_prompt.write_text(USER_PROMPT_TEMPLATE)
    config_overlay.write_text(yaml.safe_dump(_overlay_dict(system_prompt, user_prompt), sort_keys=False))

    return RenderedOpenHandsCandidate(
        root=output_dir,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config_overlay=config_overlay,
    )


def build_ng_run_command(config: OpenHandsEvaluationConfig) -> list[str]:
    """Build the command that starts Gym with the candidate overlay."""
    config_paths = list(config.base_config_paths)
    if config.config_overlay is not None:
        config_paths.append(str(config.config_overlay))

    return [
        str(config.ng_run),
        f"+config_paths=[{','.join(config_paths)}]",
        f"+policy_base_url={config.policy_base_url}",
        f"+policy_api_key={config.policy_api_key}",
        f"+policy_model_name={config.policy_model_name}",
        *config.extra_ng_run_args,
    ]


def build_collect_rollouts_command(config: OpenHandsEvaluationConfig) -> list[str]:
    """Build the command that collects rollout(s) from a running Gym agent server."""
    return [
        str(config.ng_collect_rollouts),
        f"+agent_name={config.agent_name}",
        f"+input_jsonl_fpath={config.input_jsonl_fpath}",
        f"+output_jsonl_fpath={config.output_jsonl}",
        f"+limit={config.limit}",
        f"+num_samples_in_parallel={config.num_samples_in_parallel}",
        f"+upload_rollouts_to_wandb={str(config.upload_rollouts_to_wandb).lower()}",
        *config.extra_collect_args,
    ]


def evaluate_candidate(
    candidate_path: str | Path,
    output_dir: str | Path,
    config: OpenHandsEvaluationConfig | None = None,
    *,
    execute: bool = True,
) -> dict[str, Any]:
    """OpenEvolve-facing evaluator for one candidate instruction file."""
    config = config or OpenHandsEvaluationConfig()
    output_dir = Path(output_dir).expanduser().resolve()
    rendered = render_openhands_candidate(candidate_path, output_dir)
    output_jsonl = _candidate_output_jsonl(config.output_jsonl, output_dir)
    evaluation_config = replace(
        config,
        config_overlay=rendered.config_overlay,
        output_jsonl=output_jsonl,
    )

    if execute:
        score = run_gym_rollout(evaluation_config)
    else:
        score = {
            "combined_score": 0.0,
            "metrics": {
                "dry_run": True,
            },
            "artifacts": {},
        }

    score["artifacts"].update(
        {
            "candidate_path": str(Path(candidate_path).expanduser().resolve()),
            "system_prompt_path": str(rendered.system_prompt),
            "user_prompt_path": str(rendered.user_prompt),
            "config_overlay_path": str(rendered.config_overlay),
            "ng_run_command": build_ng_run_command(evaluation_config),
            "collect_rollouts_command": build_collect_rollouts_command(evaluation_config),
        }
    )
    return score


def run_gym_rollout(
    config: OpenHandsEvaluationConfig,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    run_command: Callable[..., Any] = subprocess.run,
    wait_for_servers: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Start Gym with the candidate overlay, collect rollout(s), then parse the score."""
    assert_live_readiness(config)
    wait_for_servers = wait_for_servers or wait_for_gym_servers
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with suppress(FileNotFoundError):
        config.output_jsonl.unlink()

    log_path = config.ng_run_log or config.output_jsonl.with_suffix(".ng_run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log:
        process = popen_factory(
            build_ng_run_command(config),
            cwd=config.nemo_gym_repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_servers(
                config.head_server_url,
                {config.agent_name, "policy_model"},
                timeout_seconds=config.server_ready_timeout_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
                process=process,
            )
            run_command(
                build_collect_rollouts_command(config),
                cwd=config.nemo_gym_repo,
                check=True,
            )
            score = parse_rollout_scores(config.output_jsonl)
            score["artifacts"]["ng_run_log"] = str(log_path)
            return score
        finally:
            _terminate_process(process, config.shutdown_timeout_seconds)


def wait_for_gym_servers(
    head_server_url: str,
    process_names: set[str],
    *,
    timeout_seconds: int,
    poll_interval_seconds: float,
    process: Any | None = None,
) -> None:
    """Wait until the Gym head server lists all requested server processes."""
    deadline = time.monotonic() + timeout_seconds
    head_server_url = head_server_url.rstrip("/")

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"ng_run exited before servers became ready: {process.returncode}")
        if _servers_ready(head_server_url, process_names):
            return
        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for Gym servers: {sorted(process_names)}")


def parse_rollout_scores(output_jsonl: str | Path) -> dict[str, Any]:
    """Parse Gym rollout JSONL into the scalar fitness contract used by OpenEvolve."""
    output_jsonl = Path(output_jsonl).expanduser().resolve()
    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        return _empty_score(output_jsonl)

    candidate_scores = [_extract_score(row) for row in rows]
    rewards = [_as_float(row.get("reward")) for row in rows]
    task_passed = [_task_passed(row) for row in rows]
    tests_passed_ratios = [ratio for row in rows if (ratio := _extract_metric(row, "tests_passed_ratio")) is not None]

    metrics: dict[str, Any] = {
        "num_rollouts": len(rows),
        "mean_reward": statistics.fmean(rewards),
        "task_passed_rate": statistics.fmean(task_passed),
    }
    if tests_passed_ratios:
        metrics["tests_passed_ratio"] = statistics.fmean(tests_passed_ratios)

    return {
        "combined_score": statistics.fmean(candidate_scores),
        "metrics": metrics,
        "artifacts": {
            "rollout_path": str(output_jsonl),
        },
    }


def assert_live_readiness(config: OpenHandsEvaluationConfig) -> None:
    """Raise if live Gym execution is missing required local/runtime inputs."""
    errors = _live_readiness_errors(config)
    if errors:
        message = "Live OpenEvolve evaluation is not ready:\n" + "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(message)


def _overlay_dict(system_prompt: Path, user_prompt: Path) -> dict[str, Any]:
    return {
        "swe_agents": {
            "responses_api_agents": {
                "swe_agents": {
                    "agent_prompt_overrides": [
                        {
                            "user_prompt_template": str(user_prompt),
                            "system_prompt_template": str(system_prompt),
                            "agent_cls": "CodexAgent",
                            "diversify_tool_names": False,
                            "camel_case_tool_names": False,
                        }
                    ],
                    "agent_prompt_override_random": False,
                    "concurrency": 1,
                }
            }
        }
    }


def _extract_score(row: dict[str, Any]) -> float:
    combined_score = _extract_metric(row, "combined_score")
    if combined_score is not None:
        return combined_score
    return _as_float(row.get("reward"))


def _extract_metric(row: dict[str, Any], key: str) -> float | None:
    candidates = [
        row.get(key),
        row.get("metrics", {}).get(key) if isinstance(row.get("metrics"), dict) else None,
        _nested_get(row, ("metadata", "metrics", key)),
        _nested_get(row, ("response", "metadata", "metrics", key)),
    ]
    for candidate in candidates:
        if candidate is not None:
            return _as_float(candidate)
    return None


def _nested_get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = row
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _task_passed(row: dict[str, Any]) -> float:
    resolved = row.get("resolved", row.get("task_passed"))
    return 1.0 if bool(resolved) else 0.0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _empty_score(output_jsonl: Path) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "metrics": {
            "num_rollouts": 0,
            "mean_reward": 0.0,
            "task_passed_rate": 0.0,
        },
        "artifacts": {
            "rollout_path": str(output_jsonl),
        },
    }


def _candidate_output_jsonl(configured_output: Path, output_dir: Path) -> Path:
    if configured_output.is_absolute():
        return configured_output
    return output_dir / configured_output.name


def _live_readiness_errors(config: OpenHandsEvaluationConfig) -> list[str]:
    errors: list[str] = []
    if not config.nemo_gym_repo.is_dir():
        errors.append(f"nemo_gym_repo does not exist or is not a directory: {config.nemo_gym_repo}")
    if not config.ng_run.is_file():
        errors.append(f"ng_run is missing: {config.ng_run}")
    if not config.ng_collect_rollouts.is_file():
        errors.append(f"ng_collect_rollouts is missing: {config.ng_collect_rollouts}")
    if config.config_overlay is None:
        errors.append("config_overlay is required for live execution")
    elif not config.config_overlay.is_file():
        errors.append(f"config_overlay is missing: {config.config_overlay}")
    for config_path in config.base_config_paths:
        resolved_config_path = _resolve_repo_path(config.nemo_gym_repo, config_path)
        if not resolved_config_path.is_file():
            errors.append(f"base_config_path is missing: {resolved_config_path}")
    input_jsonl = _resolve_repo_path(config.nemo_gym_repo, config.input_jsonl_fpath)
    if not input_jsonl.is_file():
        errors.append(f"input_jsonl_fpath is missing: {input_jsonl}")
    if not config.policy_base_url:
        errors.append("policy_base_url is required")
    if config.policy_api_key in {"", "dummy"}:
        errors.append("policy_api_key must be explicitly configured for live execution")
    if config.policy_model_name in {"", "dummy-model"}:
        errors.append("policy_model_name must be explicitly configured for live execution")
    if config.limit < 1:
        errors.append("limit must be at least 1 for live execution")
    if config.num_samples_in_parallel < 1:
        errors.append("num_samples_in_parallel must be at least 1 for live execution")
    return errors


def _resolve_repo_path(repo: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo / path


def _terminate_process(process: Any, timeout_seconds: int) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


def _servers_ready(head_server_url: str, process_names: set[str]) -> bool:
    try:
        with urllib.request.urlopen(f"{head_server_url}/server_instances", timeout=2) as response:
            instances = json.loads(response.read().decode())
    except Exception:
        return False

    by_name = {instance.get("process_name"): instance for instance in instances}
    if not process_names.issubset(by_name):
        return False

    for process_name in process_names:
        url = by_name[process_name].get("url")
        if not url:
            return False
        try:
            urllib.request.urlopen(url, timeout=2).close()
        except urllib.error.HTTPError:
            pass
        except Exception:
            return False
    return True
