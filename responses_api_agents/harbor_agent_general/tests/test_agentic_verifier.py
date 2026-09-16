from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import TaskOS
from harbor.models.trial.paths import TrialPaths

from responses_api_agents.harbor_agent_general.agentic_verifier import (
    AgenticVerifier,
    AgenticVerifierConfig,
    _validate_score_trajectory,
    _WorkingDirectoryEnvironment,
)


class FakeJudge:
    SUPPORTS_RESUME = False

    def __init__(self, reward_path: Path, logs_dir: Path, config) -> None:
        self.reward_path = reward_path
        self.logs_dir = logs_dir
        self.config = config
        self.extra_env = dict(config.env)
        self.session_id = None
        self.context_id = None
        self.instructions: list[str] = []

    async def setup(self, environment) -> None:
        await environment.exec("judge-setup")

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        self.instructions.append(instruction)
        await environment.exec("judge-run")
        self.reward_path.write_text(json.dumps({"reward": 0.75}))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "tool_calls": [
                                {
                                    "tool_call_id": "score-1",
                                    "function_name": "harbor_score_score_solution",
                                }
                            ],
                            "observation": {
                                "results": [
                                    {
                                        "source_call_id": "score-1",
                                        "content": '{"accepted": true}',
                                    }
                                ]
                            },
                        },
                        {"message": "done", "tool_calls": None},
                    ]
                }
            )
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        context.metadata = {"judge": "complete"}


class FailingAfterScoreJudge(FakeJudge):
    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        await super().run(instruction, environment, context)
        raise TimeoutError("judge wrapper timed out during shutdown")


class FailingBeforeScoreJudge(FakeJudge):
    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        self.instructions.append(instruction)
        await environment.exec("judge-run")
        raise TimeoutError("judge timed out before scoring")


class ResumeAfterMissingScoreJudge(FailingBeforeScoreJudge):
    SUPPORTS_RESUME = True

    def __init__(self, reward_path: Path, logs_dir: Path, config) -> None:
        super().__init__(reward_path, logs_dir, config)
        self.resume_instructions: list[str] = []

    async def resume(self, instruction: str, environment, context: AgentContext) -> None:
        self.resume_instructions.append(instruction)
        await FakeJudge.run(self, instruction, environment, context)


class ResumeAfterNonterminalScoreJudge(FakeJudge):
    SUPPORTS_RESUME = True

    def __init__(self, reward_path: Path, logs_dir: Path, config) -> None:
        super().__init__(reward_path, logs_dir, config)
        self.resume_instructions: list[str] = []
        self.reward_existed_before_resume: bool | None = None

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        await super().run(instruction, environment, context)
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["steps"].append({"tool_calls": [{"tool_call_id": "later", "function_name": "bash"}]})
        trajectory_path.write_text(json.dumps(trajectory))

    async def resume(self, instruction: str, environment, context: AgentContext) -> None:
        self.resume_instructions.append(instruction)
        self.reward_existed_before_resume = self.reward_path.exists()
        await FakeJudge.run(self, instruction, environment, context)


class FailingSetupJudge(FakeJudge):
    async def setup(self, environment) -> None:
        await environment.exec("judge-setup")
        raise RuntimeError("judge setup failed")


def make_task(tmp_path: Path):
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "judge_instruction.md").write_text("Inspect and score /app.\n")
    paths = SimpleNamespace(
        tests_dir=tests_dir,
        step_tests_dir=lambda _name: tmp_path / "missing-step-tests",
    )
    verifier = SimpleNamespace(
        env={
            "RUBRIC_MODEL_API_KEY": "${TEST_JUDGE_KEY}",
            "RUBRIC_MODEL_API_BASE": "${TEST_JUDGE_BASE}",
        }
    )
    return SimpleNamespace(paths=paths, config=SimpleNamespace(verifier=verifier))


def make_environment(*, separate: bool = True):
    @contextlib.contextmanager
    def scoped_exec_env(_env):
        yield

    environment = SimpleNamespace(
        session_id=("task__trial__verifier__rollout" if separate else "task__trial__env"),
        context_id=uuid4(),
        os=TaskOS.LINUX,
        capabilities=SimpleNamespace(mounted=True),
        upload_dir=AsyncMock(),
        download_dir=AsyncMock(),
        ensure_dirs=AsyncMock(),
        exec=AsyncMock(return_value=ExecResult(stdout="", stderr="", return_code=0)),
        scoped_exec_env=scoped_exec_env,
    )
    return environment


def make_config() -> dict:
    return {
        "judge_agent": {
            "name": "nop",
            "model_name": "judge-model",
            "mcp_servers": [
                {
                    "name": "harbor_score",
                    "transport": "stdio",
                    "command": "python3",
                    "args": ["/tests/score_solution.py"],
                }
            ],
        },
        "judge_env_aliases": {
            "OPENAI_API_KEY": "RUBRIC_MODEL_API_KEY",
            "OPENAI_BASE_URL": "RUBRIC_MODEL_API_BASE",
        },
    }


@pytest.mark.asyncio
async def test_working_directory_environment_clamps_explicit_timeout() -> None:
    environment = make_environment()
    judge_environment = _WorkingDirectoryEnvironment(environment, "/judge", 3)

    await judge_environment.exec("long", timeout_sec=99)
    await judge_environment.exec("short", timeout_sec=1)
    await judge_environment.exec("default")

    assert [call.kwargs["timeout_sec"] for call in environment.exec.await_args_list] == [3, 1, 3]


@pytest.mark.parametrize(
    ("api_mode", "package"),
    [
        ("chat_completions", "@ai-sdk/openai-compatible"),
        ("responses", "@ai-sdk/openai"),
    ],
)
def test_configures_opencode_judge_api_transport(
    monkeypatch,
    tmp_path: Path,
    api_mode: str,
    package: str,
) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    config = make_config()
    config["judge_agent"] = {
        "name": "opencode",
        "model_name": "org/judge-model",
    }
    config["judge_opencode_provider"] = {
        "api_mode": api_mode,
        "base_url": "https://judge.test/v1",
    }
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(),
        config=config,
    )

    judge = verifier._resolved_judge_agent()

    assert judge.model_name == "rubric/org/judge-model"
    opencode_config = judge.kwargs["opencode_config"]
    assert opencode_config["small_model"] == "rubric/org/judge-model"
    assert opencode_config["provider"] == {
        "rubric": {
            "npm": package,
            "name": "Rubric model",
            "options": {
                "baseURL": "https://judge.test/v1",
                "apiKey": "{env:OPENAI_API_KEY}",
            },
            "models": {
                "org/judge-model": {
                    "name": "org/judge-model",
                    "reasoning": True,
                    "tool_call": True,
                }
            },
        }
    }


def test_configures_preinstalled_opencode_judge_api_transport(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    config = make_config()
    config["judge_agent"] = {
        "name": None,
        "import_path": ("responses_api_agents.harbor_agent_general.audited_opencode:PreinstalledOpenCode"),
        "model_name": "org/judge-model",
    }
    config["judge_opencode_provider"] = {
        "api_mode": "chat_completions",
        "base_url": "https://judge.test/v1",
    }
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(),
        config=config,
    )

    judge = verifier._resolved_judge_agent()

    assert judge.name is None
    assert judge.import_path.endswith(":PreinstalledOpenCode")
    assert judge.model_name == "rubric/org/judge-model"


@pytest.mark.asyncio
async def test_runs_judge_through_harbor_agent_factory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = make_environment()
    created = {}

    def create_agent(config, **_kwargs):
        created["config"] = config
        created["kwargs"] = _kwargs
        created["judge"] = FakeJudge(trial_paths.reward_json_path, _kwargs["logs_dir"], config)
        return created["judge"]

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=environment,
        config=make_config(),
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.75}
    assert created["config"].env == {
        "OPENAI_API_KEY": "secret-key",
        "OPENAI_BASE_URL": "https://judge.test/v1",
    }
    assert created["config"].mcp_servers[0].name == "harbor_score"
    assert created["kwargs"]["mcp_servers"] == created["config"].mcp_servers
    assert created["judge"].instructions == ["Inspect and score /app.\n"]
    environment.upload_dir.assert_not_awaited()
    assert [call.kwargs["cwd"] for call in environment.exec.await_args_list] == [
        "/judge",
        "/judge",
    ]
    assert json.loads((trial_paths.verifier_dir / "judge" / "context.json").read_text())["metadata"] == {
        "judge": "complete"
    }
    assert json.loads((trial_paths.verifier_dir / "score_integrity.json").read_text()) == {
        "accepted_call_count": 1,
        "reason": "",
        "terminal": True,
    }


@pytest.mark.asyncio
async def test_judge_command_timeout_precedes_client_backstop(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    def create_agent(config, **kwargs):
        return FakeJudge(trial_paths.reward_json_path, kwargs["logs_dir"], config)

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    wait_for_timeouts: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable, timeout=None):
        wait_for_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", recording_wait_for)
    config = make_config()
    config.update(
        {
            "setup_timeout_sec": 2,
            "run_timeout_sec": 3,
            "client_timeout_grace_sec": 0.5,
        }
    )
    environment = make_environment()
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=environment,
        config=config,
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.75}
    assert [call.kwargs["timeout_sec"] for call in environment.exec.await_args_list] == [2, 3]
    assert wait_for_timeouts == [2.5, 3.5]


@pytest.mark.asyncio
async def test_accepts_terminal_score_recovered_after_judge_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    def create_agent(config, **kwargs):
        return FailingAfterScoreJudge(trial_paths.reward_json_path, kwargs["logs_dir"], config)

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(),
        config=make_config(),
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.75}
    assert json.loads((trial_paths.verifier_dir / "score_integrity.json").read_text())["terminal"] is True


@pytest.mark.asyncio
async def test_resumes_judge_after_missing_score_without_rerunning_setup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    created = {}

    def create_agent(config, **kwargs):
        created["judge"] = ResumeAfterMissingScoreJudge(
            trial_paths.reward_json_path,
            kwargs["logs_dir"],
            config,
        )
        return created["judge"]

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    config = make_config()
    config["max_attempts"] = 2
    environment = make_environment()
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=environment,
        config=config,
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.75}
    assert created["judge"].instructions == [
        "Inspect and score /app.\n",
        created["judge"].resume_instructions[0],
    ]
    assert "policy submission is unchanged" in created["judge"].resume_instructions[0]
    assert "call score_solution exactly once" in created["judge"].resume_instructions[0]
    assert [call.kwargs["cwd"] for call in environment.exec.await_args_list] == [
        "/judge",
        "/judge",
        "/judge",
    ]


@pytest.mark.asyncio
async def test_returns_zero_after_exhausting_missing_score_retries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    created = {}

    def create_agent(config, **kwargs):
        created["judge"] = FailingBeforeScoreJudge(trial_paths.reward_json_path, kwargs["logs_dir"], config)
        return created["judge"]

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    config = make_config()
    config["max_attempts"] = 2
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(),
        config=config,
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.0}
    assert len(created["judge"].instructions) == 2
    assert "Inspect and score /app." in created["judge"].instructions[1]
    assert "call score_solution exactly once" in created["judge"].instructions[1]
    assert json.loads((trial_paths.verifier_dir / "score_integrity.json").read_text()) == {
        "accepted_call_count": 0,
        "host_fallback_zero": True,
        "reason": "judge trajectory is missing",
        "terminal": False,
    }


@pytest.mark.asyncio
async def test_resumes_judge_after_nonterminal_score_on_fixed_submission(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    created = {}

    def create_agent(config, **kwargs):
        created["judge"] = ResumeAfterNonterminalScoreJudge(
            trial_paths.reward_json_path,
            kwargs["logs_dir"],
            config,
        )
        return created["judge"]

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    config = make_config()
    config["max_attempts"] = 2
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(),
        config=config,
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.75}
    assert created["judge"].reward_existed_before_resume is False
    assert len(created["judge"].resume_instructions) == 1
    assert "nonterminal score sink" in created["judge"].resume_instructions[0]
    assert json.loads((trial_paths.verifier_dir / "score_integrity.json").read_text()) == {
        "accepted_call_count": 1,
        "reason": "",
        "terminal": True,
    }


@pytest.mark.asyncio
async def test_does_not_convert_judge_setup_failure_to_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-key")
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.test/v1")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    def create_agent(config, **kwargs):
        return FailingSetupJudge(trial_paths.reward_json_path, kwargs["logs_dir"], config)

    monkeypatch.setattr(
        "responses_api_agents.harbor_agent_general.agentic_verifier.AgentFactory.create_agent_from_config",
        create_agent,
    )
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(),
        config=make_config(),
    )

    with pytest.raises(RuntimeError, match="judge setup failed"):
        await verifier.verify()


def test_score_integrity_allows_failed_attempt_before_terminal_accepted_call(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "tool_calls": [{"tool_call_id": "failed", "function_name": "harbor_score_score_solution"}],
                        "observation": None,
                    },
                    {
                        "tool_calls": [{"tool_call_id": "accepted", "function_name": "harbor_score_score_solution"}],
                        "observation": {"results": [{"source_call_id": "accepted", "content": '{"accepted": true}'}]},
                    },
                    {"message": "done", "tool_calls": None},
                ]
            }
        )
    )

    result = _validate_score_trajectory(trajectory_path)

    assert result.terminal is True
    assert result.accepted_call_count == 1


def test_score_integrity_rejects_tool_call_after_accepted_score(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "tool_calls": [{"tool_call_id": "accepted", "function_name": "score_solution"}],
                        "observation": {"results": [{"source_call_id": "accepted", "content": '{"accepted": true}'}]},
                    },
                    {"tool_calls": [{"tool_call_id": "later", "function_name": "bash"}]},
                ]
            }
        )
    )

    result = _validate_score_trajectory(trajectory_path)

    assert result.terminal is False
    assert result.reason == "accepted score_solution call was not the judge's final tool call"


@pytest.mark.asyncio
async def test_rejects_shared_verifier_environment(tmp_path: Path) -> None:
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    verifier = AgenticVerifier(
        task=make_task(tmp_path),
        trial_paths=trial_paths,
        environment=make_environment(separate=False),
        config=make_config(),
    )

    with pytest.raises(ValueError, match="environment_mode='separate'"):
        await verifier.verify()


def test_config_rejects_reward_path_outside_verifier_logs() -> None:
    config = make_config()
    config["reward_path"] = "/app/reward.json"

    with pytest.raises(ValueError, match="under /logs/verifier"):
        AgenticVerifierConfig.model_validate(config)
