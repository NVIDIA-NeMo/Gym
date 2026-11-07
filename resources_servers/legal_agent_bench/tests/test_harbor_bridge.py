# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge tests run in the Harbor agent environment; LAB preparation needs no Harbor."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError


pytest.importorskip("harbor")

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from resources_servers.legal_agent_bench.harbor_bridge import (
    LegalAgentBenchHarborBridge,
    LegalAgentBenchHarborConfig,
    LegalAgentBenchHarborRequest,
)
from resources_servers.legal_agent_bench.legal_harbor_agent import LegalAgentBenchHarborAgent


def _bridge() -> LegalAgentBenchHarborBridge:
    config = LegalAgentBenchHarborConfig(
        name="legal_agent_bench_harbor_agent",
        host="127.0.0.1",
        port=0,
        entrypoint="../../resources_servers/legal_agent_bench/harbor_bridge.py",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        harbor_resolve_paths_from_gym_root=True,
        job={
            "jobs_dir": "results/legal_agent_bench/harbor_jobs",
            "datasets": [{"path": "/tmp/tasks"}],
            "agents": [
                {"import_path": "resources_servers.legal_agent_bench.legal_harbor_agent:LegalAgentBenchHarborAgent"}
            ],
        },
    )
    return LegalAgentBenchHarborBridge.model_construct(config=config, server_client=MagicMock())


def test_lab_bridge_uses_shared_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("responses_api_agents.harbor_agent_general.app.get_global_config_dict", lambda: {})
    bridge = _bridge()
    routes = {route.path for route in bridge.setup_webserver().routes}
    assert {"/v1/responses", "/run", "/aggregate_metrics"} <= routes


@pytest.mark.parametrize("pointer", ["other::task", "legal_agent_bench::", "legal_agent_bench::../task"])
def test_lab_bridge_rejects_invalid_pointer(pointer) -> None:
    with pytest.raises(ValidationError):
        LegalAgentBenchHarborRequest.model_validate(
            {
                "instance_id": pointer,
                "responses_create_params": {"input": []},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
        )


def test_lab_bridge_preserves_pointer_and_translates_task() -> None:
    request = LegalAgentBenchHarborRequest.model_validate(
        {
            "instance_id": "legal_agent_bench::domain__task",
            "responses_create_params": {"input": []},
            TASK_INDEX_KEY_NAME: 3,
            ROLLOUT_INDEX_KEY_NAME: 2,
        }
    )
    assert request.task_name == "domain__task"
    assert request.model_dump()["instance_id"] == "legal_agent_bench::domain__task"
    assert request.task_index == 3 and request.rollout_index == 2


def test_lab_bridge_rejects_conflicting_explicit_task_name() -> None:
    with pytest.raises(ValidationError, match="task_name does not match instance_id"):
        LegalAgentBenchHarborRequest.model_validate(
            {
                "instance_id": "legal_agent_bench::domain__task",
                "task_name": "different__task",
                "responses_create_params": {"input": []},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
        )


def test_lab_agent_declares_atif_capability() -> None:
    assert LegalAgentBenchHarborAgent.capabilities.atif


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "error_type", "error_message", "max_turns"),
    [
        ("model-error", RuntimeError, "HTTP 503: model unavailable", 3),
        ("empty-error", ConnectionResetError, "", 3),
        ("context-limit", RuntimeError, "context_length_exceeded", 3),
        ("turn-limit", None, None, 1),
        ("completed", None, None, 3),
    ],
)
async def test_agent_run_preserves_artifacts_and_distinguishes_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
    error_type: type[Exception] | None,
    error_message: str | None,
    max_turns: int,
) -> None:
    from harbor.models.agent.context import AgentContext

    from resources_servers.legal_agent_bench.harness.adapters.base import ModelResponse, ToolCall
    from resources_servers.legal_agent_bench.harness.adapters.openai_compatible import OpenAICompatibleAdapter
    from resources_servers.legal_agent_bench.prepare import REQUIRED_SKILLS, _marker

    task_dir = tmp_path / "task"
    (task_dir / "documents").mkdir(parents=True)
    (task_dir / "documents/input.txt").write_text("document contents", encoding="utf-8")
    (task_dir / "task.toml").touch()
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "title": "Review",
                "instructions": "Review input.txt",
                "criteria": [{"id": "C1", "title": "Review", "match_criteria": "Correct review"}],
            }
        ),
        encoding="utf-8",
    )
    skills = tmp_path / "skills"
    for name in REQUIRED_SKILLS:
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(f"# {name}", encoding="utf-8")
    (skills / ".nemo_gym_asset.json").write_text(json.dumps(_marker("skills")), encoding="utf-8")
    model_calls = 0

    async def chat(self: OpenAICompatibleAdapter, messages: list[dict], tools: list[dict]) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                message={"role": "assistant", "content": "Reading"},
                tool_calls=[ToolCall(id="read-1", name="read", arguments='{"file_path":"input.txt"}')],
                input_tokens=10,
                output_tokens=2,
            )
        if error_type is not None:
            raise error_type(error_message)
        return ModelResponse(
            message={"role": "assistant", "content": "Done"}, text="Done", input_tokens=20, output_tokens=3
        )

    class Environment:
        environment_dir = task_dir / "environment"

        def __init__(self) -> None:
            self.downloads: list[str] = []

        async def upload_dir(self, source: Path, destination: str) -> None:
            pass

        async def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
            result = ""
            if command.startswith("python "):
                tool = shlex.split(command)[2]
                assert tool in {"preflight", "read"}
                result = json.dumps({"result": "OK" if tool == "preflight" else "document contents", "metrics": {}})
            return SimpleNamespace(stdout=result, stderr="", return_code=0)

        async def download_dir(self, source: str, destination: Path) -> None:
            self.downloads.append(source)
            destination.mkdir(parents=True)
            (destination / "retained.txt").write_text(source, encoding="utf-8")

    monkeypatch.setattr(OpenAICompatibleAdapter, "chat", chat)
    # Only model and sandbox I/O are substituted; the agent lifecycle remains real.
    environment = Environment()
    context = AgentContext()
    agent = LegalAgentBenchHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="policy",
        api_base="http://unused/v1",
        max_turns=max_turns,
        skills_dir=skills,
    )
    instruction = "<!-- lab_task_id:area/task -->"
    if outcome in {"model-error", "empty-error"}:
        with pytest.raises(RuntimeError, match="LAB model request failed") as raised:
            await agent.run(instruction, environment, context)
        assert (error_message or "ConnectionResetError") in str(raised.value)
    else:
        await agent.run(instruction, environment, context)

    artifact_dir = tmp_path / "logs/artifacts/lab-run"
    assert environment.downloads == ["/workspace/output", "/workspace/workspace"]
    for name in ("output", "workspace"):
        assert (artifact_dir / name / "retained.txt").read_text() == f"/workspace/{name}"
    assert json.loads((artifact_dir / "config.json").read_text())["task"] == "area/task"
    metrics = json.loads((artifact_dir / "metrics.json").read_text())
    assert metrics["context_overflow"] is (outcome == "context-limit")
    assert metrics["finished_cleanly"] is (outcome == "completed")
    assert metrics["model_error"] == (
        error_message or "ConnectionResetError" if outcome in {"model-error", "empty-error"} else None
    )
    assert metrics["turn_count"] == model_calls == (1 if outcome == "turn-limit" else 2)
    transcript = [json.loads(line) for line in (artifact_dir / "transcript.jsonl").read_text().splitlines()]
    assert any(entry.get("tool_call_id") == "read-1" for entry in transcript)
    if error_type is not None:
        failure = next(entry for entry in transcript if entry["role"] == "model_error")
        assert failure["error_type"] == error_type.__name__
    trajectory = json.loads((tmp_path / "logs/trajectory.json").read_text())
    assert trajectory["steps"][1]["observation"]["results"][0]["source_call_id"] == "read-1"
    assert context.n_input_tokens == metrics["input_tokens"] == (30 if outcome == "completed" else 10)
    assert context.n_output_tokens == metrics["output_tokens"] == (5 if outcome == "completed" else 2)
    assert context.metadata["artifact_dir"] == str(artifact_dir)
    assert context.metadata["model_error"] == metrics["model_error"]
    flags = json.loads((tmp_path / "logs/agent_error_flags.json").read_text())
    assert flags["context_length_exceeded"] is (outcome == "context-limit")


def test_lab_trajectory_keeps_tool_ids_and_full_observations(tmp_path: Path) -> None:
    from harbor.models.trajectories import Trajectory

    from resources_servers.legal_agent_bench.harness.adapters.base import ModelResponse, ToolCall
    from resources_servers.legal_agent_bench.legal_harbor_agent import _log_tool, _log_turn, _write_trajectory
    from responses_api_agents.harbor_agent_general.app import HarborAgent

    transcript = tmp_path / "transcript.jsonl"
    observations = {"read-1": "document contents " * 200, "read-2": "second document " * 200}
    calls = [
        ToolCall(id=call_id, name="read", arguments=json.dumps({"file_path": f"{call_id}.docx"}))
        for call_id in observations
    ]
    with transcript.open("w", encoding="utf-8") as stream:
        _log_turn(stream, 1, "assistant", ModelResponse(message={"content": ""}, tool_calls=calls))
        for call in calls:
            _log_tool(stream, 1, call.id, call.name, call.arguments, observations[call.id])
    _write_trajectory(
        logs_dir=tmp_path,
        instruction="review memo",
        transcript_path=transcript,
        metrics={},
        model_name="policy",
        agent_name="legal-agent-bench",
    )
    trajectory = Trajectory.model_validate_json((tmp_path / "trajectory.json").read_text())
    output = HarborAgent.convert_atif_to_gym_responses(trajectory)
    assert {
        item["call_id"]: item["output"] for item in output if item["type"] == "function_call_output"
    } == observations
    assert [item["call_id"] for item in output if item["type"] == "function_call"] == list(observations)


@pytest.mark.parametrize(("model_name", "port"), [("policy_model", 12345), ("alternate_model", 23456)])
@pytest.mark.parametrize(("explicit_id", "capture_key"), [(None, "3-2"), ("lab-explicit", "lab-explicit")])
def test_lab_bridge_routes_model_calls_without_mutating_template(
    monkeypatch: pytest.MonkeyPatch, model_name: str, port: int, explicit_id: str | None, capture_key: str
) -> None:
    global_config = {
        "observability_enabled": True,
        "policy_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 12345}}},
        "alternate_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 23456}}},
    }
    monkeypatch.setattr("responses_api_agents.harbor_agent_general.app.get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(
        "resources_servers.legal_agent_bench.harbor_bridge.get_global_config_dict", lambda: global_config
    )
    bridge = _bridge()
    bridge.server_client = SimpleNamespace(global_config_dict=global_config)
    bridge.config.model_server.name = model_name
    template = bridge.config.job.model_dump(mode="json")
    request = LegalAgentBenchHarborRequest.model_validate(
        {
            "instance_id": "legal_agent_bench::domain__task",
            "responses_create_params": {"input": [], "temperature": 0.25, "top_p": 0.8, "max_output_tokens": 2048},
            TASK_INDEX_KEY_NAME: 3,
            ROLLOUT_INDEX_KEY_NAME: 2,
            "_ng_rollout_id": explicit_id,
        }
    )
    job = bridge.config.build_job_config(request.task_name, "t3-r2")
    bridge._configure_job_for_run(job, request)
    assert job.agents[0].kwargs["api_base"] == f"http://127.0.0.1:{port}/ng-rollout/{capture_key}/v1"
    assert job.agents[0].kwargs["responses_create_params"] == {
        "input": [],
        "temperature": 0.25,
        "top_p": 0.8,
        "max_output_tokens": 2048,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    assert bridge.config.job.model_dump(mode="json") == template


@pytest.mark.parametrize(
    ("config_path", "agent_name", "resources_name"),
    [
        (
            "resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml",
            "legal_agent_bench_harbor_agent",
            "legal_agent_bench",
        ),
        (
            "benchmarks/legal_agent_bench/config.yaml",
            "legal_agent_bench_benchmark_harbor_agent",
            "legal_agent_bench_benchmark_resources_server",
        ),
    ],
)
def test_lab_configs_keep_schema_and_paths_when_loaded_outside_gym(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: str, agent_name: str, resources_name: str
) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from nemo_gym.data import TrainDataProcessor

    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)
    for key in (
        "LEGAL_AGENT_BENCH_TASK_CACHE_DIR",
        "LEGAL_AGENT_BENCH_RUNTIME_TASKS_DIR",
        "LEGAL_AGENT_BENCH_SKILLS_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    config = GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=OmegaConf.merge(
            OmegaConf.load(root / config_path),
            GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            {"default_host": "127.0.0.1"},
        )
    )
    instances = GlobalConfigDictParser().filter_for_server_instance_configs(config)
    agent = next(instance for instance in instances if instance.name == agent_name)
    agent_config = LegalAgentBenchHarborConfig.model_validate(
        {
            "name": agent_name,
            **OmegaConf.to_container(config[agent_name].responses_api_agents.harbor_agent_general, resolve=True),
        }
    )
    assert agent.get_inner_run_server_config().resources_server["name"] == resources_name
    assert TrainDataProcessor._schema_home_for(agent, instances) == ("legal_agent_bench", "resources_servers")
    assert agent_config.harbor_resolve_paths_from_gym_root is True
    assert agent_config.harbor_unique_job_names is True
    job = agent_config.build_job_config("domain__task", "t0-r0")
    assert job.jobs_dir == root / "results/legal_agent_bench/harbor_jobs"
    assert (
        job.datasets[0].path
        == root / "resources_servers/legal_agent_bench/data/runtime/harbor_tasks/legal_agent_bench"
    )
    assert job.datasets[0].task_names == ["domain__task"]
    assert not agent_config.job.datasets[0].task_names
