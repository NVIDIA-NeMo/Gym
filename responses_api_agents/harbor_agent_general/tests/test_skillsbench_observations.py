# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.task import config as harbor_task_config
from harbor.models.task.task import Task
from harbor.verifier.verifier import Verifier
from openhands.sdk import Agent
from openhands.sdk.conversation import LocalConversation
from pydantic import SecretStr

from benchmarks.skillsbench.environment import SkillsBenchSandboxEnvironment
from benchmarks.skillsbench.observations import (
    VERIFIER_EXECUTION_FILE,
    agent_observations,
    verifier_command,
    verifier_exit_code,
)
from benchmarks.skillsbench.openhands_agent import SkillsBenchAgent, SkillsBenchLLM
from nemo_gym.config import ModelServerRef
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.harbor_agent.tests.test_environment import _make_environment, _provider
from responses_api_agents.harbor_agent.tests.test_environment import _reset_fake_provider as _reset_fake_provider
from responses_api_agents.harbor_agent_general import app
from responses_api_agents.harbor_agent_general.skillsbench_app import SkillsBenchHarborAgent
from responses_api_agents.harbor_agent_general.tests.test_skillsbench_config import _config


@pytest.mark.asyncio
async def test_legacy_harbor_exec_does_not_claim_modern_verifier_evidence(tmp_path, monkeypatch):
    monkeypatch.delattr(harbor_task_config, "TaskOS")
    assert verifier_command() is None
    env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
    await env.start(force_build=False)
    try:
        result = await env.exec("echo legacy")
        assert result.return_code == 0
        assert not (env.trial_paths.result_path.parent / VERIFIER_EXECUTION_FILE).exists()
        assert verifier_exit_code(env.trial_paths.result_path.parent, SimpleNamespace()) is None
    finally:
        await env.stop(delete=True)


@pytest.mark.asyncio
async def test_real_harbor_verifier_records_host_execution_without_changing_reward(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Create the answer.")
    (task_dir / "task.toml").write_text('version = "1.0"\n[environment]\ndocker_image = "example/task:1.0"\n')
    (task_dir / "tests/test.sh").write_text("#!/bin/bash\necho 0.25 > /logs/verifier/reward.txt\n")
    env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")

    async def download_logs(source_dir, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "reward.txt").write_text("0.25")

    env.download_dir = AsyncMock(side_effect=download_logs)
    await env.start(force_build=False)
    try:
        start = datetime.now(timezone.utc)
        result = await Verifier(task=Task(task_dir=task_dir), trial_paths=env.trial_paths, environment=env).verify()
        end = datetime.now(timezone.utc)
        trial = SimpleNamespace(
            step_results=[],
            verifier_result=result,
            exception_info=None,
            verifier=SimpleNamespace(started_at=start, finished_at=end),
        )
        assert result.rewards == {"reward": 0.25}
        assert verifier_exit_code(env.trial_paths.result_path.parent, trial) == 0
        assert not (env.trial_paths.verifier_dir / VERIFIER_EXECUTION_FILE).exists()
    finally:
        await env.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("transport"), asyncio.CancelledError()])
async def test_failed_verifier_exec_cannot_leave_stale_success(tmp_path, error):
    env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
    await env.start(force_build=False)
    path = env.trial_paths.result_path.parent / VERIFIER_EXECUTION_FILE
    path.write_text('{"return_code": 0}')
    _provider().exec = AsyncMock(side_effect=error)
    try:
        with pytest.raises(type(error)):
            await env.exec(verifier_command())
        assert not path.exists()
    finally:
        await env.stop(delete=True)


@pytest.mark.asyncio
async def test_unknown_verifier_exit_discards_sandbox_without_witness(tmp_path):
    env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
    await env.start(force_build=False)
    path = env.trial_paths.result_path.parent / VERIFIER_EXECUTION_FILE
    path.write_text('{"return_code": 0}')
    _provider().queue_exec_result(SandboxExecResult(return_code=-1, error_type=None, stdout=None, stderr=None))
    with pytest.raises(RuntimeError, match="Verifier command outcome is unknown"):
        await env.exec(verifier_command())
    assert env._sandbox is None
    assert _provider().closed_handles == ["sbx-123"]
    assert not path.exists()


@pytest.mark.asyncio
async def test_agent_command_cannot_create_verifier_witness(tmp_path):
    env = _make_environment(tmp_path, environment_class=SkillsBenchSandboxEnvironment, skill_mode="no-skill")
    await env.start(force_build=False)
    try:
        await env.exec("bash -c " + repr(verifier_command()))
        assert not (env.trial_paths.result_path.parent / VERIFIER_EXECUTION_FILE).exists()
        _provider().queue_exec_result(SandboxExecResult(return_code=125, error_type=None, stdout=None, stderr=None))
        await env.exec(verifier_command())
        assert (
            json.loads((env.trial_paths.result_path.parent / VERIFIER_EXECUTION_FILE).read_text())["return_code"]
            == 125
        )
    finally:
        await env.stop(delete=True)


@pytest.mark.parametrize(
    "defect", ["missing", "stale", "boolean_exit", "unknown_exit", "unfinished", "no_reward", "timeout"]
)
def test_incomplete_verifier_evidence_is_not_certified(tmp_path, defect):
    phase = SimpleNamespace(
        started_at=datetime.fromtimestamp(100, timezone.utc),
        finished_at=datetime.fromtimestamp(200, timezone.utc),
    )
    trial = SimpleNamespace(step_results=[], verifier_result=object(), exception_info=None, verifier=phase)
    witness = dict(command=verifier_command(), return_code=0, started_at=110, finished_at=120)
    if defect == "stale":
        witness.update(started_at=1, finished_at=2)
    elif defect == "boolean_exit":
        witness["return_code"] = True
    elif defect == "unknown_exit":
        witness["return_code"] = -1
    elif defect == "unfinished":
        phase.finished_at = None
    elif defect == "no_reward":
        trial.verifier_result = None
    elif defect == "timeout":
        trial.exception_info = SimpleNamespace(exception_type="AgentTimeoutError")
    if defect != "missing":
        (tmp_path / VERIFIER_EXECUTION_FILE).write_text(json.dumps(witness))
    assert verifier_exit_code(tmp_path, trial) is None


def test_openhands_events_identify_policy_calls_without_claiming_auxiliary_calls(tmp_path):
    (tmp_path / "agent").mkdir()
    events = [
        dict(kind="ActionEvent", llm_response_id="policy-1"),
        dict(kind="ActionEvent", llm_response_id="policy-1"),
        dict(kind="MessageEvent", llm_response_id="policy-2"),
        dict(kind="CondenserEvent", llm_response_id="summary-1"),
        dict(kind="MessageEvent", llm_response_id=None),
    ]
    (tmp_path / "agent/openhands-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    response = SimpleNamespace(
        failure_kind=None, responses_create_params=SimpleNamespace(input=[]), response=SimpleNamespace(output=[])
    )
    result = agent_observations(
        tmp_path, SimpleNamespace(id="trial-1"), response, ModelServerRef(type="responses_api_models", name="policy")
    )
    assert [r.response_id for r in result.records[0].model_calls] == ["policy-1", "policy-2"]
    assert result.records[0].status == "completed"


def test_compaction_tokens_are_accounted_without_creating_policy_turns(tmp_path):
    from nemo_gym.rollouts.rollout_health import evaluate_record
    from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record

    (tmp_path / "agent").mkdir()
    events = [
        {"kind": "MessageEvent", "llm_response_id": "policy-1"},
        {"kind": "Condensation", "llm_response_id": "summary-1"},
        {"kind": "MessageEvent", "llm_response_id": "policy-2"},
    ]
    (tmp_path / "agent/openhands-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    response = SimpleNamespace(
        failure_kind=None, responses_create_params=SimpleNamespace(input=[]), response=SimpleNamespace(output=[])
    )
    ref = ModelServerRef(type="responses_api_models", name="policy")
    observations = agent_observations(tmp_path, SimpleNamespace(id="trial-1"), response, ref)
    calls = [
        {
            "model_call_id": f"call-{index}",
            "model_ref": ref.model_dump(),
            "response_id": event["llm_response_id"],
            "status_code": 200,
            "tokens_in": 10,
            "tokens_out": 5,
            "dialect": "chat_completions",
            "request": {"messages": [{"role": "user", "content": "Task"}]},
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}]
            },
        }
        for index, event in enumerate(events)
    ]
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": observations.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": calls},
        "response": {"usage": {"input_tokens": 30, "output_tokens": 15}},
    }
    trajectory = _build_trajectory_record(row, row)
    assert [turn.model_calls[0].response_id for turn in trajectory.turns] == ["policy-1", "policy-2"]
    assert len(trajectory.invocations[0].model_calls) == 3
    row["ng_trajectory"] = trajectory.model_dump(mode="json")
    health = evaluate_record(row, frozenset())
    assert "rollout_token_count_mismatch" not in health.unobserved
    assert "rollout_token_count_mismatch" not in [finding.check for finding in health.findings]


def test_sdk_registers_wrapped_policy_metrics_in_real_conversation(tmp_path):
    llm = SkillsBenchLLM(model="openai/test", api_key=SecretStr("dummy"))
    agent = SkillsBenchAgent(llm=llm, tools=[])
    original = Agent(llm=llm, tools=[])
    assert agent.prompt_dir == original.prompt_dir
    assert agent._prompt_preset == original._prompt_preset
    conversation = LocalConversation(
        agent=agent,
        workspace=tmp_path,
        persistence_dir=tmp_path / "state",
        profile_store_dir=tmp_path / "profiles",
        visualizer=None,
    )
    try:
        conversation.send_message("hello")
        conversation.agent.llm.metrics.add_token_usage(
            prompt_tokens=123,
            completion_tokens=17,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=4,
            context_window=4096,
            response_id="reply-1",
        )
        usage = conversation.state.stats.get_combined_metrics().get_snapshot().accumulated_token_usage
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (123, 17)
    finally:
        conversation.close()


@pytest.mark.parametrize("exit_code", [None, -1, 0, 1, 124, 125])
def test_host_verifier_evidence_survives_response_and_collection(tmp_path, monkeypatch, exit_code):
    from nemo_gym.orchestration.recovery import grading_artifact
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import (
        SAVED_ANSWER,
        collect_with_score,
        write_job_dir,
    )

    monkeypatch.setattr(SkillsBenchHarborAgent, "model_post_init", lambda self, context: None)
    agent = SkillsBenchHarborAgent.model_construct(config=_config(opensandbox=True))
    body = app.HarborRunRequest(task_name="task", task_index=0, rollout_index=0, responses_create_params={"input": []})
    official = app.HarborVerifyResponse.model_validate(
        body.model_dump(by_alias=True)
        | {
            "reward": 0.25,
            "response": SAVED_ANSWER
            | {
                "output": [
                    {"type": "function_call", "call_id": "tool-1", "name": "terminal", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "tool-1", "output": "done"},
                    *SAVED_ANSWER["output"],
                ],
            },
        }
    )
    monkeypatch.setattr(app.HarborAgent, "success_response", lambda *args: official)
    trial = SimpleNamespace(
        id="trial-1",
        step_results=[],
        verifier_result=object(),
        exception_info=None,
        verifier=SimpleNamespace(
            started_at=datetime.fromtimestamp(100, timezone.utc),
            finished_at=datetime.fromtimestamp(200, timezone.utc),
        ),
    )
    (tmp_path / "result.json").write_text("{}")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent/openhands-events.jsonl").write_text(
        json.dumps({"kind": "ActionEvent", "llm_response_id": "policy-1"}) + "\n"
    )
    monkeypatch.setattr(app.TrialResult, "model_validate_json", lambda _: trial)
    if exit_code is not None:
        (tmp_path / VERIFIER_EXECUTION_FILE).write_text(
            json.dumps(
                {
                    "command": verifier_command(),
                    "return_code": exit_code,
                    "started_at": 110,
                    "finished_at": 120,
                }
            )
        )
    response = agent.success_response(body, tmp_path).model_dump(mode="json", by_alias=True)
    assert response["reward"] == 0.25
    assert response["ng_agent_observations"]["records"][0]["conversation"][1]["output"] == "done"
    job = write_job_dir(tmp_path / "collected", n=1, results=[response], failures=[])
    [collected] = collect_with_score(job, "[]").results
    assert grading_artifact(collected) == (None if exit_code in (None, -1) else "test_execution")
    if exit_code not in (None, -1):
        assert collected["_ng_collected_grading_facts"]["verifier_exit_code"] == exit_code
        assert collected["_ng_collected_grading_binding"]
