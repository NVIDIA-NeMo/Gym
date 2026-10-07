# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import subprocess
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.interactive_agent_types import AgentActivationResponse, InteractionBudget, ResourcesStepRequest
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.swe_together import snapshot_worker
from resources_servers.swe_together.app import Session, SWETResourcesServer
from resources_servers.swe_together.metrics import aggregate
from resources_servers.swe_together.simulator import UserAgent
from resources_servers.swe_together.task import TaskData


def activation(index, **kwargs):
    return AgentActivationResponse(
        activation_id=index,
        response=NeMoGymResponse(
            id="r",
            object="response",
            created_at=1,
            model="test",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        ),
        **kwargs,
    )


def session(tmp_path, decisions):
    task = TaskData(task_id="test", image="image", image_digest="sha256:" + "1" * 64)
    request = ResourcesSeedSessionRequest(
        resources_session_id="s",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "test", "task_id": "test"},
        task_data=task.model_dump(),
    )
    state = Session(request, task, tmp_path, tmp_path, {})
    state.snapshots = SimpleNamespace(capture=AsyncMock(return_value=""))
    state.prompt = "Task"
    llm = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    content="",
                    tool_calls=[{"function": {"name": action, "arguments": json.dumps({"content": message})}}],
                )
                for action, message in decisions
            ]
        )
    )
    state.simulator = UserAgent(llm, original_user_messages=["a", "b"])
    state.simulator_model = SimpleNamespace(calls=[])
    server = SWETResourcesServer.model_construct(
        config=SimpleNamespace(max_resumes=15, user_context_chars=3000, trial_budget_seconds=5400)
    )
    return server, state, llm


@pytest.mark.asyncio
async def test_retries_do_not_duplicate_user_and_fourth_noop_stops(tmp_path):
    server, state, llm = session(tmp_path, [("redirect", "please fix it")] + [("no-op", "")] * 4)

    async def invoke(i):
        request = ResourcesStepRequest(
            resources_session_id="s", episode_id=state.request.episode_id, activation=activation(i)
        )
        return await state.steps.execute(index=i, request=request, operation=lambda: server._step(state, request))

    first, retry = await asyncio.gather(invoke(0), invoke(0))
    assert first == retry and not first.synthetic
    assert llm.call.await_count == 1 and state.simulator._cursor == 1 and len(state.messages) == 1
    assert state.simulator.last_turn_content.startswith("## Turn 1\n")
    for i in range(1, 4):
        result = await invoke(i)
        assert result.continue_episode and result.synthetic
        assert result.responses_create_params.input[0].content == "continue"
    final = await invoke(4)
    assert not final.continue_episode and final.stop_reason == "consecutive_noops"
    assert llm.call.await_count == 5


@pytest.mark.asyncio
async def test_timeout_rescue_and_final_resume_do_not_consult_simulator(tmp_path):
    server, state, llm = session(tmp_path, [])
    request = ResourcesStepRequest(
        resources_session_id="s",
        episode_id=state.request.episode_id,
        activation=activation(0, turn_complete=False, stop_reason="timeout"),
    )
    result = await server._step(state, request)
    assert result.synthetic and result.metadata["cap_rescue"]
    assert "interrupted" in result.responses_create_params.input[0].content
    assert not llm.call.called and state.noops == 0
    request.activation = activation(15)
    result = await server._step(state, request)
    assert not result.continue_episode and result.stop_reason == "max_resumes"
    assert not llm.call.called


def init_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    (path / "tracked.txt").write_text("original\n")
    snapshot_worker.git(str(path), "add", "-A")
    snapshot_worker.git(
        str(path), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "initial"
    )


def test_snapshot_preserves_dirty_baseline_untracked_and_agent_commits(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / "tracked.txt").write_text("prepared dirty\n")
    (repo / "preexisting.txt").write_text("prepared untracked\n")
    baseline = snapshot_worker.snapshot(str(repo))
    snapshot_worker.git(str(repo), "update-ref", "refs/nemo-gym/test/baseline", baseline)
    snapshot_worker.git(str(repo), "gc", "--prune=now")
    assert snapshot_worker.git(str(repo), "cat-file", "-t", baseline).strip() == "tree"
    snapshot_worker.git(str(repo), "log", "--all", "--oneline")
    assert snapshot_worker.git(str(repo), "status", "--porcelain").strip() == "M tracked.txt\n?? preexisting.txt"
    (repo / "tracked.txt").write_text("candidate\n")
    (repo / "new.txt").write_text("new file\n")
    snapshot_worker.git(str(repo), "add", "-A")
    snapshot_worker.git(
        str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "candidate"
    )
    (repo / "new.txt").write_text("new file\nsecond turn\n")
    final = snapshot_worker.snapshot(str(repo))
    patch = snapshot_worker.git(str(repo), "diff", baseline, final)
    assert "-prepared dirty" in patch and "+candidate" in patch and "+second turn" in patch
    assert "preexisting.txt" not in patch
    assert snapshot_worker.git(str(repo), "status", "--porcelain").strip() == "M new.txt"


def test_failure_aware_repeat_aggregation():
    result = aggregate(
        [
            {"task_id": "a", "judge_score": 1.0, "user_correction": 0.0},
            {"task_id": "a", "judge_score": 0.8},
            {"task_id": "b", "judge_score": None, "mask_sample": True},
        ],
        planned=4,
    )
    assert result["graded"] == 2 and result["masked"] == 1 and result["missing"] == 1
    assert result["MeanJudge"] == 0.9 and result["pass@1"] == 0.5
    assert result["stable_solve_rate"] == 1 and result["pass2"] == 0
    assert result["upstream_compatibility_mean_zero_filled"] == 0.45


@pytest.mark.asyncio
async def test_declared_session_budget_stops_before_simulator(tmp_path):
    server, state, llm = session(tmp_path, [])
    request = ResourcesStepRequest(
        resources_session_id="s",
        episode_id=state.request.episode_id,
        activation=activation(0, turn_complete=False, stop_reason="session_budget_exhausted"),
    )
    result = await server._step(state, request)
    assert not result.continue_episode and not llm.call.called
    assert state.snapshots.capture.await_count == 1


@pytest.mark.asyncio
async def test_permission_denied_consults_simulator_as_completion_and_resumes(tmp_path):
    server, state, llm = session(tmp_path, [("redirect", "Use the repository copy instead.")])
    request = ResourcesStepRequest(
        resources_session_id="s",
        episode_id=state.request.episode_id,
        activation=activation(0, turn_complete=False, stop_reason="permission_denied"),
    )
    result = await server._step(state, request)
    assert result.continue_episode and not result.synthetic
    assert result.responses_create_params.input[0].content == "Use the repository copy instead."
    assert "The agent is signaling completion." in llm.call.call_args.kwargs["prompt"]
    assert state.simulator._cursor == 1 and len(state.messages) == 1 and state.noops == 0


@pytest.mark.asyncio
async def test_close_retains_late_creation_until_owned_handle_is_stopped(tmp_path):
    server, state, _ = session(tmp_path, [])
    server.config.close_timeout = 0.01
    server._sessions["s"] = state
    ready = asyncio.Event()
    sandbox = SimpleNamespace(serialize=AsyncMock(return_value={"id": "late"}), stop=AsyncMock())
    state.sandbox = sandbox
    state.creations["candidate"] = asyncio.create_task(ready.wait())
    body = ResourcesCloseSessionRequest(resources_session_id="s", episode_id=state.request.episode_id)
    with pytest.raises(TimeoutError):
        await server.close_resources_session(None, body)
    assert "s" in server._sessions and not state.creations["candidate"].cancelled()
    assert not sandbox.stop.called
    assert not json.loads((tmp_path / "candidate-sandbox-close.json").read_text())["cleanup_confirmed"]
    ready.set()
    receipt = await server.close_resources_session(None, body)
    assert receipt.resources_session_id == "s" and "s" not in server._sessions
    sandbox.stop.assert_awaited_once()
    assert json.loads((tmp_path / "candidate-sandbox-close.json").read_text())["cleanup_confirmed"]


@pytest.mark.asyncio
async def test_simulator_failure_retains_exact_inputs_without_noop(tmp_path):
    server, state, llm = session(tmp_path, [])
    llm.call.side_effect = RuntimeError("transport unavailable")
    state.simulator_model = SimpleNamespace(calls=[{"error": "transport unavailable"}])
    request = ResourcesStepRequest(
        resources_session_id="s", episode_id=state.request.episode_id, activation=activation(0)
    )
    with pytest.raises(RuntimeError, match="User simulator request failed"):
        await server._step(state, request)
    evidence = json.loads((tmp_path / "turn-0-simulator.json").read_text())
    assert evidence["simulator_messages"][-1]["content"] == llm.call.call_args.kwargs["prompt"]
    assert state.simulator.last_messages_sent == []
    assert evidence["error"] == "User simulator request failed"
    assert state.noops == 0 and state.messages == []
    assert json.loads((tmp_path / "simulator-model-calls.json").read_text())[0]["error"] == "transport unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{}, {"disallowed_tools": "WebFetch,WebSearch"}])
async def test_prepare_forwards_canonical_agent_kwargs_without_harness_translation(tmp_path, monkeypatch, kwargs):
    from resources_servers.swe_together import app

    server, state, _ = session(tmp_path, [])
    server.config = SimpleNamespace(
        sandbox_provider="sandbox",
        python_runtime_url=None,
        python_runtime_sha256=None,
        simulator_model=SimpleNamespace(name="simulator"),
        simulator_temperature=0.5,
        protocol_profile="smoke",
        network_qualification={},
        scoring_profile="judge_all",
        judge_agent=SimpleNamespace(name="fixed_judge"),
    )
    server.server_client = SimpleNamespace()
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="prepared", stderr="")),
        serialize=AsyncMock(return_value={"sandbox_id": "fixture"}),
    )
    monkeypatch.setattr(SWETResourcesServer, "_new_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(
        app, "load_task", lambda *args: {"agent_kwargs": kwargs, "record": {"history_policy": "upstream"}}
    )
    monkeypatch.setattr(app, "ensure_python", AsyncMock(return_value="python3"))
    monkeypatch.setattr(app, "discover_repo_config_files", AsyncMock(return_value=""))
    monkeypatch.setattr(app, "session_analysis", lambda path: ("", []))
    monkeypatch.setattr(app, "RepositorySnapshots", lambda *args, **kw: state.snapshots)
    (tmp_path / "instruction.md").write_text("Public task instruction")

    result = await server._prepare(state)
    if kwargs:
        assert result.runtime_policy.format == "harbor.agent-kwargs.v1"
        assert result.runtime_policy.settings == kwargs
        assert "permission" not in result.runtime_policy.settings
    else:
        assert result.runtime_policy is None
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["agent_kwargs"] == kwargs
    assert result.sandbox_access.connection.descriptor == {"sandbox_id": "fixture"}
    assert result.supports_interaction_budget


def budget_request(state, *, seconds=0.01, index=0):
    now = time()
    return ResourcesStepRequest(
        resources_session_id="s",
        episode_id=state.request.episode_id,
        activation=activation(index),
        interaction_budget=InteractionBudget(started_at_unix_seconds=now, deadline_unix_seconds=now + seconds),
    )


@pytest.mark.asyncio
async def test_budget_expiry_cancels_simulator_once_and_replays_terminal_step(tmp_path):
    server, state, llm = session(tmp_path, [])
    canceled = asyncio.Event()

    async def wait_for_cancellation(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    llm.call.side_effect = wait_for_cancellation
    request = budget_request(state)

    async def invoke():
        return await state.steps.execute(index=0, request=request, operation=lambda: server._step(state, request))

    first, duplicate = await asyncio.gather(invoke(), invoke())
    budget_path = tmp_path / "interaction-budget.json"
    budget_mtime = budget_path.stat().st_mtime_ns
    assert first == duplicate == await invoke()
    assert budget_path.stat().st_mtime_ns == budget_mtime
    assert json.loads(budget_path.read_text()) == request.interaction_budget.model_dump(mode="json")
    assert first.stop_reason == "session_budget_exhausted" and not first.continue_episode
    assert canceled.is_set() and llm.call.await_count == 1
    assert state.stopped and state.messages == [] and state.noops == 0
    assert state.simulator._cursor == 0 and state.simulator._messages == []
    assert state.snapshots.capture.await_count == 1
    evidence = json.loads((tmp_path / "turn-0-simulator.json").read_text())
    assert evidence["discarded"] and evidence["simulator_messages"]
    assert evidence["discarded_at_unix_seconds"] >= request.interaction_budget.started_at_unix_seconds
    assert "accepted_at_unix_seconds" not in evidence


@pytest.mark.asyncio
async def test_late_simulator_reply_cannot_commit_after_cancel_is_suppressed(tmp_path):
    server, state, llm = session(tmp_path, [])

    async def delayed_reply(**kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            return SimpleNamespace(
                content="",
                tool_calls=[{"function": {"name": "redirect", "arguments": '{"content":"late instruction"}'}}],
            )

    llm.call.side_effect = delayed_reply
    response = await server._step(state, budget_request(state))
    assert response.stop_reason == "session_budget_exhausted"
    assert state.messages == [] and state.noops == 0
    assert state.simulator._cursor == 0 and state.simulator._messages == []
    assert state.simulator.message_count == 0 and state.simulator._counts == {}


@pytest.mark.asyncio
async def test_post_await_deadline_check_discards_reply_before_timer_callback(tmp_path, monkeypatch):
    from resources_servers.swe_together import app

    server, state, llm = session(tmp_path, [])
    request = budget_request(state, seconds=60)
    clock = [request.interaction_budget.started_at_unix_seconds]
    monkeypatch.setattr(app, "time", lambda: clock[0])

    async def reply_after_clock_advance(**kwargs):
        clock[0] = request.interaction_budget.deadline_unix_seconds + 1
        return SimpleNamespace(
            content="", tool_calls=[{"function": {"name": "redirect", "arguments": '{"content":"too late"}'}}]
        )

    llm.call.side_effect = reply_after_clock_advance
    response = await server._step(state, request)
    assert response.stop_reason == "session_budget_exhausted"
    assert state.messages == [] and state.simulator._messages == [] and state.simulator._cursor == 0


@pytest.mark.asyncio
async def test_model_timeout_before_interaction_deadline_remains_failure(tmp_path):
    server, state, llm = session(tmp_path, [])
    llm.call.side_effect = TimeoutError("model request timed out")
    with pytest.raises(RuntimeError, match="User simulator request failed"):
        await server._step(state, budget_request(state, seconds=60))
    assert not state.stopped and state.messages == [] and state.noops == 0
    assert "error" in json.loads((tmp_path / "turn-0-simulator.json").read_text())


@pytest.mark.asyncio
async def test_wrapper_budget_starts_at_execution_and_is_not_reset_by_steps(tmp_path, monkeypatch):
    from resources_servers.swe_together import app

    server, state, llm = session(tmp_path, [("redirect", "continue implementation")])
    server.config.trial_budget_seconds = 20
    clock = [5000.0]
    monkeypatch.setattr(app, "monotonic", lambda: clock[0])
    request = ResourcesStepRequest(
        resources_session_id="s",
        episode_id=state.request.episode_id,
        activation=activation(0, observation={"elapsed_seconds": 10}),
    )
    first = await server._step(state, request)
    assert first.continue_episode and state.started_at == 4990
    clock[0] = 5011
    request.activation = activation(1)
    final = await server._step(state, request)
    assert final.stop_reason == "session_budget_exhausted" and llm.call.await_count == 1


@pytest.mark.asyncio
async def test_resources_rejects_interaction_budget_change(tmp_path):
    from fastapi import HTTPException

    from nemo_gym.server_utils import SESSION_ID_KEY

    server, state, llm = session(tmp_path, [("redirect", "continue implementation")] * 2)
    server._sessions["s"] = state
    http_request = SimpleNamespace(session={SESSION_ID_KEY: "s"})
    request = budget_request(state, seconds=60)
    assert (await server.step(http_request, request)).continue_episode
    budget_path = tmp_path / "interaction-budget.json"
    budget_text = budget_path.read_text()
    budget_mtime = budget_path.stat().st_mtime_ns
    evidence = json.loads((tmp_path / "turn-0-simulator.json").read_text())
    assert evidence["accepted_at_unix_seconds"] < request.interaction_budget.deadline_unix_seconds
    changed = budget_request(state, seconds=120, index=1)
    with pytest.raises(HTTPException, match="Interaction budget changed"):
        await server.step(http_request, changed)
    assert llm.call.await_count == 1 and len(state.messages) == 1
    request.activation = activation(1)
    assert (await server.step(http_request, request)).continue_episode
    assert llm.call.await_count == 2 and len(state.messages) == 2
    assert budget_path.read_text() == budget_text and budget_path.stat().st_mtime_ns == budget_mtime
