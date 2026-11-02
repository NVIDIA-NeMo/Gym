# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from itertools import count
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_runtime import ResourcesCloseSessionRequest
from resources_servers.swe_atlas_qa import app as module


TASK = "task-" + "a" * 24
OTHER_TASK = "task-" + "b" * 24
PRIVATE_FILES = ("evaluate_answer.py", "rubrics.json", "prompt.txt", "system_prompt.txt", "user_prompt_template.txt")


def _request(session_id: str = "session") -> Request:
    headers = [(b"cookie", f"session={session_id}".encode())]
    return Request({"type": "http", "headers": headers, "session": {SESSION_ID_KEY: session_id}})


@pytest.fixture
def body() -> module.SWEAtlasQAVerifyRequest:
    response = dict(
        id="test",
        created_at=0,
        model="test",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return module.SWEAtlasQAVerifyRequest.model_validate(
        {"task_id": TASK, "responses_create_params": {"input": []}, "response": response}
    )


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> module.SWEAtlasQAResourcesServer:
    for task_id in (TASK, OTHER_TASK):
        task_dir = tmp_path / task_id
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            '[environment]\ndocker_image = "registry.invalid/task@sha256:' + "a" * 64 + '"\n'
            "cpus = 2\nmemory_mb = 4096\nstorage_mb = 1025\n[verifier]\ntimeout_sec = 90\n"
        )
        for name in PRIVATE_FILES:
            (task_dir / "tests" / name).write_text("private-placeholder")
    provider = MagicMock()
    ids = count()
    provider.create = AsyncMock(side_effect=lambda spec: SandboxHandle(f"box-{next(ids)}", "test", None))
    provider.serialize_handle = AsyncMock(side_effect=lambda h: {"sandbox_id": h.sandbox_id})
    provider.connect = AsyncMock()
    provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    provider.upload_file = AsyncMock()
    provider.close = AsyncMock()
    provider.aclose = AsyncMock()
    provider.answers = {}
    provider.reward_artifact = None
    provider.results = dict(reward=1.0, agg_score=0.75, num_rubrics=4, num_scored=4, num_unscored=0, num_passed=3)

    async def download(handle: SandboxHandle, source: str, target: Path) -> None:
        if source.endswith("answer.txt"):
            target.write_text(provider.answers.get(handle.sandbox_id, "answer-placeholder"))
        elif source.endswith("evaluation_results.json"):
            target.write_text(json.dumps(provider.results))
        elif source.endswith("reward.txt"):
            reward = provider.results["reward"] if provider.reward_artifact is None else provider.reward_artifact
            target.write_text(str(reward))
        else:
            raise AssertionError(source)

    provider.download_file = AsyncMock(side_effect=download)
    monkeypatch.setattr(module, "create_provider", lambda config: provider)
    monkeypatch.setattr(module, "validate_source", MagicMock(return_value=None))
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {
                "sandbox": {"test": {}},
                "policy": {
                    "responses_api_models": {
                        "test": {"host": "model.internal", "port": 8000, "base_url": "https://policy.invalid/v1"}
                    }
                },
            }
        ),
    )
    config = module.SWEAtlasQAResourcesServerConfig(
        host="127.0.0.1",
        port=8001,
        entrypoint="",
        name="swe_atlas_qa",
        tasks_dir=tmp_path,
        sandbox_provider="sandbox",
        model_server={"type": "responses_api_models", "name": "policy"},
        judge_base_url="http://judge.internal/v1",
        judge_model="judge-placeholder",
        judge_api_key="dummy-judge-secret",
    )
    return module.SWEAtlasQAResourcesServer(config=config, server_client=client)


def test_startup_validates_source_checkout(server) -> None:
    module.validate_source.assert_called_once_with(server.config.tasks_dir.parent.parent)


async def test_seed_uses_server_task_and_exposes_only_agent_access(server, body) -> None:
    untrusted = module.SWEAtlasQARequest.model_validate(
        body.model_dump()
        | {
            "metadata": {"docker_image": "untrusted-image", "workdir": "/tests"},
            "verifier_metadata": {"task_id": OTHER_TASK},
        }
    )
    seed = await server.seed_session(_request(), untrusted)
    provider = server._provider
    spec = provider.create.await_args.args[0]
    assert spec.image == "registry.invalid/task@sha256:" + "a" * 64
    assert spec.workdir == "/app"
    assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (2, 4096, 2)
    assert spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "model.internal"}],
    }
    assert spec.env == spec.files == {}
    assert seed.sandbox_descriptor == {"sandbox_id": "box-0"}
    exposed = repr((spec, seed.model_dump(), provider.exec.await_args_list))
    assert all(value not in exposed for value in (*PRIVATE_FILES, "dummy-judge-secret", "judge.internal"))
    provider.upload_file.assert_not_awaited()


@pytest.mark.parametrize(
    ("reward", "scored", "passed", "agg"),
    [(1.0, 4, 3, 0.75), (1.0, 3, 3, 1.0), (0.0, 3, 1, 1 / 3), (0.0, 1, 0, 0.0)],
)
async def test_verify_preserves_upstream_scores_and_stages_only_after_agent_close(
    server, body, reward, scored, passed, agg
):
    provider = server._provider
    provider.results.update(
        reward=reward, num_scored=scored, num_unscored=4 - scored, num_passed=passed, agg_score=agg
    )
    await server.seed_session(_request(), body)
    agent = server._sessions["session"][1]
    result = await server.verify(_request(), body)
    assert (result.reward, result.agg_score, result.num_rubrics, result.num_scored, result.num_passed) == (
        reward,
        agg,
        4,
        scored,
        passed,
    )
    assert result.num_unscored == 4 - scored
    assert result.all_rubrics_scored is (scored == 4)
    assert result.evaluation_completed and not result.mask_sample
    assert result.answer == "answer-placeholder"
    verifier = provider.close.await_args.args[0]
    assert verifier != agent
    assert provider.create.await_args_list[1].args[0].image == server.config.verifier_image
    events = provider.mock_calls
    collected = next(c for c in events if c[0] == "download_file" and c.args[0] == agent)
    staged = next(c for c in events if c[0] == "upload_file")
    assert events.index(collected) < events.index(call.close(agent)) < events.index(staged)
    creations = [c for c in events if c[0] == "create"]
    assert events.index(call.close(agent)) < events.index(creations[1])
    assert {c.args[2] for c in provider.upload_file.await_args_list} == {
        *(f"/tests/{name}" for name in PRIVATE_FILES),
        "/logs/agent/answer.txt",
    }
    assert all(c.args[0] == verifier for c in provider.upload_file.await_args_list)
    assert all(not c.kwargs.get("env") for c in provider.exec.await_args_list if c.args[0] == agent)
    judge_call = provider.exec.await_args
    assert judge_call.args == (verifier, "python /tests/evaluate_answer.py")
    assert judge_call.kwargs["env"] == {
        "EVAL_BASE_URL": "http://judge.internal/v1",
        "EVAL_MODEL": "judge-placeholder",
        "EVAL_API_KEY": "dummy-judge-secret",
    }
    assert provider.close.await_args_list == [call(agent), call(verifier)]
    assert not server._sessions


async def test_verify_masks_zero_scored_rubrics(server, body) -> None:
    provider = server._provider
    provider.results.update(reward=0.0, agg_score=0.0, num_scored=0, num_unscored=4, num_passed=0)
    await server.seed_session(_request(), body)
    result = await server.verify(_request(), body)
    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    assert result.failure_reason
    assert provider.close.await_count == 2
    assert not server._sessions and not server._sandboxes


@pytest.mark.parametrize("answer", [None, "", " \n", "<<FINAL_ANSWER>>", "analysis<<FINAL_ANSWER>>   "])
async def test_missing_or_empty_answer_scores_zero_without_judge(server, body, answer) -> None:
    provider = server._provider
    await server.seed_session(_request(), body)
    agent = server._sessions["session"][1]
    if answer is None:
        provider.exec.return_value = SandboxExecResult("", "", 1)
    else:
        provider.answers[agent.sandbox_id] = answer
    result = await server.verify(_request(), body)
    assert result.reward == 0 and result.evaluation_completed and not result.mask_sample
    assert provider.create.await_count == 1
    provider.upload_file.assert_not_awaited()
    provider.close.assert_awaited_once_with(agent)
    if answer is None:
        provider.download_file.assert_not_awaited()


@pytest.mark.parametrize(
    ("return_code", "error_type"),
    [(125, "sandbox"), (124, "timeout"), (127, None), (0, "sandbox"), (1, "sandbox")],
)
async def test_answer_probe_failure_masks_sample(server, body, return_code, error_type) -> None:
    provider = server._provider
    await server.seed_session(_request(), body)
    agent = server._sessions["session"][1]
    provider.exec.return_value = SandboxExecResult(None, "runtime failed", return_code, error_type=error_type)
    result = await server.verify(_request(), body)
    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    assert provider.create.await_count == 1
    provider.download_file.assert_not_awaited()
    provider.upload_file.assert_not_awaited()
    provider.close.assert_awaited_once_with(agent)
    assert not server._sessions


async def test_task_mismatch_closes_without_collecting_or_grading(server, body) -> None:
    await server.seed_session(_request(), body)
    agent = server._sessions["session"][1]
    result = await server.verify(_request(), body.model_copy(update={"task_id": OTHER_TASK}))
    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    server._provider.close.assert_awaited_once_with(agent)
    server._provider.exec.assert_not_awaited()
    server._provider.upload_file.assert_not_awaited()
    assert not server._sessions


async def test_grader_failure_masks_sample_and_closes_both_sandboxes(server, body) -> None:
    await server.seed_session(_request(), body)
    server._provider.exec.side_effect = [
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
        RuntimeError("judge unavailable"),
    ]
    result = await server.verify(_request(), body)
    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    assert result.failure_reason
    assert server._provider.close.await_count == 2
    assert not server._sessions


@pytest.mark.parametrize(
    ("changes", "reward_artifact", "reason"),
    [
        ({}, 0.0, "inconsistent rewards"),
        ({"reward": 0.5}, None, "inconsistent rewards"),
        ({"num_rubrics": 0}, None, "inconsistent rubric counts"),
        ({"num_scored": 5}, None, "inconsistent rubric counts"),
        ({"num_passed": 5}, None, "inconsistent rubric counts"),
        ({"num_unscored": 1}, None, "inconsistent rubric counts"),
        ({"agg_score": float("nan")}, None, "inconsistent rubric counts"),
    ],
)
async def test_grade_rejects_invalid_artifacts_and_closes_verifier(server, tmp_path, changes, reward_artifact, reason):
    provider = server._provider
    provider.results.update(changes)
    provider.reward_artifact = reward_artifact
    answer = tmp_path / "answer.txt"
    answer.write_text("answer-placeholder")
    with pytest.raises(ValueError, match=reason):
        await server._grade(TASK, answer, tmp_path)
    assert provider.download_file.await_count == 2
    provider.close.assert_awaited_once_with(provider.exec.await_args_list[0].args[0])


@pytest.mark.parametrize(("exit_codes", "reason"), [([1], "dependencies"), ([0, 1], "evaluator failed")])
async def test_grade_command_failure_closes_verifier(server, tmp_path, exit_codes, reason) -> None:
    provider = server._provider
    provider.exec.side_effect = [SandboxExecResult("", "", code) for code in exit_codes]
    answer = tmp_path / "answer.txt"
    answer.write_text("answer-placeholder")
    with pytest.raises(RuntimeError, match=reason):
        await server._grade(TASK, answer, tmp_path)
    assert provider.exec.await_count == len(exit_codes)
    provider.download_file.assert_not_awaited()
    provider.close.assert_awaited_once_with(provider.exec.await_args_list[0].args[0])
    if len(exit_codes) == 1:
        provider.upload_file.assert_not_awaited()


async def test_lifespan_shutdown_closes_live_sessions_and_provider(server, body) -> None:
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        await asyncio.gather(*(server.seed_session(_request(s), body) for s in ("one", "two")))
        handles = [handle for _, handle in server._sessions.values()]
        server._provider.close.assert_not_awaited()
        server._provider.aclose.assert_not_awaited()
    assert server._provider.close.await_args_list == [call(handle) for handle in handles]
    assert not server._sessions and not server._sandboxes
    server._provider.aclose.assert_awaited_once()


async def test_lifespan_shutdown_waits_for_all_deletions_after_failure(server, body) -> None:
    app = server.setup_webserver()
    provider = server._provider
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    provider_closed_after_deletions = []
    provider.aclose.side_effect = lambda: provider_closed_after_deletions.append(finished.is_set())

    async def close(handle):
        if handle == failed:
            raise RuntimeError("close unavailable")
        entered.set()
        await release.wait()
        finished.set()

    async def shutdown():
        async with app.router.lifespan_context(app):
            pass

    await asyncio.gather(*(server.seed_session(_request(s), body) for s in ("one", "two")))
    failed = server._sessions["one"][1]
    provider.close.side_effect = close
    async with asyncio.timeout(2):
        pending = asyncio.create_task(shutdown())
        await entered.wait()
        try:
            done, _ = await asyncio.wait({pending}, timeout=0.01)
            assert not done
            provider.aclose.assert_not_awaited()
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="close unavailable"):
                await pending
            await finished.wait()
    assert provider.close.await_count == 2
    provider.aclose.assert_awaited_once()
    assert provider_closed_after_deletions == [True]


async def test_parallel_cookie_sessions_keep_answers_and_cleanup_isolated(server, body) -> None:
    provider = server._provider
    requests = [_request("one"), _request("two")]
    bodies = [body, body.model_copy(update={"task_id": OTHER_TASK})]
    await asyncio.gather(*(server.seed_session(r, b) for r, b in zip(requests, bodies)))
    agents = [server._sessions[s][1] for s in ("one", "two")]
    provider.answers.update({h.sandbox_id: s for h, s in zip(agents, ("first-placeholder", "second-placeholder"))})
    uploads = []
    task_files = {}

    async def upload(handle: SandboxHandle, source: Path, target: str) -> None:
        await asyncio.sleep(0)
        if target.endswith("answer.txt"):
            uploads.append((handle.sandbox_id, source.read_text()))
        elif target.endswith("rubrics.json"):
            task_files[handle.sandbox_id] = source.parent.parent.name

    provider.upload_file.side_effect = upload
    results = await asyncio.gather(*(server.verify(r, b) for r, b in zip(requests, bodies)))
    assert [r.task_id for r in results] == [TASK, OTHER_TASK]
    assert all(r.evaluation_completed and not r.mask_sample for r in results)
    assert {(task_files[h], text) for h, text in uploads} == {
        (TASK, "first-placeholder"),
        (OTHER_TASK, "second-placeholder"),
    }
    assert len({handle for handle, _ in uploads}) == 2
    assert len({c.args[0].sandbox_id for c in provider.close.await_args_list}) == 4
    assert not server._sessions


@pytest.mark.parametrize(("seeded", "cookie"), [(False, "session"), (True, "session"), (True, "fresh")])
async def test_explicit_close_is_idempotent_and_blocks_late_seed(server, body, seeded, cookie) -> None:
    if seeded:
        await server.seed_session(_request(), body)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id={"rollout_id": "test"})
    for _ in range(2):
        assert (await server.close_resources_session(_request(cookie), close)).resources_session_id == "session"
    with pytest.raises(ValueError, match="closed"):
        await server.seed_session(_request(), body)
    assert server._provider.create.await_count == int(seeded)
    assert server._provider.close.await_count == int(seeded)
    assert not server._sessions and not server._sandboxes


async def test_close_rejects_conflicting_active_cookie(server, body) -> None:
    await asyncio.gather(*(server.seed_session(_request(s), body) for s in ("session", "other")))
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id={"rollout_id": "test"})
    with pytest.raises(ValueError, match="cookie conflicts"):
        await server.close_resources_session(_request("other"), close)
    assert set(server._sessions) == {"session", "other"}
    server._provider.close.assert_not_awaited()


@pytest.mark.parametrize("follower", ["close", "duplicate", "conflict"])
async def test_seed_lock_serializes_close_and_concurrent_seeds(server, body, follower) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def serialize(handle):
        entered.set()
        await release.wait()
        return {"sandbox_id": handle.sandbox_id}

    server._provider.serialize_handle.side_effect = serialize
    async with asyncio.timeout(2):
        seed = asyncio.create_task(server.seed_session(_request(), body))
        await entered.wait()
        if follower == "close":
            close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id={"rollout_id": "test"})
            action = server.close_resources_session(_request(), close)
        else:
            task = body if follower == "duplicate" else body.model_copy(update={"task_id": OTHER_TASK})
            action = server.seed_session(_request(), task)
        pending = asyncio.create_task(action)
        try:
            await asyncio.sleep(0)
            assert not seed.done() and not pending.done()
        finally:
            release.set()
        first, second = await asyncio.gather(seed, pending, return_exceptions=True)
    server._provider.create.assert_awaited_once()
    assert first.sandbox_descriptor == {"sandbox_id": "box-0"}
    if follower == "close":
        assert second.resources_session_id == "session"
        server._provider.close.assert_awaited_once()
        assert not server._sessions and not server._sandboxes
    else:
        if follower == "duplicate":
            assert second.sandbox_descriptor == first.sandbox_descriptor
        else:
            assert isinstance(second, ValueError) and "different task" in str(second)
        assert server._sessions["session"][0] == TASK
        assert len(server._sandboxes) == 1
        server._provider.close.assert_not_awaited()


@pytest.mark.parametrize("operation", ["verify", "explicit", "failed_cleanup"])
async def test_cancellation_during_close_waits_and_preserves_cleanup_state(server, body, operation) -> None:
    await server.seed_session(_request(), body)
    handle = server._sessions["session"][1]
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close(handle: SandboxHandle) -> None:
        entered.set()
        await release.wait()
        finished.set()
        if operation == "failed_cleanup":
            raise RuntimeError("close unavailable")

    server._provider.close.side_effect = close
    async with asyncio.timeout(2):
        if operation == "explicit":
            request = ResourcesCloseSessionRequest(resources_session_id="session", episode_id={"rollout_id": "test"})
            action = server.close_resources_session(_request(), request)
        elif operation == "failed_cleanup":
            action = server._close(handle)
        else:
            action = server.verify(_request(), body)
        pending = asyncio.create_task(action)
        await entered.wait()
        pending.cancel()
        try:
            await asyncio.sleep(0)
            assert not pending.done() and not finished.is_set()
            assert server._sandboxes == {handle.sandbox_id: handle}
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert finished.is_set()
    server._provider.close.assert_awaited_once_with(handle)
    if operation == "failed_cleanup":
        assert server._sandboxes == {handle.sandbox_id: handle}
    else:
        assert not server._sessions and not server._sandboxes
        assert "session" in server._closed_sessions
        with pytest.raises(ValueError, match="closed"):
            await server.seed_session(_request(), body)


async def test_failed_close_retains_handle_for_retry(server, body) -> None:
    await server.seed_session(_request(), body)
    handle = server._sessions["session"][1]
    server._provider.close.side_effect = [RuntimeError("close unavailable"), None]
    with pytest.raises(RuntimeError, match="close unavailable"):
        await server._close(handle)
    assert server._sandboxes == {handle.sandbox_id: handle}
    await server._close(handle)
    assert not server._sandboxes
    assert server._provider.close.await_args_list == [call(handle), call(handle)]


async def test_verify_delete_failure_allows_explicit_close_retry(server, body) -> None:
    await server.seed_session(_request(), body)
    handle = server._sessions["session"][1]
    provider = server._provider
    provider.close.side_effect = [RuntimeError("close unavailable"), None]
    result = await server.verify(_request(), body)
    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    calls = list(provider.mock_calls)
    duplicate = await server.verify(_request(), body)
    assert duplicate.mask_sample and not duplicate.evaluation_completed
    assert provider.mock_calls == calls
    assert provider.create.await_count == 1
    provider.upload_file.assert_not_awaited()
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id={"rollout_id": "test"})
    for _ in range(2):
        assert (await server.close_resources_session(_request(), close)).resources_session_id == "session"
    assert provider.close.await_args_list == [call(handle), call(handle)]
    assert not server._sessions and not server._sandboxes


@pytest.mark.parametrize("phase", ["seed", "collect", "judge"])
async def test_cancellation_propagates_and_closes_owned_sandboxes(server, body, phase) -> None:
    provider = server._provider
    if phase == "seed":
        provider.serialize_handle.side_effect = asyncio.CancelledError
        action = server.seed_session(_request(), body)
    else:
        await server.seed_session(_request(), body)
        provider.exec.side_effect = ([SandboxExecResult("", "", 0)] * 2 if phase == "judge" else []) + [
            asyncio.CancelledError()
        ]
        action = server.verify(_request(), body)
    with pytest.raises(asyncio.CancelledError):
        await action
    assert provider.close.await_count == (2 if phase == "judge" else 1)
    assert not server._sessions and not server._sandboxes
