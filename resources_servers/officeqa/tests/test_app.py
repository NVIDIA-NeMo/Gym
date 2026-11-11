# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted scoring and ownership through Gym's existing resource-session contract."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError
from starlette.requests import Request
from starlette.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import (
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    AggregateMetricsRequest,
    BaseServerConfig,
    ModelServerRef,
)
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxSpec
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.server_runtime.base_server import refuse_unsupported_num_workers
from resources_servers.officeqa import app as officeqa_app
from resources_servers.officeqa.app import (
    VERIFIER_FIXTURE,
    OfficeQAResourcesServer,
    OfficeQAResourcesServerConfig,
    OfficeQAVerifyRequest,
)
from resources_servers.officeqa.task_data import TaskData
from resources_servers.officeqa.verifiers.officeqa_pro_task import OfficeQAProTask


@pytest.fixture
def server(tmp_path):
    config = OfficeQAResourcesServerConfig(
        host="127.0.0.1", port=9000, name="officeqa", entrypoint="app.py", workspace_root=str(tmp_path)
    )
    task = OfficeQAProTask(
        task_id="officeqa_pro_" + "0" * 24,
        workspace_id="workspace_" + "0" * 24,
        uid="synthetic",
        question="Synthetic scorer case.",
        expected_answer="42",
        source_revision="0" * 40,
    )
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / (task.task_id + ".json")).write_text(task.model_dump_json())
    return OfficeQAResourcesServer.model_construct(
        config=config,
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=OmegaConf.create({})
        ),
    )


def run_body():
    return {
        "task_id": "officeqa_pro_" + "0" * 24,
        "workspace_id": "workspace_" + "0" * 24,
        "prompt_version": "synthetic-v1",
        "uid": "synthetic",
        "question": "Synthetic scorer case.",
        "source_revision": "0" * 40,
        "responses_create_params": {"input": []},
    }


def verify_body():
    case = json.loads(VERIFIER_FIXTURE.cases_path.read_text().splitlines()[0])["request"]
    case.pop("expected_answer")
    return OfficeQAVerifyRequest.model_validate(case)


@pytest.mark.parametrize(
    "text,reward,final_answer,has_final_answer",
    [
        ("<think>Use <FINAL_ANSWER> tags.</think><FINAL_ANSWER>42</FINAL_ANSWER>", 1.0, "42", True),
        ("<think>First line.\nSecond line.</think><FINAL_ANSWER>42</FINAL_ANSWER>", 1.0, "42", True),
        # An answer without the tag gets the reward but counts toward no_answer.
        ("<think>First.</think>\n<think>Second.</think>\n42", 1.0, "42", False),
        ("<think><FINAL_ANSWER>42</FINAL_ANSWER></think><FINAL_ANSWER>41</FINAL_ANSWER>", 0.0, "41", True),
        ("<think><FINAL_ANSWER>42</FINAL_ANSWER></think>", 0.0, "", False),
        ("<FINAL_ANSWER>42</FINAL_ANSWER>", 1.0, "42", True),
    ],
)
@pytest.mark.asyncio
async def test_verify_excludes_inline_reasoning_and_preserves_raw_response(
    server: OfficeQAResourcesServer, text: str, reward: float, final_answer: str, has_final_answer: bool
) -> None:
    body = verify_body()
    body.response.output[0].content[0].text = text

    result = await server.verify(http_request(), body)

    assert result.reward == reward
    assert result.final_answer == final_answer
    assert result.has_final_answer is has_final_answer
    assert result.evaluation_completed is True
    assert result.response.output[0].content[0].text == text


def seed_body():
    return ResourcesSeedSessionRequest(
        resources_session_id="synthetic-session",
        episode_id=EpisodeId(rollout_id="synthetic-rollout"),
        task_id=TaskId(taskset="officeqa_pro", task_id=run_body()["task_id"]),
        task_data={k: v for k, v in run_body().items() if k != "responses_create_params"},
    )


def http_request():
    return Request({"type": "http", "session": {}})


@pytest.mark.asyncio
@pytest.mark.parametrize("num_repeats", [1, 3, 5])
@pytest.mark.parametrize("network_access,score_name", [("model_only", "accuracy_no_net"), ("inherit", "accuracy")])
async def test_result_is_mean_accuracy(
    server: OfficeQAResourcesServer, num_repeats: int, network_access: str, score_name: str
) -> None:
    server.config.network_access = network_access
    rewards = [[1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]
    result = await server.aggregate_metrics(
        AggregateMetricsRequest(
            verify_responses=[
                {TASK_INDEX_KEY_NAME: task_index, ROLLOUT_INDEX_KEY_NAME: repeat, "reward": float(reward)}
                for task_index, task_rewards in enumerate(rewards)
                for repeat, reward in enumerate(task_rewards[:num_repeats])
            ]
        )
    )
    task_means = [sum(task_rewards[:num_repeats]) / num_repeats for task_rewards in rewards]
    mean = sum(task_means) / len(task_means)
    assert result.key_metrics == {f"pass@1[avg-of-{num_repeats}]/{score_name}": pytest.approx(100 * mean)}
    assert result.agent_metrics["mean/reward"] == pytest.approx(mean)
    assert sorted(task["mean/reward"] for task in result.group_level_metrics) == pytest.approx(sorted(task_means))


@pytest.fixture
def sandbox(server, monkeypatch):
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "synthetic-sandbox"}
    monkeypatch.setattr(
        server,
        "_create_sandbox",
        AsyncMock(return_value=(sandbox, SandboxSpec(image="synthetic", workdir="/workspace/task", ttl_s=180))),
    )
    return sandbox


@pytest.mark.asyncio
async def test_seed_validates_identity_and_reuses_the_existing_sandbox(server, sandbox):
    body = seed_body()
    first = await server.seed_session(http_request(), body)
    again = await server.seed_session(http_request(), body)
    assert first == again
    assert first.sandbox_access.connection.descriptor == {"sandbox_id": "synthetic-sandbox"}
    sandbox.start.assert_awaited_once()
    changed = body.model_copy(update={"task_data": body.task_data | {"workspace_id": "workspace_" + "1" * 24}})
    with pytest.raises(ValueError, match="identity"):
        await server.seed_session(http_request(), changed)
    changed = body.model_copy(update={"episode_id": EpisodeId(rollout_id="other-rollout")})
    with pytest.raises(ValueError, match="another episode"):
        await server.seed_session(http_request(), changed)


@pytest.mark.asyncio
async def test_close_failure_retains_ownership_for_a_retry(server, sandbox):
    body = seed_body()
    await server.seed_session(http_request(), body)
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    sandbox.stop.side_effect = [RuntimeError("synthetic stop failure"), None]
    with pytest.raises(RuntimeError, match="stop failure"):
        await server.close_resources_session(http_request(), close)
    assert body.resources_session_id in server._sessions
    await server.close_resources_session(http_request(), close)
    await server.close_resources_session(http_request(), close)
    assert not server._sessions and sandbox.stop.await_count == 2
    with pytest.raises(ValueError, match="already closed"):
        await server.seed_session(http_request(), body)


@pytest.mark.asyncio
async def test_failed_start_is_stopped_and_shutdown_releases_remaining_sandboxes(server, sandbox):
    sandbox.start.side_effect = RuntimeError("synthetic startup failure")
    with pytest.raises(RuntimeError, match="startup failure"):
        await server.seed_session(http_request(), seed_body())
    assert not server._sessions
    sandbox.stop.assert_awaited_once()
    sandbox.start.side_effect = None
    await server.seed_session(http_request(), seed_body())
    await server.shutdown()
    assert not server._sessions and sandbox.stop.await_count == 2


@pytest.mark.asyncio
async def test_failed_start_and_release_cannot_return_access_on_reseed(server, sandbox):
    body = seed_body()
    sandbox.start.side_effect = RuntimeError("synthetic startup failure")
    sandbox.stop.side_effect = [RuntimeError("synthetic release failure"), None]
    with pytest.raises(RuntimeError, match="release failure"):
        await server.seed_session(http_request(), body)
    assert body.resources_session_id in server._sessions
    with pytest.raises(RuntimeError, match="not ready"):
        await server.seed_session(http_request(), body)
    sandbox.serialize.assert_not_awaited()
    sandbox.start.assert_awaited_once()
    server._create_sandbox.assert_awaited_once()
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    await server.close_resources_session(http_request(), close)
    assert not server._sessions and sandbox.stop.await_count == 2
    with pytest.raises(ValueError, match="already closed"):
        await server.seed_session(http_request(), body)


def test_normal_http_resource_session_contract(server, sandbox):
    check_resources_session_contract(
        server.setup_webserver(), seed_body(), keeps_state=True, holds_state=server._sessions.__contains__
    )
    assert sandbox.start.await_count == 2
    assert sandbox.stop.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_seed_and_close_cannot_leave_a_sandbox(server, sandbox):
    body = seed_body()
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    started, release = asyncio.Event(), asyncio.Event()

    async def start(spec):
        started.set()
        await release.wait()

    sandbox.start.side_effect = start
    seeding = asyncio.create_task(server.seed_session(http_request(), body))
    await started.wait()
    closing = asyncio.create_task(server.close_resources_session(http_request(), close))
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.gather(seeding, closing)
    assert not server._sessions
    assert not server._locks._locks
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_verification_uses_trusted_gold_and_uid(server):
    body = verify_body().model_copy(
        update={"expected_answer": "43", "uid": "untrusted", "evaluation_completed": False}
    )
    result = await server.verify(Request({"type": "http"}), body)
    assert result.reward == 1 and result.uid == "synthetic" and result.final_answer == "42"
    assert not result.mask_sample
    assert result.evaluation_completed


@pytest.mark.asyncio
async def test_last_assistant_answer_determines_score(server):
    body = verify_body()
    last = body.response.output[0].model_copy(deep=True)
    last.content[0].text = "43"
    body.response.output.append(last)
    result = await server.verify(Request({"type": "http"}), body)
    assert result.reward == 0 and result.final_answer == "43"
    assert result.evaluation_completed


@pytest.mark.asyncio
@pytest.mark.parametrize("field,prefix", [("task_id", "officeqa_pro_"), ("workspace_id", "workspace_")])
async def test_invalid_trusted_reference_is_masked_without_trusting_the_public_uid(server, field, prefix):
    body = verify_body().model_copy(update={field: prefix + "1" * 24})
    result = await server.verify(Request({"type": "http"}), body)
    assert result.reward == 0 and result.mask_sample and result.uid is None
    assert result.failure_kind == "officeqa:reference_unavailable" and result.failure_reason
    assert not result.evaluation_completed


@pytest.mark.asyncio
async def test_corrupt_reference_cannot_turn_into_a_model_error(server):
    path = next((Path(server.config.workspace_root) / "tasks").glob("*.json"))
    path.write_text("not json")
    result = await server.verify(Request({"type": "http"}), verify_body())
    assert result.mask_sample and result.failure_kind == "officeqa:reference_unavailable"
    assert not result.evaluation_completed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "in_progress", "queued"])
async def test_nonterminal_or_failed_response_cannot_receive_credit(server, status):
    body = verify_body()
    body.response.status = status
    result = await server.verify(Request({"type": "http"}), body)
    assert result.reward == 0
    assert result.evaluation_completed
    assert not result.has_final_answer
    assert all(result.model_dump()[name] == 0.0 for name in officeqa_app.TOLERANCES)


@pytest.mark.asyncio
async def test_tolerance_and_missing_answer_metrics_preserve_headline_accuracy(server):
    # The eight attempts include an empty tag, an incomplete correct answer and a correct answer without the tag.
    texts = [
        "<FINAL_ANSWER>42</FINAL_ANSWER>",
        "<FINAL_ANSWER>42.02</FINAL_ANSWER>",
        "<FINAL_ANSWER>42.2</FINAL_ANSWER>",
        "<FINAL_ANSWER>44</FINAL_ANSWER>",
        "<FINAL_ANSWER>45</FINAL_ANSWER>",
        "<FINAL_ANSWER></FINAL_ANSWER>",
        "<FINAL_ANSWER>42</FINAL_ANSWER>",
        "42",
    ]
    rows = []
    for index, text in enumerate(texts):
        body = verify_body()
        body.response.output[0].content[0].text = text
        if index == 6:
            body.response.status = "incomplete"
        result = await server.verify(http_request(), body)
        rows.append(result.model_dump() | {TASK_INDEX_KEY_NAME: index // 4, ROLLOUT_INDEX_KEY_NAME: index % 4})

    assert [row["reward"] for row in rows] == [1, 0, 0, 0, 0, 0, 0, 1]
    assert [row["has_final_answer"] for row in rows] == [True, True, True, True, True, False, False, False]
    aggregate = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    metrics = aggregate.agent_metrics
    assert aggregate.key_metrics == {"pass@1[avg-of-4]/accuracy_no_net": 25.0}
    assert metrics["pass@1[avg-of-4]/correct_tol_0"] == 25.0
    assert metrics["pass@1[avg-of-4]/correct_tol_0p1"] == 37.5
    assert metrics["pass@1[avg-of-4]/correct_tol_1"] == 50.0
    assert metrics["pass@1[avg-of-4]/correct_tol_5"] == 62.5
    assert metrics["pass@1[avg-of-4]/no_answer"] == 37.5


@pytest.mark.asyncio
async def test_diagnostics_use_trusted_references_and_respect_configured_reward_tolerance(server):
    server.config.numeric_tolerance = 0.01
    body = verify_body().model_copy(update={"expected_answer": "43", "correct_tol_0": 1.0, "has_final_answer": False})
    body.response.output[0].content[0].text = "<FINAL_ANSWER>42.2</FINAL_ANSWER>"
    result = await server.verify(http_request(), body)
    assert result.reward == 1.0
    assert result.correct_tol_0 == result.correct_tol_0p1 == 0.0
    assert result.correct_tol_1 == result.correct_tol_5 == 1.0
    assert result.has_final_answer

    body.task_id = "officeqa_pro_" + "1" * 24
    unavailable = await server.verify(http_request(), body)
    assert unavailable.mask_sample and not unavailable.evaluation_completed
    assert not unavailable.has_final_answer
    assert all(unavailable.model_dump()[name] is None for name in officeqa_app.TOLERANCES)


def test_older_rollouts_do_not_gain_fabricated_diagnostics(server):
    metrics = server.compute_metrics([[{"reward": 1.0}], [{"reward": 0.0}]])
    assert metrics["pass@1[avg-of-1]/accuracy_no_net"] == 50.0
    assert not any("correct_tol_" in key or "no_answer" in key for key in metrics)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [json.loads(line) for line in VERIFIER_FIXTURE.cases_path.read_text().splitlines()])
async def test_production_verifier_cases(case):
    if "expected_error" in case:
        with pytest.raises(ValueError, match=case["expected_error"]):
            VERIFIER_FIXTURE.request_model.model_validate(case["request"])
    else:
        body = VERIFIER_FIXTURE.request_model.model_validate(case["request"])
        result = await VERIFIER_FIXTURE.invoke(VERIFIER_FIXTURE.server_factory(), body)
        assert result.reward == case["expected_reward"]
        assert result.mask_sample == case["request"].get("missing_reference", False)
        assert result.evaluation_completed is not result.mask_sample
        assert "expected_answer" not in result.model_dump()


@pytest.mark.parametrize("field", ["task_id", "workspace_id", "prompt_version", "uid", "question", "source_revision"])
def test_catalog_and_verification_require_the_same_fields(field: str) -> None:
    payload = verify_body().model_dump()
    payload.pop(field)
    for schema in (TaskData, OfficeQAVerifyRequest):
        with pytest.raises(ValidationError, match=field):
            schema.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["uid", "question", "source_revision"])
async def test_seed_requires_provenance_before_starting_a_sandbox(server, sandbox, field: str) -> None:
    body = seed_body()
    body.task_data.pop(field)
    with pytest.raises(ResourcesSeedRejectedError, match=field):
        await server.seed_session(http_request(), body)
    sandbox.start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("network_access", [None, "inherit"])
async def test_network_access_controls_the_sandbox_policy(server, monkeypatch, network_access: str | None) -> None:
    server.server_client.global_config_dict = OmegaConf.create({"sandbox": {"opensandbox": {}}})
    server.config.sandbox_config = {"image": "synthetic", "workdir": "/workspace/task", "ttl_s": 180}
    server.config.model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    if network_access is not None:
        server.config.network_access = network_access
    endpoint = Mock(return_value="https://model.example.test:8000/v1")
    monkeypatch.setattr(officeqa_app, "sandbox_server_url", endpoint)
    monkeypatch.setattr(officeqa_app, "create_provider", Mock(return_value=Mock()))

    _, spec = await server._create_sandbox()

    if network_access == "inherit":
        assert "network_policy" not in spec.provider_options
        endpoint.assert_not_called()
    else:
        assert spec.provider_options["network_policy"] == {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": "model.example.test"}],
        }
        endpoint.assert_called_once_with("policy_model", require_reachable=True)


@pytest.mark.parametrize("invalid", ["provenance", "task_id", "workspace_id"])
def test_invalid_seed_returns_422_without_creating_a_sandbox(server, sandbox, invalid: str) -> None:
    body = seed_body()
    if invalid == "provenance":
        body.task_data.pop("source_revision")
    elif invalid == "task_id":
        body.task_id = body.task_id.model_copy(update={"task_id": "officeqa_pro_" + "1" * 24})
    else:
        body.task_data["workspace_id"] = "workspace_" + "1" * 24
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=body.model_dump(mode="json"))
    assert response.status_code == 422
    server._create_sandbox.assert_not_awaited()
    assert not server._sessions


@pytest.mark.parametrize("failure", ["missing_reference", "corrupt_reference", "provider"])
def test_seed_infrastructure_failures_remain_retryable(server, sandbox, failure: str) -> None:
    body = seed_body()
    record = Path(server.config.workspace_root) / "tasks" / (body.task_data["task_id"] + ".json")
    if failure == "missing_reference":
        record.unlink()
    elif failure == "corrupt_reference":
        record.write_text("{}")
    else:
        sandbox.start.side_effect = RuntimeError("provider unavailable")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=body.model_dump(mode="json"))
    assert response.status_code == 500
    assert not server._sessions
    if failure == "provider":
        sandbox.stop.assert_awaited_once()
    else:
        server._create_sandbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_session_cycles_release_inactive_locks(server, sandbox) -> None:
    for index in range(30):
        body = seed_body().model_copy(update={"resources_session_id": f"session-{index}"})
        await server.seed_session(http_request(), body)
        await server.close_resources_session(
            http_request(),
            ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id),
        )
        assert not server._sessions
        assert not server._locks._locks
    assert sandbox.start.await_count == sandbox.stop.await_count == 30


def test_the_server_runs_with_several_workers(server):
    # Document sandboxes are keyed by resources_session_id, which session routing follows to the worker that holds them.
    # Built afresh, so the server's own setup sees the worker count.
    two_workers = OfficeQAResourcesServer.model_construct(
        config=server.config.model_copy(update={"num_workers": 2}), server_client=server.server_client
    )
    assert two_workers.multiple_workers_unsupported_reason() is None
    refuse_unsupported_num_workers(two_workers)


def test_uvicorn_workers_import_a_module_level_app(monkeypatch):
    # With num_workers > 1, each uvicorn worker imports the entrypoint module's `app`.
    import runpy
    from pathlib import Path
    from unittest.mock import MagicMock

    from nemo_gym.server_runtime import SimpleServer
    from nemo_gym.utils import IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME

    path = str(Path(__file__).parents[1] / "app.py")
    started = MagicMock(return_value=object())
    monkeypatch.setattr(SimpleServer, "run_webserver", started)
    for key in (IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME):
        monkeypatch.delenv(key, raising=False)
    assert "app" not in runpy.run_path(path, run_name="resources_servers.officeqa.app")
    started.assert_not_called()
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, "resources_servers/officeqa/app.py")
    assert runpy.run_path(path, run_name="resources_servers.officeqa.app")["app"] is started.return_value
