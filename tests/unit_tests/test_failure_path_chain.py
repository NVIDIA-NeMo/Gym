# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One failure's path, carried hop by hop: resources server, agent, collector."""

import json
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

import nemo_gym.rollout_collection
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    HandledEpisodeError,
)
from nemo_gym.config_types import AggregateMetrics
from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    EpisodeId,
    MaterializedTask,
    TaskId,
)
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.judge import JudgeError, call_judge, judge_failsafe
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_collection import RolloutCollectionHelper, _episode_record
from nemo_gym.sandbox.providers.base import SandboxCreateError
from nemo_gym.server_utils import ServerClient
from nemo_gym.typed_exceptions import FAILURE_EVENT_HEADER


class _Verdict(BaseModel):
    verdict: str


class _VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    response: dict = {}


_VERIFY_BODY = {
    "responses_create_params": {"input": "task"},
    "response": NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    ).model_dump(mode="json"),
}


async def _judge_row(status: int) -> dict:
    """Hop 1: the judge client raises inside a resources server; judge_failsafe returns the row."""

    async def handle(_: web.Request) -> web.Response:
        return web.json_response({"error": "refused"}, status=status)

    app = web.Application()
    app.router.add_post("/v1/responses", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    import aiohttp

    async with aiohttp.ClientSession() as session:

        class _Client:
            async def post(self, *, server_name, url_path, json):
                return await session.post(url + url_path, json=json)

        async def verify(body: _VerifyRequest):
            await call_judge(
                _Client(), server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict
            )

        reply = await judge_failsafe(verify)(body=_VerifyRequest(**_VERIFY_BODY))
    await runner.cleanup()
    return json.loads(reply.body)


async def test_hop1_resources_server_puts_the_path_on_the_judge_row() -> None:
    row = await _judge_row(429)
    # One kind per failure: the kind the client observed, with the status on the path.
    assert row["failure_kind"] == "judge_rate_limited"
    assert row["_ng_failure_path"][0] == {
        "operation": "judge.request",
        "exception_type": "nemo_gym.judge.JudgeRateLimited",
    }
    assert row["_ng_failure_path"][-1]["http_status"] == 429
    # The routing class, reward and mask are unchanged.
    assert row["_ng_failure_class"] == "judge_failed" and row["reward"] == 0.0 and row["mask_sample"] is True


async def test_an_untyped_judge_error_carries_its_type_on_the_path() -> None:
    async def verify(body: _VerifyRequest):
        raise JudgeError("ValueError: unparsable verdict")

    row = json.loads((await judge_failsafe(verify)(body=_VerifyRequest())).body)
    assert row["failure_kind"] == "judge_failed"
    assert row["_ng_failure_path"] == [{"exception_type": "nemo_gym.judge.JudgeError"}]


class _Response:
    def __init__(self, payload: dict, headers: CIMultiDict | None = None) -> None:
        self.status, self.ok, self.payload, self.headers = 200, True, payload, headers or CIMultiDict()

    def release(self) -> None:
        pass


def _collector(monkeypatch, response: _Response) -> None:
    server_client = MagicMock()
    server_client.post = AsyncMock(return_value=response)
    server_client.global_config_dict = OmegaConf.create(
        {
            "my_agent": {"responses_api_agents": {"impl": {}}},
            "my_environment_server": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}},
        }
    )
    monkeypatch.setattr(nemo_gym.rollout_collection, "setup_server_client_utils", lambda *a, **k: server_client)

    async def get_response_json(res):
        return res.payload

    async def raise_for_status(res):
        return None

    monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", get_response_json)
    monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", raise_for_status)


_ROW = {
    AGENT_REF_KEY_NAME: {"name": "my_agent"},
    TASK_INDEX_KEY_NAME: 0,
    ROLLOUT_INDEX_KEY_NAME: 0,
    "responses_create_params": {},
}


class _TaskInput(BaseModel):
    value: str


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _EpisodeResponse(BaseEpisodeResponse[str]):
    pass


class _FailingServer(BaseEnvironmentServer[_Request, _EpisodeResponse]):
    request_model = _Request
    response_model = _EpisodeResponse

    async def run(self, request: _Request, cleanup: CleanupContext) -> _EpisodeResponse:
        try:
            raise SandboxCreateError("no capacity")
        except SandboxCreateError as error:
            raise HandledEpisodeError(
                EpisodeFailure(failure_reason="Seed failed", terminal=False, stage="seed")
            ) from error

    async def aggregate_metrics(self, body=None) -> AggregateMetrics:
        return AggregateMetrics()


def _episode_request() -> dict:
    return _Request(
        episode_id=EpisodeId(rollout_id="r", attempt=0),
        task=MaterializedTask(task_id=TaskId(taskset="t:d", task_id="1"), task_input=_TaskInput(value="v")),
    ).model_dump(mode="json")


async def test_an_episode_failure_carries_its_path_in_the_reply_and_the_saved_row(monkeypatch) -> None:
    config = BaseEnvironmentServerConfig(
        name="environment", host="127.0.0.1", port=1, entrypoint="app.py", cleanup_timeout_seconds=0.01
    )
    server = _FailingServer(config=config, server_client=MagicMock(spec=ServerClient))
    reply = TestClient(server.setup_webserver()).post("/run", json=_episode_request())
    assert reply.status_code == 200 and FAILURE_EVENT_HEADER not in reply.headers
    failure = reply.json()["failure"]
    # The producer's fields are kept; the base server fills the path, and the unset kind, in the same EpisodeFailure.
    assert (failure["failure_reason"], failure["terminal"], failure["stage"]) == ("Seed failed", False, "seed")
    assert failure["failure_kind"] == "provider_unavailable"
    assert failure["failure_path"] == [
        {"exception_type": "nemo_gym.base_environment_server.HandledEpisodeError"},
        {"operation": "sandbox.create", "exception_type": "nemo_gym.sandbox.providers.base.SandboxCreateError"},
    ]

    _collector(monkeypatch, _Response(reply.json()))
    completed = await next(iter(RolloutCollectionHelper()._run_examples_with_metadata([dict(_ROW)])))
    record = _episode_record(completed.result)
    assert record["_ng_failure_class"] == "environment_server_failed" and record["_ng_failure_terminal"] is False
    assert record["_ng_failure_stage"] == "seed" and record["_ng_failure_path"] == failure["failure_path"]
    # The reason reaches the saved row beside the routing class, which stays main's.
    assert record["_ng_failure_kind"] == "provider_unavailable"


def test_an_episode_failure_without_a_path_saves_the_row_main_saves() -> None:
    record = _episode_record(
        {
            "episode_id": {"rollout_id": "r", "attempt": 0},
            "task_id": {"taskset": "t:d", "task_id": "1"},
            "failure": {"failure_reason": "Episode timed out", "terminal": False},
        }
    )
    assert record == {
        "_ng_task_id": {"taskset": "t:d", "task_id": "1"},
        "_ng_failure_class": "environment_server_failed",
        "_ng_failure_terminal": False,
        "_ng_failure_message": "Episode timed out",
    }


def test_a_judge_failed_result_keeps_its_path_through_the_episode_record() -> None:
    facts = {"_ng_failure_path": [{"exception_type": "nemo_gym.judge.JudgeRateLimited"}]}
    record = _episode_record(
        {
            "episode_id": {"rollout_id": "r", "attempt": 0},
            "task_id": {"taskset": "t:d", "task_id": "1"},
            "result": {
                "reward": 0.0,
                "failure_kind": "judge_failed",
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": "x",
                **facts,
            },
        }
    )
    assert {key: record[key] for key in facts} == facts
