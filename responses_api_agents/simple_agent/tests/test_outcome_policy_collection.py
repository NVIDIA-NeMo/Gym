# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SimpleAgent's outcome-policy configuration and its confirmed-timeout path through rollout collection."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError
from omegaconf import OmegaConf
from pydantic import ValidationError

import nemo_gym.client.http_client
import nemo_gym.rollout_collection
from nemo_gym.agent_outcome import AgentOutcomePolicyConfig, OutcomeAction
from nemo_gym.client import ServerClient
from nemo_gym.config import ATTEMPT_INDEX_KEY_NAME
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.core.failure_kinds import MODEL_POLICY_RESPONSE_TIMEOUT
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    NG_FAILURE_COUNTS_AS_ZERO_KEY,
    NG_TERMINAL_KEY,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _failures_path_for,
)
from responses_api_agents.simple_agent.app import SimpleAgentConfig, SimpleAgentOutcomePolicyConfig


@pytest.fixture
def empty_global_config(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    get_global_config_dict = MagicMock(return_value={})
    monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", get_global_config_dict)
    return get_global_config_dict


class FakeResponse:
    """The parts of aiohttp's ClientResponse that the rollout dispatcher touches."""

    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status = status
        self.ok = 200 <= status < 300
        self.payload = payload
        self.released = False

    def release(self) -> None:
        self.released = True

    async def read(self) -> bytes:
        return orjson.dumps(self.payload)


def http_error(status: int, message: str = "boom", body: bytes | None = None) -> ClientResponseError:
    request_info = SimpleNamespace(method="POST", url="http://agent/run", real_url="http://agent/run")
    error = ClientResponseError(request_info=request_info, history=(), status=status, message=message)
    if body is not None:
        error.response_content = body
    return error


def install_fake_server_client(monkeypatch: pytest.MonkeyPatch, post: AsyncMock) -> MagicMock:
    """Route every dispatcher HTTP call through `post` and unwrap FakeResponse."""
    server_client = MagicMock()
    server_client.post = post
    server_client.global_config_dict = OmegaConf.create(
        {
            "my_agent": {"responses_api_agents": {"impl": {}}},
            "my_environment_server": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}},
        }
    )
    monkeypatch.setattr(
        nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: server_client
    )

    async def raise_for_status(response: FakeResponse) -> None:
        if not response.ok:
            raise http_error(response.status)

    async def get_response_json(response: FakeResponse) -> dict | None:
        return response.payload

    monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", raise_for_status)
    monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", get_response_json)
    return server_client


@pytest.mark.parametrize("field", ["default_action", "by_failure_kind", "by_status_code"])
def test_boundary_rejects_unsupported_actions_at_load(field):
    def value(action):
        return (
            action
            if field == "default_action"
            else {MODEL_POLICY_RESPONSE_TIMEOUT if field == "by_failure_kind" else 500: action}
        )

    with pytest.raises(ValidationError, match="does not support verify"):
        SimpleAgentOutcomePolicyConfig(**{field: value("verify")})
    with pytest.raises(ValidationError, match="does not support count_as_zero"):
        AgentOutcomePolicyConfig(**{field: value("count_as_zero")})


def test_alias_resolves_once_conflicts_are_rejected_and_config_roundtrips():
    base = dict(
        host="127.0.0.1",
        port=8000,
        entrypoint="",
        name="agent",
        model_server={"type": "responses_api_models", "name": "model"},
    )
    legacy = SimpleAgentConfig(**base, final_on_model_timeout=True)
    assert legacy.outcome_policy.by_failure_kind[MODEL_POLICY_RESPONSE_TIMEOUT] == OutcomeAction.COUNT_AS_ZERO
    assert "final_on_model_timeout" not in legacy.model_dump()
    assert (
        SimpleAgentConfig.model_validate(legacy.model_dump(exclude_none=True)).outcome_policy == legacy.outcome_policy
    )
    explicit = SimpleAgentConfig(**base, outcome_policy={"by_failure_kind": {MODEL_POLICY_RESPONSE_TIMEOUT: "fail"}})
    assert explicit.outcome_policy.default_action == OutcomeAction.RETRY
    assert explicit.outcome_policy.by_failure_kind[MODEL_POLICY_RESPONSE_TIMEOUT] == OutcomeAction.FAIL
    for alias, action in [(True, "retry"), (False, "count_as_zero")]:
        with pytest.raises(ValidationError, match="conflicts"):
            SimpleAgentConfig(
                **base,
                final_on_model_timeout=alias,
                outcome_policy={"by_failure_kind": {MODEL_POLICY_RESPONSE_TIMEOUT: action}},
            )


async def test_confirmed_timeout_policy_actions_keep_facts_and_collection_budgets(
    tmp_path,
    monkeypatch,
    empty_global_config,
):
    """Actual loopback timeout producer, legacy /run executor, Collection writer and bounded resume.

    The endpoint is a scripted CPU fixture, not a model evaluation. The same classified failure is reused for
    all actions; the transport adapters deliver its report without replacing it with normalized health data.
    """
    import aiohttp
    import httpx
    from fastapi import FastAPI, Request, Response

    from nemo_gym.agent_outcome import OutcomeAction
    from nemo_gym.core.failure_kinds import MODEL_POLICY_RESPONSE_TIMEOUT
    from nemo_gym.failure_transport import FailureHeader, RequestFailureLog
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
    from nemo_gym.server_runtime.episode_types import FailureBoundary
    from nemo_gym.typed_exceptions import NgFailureEvent, failure_report_for
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest

    connections = []
    hits = 0

    async def never_answer(reader, writer):
        nonlocal hits
        hits += 1
        connections.append(writer)
        await reader.read()
        writer.close()

    listener = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    session = aiohttp.ClientSession()
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: session)
    monkeypatch.setattr(nemo_gym.server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(
        api_key="test",
        base_url=f"http://127.0.0.1:{port}",
        max_connection_retries=2,
        max_http_attempts=2,
        request_timeout_seconds=0.04,
    )
    try:
        with RequestFailureLog.capture() as log:
            with pytest.raises(TimeoutError) as timeout:
                await model._request(
                    method="POST", url=f"http://127.0.0.1:{port}/responses", json={"input": "fixture"}
                )
            upstream = log.reply_report(timeout.value)
    finally:
        await session.close()
        listener.close()
        await listener.wait_closed()
        for writer in connections:
            writer.close()
    assert hits == 2 and len(upstream.events) == 2
    assert all(e.request_upload_complete is True and e.attempt_id and e.operation_id for e in upstream.events.values())

    def reply(report=None):
        response = MagicMock(
            status=500 if report else 200,
            ok=not report,
            cookies={},
            headers=FailureHeader.encode(report) if report else {},
        )
        response.read = AsyncMock(return_value=b"{}")
        response.content.read = AsyncMock(return_value=b"controlled failure")
        if report:
            error = http_error(500)
            error.request_info.headers = {}
            error.headers = response.headers
            response.raise_for_status = MagicMock(side_effect=error)
        else:
            response.raise_for_status = MagicMock()
        return response

    def config(action):
        return SimpleAgentConfig(
            host="127.0.0.1",
            port=8000,
            entrypoint="",
            name="my_agent",
            model_server={"type": "responses_api_models", "name": "model"},
            resources_server={"type": "resources_servers", "name": "resources"},
            outcome_policy={"by_failure_kind": {MODEL_POLICY_RESPONSE_TIMEOUT: action}},
        )

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    client.post = AsyncMock(return_value=reply(upstream))
    agent = SimpleAgent(config=config(OutcomeAction.RETRY), server_client=client)
    with pytest.raises(NgFailureEvent) as classified:
        await agent.responses(
            MagicMock(session={}, path_params={}, cookies={}),
            Response(),
            NeMoGymResponseCreateParamsNonStreaming(input="fixture"),
        )
    assert client.post.await_count == 1  # The model's two attempts did not become another agent model call.
    failure = classified.value
    assert failure.boundary == FailureBoundary.AGENT and failure.failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT
    report = failure_report_for(failure).merged(upstream)
    facts = {k: v.model_dump(mode="json") for k, v in report.events.items()}
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def wire_endpoint(request):
        if request.path == "/seed_session":
            return web.json_response({})
        assert request.path.endswith("/v1/responses")
        return web.Response(status=500, text="controlled failure", headers=FailureHeader.encode(report))

    app = web.Application()
    app.router.add_post("/{tail:.*}", wire_endpoint)
    wire_server = TestServer(app)
    await wire_server.start_server()
    wire_session = aiohttp.ClientSession()
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: wire_session)
    # Use the actual SimpleServer HTTP middleware: its request capture retains recovered remote attempts.
    run_app = FastAPI()
    agent.setup_exception_middleware(run_app)

    @run_app.post("/run")
    async def run_endpoint(request: Request, body: SimpleAgentRunRequest):
        return await agent.run(request, body)

    run_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=run_app), base_url="http://agent")
    try:
        for action in [OutcomeAction.RETRY, OutcomeAction.FAIL, OutcomeAction.COUNT_AS_ZERO]:
            agent.config = config(action)

            async def agent_post(*, url_path, **kwargs):
                payload = kwargs.get("json")
                if isinstance(payload, NeMoGymResponseCreateParamsNonStreaming):
                    payload = payload.model_dump(mode="json")
                return await nemo_gym.client.http_client.request(
                    method="POST",
                    url=str(wire_server.make_url(url_path)),
                    json=payload,
                    _max_num_tries=1,
                )

            client.post = AsyncMock(side_effect=agent_post)
            calls = 0
            aggregate_rows = []

            async def collect_post(server_name, url_path, json, **kwargs):
                nonlocal calls
                if url_path == "/run":
                    if json.get("control") == "healthy":
                        return FakeResponse(200, {"reward": 1.0, "response": {"usage": {}}})
                    calls += 1
                    result = await run_client.post("/run", json=json)
                    assert result.status_code == 200
                    forwarded = FakeResponse(200, result.json())
                    forwarded.headers = result.headers
                    return forwarded
                aggregate_rows.extend(dict(r) for r in json.verify_responses)
                return FakeResponse(200, compute_aggregate_metrics(json.verify_responses).model_dump())

            install_fake_server_client(monkeypatch, AsyncMock(side_effect=collect_post))
            case = tmp_path / action.value
            case.mkdir()
            inputs = case / "input.jsonl"
            inputs.write_text(
                "".join(
                    orjson.dumps(
                        {
                            "responses_create_params": {"input": "fixture"},
                            "agent_ref": {"name": "my_agent"},
                            "control": control,
                        }
                    ).decode()
                    + "\n"
                    for control in ["healthy", "timeout"]
                )
            )
            cfg = RolloutCollectionConfig(
                input_jsonl_fpath=str(inputs), output_jsonl_fpath=str(case / "output.jsonl"), disable_health_check=True
            )
            await RolloutCollectionHelper().run_from_config(cfg)
            assert (
                calls == 1 and client.post.await_count == 2
            )  # Seed + one loop, no action-dependent immediate model retry.
            rows = [
                orjson.loads(line)
                for line in _failures_path_for(Path(cfg.output_jsonl_fpath)).read_bytes().splitlines()
            ]
            assert len(rows) == 1
            row = rows[0]
            assert row["_ng_failure_id"] == report.failure_id
            assert row["_ng_failure_events"] == facts
            assert row[NG_TERMINAL_KEY] is (action != OutcomeAction.RETRY)
            assert row[NG_FAILURE_COUNTS_AS_ZERO_KEY] is (action == OutcomeAction.COUNT_AS_ZERO)
            assert row.get(ATTEMPT_INDEX_KEY_NAME, 0) == 0
            assert "reward" not in row
            assert sorted(r["reward"] for r in aggregate_rows) == (
                [0.0, 1.0] if action == OutcomeAction.COUNT_AS_ZERO else [1.0]
            )
            # Resume must spend the next episode attempt only for RETRY; then exhaustion prevents another dispatch.
            cfg.resume_from_cache = True
            await RolloutCollectionHelper().run_from_config(cfg)
            assert calls == (2 if action == OutcomeAction.RETRY else 1)
            await RolloutCollectionHelper().run_from_config(cfg)
            assert calls == (2 if action == OutcomeAction.RETRY else 1)
            if action == OutcomeAction.RETRY:
                rows = [
                    orjson.loads(line)
                    for line in _failures_path_for(Path(cfg.output_jsonl_fpath)).read_bytes().splitlines()
                ]
                assert [r.get(ATTEMPT_INDEX_KEY_NAME, 0) for r in rows] == [0, 1]
                assert all(r["_ng_failure_events"] == facts for r in rows)
    finally:
        await run_client.aclose()
        await wire_session.close()
        await wire_server.close()
    assert hits == 2  # All episode-policy decisions reused the original classified model operation.
