# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from importlib import import_module
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import Request

from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.failure_transport import FailureHeader, FailureReportRow
from nemo_gym.typed_exceptions import NgFailureEvent


async def assert_real_verifier_reply_preserves_the_original_judge_model_cause(tmp_path, monkeypatch, adapter):
    multiple = adapter == "rolemrc_multi"
    adapter = "rolemrc" if multiple else adapter
    module = import_module(f"resources_servers.{adapter}.app")
    fixtures = import_module(f"resources_servers.{adapter}.tests.test_app")
    model_error = NgFailureEvent(
        boundary="model",
        failure_kind="model_request_error",
        failure_reason="controlled unavailable",
        operation="model.request",
        operation_id="model-operation",
        attempt_id="model-attempt",
        http_status=503,
    )
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(side_effect=model_error)
    second_error = None
    if multiple:
        second_error = NgFailureEvent(
            boundary="model",
            failure_kind="model_request_error",
            failure_reason="second controlled failure",
            operation="model.request",
            operation_id="second-model-operation",
            attempt_id="second-model-attempt",
            http_status=503,
        )
        client.post = AsyncMock(side_effect=[model_error, second_error])
        monkeypatch.setattr(
            module, "_build_judge_prompts", lambda *args: [("first", "first prompt"), ("second", "second prompt")]
        )
    if adapter == "rolemrc":
        server = module.RoleMRCResourcesServer(config=fixtures._judge_config(), server_client=client)
        body = module.RoleMRCVerifyRequest(
            responses_create_params=fixtures.NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=fixtures._make_response("saved answer"),
            task="role_related_mrc_answer_no_narration",
        )
    else:
        factory = fixtures.TestApp()
        config = fixtures.TestApp.config.__wrapped__(factory)
        server = module.TavilySearchResourcesServer(config=config, server_client=client)
        body = module.TavilySearchVerifyRequest(
            responses_create_params=fixtures.NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=factory._create_model_response("saved answer"),
            ground_truth="reference",
            question="synthetic question",
        )
        if adapter == "browsecomp_advanced_harness":
            monkeypatch.setattr(type(server), "JUDGE_MAX_ATTEMPTS", 1)
            monkeypatch.setattr(module, "sleep", AsyncMock())
            server.config.workspace = "per_session"
            monkeypatch.setattr(server, "_cleanup_workspace", MagicMock())
    app = server.setup_webserver()
    server.setup_exception_middleware(app)

    @app.post("/seed-cause-probe")
    async def seed(request: Request):
        request.session[SESSION_ID_KEY] = "synthetic-session"
        return {}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as http:
        assert (await http.post("/seed-cause-probe")).status_code == 200
        reply = await http.post("/ng-rollout/synthetic/verify", json=body.model_dump(mode="json"))
    assert reply.status_code == 200
    assert reply.json()["_ng_failure_class"] == "judge_failed"
    assert reply.json()["mask_sample"] is True
    report = FailureReportRow.report(reply.json())
    observed = FailureHeader.decode(reply.headers)
    if observed is not None:
        report = report.merged(observed)
    assert report is not None and report.failure_id is not None
    selected = report.failure_id
    chain = []
    while selected is not None:
        chain.append(selected)
        selected = report.events[selected].cause_id
    assert model_error.event_id in chain
    assert sum(event.event_id == model_error.event_id for event in report.events.values()) == 1
    assert client.post.await_count == (2 if multiple else 1)
    if multiple:
        assert second_error.event_id in report.events and second_error.event_id not in chain
        assert "first" in reply.json()["_ng_failure_judge_error"]
        assert "second" in reply.json()["_ng_failure_judge_error"]
    if adapter == "browsecomp_advanced_harness":
        server._cleanup_workspace.assert_called_once_with("synthetic-session")

    # Persist the actual registered verifier reply through the normal collection writer.
    import nemo_gym.rollout_collection as collection
    from nemo_gym.rollout_store import RolloutStore
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client
    from tests.unit_tests.test_rollout_recovery import failing_row

    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(failing_row()) + "\n")
    output = tmp_path / "rollouts.jsonl"
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        route_failures_to_sidecar=True,
        disable_aggregation=True,
        disable_health_check=True,
    )
    wire_reply = FakeResponse(200, reply.json())
    wire_reply.headers = reply.headers
    dispatch = AsyncMock(return_value=wire_reply)
    install_fake_server_client(monkeypatch, dispatch)
    with pytest.raises(RuntimeError, match="None of the 1 dispatched"):
        await collection.RolloutCollectionHelper().run_from_config(config)
    store = RolloutStore.read(output)
    [saved] = store.failures()
    assert FailureReportRow.report(saved) == report
    assert saved["response"] == reply.json()["response"]
    assert "reward" not in saved and store.pending() == []
    dispatch.reset_mock()
    try:
        await collection.RolloutCollectionHelper().run_from_config(
            config.model_copy(update={"resume_from_cache": True})
        )
    except RuntimeError:
        pass
    dispatch.assert_not_called()


async def assert_browsecomp_received_unparseable_verdict_keeps_zero_and_transport_observations(
    monkeypatch, error_first
):
    from tests.unit_tests.test_rollout_collection import FakeResponse

    module = import_module("resources_servers.browsecomp_advanced_harness.app")
    fixtures = import_module("resources_servers.browsecomp_advanced_harness.tests.test_app")
    factory = fixtures.TestApp()
    config = fixtures.TestApp.config.__wrapped__(factory)
    model_error = NgFailureEvent(
        boundary="model",
        failure_kind="model_request_error",
        failure_reason="controlled retry",
        operation="model.request",
        operation_id="mixed-operation",
        attempt_id="mixed-attempt",
        http_status=503,
    )
    received = FakeResponse(200, factory._create_judge_response("received without a usable verdict"))
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(side_effect=[model_error, received] if error_first else [received, model_error])
    server = module.TavilySearchResourcesServer(config=config, server_client=client)
    monkeypatch.setattr(type(server), "JUDGE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    app = server.setup_webserver()
    server.setup_exception_middleware(app)

    @app.post("/seed-mixed-probe")
    async def seed(request: Request):
        request.session[SESSION_ID_KEY] = "mixed-session"
        return {}

    body = module.TavilySearchVerifyRequest(
        responses_create_params=fixtures.NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=factory._create_model_response("saved answer"),
        ground_truth="reference",
        question="question",
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as http:
        assert (await http.post("/seed-mixed-probe")).status_code == 200
        reply = await http.post("/ng-rollout/mixed/verify", json=body.model_dump(mode="json"))
    assert reply.status_code == 200 and reply.json()["reward"] == 0.0
    assert reply.json().get("_ng_failure_class") is None
    report = FailureHeader.decode(reply.headers)
    assert report is not None and report.failure_id is None
    assert model_error.event_id in report.events
    assert sum(event.event_id == model_error.event_id for event in report.events.values()) == 1
    assert client.post.await_count == 2
