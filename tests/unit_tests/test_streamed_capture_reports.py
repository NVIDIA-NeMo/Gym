# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import nemo_gym.rollout_collection as collection
from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, CaptureStore
from nemo_gym.client.server_client import ServerClient
from nemo_gym.failure_transport import FailureReportRow
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client
from tests.unit_tests.test_streamed_failure_reports import FINAL, RECOVERED, _LateChat, _LateResponses


@pytest.mark.parametrize("dialect", ["responses", "chat"])
@pytest.mark.parametrize("fail", [False, True])
async def test_actual_stream_capture_survives_collection_and_store(tmp_path, monkeypatch, dialect, fail):
    capture = tmp_path / "capture"
    global_config = {"observability_enabled": True, "model_call_capture_dir": str(capture)}
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr("nemo_gym.base_responses_api_model._CHAT_KEEPALIVE_SECONDS", 0.005)
    cls = _LateResponses if dialect == "responses" else _LateChat
    server = cls(
        config=BaseResponsesAPIModelConfig(
            host="0.0.0.0",
            port=8099,
            entrypoint="",
            name="",
            **({"stream_keepalive_s": 0.005} if dialect == "responses" else {}),
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config),
    )
    object.__setattr__(server, "fail", fail)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    model_http = TestClient(app)
    calls = []

    async def dispatch(*args, **kwargs):
        row = kwargs["json"]
        rollout_id = row.get("_ng_rollout_id", "0-0")
        suffix = "responses" if dialect == "responses" else "chat/completions"
        body = {"stream": True, "input" if dialect == "responses" else "messages": [{"role": "user", "content": "hi"}]}
        reply = model_http.post(f"/ng-rollout/{rollout_id}/v1/{suffix}", json=body)
        assert reply.status_code == 200
        calls.append(reply)
        # The agent's already-settled response is a separate operation from this model-call report.
        return FakeResponse(
            200,
            {
                "response": {"id": "settled-answer", "status": "completed", "output": []},
                "reward": 0.0,
                "evaluation_completed": True,
            },
        )

    post = AsyncMock(side_effect=dispatch)
    install_fake_server_client(monkeypatch, post)
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(failing_row(0)) + "\n")
    output = tmp_path / "rollouts.jsonl"
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        disable_aggregation=True,
        disable_health_check=True,
    )
    await collection.RolloutCollectionHelper().run_from_config(config)
    store = RolloutStore.read(output)
    [saved] = store.selected("success")
    report = FailureReportRow.report(saved)
    assert RECOVERED.event_id in report.events
    assert (FINAL.event_id in report.events) is fail
    assert report.failure_id is None and not report.omitted
    assert saved["reward"] == 0.0 and saved["response"]["id"] == "settled-answer"
    assert store.pending() == []
    [model_call] = saved["ng_model_call_capture"]["calls"]
    assert model_call["error_category"] == ("upstream_error" if fail else None)
    assert saved["ng_trajectory"]["model_calls"]
    before = store.coverage()
    await collection.RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": True}))
    assert post.await_count == len(calls) == 1
    assert RolloutStore.read(output).coverage() == before


@pytest.mark.parametrize(
    "mutation", ["null", "malformed", "omitted", "truncated", "nonterminal", "conflict", "omitted_then_valid"]
)
def test_capture_report_negatives_preserve_episode_selection_and_saved_zero(tmp_path, mutation):
    from nemo_gym.failure_transport import FailureHeader, report_for_current_request
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
    from nemo_gym.typed_exceptions import NgFailureEvent

    episode = NgFailureEvent(
        boundary="resources_server",
        failure_kind="judge_failed",
        failure_reason="saved judge error",
        operation="verify",
        operation_id="verify-operation",
    )
    selected = report_for_current_request(episode)
    valid = report_for_current_request(RECOVERED).model_dump(mode="json")
    field = FailureHeader.STREAM_FIELD
    payload = {"type": "response.completed", field: valid}
    if mutation == "null":
        payload[field] = None
    if mutation == "malformed":
        payload[field] = {"events": 4}
    if mutation == "omitted":
        payload[field] = {"omitted": True}
    if mutation == "nonterminal":
        payload["type"] = "response.in_progress"
    if mutation == "malformed_type":
        payload["type"] = []
    raw = "data: " + json.dumps(payload) + "\n\n"
    if mutation == "truncated":
        raw = raw[:-2]
    if mutation == "omitted_then_valid":
        raw = 'data: {"type":"response.completed","' + field + '":{"omitted":true}}\n\n' + raw
    if mutation == "conflict":
        altered = json.loads(json.dumps(payload))
        altered[field]["events"][RECOVERED.event_id]["failure_reason"] = "contradiction"
        raw += "data: " + json.dumps(altered) + "\n\n"
    capture = tmp_path / "capture"
    CaptureStore(capture).record(
        "0-0",
        {"model_call_id": "bounded-call", "dialect": "responses", "response_raw": raw, "response": {}, "request": {}},
    )
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 0.0,
        "response": {"id": "saved"},
        **FailureReportRow.fields(selected),
    }
    result = merge_model_call_capture_into_record(row, [capture])
    report = FailureReportRow.report(result)
    assert report.omitted is True and report.failure_id == episode.event_id
    assert result["reward"] == 0.0 and result["response"] == {"id": "saved"}


@pytest.mark.parametrize(
    "raw",
    [
        'data: {"type":"response.completed","nemo_gym_failure_report":' + "[" * 1500 + "0" + "]" * 1500 + "}\n\n",
        'data: {"choices":[{"finish_reason":null}],"nemo_gym_failure_report":{}}\n\n',
    ],
)
def test_malformed_saved_stream_diagnostics_do_not_change_score_or_crash(tmp_path, raw):
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

    capture = tmp_path / "capture"
    CaptureStore(capture).record(
        "0-0",
        {"model_call_id": "malformed-call", "dialect": "chat", "response_raw": raw, "response": {}, "request": {}},
    )
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0.0, "response": {"id": "saved"}}
    result = merge_model_call_capture_into_record(row, [capture])
    assert FailureReportRow.report(result).omitted
    assert result["reward"] == 0.0 and result["response"] == {"id": "saved"}


def test_unencodable_capture_text_is_omitted_at_the_reader_boundary():
    from nemo_gym.failure_transport import StreamFailureReportReader

    reader = StreamFailureReportReader()
    reader.feed_text("\ud800")
    assert reader.finish().omitted
