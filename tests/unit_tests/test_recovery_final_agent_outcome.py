# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from copy import deepcopy

import pytest

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.config.config_types import ConfigError
from nemo_gym.failure_transport import FailureReportRow
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollout_records import logical_rollout_id
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from nemo_gym.server_runtime.episode_types import FailureEvent, FailureReport


def _failure(case):
    event = FailureEvent(
        event_id="model-error",
        boundary="model",
        failure_kind="model_request_error",
        failure_reason="controlled rejection",
        operation="model_request",
        operation_id="request",
        attempt_id="request-1",
        http_status=503,
    )
    response = {
        "id": "failed-response",
        "object": "response",
        "status": "failed",
        "output": [],
        "error": {"code": "server_error", "message": "diagnostic"},
        "metadata": {
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                failure_kind="model_request_error", status_code=503
            ).model_dump_json(),
            "private_extra": "must stay on host",
        },
    }
    row = {
        "_ng_failure_class": "environment_server_failed",
        "_ng_failure_terminal": False,
        "_ng_failure_stage": "agent",
        "_ng_failure_max_attempts": 3,
        "_ng_failure_http_status": None,
        "_ng_failure_record": {"source": "environment", "delivery": "delivered", "failure": {"stage": "agent"}},
        "_ng_failure_partial_response": response,
        **FailureReportRow.fields(FailureReport(events={event.event_id: event})),
    }
    if case == "timeout":
        response["metadata"][AGENT_OUTCOME_METADATA_KEY] = AgentOutcome(
            failure_kind="model_request_error", status_code=500
        ).model_dump_json()
        row["_ng_failure_events"]["model-error"]["http_status"] = None
        row["_ng_failure_events"]["model-error"]["request_upload_complete"] = True
    elif case == "recovered_only":
        response["metadata"].pop(AGENT_OUTCOME_METADATA_KEY)
    elif case == "completed":
        response.update(status="completed", error=None)
    elif case == "omitted":
        row["_ng_failure_events_omitted"] = True
    elif case == "mixed":
        other = event.model_copy(update={"event_id": "earlier-error", "operation_id": "earlier"})
        row["_ng_failure_events"][other.event_id] = other.model_dump(mode="json")
    elif case == "dangling":
        row["_ng_failure_events"]["model-error"]["cause_id"] = "missing"
    elif case == "cycle":
        row["_ng_failure_events"]["model-error"]["cause_id"] = "model-error"
    elif case == "foreign":
        row["_ng_failure_events"]["model-error"]["boundary"] = "resources_server"
    elif case == "unobserved":
        row.pop("_ng_failure_events")
    elif case == "missing_operation":
        row["_ng_failure_events"]["model-error"]["operation_id"] = None
    elif case == "missing_attempt":
        row["_ng_failure_events"]["model-error"]["attempt_id"] = None
    elif case == "unbound_recovered_later_failure":
        response["metadata"][AGENT_OUTCOME_METADATA_KEY] = AgentOutcome(failure_kind="unknown").model_dump_json()
    elif case == "wrong_status":
        row["_ng_failure_events"]["model-error"]["http_status"] = 429
    elif case == "malformed":
        response["metadata"][AGENT_OUTCOME_METADATA_KEY] = '{"failure_kind":"model_request_error","status_code":true}'
    elif case == "extra_metadata":
        response["metadata"][AGENT_OUTCOME_METADATA_KEY] = (
            '{"failure_kind":"model_request_error","status_code":503,"private":"must not leave host"}'
        )
    elif case == "unknown_delivery":
        row["_ng_failure_record"]["delivery"] = "possibly_delivered"
    elif case == "cleanup":
        row["_ng_failure_record"]["failure"]["stage"] = "cleanup"
    elif case == "invalid_selected":
        row["_ng_failure_id"] = "missing"
    elif case == "explicit_unknown":
        row["failure_kind"] = "unknown"
    elif case == "terminal":
        row["_ng_failure_terminal"] = True
    elif case not in ("503", "exhausted"):
        raise AssertionError(case)
    return row


@pytest.mark.parametrize(
    "case",
    [
        "503",
        "exhausted",
        "timeout",
        "recovered_only",
        "completed",
        "omitted",
        "mixed",
        "dangling",
        "cycle",
        "foreign",
        "unobserved",
        "missing_operation",
        "missing_attempt",
        "unbound_recovered_later_failure",
        "wrong_status",
        "malformed",
        "extra_metadata",
        "unknown_delivery",
        "cleanup",
        "invalid_selected",
        "explicit_unknown",
        "terminal",
    ],
)
def test_final_agent_outcome_through_store_and_stdlib_collector(tmp_path, case):
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(inputs[0]) + "\n")
    output = tmp_path / "artifacts/rollouts.jsonl"

    def prepare():
        return deepcopy(inputs), RunManifest.create(source, inputs, {}, {}).model_copy(
            update={"max_rollout_attempts": 3}
        )

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for _ in range(3 if case == "exhausted" else 1):
            row = deepcopy(inputs[0])
            store.allocate_attempt(row)
            failure = _failure(case)
            record = failure["_ng_failure_record"]
            record.update(
                episode_id={"rollout_id": logical_rollout_id(row), "attempt": row["_ng_attempt_index"]},
                run_id=store.manifest.run_id,
            )
            record["failure"].update(failure_reason="controlled failure", terminal=failure["_ng_failure_terminal"])
            if case == "unknown_delivery":
                with pytest.raises(ConfigError, match="requires delivery"):
                    store.record_outcome(row | failure)
                return
            store.record_outcome(row | failure)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    raw = subprocess.check_output(["/usr/bin/python3", "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], text=True)
    assert "must stay on host" not in raw and "must not leave host" not in raw
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw), tmp_path)
    assert collected.managed_reader_error is None
    decision = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert decision.action.value == ("resubmit" if case == "503" else "stop")
    assert decision.evidence[0].attempts == (3 if case == "exhausted" else 0 if case == "terminal" else 1)
    assert decision.evidence[0].failure_id is None or case == "invalid_selected"
    assert decision.evidence[0].state.value == (
        "retryable"
        if case == "503"
        else "exhausted"
        if case == "exhausted"
        else "terminal"
        if case == "terminal"
        else "unknown"
    )
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("legacy", [False, True])
async def test_failed_agent_outcome_through_environment_and_collection(tmp_path, monkeypatch, legacy):
    from unittest.mock import AsyncMock

    import httpx
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn_legacy.app import (
        SingleAgentTurnLegacyEnvironmentServer,
        SingleAgentTurnLegacyEnvironmentServerConfig,
    )
    from nemo_gym.failure_transport import FailureHeader, observe_reply
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from tests.single_agent_turn_helpers import _Client, _environment_server, _request, _Response
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client
    from tests.unit_tests.test_saved_generation_admission import response

    class ObservingClient(_Client):
        async def post(self, *args, **kwargs):
            reply = await super().post(*args, **kwargs)
            observe_reply(reply.headers)
            return reply

    environment, dependencies = _environment_server(client_type=ObservingClient)
    if legacy:
        environment = SingleAgentTurnLegacyEnvironmentServer(
            config=SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(environment.config),
            server_client=dependencies,
        )
    failure = _failure("503")
    failed = response() | failure["_ng_failure_partial_response"]
    dependencies.responses[2] = _Response(failed, FailureHeader.encode(FailureReportRow.report(failure)))
    del dependencies.responses[4]
    source = tmp_path / "source.jsonl"
    source.write_text(
        (
            json.dumps({"responses_create_params": {"input": "task"}, "instance_id": "task"})
            if legacy
            else _request().task.model_dump_json()
        )
        + "\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = artifacts / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        environment_server_routes={"test:source": "environment"},
        route_failures_to_sidecar=True,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    if legacy:
        config = config.model_copy(
            update={"environment_routing_mode": "legacy", "environment_server_name": "environment"}
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as http:

        async def post(**kwargs):
            result = await http.post("/run", json=kwargs["json"])
            assert result.status_code == 200
            return FakeResponse(result.status_code, result.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {
                "environment": {
                    "environment_servers": {"single_agent_turn_legacy" if legacy else "single_agent_turn": {}}
                }
            }
        )
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
        from nemo_gym.rollout_recovery import IncompleteEvaluationError

        with pytest.raises(IncompleteEvaluationError):
            await RolloutCollectionHelper().run_from_config(config)
    store = RolloutStore.read(output)
    rows = store.failures()
    assert len(rows) == 1
    assert (
        rows[0]["_ng_failure_partial_response"]["metadata"][AGENT_OUTCOME_METADATA_KEY]
        == failed["metadata"][AGENT_OUTCOME_METADATA_KEY]
    )
    assert len(store.pending()) == 1
    raw = subprocess.check_output(["/usr/bin/python3", "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], text=True)
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw), tmp_path)
    decision = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert decision.action.value == "resubmit", decision.model_dump()
    assert decision.evidence[0].attempts == 1
    assert client.post.await_count == 1
    assert not any(path == "/verify" for _, path, _ in dependencies.calls)
