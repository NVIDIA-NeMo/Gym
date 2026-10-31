# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_recovery import IncompleteEvaluationError
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client


async def persist_producer_result(job, monkeypatch, result):
    """Feed an actual producer result through collection, close/reload, stdlib projection and installed reader."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True)
    source = job / "input.jsonl"
    source.write_text(json.dumps(failing_row(0)) + "\n")
    output = artifacts / "rollouts.jsonl"
    post = AsyncMock(side_effect=lambda **kwargs: FakeResponse(200, deepcopy(result)))
    client = install_fake_server_client(monkeypatch, post)
    monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        route_failures_to_sidecar=True,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    try:
        await RolloutCollectionHelper().run_from_config(config)
    except IncompleteEvaluationError:
        pass
    first_calls = post.await_count
    assert first_calls == 1
    store = RolloutStore.read(output)
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    selected = select_managed_evidence(collected, job)
    assert selected.managed_reader_error is None
    decision = decide_from_collected(selected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    if not store.pending():
        try:
            await RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": True}))
        except IncompleteEvaluationError:
            pass
        assert post.await_count == first_calls
    return decision, store, collected


def response(output=None):
    return NeMoGymResponse(
        id="resp_saved",
        object="response",
        created_at=0,
        model="controlled",
        output=[] if output is None else output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")


@pytest.mark.parametrize("slot", ["response", "_ng_failure_response", "_ng_failure_partial_response"])
def test_retained_generation_is_structural_not_placeholder_presence(slot):
    from nemo_gym.rollout_recovery import has_saved_generation

    assert has_saved_generation({slot: response()})
    for placeholder in (
        {},
        {"error": "failed"},
        {"output": []},
        {"status": "failed", "output": []},
        response() | {"status": "failed", "error": {"code": "server_error", "message": "diagnostic only"}},
    ):
        assert not has_saved_generation({slot: placeholder})


@pytest.mark.parametrize("saved,expected", [(True, "reverify"), (False, "stop")])
async def test_judge_saved_generation_is_not_regenerated(tmp_path, monkeypatch, saved, expected):
    result = {
        "_ng_failure_class": "judge_failed",
        "failure_kind": "judge_failed",
        "response": response() if saved else {},
    }
    decision, store, projected = await persist_producer_result(tmp_path, monkeypatch, result)
    assert decision.action.value == expected
    assert store.pending() == []
    assert decision.evidence[0].attempts == 1
    assert "resp_saved" not in json.dumps(projected.model_dump()) or saved


@pytest.mark.parametrize("kind", ["wrong", "final_zero"])
async def test_settled_zeros_are_not_replayed(tmp_path, monkeypatch, kind):
    result = {"response": response(), "reward": 0.0}
    if kind == "final_zero":
        result.update(
            _ng_failure_class="agent_run_error",
            failure_kind="model_context_limit",
            _ng_failure_terminal=True,
            _ng_failure_counts_as_zero=True,
        )
    decision, store, _ = await persist_producer_result(tmp_path, monkeypatch, result)
    assert decision.action.value == "noop"
    assert store.pending() == []


@pytest.mark.parametrize("empty_output", [False, True])
async def test_native_cleanup_partial_response_survives_collection(tmp_path, monkeypatch, empty_output):
    import httpx
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.tests.test_app import _environment_server, _request, _Response

    environment, dependencies = _environment_server()
    if empty_output:
        dependencies.responses[2] = _Response(response())
    dependencies.responses = dependencies.responses[:3] + [
        TimeoutError("controlled lost agent cleanup reply"),
        _Response({"resources_session_id": "resource"}),
    ]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as http:

        async def post(**kwargs):
            result = await http.post("/run", json=kwargs["json"])
            assert result.status_code == 200
            return FakeResponse(result.status_code, result.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {"environment": {"environment_servers": {"single_agent_turn": {}}}}
        )
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
        with pytest.raises(IncompleteEvaluationError):
            await RolloutCollectionHelper().run_from_config(config)
        store = RolloutStore.read(output)
        assert store.pending() == []
        failure = store.failures()[0]
        assert "_ng_failure_partial_response" in failure
        assert has_actual_generation(failure)
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        selected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), tmp_path)
        assert selected.managed_reader_error is None
        decision = decide_from_collected(selected, max_infra_retries=3, max_walltime_s=None, live_job=False)
        assert decision.action.value == "stop"
        assert decision.evidence[0].failure_id == failure["_ng_failure_id"]
        assert decision.evidence[0].attempts == 1
        assert decision.evidence[0].max_attempts == 3
        try:
            await RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": True}))
        except IncompleteEvaluationError:
            pass
        assert client.post.await_count == 1
        assert len([call for call in dependencies.calls if call[1].endswith("/v1/responses")]) == 1


def has_actual_generation(row):
    from nemo_gym.rollout_recovery import has_saved_generation

    return has_saved_generation(row)


@pytest.mark.parametrize("error_placeholder", [False, True])
async def test_confirmed_preanswer_provider_failure_keeps_its_budget(tmp_path, monkeypatch, error_placeholder):
    from tests.unit_tests.test_rollout_recovery import known_pre_answer_provider_failure

    result = known_pre_answer_provider_failure()
    if error_placeholder:
        result["response"] = response() | {
            "status": "failed",
            "error": {"code": "server_error", "message": "diagnostic only"},
        }
    decision, store, projected = await persist_producer_result(tmp_path, monkeypatch, result)
    assert decision.action.value == "resubmit"
    assert len(store.pending()) == 1
    assert (decision.evidence[0].attempts, decision.evidence[0].max_attempts) == (1, 3)
    assert decision.evidence[0].failure_id == "allocation-failure"
    assert store.coverage()["counted_failures"] == 1
    assert "diagnostic only" not in json.dumps(projected.model_dump())


async def test_current_unrouted_diagnostic_blocks_projected_older_retryable_outcome(tmp_path, monkeypatch):
    from nemo_gym.omitted_rollouts import omitted_rollout_record
    from tests.unit_tests.test_rollout_recovery import known_pre_answer_provider_failure

    before, reader, _ = await persist_producer_result(tmp_path, monkeypatch, known_pre_answer_provider_failure())
    assert before.action.value == "resubmit"
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    [original] = reader.selected("failure")
    with RolloutStore.start_or_resume(output, lambda: ([], reader.manifest), resume=True) as store:
        store.record_unstarted(
            omitted_rollout_record(original | {"_ng_failure_class": "agent_run_error"})
            | {"_ng_unrouted_run_error": True}
        )
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    selected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), tmp_path)
    assert selected.managed_reader_error is None
    decision = decide_from_collected(selected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.action.value == "stop"
    assert decision.evidence[0].attempts == 1
    assert RolloutStore.read(output).selected("failure") == [original]
    assert RolloutStore.read(output).pending() == []
