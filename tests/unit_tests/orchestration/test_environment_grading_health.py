# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Taskset grading is bound to the environment's configured verifier, without an agent_ref on the row."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.core.episode_context import VERIFIER_INVOCATION_HEADER, current_rollout_id
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.orchestration.recovery import (
    _bound_artifact,
    decide_from_collected,
    grading_artifact,
    select_managed_evidence,
    trial_counts,
    trial_verdict,
)
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollouts.rollout_health import run_health_checks
from nemo_gym.server_runtime.verifier_observability import (
    observe_llm_judge_grading,
    observe_programmatic_grading,
    observe_verifier,
)
from nemo_gym.verification.judge import call_judge, record_parsed_judge_verdict
from tests.helpers.single_agent_turn import _agent_response, _Client, _environment_server, _request, _Response
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score
from tests.unit_tests.orchestration.test_grading_artifacts import _ungraded
from tests.unit_tests.orchestration.test_recovery_health_binding import row as raw_trial
from tests.unit_tests.orchestration.test_verifier_recorded_trial_coverage import _in_process, _judge_row
from tests.unit_tests.rollouts.test_model_calls import _capture_exchange
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, install_fake_server_client


def _gate(collected):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, report = trial_verdict(decision, collected)
    return passed, reason, report, trial_counts(decision, collected)


def _session_case(*, judged=False, case="healthy"):
    """Consumer-boundary controls, using the existing observed grading fixtures."""
    collected = _judge_row(captured=True, recorded=False) if judged else _in_process(case)
    servers = collected.quality["provenance"]["servers"]
    for row in collected.results:
        row.pop("agent_ref")
        row["_ng_result_type"] = "single_agent_turn"
        servers[row["_ng_environment_server"]] |= {
            "implementation": "single_agent_turn",
            "agent_server": "simple",
            "resources_server": "mcqa",
        }
    return collected


def _artifact_case(case="healthy"):
    """One registered metric artifact, with any remaining trials deliberately ungraded."""
    collected = _session_case(case=case)
    for row in collected.results:
        row["ng_trial_observations"]["verifier_invocations"][0].pop("grading_record")
        _ungraded(row)
    _ungraded(collected.results[0], sentence_metrics=["sentence_chrf"])
    return collected


@pytest.mark.parametrize("case", ["healthy", "repeats"])
def test_a_bound_public_artifact_keeps_its_count_when_another_trial_is_unobserved(case):
    collected = _artifact_case(case)
    assert grading_artifact(collected.results[0]) == "metric_record"
    passed, reason, report, counts = _gate(collected)
    if case == "healthy":
        assert passed is True and report["exact"] is True, reason
    else:
        assert passed is None and report["status"] == "UNVERIFIED", reason
    assert (counts["observed_healthy"], counts["unhealthy"], counts["unobserved"], counts["verifier_recorded"]) == (
        1,
        0,
        len(collected.results) - 1,
        0,
    )


@pytest.mark.parametrize(
    "change", ["malformed", "row_conflict", "missing_resources", "missing_agent", "missing_environment"]
)
def test_a_public_artifact_with_unresolved_ownership_is_not_counted_as_observed(change):
    collected = _artifact_case()
    row = collected.results[0]
    assert grading_artifact(row) == "metric_record"
    servers = collected.quality["provenance"]["servers"]
    environment = servers[row["_ng_environment_server"]]
    if change == "malformed":
        environment["bindings_malformed"] = True
    elif change == "row_conflict":
        servers["other_agent"] = dict(servers["simple"])
        row["agent_ref"] = {"name": "other_agent"}
    elif change == "missing_resources":
        del environment["resources_server"]
    elif change == "missing_agent":
        del environment["agent_server"]
    else:
        del servers[row["_ng_environment_server"]]
    passed, reason, report, counts = _gate(collected)
    assert passed is None and report["status"] == "UNVERIFIED", reason
    assert (counts["observed_healthy"], counts["unhealthy"], counts["unobserved"], counts["verifier_recorded"]) == (
        0,
        0,
        1,
        0,
    )


@pytest.mark.parametrize("artifact", [False, True])
def test_the_model_failure_fallback_applies_only_when_no_grading_artifact_exists(artifact):
    collected = _artifact_case()
    row = collected.results[0]
    if not artifact:
        _ungraded(row)
    row["reward"] = 0.0
    row["response"].update(status="incomplete", output=[{"type": "reasoning"}])
    collected.quality["provenance"]["servers"][row["_ng_environment_server"]]["bindings_malformed"] = True
    assert grading_artifact(row) == ("metric_record" if artifact else None)
    assert _bound_artifact(row, collected) is (not artifact)


def test_an_explicit_agent_ref_keeps_its_owner_when_fresh_provenance_has_no_primary_agent():
    collected = _in_process("healthy")
    row = collected.results[0]
    environment = collected.quality["provenance"]["servers"][row["_ng_environment_server"]]
    environment.update(agent_server=None, resources_server=None)
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert counts == {"observed_healthy": 1, "unhealthy": 0, "unobserved": 0, "verifier_recorded": 1}


@pytest.mark.parametrize("flag", [True, None, "false", 0])
def test_an_explicit_agent_ref_cannot_hide_malformed_environment_bindings(flag):
    collected = _in_process("healthy")
    row = collected.results[0]
    environment = collected.quality["provenance"]["servers"][row["_ng_environment_server"]]
    environment.update(agent_server=None, resources_server=None, bindings_malformed=flag)
    passed, reason, report, counts = _gate(collected)
    assert passed is None and report["status"] == "UNVERIFIED", reason
    assert counts["observed_healthy"] == 0 and counts["unobserved"] == 1


class _ObservedDependencies(_Client):
    """Replace dependency transport while executing the verifier and its judge grading."""

    reward: float = 1.0
    judged: bool = False
    programmatic: bool = False
    capture_dir: Path | None = None

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if server_name == "judge":
            reply = _agent_response().model_dump(mode="json")
            reply.update(id="judge-response", status="completed")
            reply["output"][0]["content"][0]["text"] = "correct" if self.reward else "incorrect"
            exchange = _capture_exchange("responses", server_name, {}, reply)
            exchange.update(
                request=kwargs["json"],
                verifier_invocation_id=kwargs["headers"][VERIFIER_INVOCATION_HEADER],
                failure_report={"events": {}, "omitted": False, "failure_id": None},
                model_operations=[
                    {
                        "operation_id": "judge-operation",
                        "outcome": "completed",
                        "final_attempt_id": "judge-attempt",
                        "failure_id": None,
                        "http_attempts": 1,
                        "transport_failures": 0,
                    }
                ],
            )
            rollout_id = current_rollout_id()
            assert rollout_id is not None and self.capture_dir is not None
            CaptureStore(self.capture_dir).record(rollout_id, exchange)
            return _Response(reply)
        if url_path != "/verify":
            return await super().post(server_name, url_path, **kwargs)

        async def grade():
            reply = await call_judge(
                self,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "Return correct or incorrect for this synthetic answer."},
                response_model=NeMoGymResponse,
            )
            verdict = reply.output[0].content[0].text
            assert verdict in {"correct", "incorrect"}
            record_parsed_judge_verdict()
            return kwargs["json"] | {"reward": float(verdict == "correct")}

        async def verify():
            if self.judged:
                return await observe_llm_judge_grading(grade)
            if self.programmatic:
                return observe_programmatic_grading(lambda: kwargs["json"] | {"reward": self.reward})
            return kwargs["json"] | {"reward": self.reward}

        result = await observe_verifier(verify, enabled=True, verifier_name=server_name)()
        return _Response(result)


async def _collect_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reward: float, empty=False, judged=False, opencode=False
):
    environment, dependencies = _environment_server(client_type=_ObservedDependencies)
    dependencies.reward = reward
    dependencies.judged = judged
    dependencies.programmatic = opencode and not judged
    dependencies.capture_dir = tmp_path / "captures"
    response = _agent_response().model_dump(mode="json")
    if empty:
        response |= {"status": "incomplete", "output": [{"type": "reasoning", "id": "reasoning", "summary": []}]}
    else:
        response["status"] = "completed"
    if opencode:
        from environment_servers.single_agent_turn.opencode import OpenCodeEnvironmentServer
        from responses_api_agents.opencode_multimodal_agent.rewards import (
            OpenCodeToolRewardConfig,
            score_opencode_tools,
        )

        rewards = OpenCodeToolRewardConfig()
        dependencies.global_config_dict.agent.responses_api_agents.test.tool_rewards = rewards.model_dump()
        response["output"][0]["content"][0]["text"] = "<think>private reasoning</think>answer"
        call = {"type": "function_call", "call_id": "bash-1", "name": "bash", "arguments": '{"command":"ls"}'}
        response["output"] = [
            call,
            {"type": "function_call_output", "call_id": "bash-1", "output": "file.txt"},
            *response["output"],
        ]
        response["opencode_evidence"] = {
            "turns_used": 1,
            "finished_naturally": True,
            "tool_reward": score_opencode_tools(
                {"bash-1": ("bash", "completed", 0)}, [call], 0.0, rewards
            ).model_dump(),
        }
        environment = OpenCodeEnvironmentServer(config=environment.config, server_client=dependencies)
    trajectory = raw_trial()["ng_trajectory"]
    trajectory["task_id"] = "task"
    trajectory["verifier_invocations"] = []
    invocation = trajectory["invocations"][0]
    invocation.update(invocation_id="root", in_session=True, verification_skipped=None)
    invocation["model_calls"][0]["response_id"] = response["id"]
    [call] = trajectory["model_calls"]
    call["response_metadata"]["response_id"] = response["id"]
    call["response"] = response
    [turn] = trajectory["turns"]
    turn.update(task_id="task", invocation_id="root", answer=response["output"])
    turn["model_calls"][0]["response_id"] = response["id"]
    dependencies.responses[2] = _Response(response)
    dependencies.responses[3] = _Response({"agent_session_id": "agent-session", "trajectory": trajectory})
    del dependencies.responses[4]  # /verify executes above instead of returning a prepared result.

    config = {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path / "captures"),
        "agent": {"responses_api_agents": {"test": {"resources_server": {"name": "resources"}}}},
        "resources": {"resources_servers": {"test": {}}},
        "environment": {
            "environment_servers": {
                "single_agent_turn": {
                    "agent_server": {"type": "responses_api_agents", "name": "agent"},
                    "resources_server": {"type": "resources_servers", "name": "resources"},
                }
            }
        },
    }
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source, output = tmp_path / "input.jsonl", artifacts / "rollouts.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    with TestClient(environment.setup_webserver()) as http:

        async def post(**kwargs):
            result = http.post("/run", json=kwargs["json"])
            assert result.status_code == 200, result.text
            return FakeResponse(result.status_code, result.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(config)
        monkeypatch.setattr(
            "nemo_gym.rollouts.rollout_collection.get_global_config_dict", lambda: client.global_config_dict
        )
        await RolloutCollectionHelper().run_from_config(
            RolloutCollectionConfig(
                input_jsonl_fpath=str(source),
                output_jsonl_fpath=str(output),
                environment_server_routes={"test:source": "environment"},
                route_failures_to_sidecar=True,
                disable_aggregation=True,
                disable_health_check=True,
            )
        )
    # Execute the real health producer and serialized collector against the persisted taskset result.
    OmegaConf.save(OmegaConf.create(config), artifacts / "rollouts_resolved_config.yaml")
    run_health_checks(output, workers=1)
    collected = select_managed_evidence(collect_with_score(tmp_path, "[]"), tmp_path)
    [saved] = [json.loads(line) for line in output.read_text().splitlines()]
    return saved, collected


@pytest.mark.parametrize("judged", [False, True])
@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_opencode_projection_and_shaping_preserve_observed_grading(tmp_path, monkeypatch, judged, reward):
    from copy import deepcopy

    from nemo_gym.trajectory import grading_result_binding
    from nemo_gym.trajectory.trajectory_record import grading_transform_source

    saved, collected = await _collect_session(tmp_path, monkeypatch, reward=reward, judged=judged, opencode=True)
    assert saved["reward"] == pytest.approx(reward + (0.1 if reward else 0.02))
    transform = saved["ng_grading_transform"]
    original = transform["input_result"]
    assert original["reward"] == reward
    assert original["response"]["output"][0]["content"][0]["text"] == "answer"
    assert saved["response"]["output"][-1]["content"][0]["text"] == "<think>private reasoning</think>answer"
    assert saved["response"]["output"][-1]["generation_token_ids"] == [3, 4]
    [invocation] = saved["ng_trajectory"]["verifier_invocations"]
    grade = invocation["grading_operations"][0]
    assert grade["result_binding_sha256"] == grading_result_binding(original)
    assert grade["result_binding_sha256"] != grading_result_binding(saved)
    assert grading_transform_source(saved) == collected.results[0]["_ng_collected_grading_transform"]
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert counts["observed_healthy"] == 1

    for field in ("response", "reward", "parameters", "input"):
        changed = deepcopy(saved)
        if field == "response":
            changed["response"]["output"][-1]["content"][0]["text"] = "different answer"
        elif field == "reward":
            changed["reward"] += 0.2
        elif field == "parameters":
            changed["ng_grading_transform"]["parameters"]["strip_reasoning_for_grading"] = False
        else:
            changed["ng_grading_transform"]["input_result"]["reward"] = 0.5
        assert grading_transform_source(changed) is None


@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_collected_taskset_score_is_observed_without_an_agent_ref(tmp_path, monkeypatch, reward):
    saved, collected = await _collect_session(tmp_path, monkeypatch, reward=reward)
    assert "agent_ref" not in saved and "agent_ref" not in collected.results[0]
    assert "ng_verifier_invocations" not in saved
    [verifier] = saved["ng_trajectory"]["verifier_invocations"]
    assert verifier["grading_record"]["verifier"] == "resources"
    assert verifier["grading_record"]["score"] == saved["reward"] == reward
    projected = collected.results[0]
    binding = projected["_ng_collected_grading_binding"]
    assert isinstance(binding, str) and len(binding) == 64
    assert (
        projected["ng_trial_observations"]["verifier_invocations"][0]["grading_record"]["result_binding_sha256"]
        == binding
    )
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert counts == {"observed_healthy": 1, "unhealthy": 0, "unobserved": 0, "verifier_recorded": 1}


@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_collected_taskset_judge_score_joins_its_owned_call_without_an_agent_ref(tmp_path, monkeypatch, reward):
    saved, collected = await _collect_session(tmp_path, monkeypatch, reward=reward, judged=True)
    assert "agent_ref" not in saved and "agent_ref" not in collected.results[0]
    assert saved["reward"] == reward
    [verifier] = saved["ng_trajectory"]["verifier_invocations"]
    assert "grading_record" not in verifier
    [operation] = verifier["grading_operations"]
    assert (operation["kind"], operation["outcome"]) == ("llm_judge", "completed")
    [dependency] = operation["dependencies"]
    assert dependency == {
        "model_call_id": None,
        "model_ref": {"type": "responses_api_models", "name": "judge"},
        "response_id": "judge-response",
    }
    [owned] = verifier["model_calls"]
    assert owned["model_call_id"] == "call-judge"
    [judge_call] = [call for call in saved["ng_trajectory"]["model_calls"] if call["model_call_id"] == "call-judge"]
    assert judge_call["response_metadata"]["response_id"] == dependency["response_id"]
    assert judge_call["response_metadata"]["model_ref"] == dependency["model_ref"]
    projected = collected.results[0]
    assert operation["result_binding_sha256"] == projected["_ng_collected_grading_binding"]
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert counts == {"observed_healthy": 1, "unhealthy": 0, "unobserved": 0, "verifier_recorded": 0}


@pytest.mark.parametrize(
    "change",
    [
        "verifier",
        "binding",
        "missing_environment",
        "missing_environment_resources",
        "unknown_environment_resources",
        "ambiguous_agent",
        "row_conflict",
    ],
)
def test_taskset_grading_without_a_bound_owner_or_result_stays_unverified(change):
    collected = _session_case()
    row = collected.results[0]
    servers = collected.quality["provenance"]["servers"]
    environment = servers[row["_ng_environment_server"]]
    record = row["ng_trial_observations"]["verifier_invocations"][0]["grading_record"]
    if change == "verifier":
        record["verifier"] = "another_resources_server"
    elif change == "binding":
        record["result_binding_sha256"] = "0" * 64
    elif change == "missing_environment":
        del servers[row["_ng_environment_server"]]
    elif change == "missing_environment_resources":
        del environment["resources_server"]
    elif change == "unknown_environment_resources":
        environment["resources_server"] = None
    elif change == "ambiguous_agent":
        environment["agent_server"] = None
    else:
        servers["other_agent"] = dict(servers["simple"])
        row["agent_ref"] = {"name": "other_agent"}
    passed, reason, report, counts = _gate(collected)
    assert passed is None and report["status"] == "UNVERIFIED", reason
    assert counts["observed_healthy"] == 0 and counts["unobserved"] == 1


@pytest.mark.parametrize(
    "change",
    ["binding", "missing_resources", "unknown_resources", "missing_agent", "ambiguous_agent", "row_conflict"],
)
def test_taskset_judge_without_a_bound_owner_or_result_is_not_counted_as_observed(change):
    collected = _session_case(judged=True)
    row = collected.results[0]
    servers = collected.quality["provenance"]["servers"]
    environment = servers[row["_ng_environment_server"]]
    if change == "binding":
        row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"][0]["result_binding_sha256"] = (
            "0" * 64
        )
    elif change == "missing_resources":
        del environment["resources_server"]
    elif change == "unknown_resources":
        environment["resources_server"] = None
    elif change == "missing_agent":
        del environment["agent_server"]
    elif change == "ambiguous_agent":
        environment["agent_server"] = None
    else:
        servers["other_agent"] = dict(servers["simple"])
        row["agent_ref"] = {"name": "other_agent"}
    passed, reason, report, counts = _gate(collected)
    assert passed is None and report["status"] == "UNVERIFIED", reason
    assert (counts["observed_healthy"], counts["unhealthy"], counts["unobserved"], counts["verifier_recorded"]) == (
        0,
        0,
        1,
        0,
    )


def test_taskset_verifier_is_owned_by_the_environment_resources_binding():
    collected = _session_case()
    row = collected.results[0]
    servers = collected.quality["provenance"]["servers"]
    servers["environment_resources"] = {"kind": "resources_servers", "implementation": "test"}
    servers[row["_ng_environment_server"]]["resources_server"] = "environment_resources"
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_record"]["verifier"] = "environment_resources"
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert counts["verifier_recorded"] == 1 and counts["unobserved"] == 0


async def test_shared_record_does_not_hide_a_collected_incomplete_empty_answer(tmp_path, monkeypatch):
    saved, collected = await _collect_session(tmp_path, monkeypatch, reward=0.0, empty=True)
    assert saved["response"]["status"] == "incomplete"
    assert saved["ng_trajectory"]["verifier_invocations"][0]["grading_record"]["score"] == 0.0
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {"empty_answer": 1}
    assert report["evaluation"]["count"] == 0
    assert counts == {"observed_healthy": 0, "unhealthy": 1, "unobserved": 0, "verifier_recorded": 0}
