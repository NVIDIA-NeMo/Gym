# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-handling conformance: the grading record, typed failures, and what a resumed run may change.

The grading tests run the real verify and call_judge; only the judges' HTTP transport is stubbed. The resume tests
classify EQ-Bench 4's own configuration with Gym's resume rules.
"""

import json as json_module
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi.responses import JSONResponse

import nemo_gym.rollouts.rollout_recovery as recovery
from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollouts.rollout_recovery import RunManifest, resume_drift_path_for, validate_resume
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification.judge import judge_failsafe
from resources_servers.eqbench4.verifier_fixture import fixture_verdict

from .test_app import MARKER, _policy_response, _reply, make_request, make_server


SERVER_DIR = Path(__file__).resolve().parents[1]


class JudgeTransport:
    """The judges' HTTP endpoint: each pairwise reading gets ``winner`` (five plus signs on rapport, attunement tied),
    or the next of ``script``; the trait judge (server ``traits``) gets ``trait_text``."""

    def __init__(self, *, winner: str = "policy", script: list[str] | None = None, trait_text: str = "") -> None:
        self.winner = winner
        self.script = list(script or [])
        self.trait_text = trait_text
        self.replies: list[tuple[str, str]] = []  # (server, response id)

    async def post(self, *, server_name: str, json: Any, **_: Any) -> MagicMock:
        prompt = json.input[-1].content
        response_id = f"judge-{len(self.replies)}"
        self.replies.append((server_name, response_id))
        if server_name == "traits":
            text = self.trait_text
        elif self.script:
            text = self.script.pop(0)
        else:
            verdict = fixture_verdict(prompt, winner=self.winner, policy_marker=MARKER, dimensions=["rapport"])
            text = json_module.dumps(json_module.loads(verdict) | {"attunement": "tie"})
        reply = MagicMock(ok=True)
        reply.read = AsyncMock(return_value=_reply(text, response_id=response_id).model_dump_json())
        return reply

    def pairwise_ids(self) -> list[str]:
        return [response_id for server, response_id in self.replies if server != "traits"]


async def verify_observed(server: Any, request: Any) -> tuple[Any, VerifierInvocation]:
    """Verify inside a verifier invocation, as the verify route does during a captured rollout."""
    invocation = VerifierInvocation(invocation_id="verify-eqbench4")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(request)
    except BaseException as error:
        result = error
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation


def judged_server(tmp_path: Path, transport: JudgeTransport, **overrides: Any) -> Any:
    server = make_server(tmp_path, **overrides)
    server.server_client.post = transport.post
    return server


# 1. A scored rollout carries the grading record of path A, bound to its response and reward.


async def test_a_judged_conversation_completes_its_grading_and_binds_the_reward(tmp_path: Path) -> None:
    transport = JudgeTransport(winner="policy", trait_text='{"analytical": 7}')
    server = judged_server(
        tmp_path,
        transport,
        trait_judge_model_server={"type": "responses_api_models", "name": "traits"},
    )
    result, invocation = await verify_observed(server, make_request())

    assert result.reward == pytest.approx(0.5 + 5 / 15 / 2)
    (operation,) = invocation.grading_operations
    assert (operation.kind, operation.outcome) == ("llm_judge", "completed")
    # Two opponents, each read in both orders. The trait judge is informational and is not a dependency.
    assert sorted(dependency.response_id for dependency in operation.dependencies) == sorted(transport.pairwise_ids())
    assert len(operation.dependencies) == 4
    assert [server for server, _ in transport.replies].count("traits") == 1
    assert operation.result_binding_sha256 == grading_result_binding(result)


async def test_a_changed_reward_or_response_no_longer_matches_the_binding(tmp_path: Path) -> None:
    result, invocation = await verify_observed(judged_server(tmp_path, JudgeTransport()), make_request())
    (operation,) = invocation.grading_operations
    assert operation.result_binding_sha256 == grading_result_binding(result)

    other_reward = result.model_copy(update={"reward": result.reward - 0.1})
    other_response = result.model_copy(
        update={"response": result.response.model_copy(update={"id": "another-policy-reply"})}
    )
    assert operation.result_binding_sha256 != grading_result_binding(other_reward)
    assert operation.result_binding_sha256 != grading_result_binding(other_response)


async def test_an_order_without_a_verdict_keeps_the_score_but_leaves_the_grading_unknown(tmp_path: Path) -> None:
    # The first reading has no verdict in its only attempt; the comparison rests on its other order.
    transport = JudgeTransport(script=["no verdict here"])
    result, invocation = await verify_observed(
        judged_server(tmp_path, transport, judge_max_attempts=1), make_request()
    )

    assert not result.mask_sample and result.evaluation_completed is True
    assert sum(len(comparison.unreadable_orders) for comparison in result.comparisons) == 1
    (operation,) = invocation.grading_operations
    assert operation.outcome == "unknown"
    assert len(operation.dependencies) == 4


async def test_a_reply_rejected_before_a_successful_retry_still_completes_the_grading(tmp_path: Path) -> None:
    transport = JudgeTransport(script=["no verdict here"])
    result, invocation = await verify_observed(
        judged_server(tmp_path, transport, judge_max_attempts=2), make_request()
    )

    assert sum(len(comparison.unreadable_orders) for comparison in result.comparisons) == 0
    (operation,) = invocation.grading_operations
    assert operation.outcome == "completed"
    # The rejected reply stays a dependency of the grade, beside the four accepted ones.
    assert len(operation.dependencies) == 5
    assert operation.result_binding_sha256 == grading_result_binding(result)


async def test_a_failed_judge_transport_is_a_failed_grading(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    server.server_client.post = AsyncMock(side_effect=ConnectionError("judge endpoint down"))
    error, invocation = await verify_observed(server, make_request())

    assert isinstance(error, BaseException)
    (operation,) = invocation.grading_operations
    assert operation.outcome == "failed" and operation.dependencies == []


# 2. Each characteristic failure is typed where it happens and keeps the policy's answer; a genuinely lost
# conversation stays an unmasked score.


async def test_a_lost_conversation_is_an_unmasked_low_score(tmp_path: Path) -> None:
    result, invocation = await verify_observed(
        judged_server(tmp_path, JudgeTransport(winner="opponent")), make_request()
    )

    assert result.reward == pytest.approx(0.5 - 5 / 15 / 2)
    assert (result.mask_sample, result.evaluation_completed, result.failure_kind) == (False, True, None)
    (operation,) = invocation.grading_operations
    assert operation.outcome == "completed"


@pytest.mark.parametrize(
    ("fields", "failure_kind"),
    [
        ({"conversation_status": "policy_failed", "policy_stop_reason": "context_limit"}, "model_context_limit"),
        (
            {"conversation_status": "policy_failed", "policy_stop_reason": "output_budget"},
            "eqbench4:policy_output_budget",
        ),
        ({"conversation_status": "persona_failed"}, "eqbench4:persona_failed"),
    ],
)
async def test_an_unfinished_conversation_is_typed_masked_and_keeps_the_answer(
    tmp_path: Path, fields: dict[str, Any], failure_kind: str
) -> None:
    transport = JudgeTransport()
    request = make_request(**fields, conversation_error="stopped")
    result, invocation = await verify_observed(judged_server(tmp_path, transport), request)

    assert (result.mask_sample, result.evaluation_completed, result.failure_kind) == (True, False, failure_kind)
    assert result.failure_reason == "stopped"
    # Nothing was judged, and the policy's answer and conversation are kept as sent.
    assert not transport.replies and not invocation.grading_operations
    assert result.response == request.response and result.transcript == request.transcript


async def test_a_judge_outage_is_a_masked_judge_failure_that_keeps_the_answer(tmp_path: Path) -> None:
    server = make_server(tmp_path, judge_max_attempts=1)
    server.server_client.post = AsyncMock(side_effect=ConnectionError("judge endpoint down"))
    request = make_request()
    reply = await judge_failsafe(server.verify)(body=request)

    assert isinstance(reply, JSONResponse)
    row = json_module.loads(reply.body)
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (0.0, True, "judge_failed")
    assert row["failure_reason"]
    # The saved answer is what a re-judge grades; the policy is not asked again.
    assert row["response"] == request.response.model_dump(mode="json") == _policy_response()
    assert row["transcript"] == [message.model_dump(mode="json") for message in request.transcript]


# 3. A resumed run may change how it runs, never what it measures.


@pytest.fixture(autouse=True)
def _fresh_digest_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registries of recorded digests are per process; each test starts with none, as a resuming job does."""
    monkeypatch.setattr(recovery, "_LEGACY_CONFIG_DIGESTS", {})
    monkeypatch.setattr(recovery, "_CONFIG_IDENTITIES", {})


def eqbench4_config() -> dict[str, Any]:
    """The EQ-Bench 4 server, agent, persona and judge configuration, with the policy `--model-type` adds."""
    config: dict[str, Any] = {
        "policy_model": {
            "responses_api_models": {"vllm_model": {"model": "the-policy", "base_url": "http://policy:8000/v1"}}
        }
    }
    for name in ("eqbench4.yaml", "eqbench4_models.yaml"):
        config |= yaml.safe_load((SERVER_DIR / "configs" / name).read_text())
    return config


def changed(config: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    out = json_module.loads(json_module.dumps(config))
    for dotted, value in change.items():
        *parents, leaf = dotted.split(".")
        node = out
        for key in parents:
            node = node.setdefault(key, {})
        if value is DROP:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return out


DROP = object()
SERVER = "eqbench4_resources_server.resources_servers.eqbench4"
AGENT = "eqbench4_agent.responses_api_agents.eqbench4_agent"
PERSONA = "eqbench4_persona_model.responses_api_models.vllm_model"
GPT_JUDGE = "eqbench4_gpt_judge_model.responses_api_models.vllm_model"


def saved_run(tmp_path: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    materialized = tmp_path / "materialized.jsonl"
    materialized.write_text("")
    path = tmp_path / "rollouts_manifest.json"
    RunManifest.create(source, [], {}, config).write(path)
    return path, materialized


@pytest.mark.parametrize(
    "drift",
    [
        {f"{SERVER}.judge_concurrency": 4},
        {f"{SERVER}.judge_timeout_secs": 600},
        {f"{SERVER}.judge_max_attempts": 5},
        {f"{AGENT}.policy_max_attempts": 3},
        {f"{PERSONA}.max_http_attempts": 12},
        # A setting saved runs did not have, now present with its default.
        {f"{SERVER}.judge_concurrency": DROP},
    ],
)
def test_operational_drift_resumes_and_is_recorded(tmp_path: Path, drift: dict[str, Any]) -> None:
    saved = changed(eqbench4_config(), {key: DROP for key, value in drift.items() if value is DROP})
    current = changed(eqbench4_config(), {key: value for key, value in drift.items() if value is not DROP})
    path, materialized = saved_run(tmp_path, saved)

    with pytest.warns(UserWarning):
        resumed = validate_resume(path, RunManifest.create(tmp_path / "input.jsonl", [], {}, current), materialized)

    assert resumed is not None and not resumed.identity_overridden
    (entry,) = [json_module.loads(line) for line in resume_drift_path_for(path).read_text().splitlines()]
    assert {key.removeprefix("servers.") for key in entry["operational_keys"]} == set(drift)


def test_a_new_judge_endpoint_or_key_is_not_drift(tmp_path: Path) -> None:
    # Where the judges are served and with which key is not part of the run's identity.
    path, materialized = saved_run(tmp_path, eqbench4_config())
    current = changed(
        eqbench4_config(),
        {f"{GPT_JUDGE}.base_url": "http://judges.internal/v1", f"{GPT_JUDGE}.api_key": "another-key"},
    )

    resumed = validate_resume(path, RunManifest.create(tmp_path / "input.jsonl", [], {}, current), materialized)

    assert resumed is not None and not resume_drift_path_for(path).exists()


@pytest.mark.parametrize(
    "drift",
    [
        {f"{GPT_JUDGE}.model": "another-judge"},
        {f"{GPT_JUDGE}.extra_body": {"reasoning_effort": "high"}},
        {f"{SERVER}.judge_responses_create_params.temperature": 0.7},
        {f"{SERVER}.trait_judge_model_server.name": "eqbench4_gpt_judge_model"},
        {f"{SERVER}.rating_bootstrap_resamples": 50},
        {f"{PERSONA}.model": "another-persona"},
        {f"{AGENT}.persona_responses_create_params.temperature": 1.0},
        {f"{AGENT}.persona_correction_message": "Try again."},
    ],
)
def test_material_drift_refuses_the_resume_and_names_the_setting(tmp_path: Path, drift: dict[str, Any]) -> None:
    path, materialized = saved_run(tmp_path, eqbench4_config())
    current = changed(eqbench4_config(), drift)

    with pytest.raises(ConfigError, match="semantic settings differ") as raised:
        validate_resume(path, RunManifest.create(tmp_path / "input.jsonl", [], {}, current), materialized)

    assert all(key in str(raised.value) for key in drift)
    assert not resume_drift_path_for(path).exists()
