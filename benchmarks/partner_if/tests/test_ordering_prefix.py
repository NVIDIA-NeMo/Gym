# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic verdicts here are fixtures; these tests validate host stopping and isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from starlette.requests import Request

from benchmarks.partner_if.replay_saved import audit_attempt
from benchmarks.partner_if.tests.test_replay import BUNDLE, PARAMS, SOURCE, TASK, call, draft, probe, response
from responses_api_agents.ngc_recorded_context import app as replay_app
from responses_api_agents.ngc_recorded_context.prefix import ordering_outcome
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, execute
from responses_api_agents.ngc_recorded_context.saved_prefix import prefix_digest


def fields(passed=True):
    return {
        "reward": float(passed),
        "mask_sample": False,
        "if_result": {"items": [{"id": "probe_before_prescription", "passed": passed}]},
    }


async def run_prefix(tmp_path, responses, verdicts, **overrides):
    queue, reviews, count = list(responses), list(verdicts), 0

    async def provider(payload, timeout):
        nonlocal count
        count += 1
        assert queue, "The runner must not request another response after a decisive event"
        return 200, queue.pop(0)

    async def review(output, captured, turn):
        assert captured.input == PARAMS["input"]
        assert json.loads(captured.artifacts["replay_receipts"]["text"])["evaluation_boundary"] == "ordering_prefix_v1"
        result = reviews.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    config = ReplayConfig(
        package_dir=SOURCE / "assets",
        artifacts_dir=tmp_path,
        evaluation_boundary="ordering_prefix_v1",
        **overrides,
    )
    result = await execute(config, PARAMS, model="test", secret="", request_api=provider, review_prefix=review)
    return result, count


@pytest.mark.asyncio
async def test_returned_exact_probe_stops_without_inventing_an_answer_and_audits_offline(tmp_path):
    (root, output, captured, diagnostics), count = await run_prefix(
        tmp_path / "capture", [response(probe())], [fields()]
    )
    assert count == 1 and diagnostics["termination"] == "ordering_passed"
    assert diagnostics["evaluation_completed"] and not diagnostics["answer_draft_or_terminal_captured"]
    assert captured.artifacts["final_reply"] == {"exists": False}
    assert all(item["type"] != "message" for item in output)
    assert json.loads((root / "drafts.json").read_text()) == {}
    audit = await audit_attempt(root, tmp_path / "offline", SOURCE / "assets")
    assert audit["calls_reconstructed"] == 1 and audit["new_actor_requests"] == 0
    assert audit["source_unchanged"] and not audit["judge_called"]


@pytest.mark.asyncio
async def test_premature_intermediate_advice_stops_before_another_policy_call(tmp_path):
    (root, _, _, diagnostics), count = await run_prefix(
        tmp_path,
        [
            response(
                call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}),
                content="Publish another artifact now.",
            )
        ],
        [fields(False)],
    )
    assert count == 1 and diagnostics["termination"] == "ordering_failed"
    assert diagnostics["evaluation_completed"] and not diagnostics["answer_draft_or_terminal_captured"]
    assert not (root / "call-02.request.json").exists()


@pytest.mark.asyncio
async def test_same_response_probe_and_prescription_fail_despite_successful_probe(tmp_path):
    (_, _, _, diagnostics), count = await run_prefix(tmp_path, [response(probe(), draft())], [fields(False)])
    assert count == 1 and diagnostics["new_sibling_probe_count"] == 1
    assert diagnostics["termination"] == "ordering_failed" and diagnostics["evaluation_completed"]


@pytest.mark.asyncio
async def test_safe_read_is_not_a_pass_and_the_next_decision_can_fail(tmp_path):
    (_, _, _, diagnostics), count = await run_prefix(
        tmp_path,
        [response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})), response(draft())],
        [fields(), fields(False)],
    )
    assert count == 2 and diagnostics["termination"] == "ordering_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["cutoff", "safe_final", "safe_draft"])
async def test_no_violation_without_required_check_is_inconclusive(tmp_path, end):
    raw = (
        response(content="I am investigating.")
        if end == "safe_final"
        else response(draft("I am investigating."))
        if end == "safe_draft"
        else response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}))
    )
    (_, _, _, diagnostics), count = await run_prefix(tmp_path, [raw], [fields()], max_new_turns=1)
    assert count == 1 and not diagnostics["evaluation_completed"]
    assert diagnostics["termination"] == ("turn_limit" if end == "cutoff" else "ordering_undecided")
    assert diagnostics["new_sibling_probe_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["judge_error", "masked", "inconsistent"])
async def test_judge_errors_stop_and_remain_unscored(tmp_path, fault):
    result = (
        RuntimeError("judge unavailable")
        if fault == "judge_error"
        else fields() | {"mask_sample": True}
        if fault == "masked"
        else fields() | {"reward": 0}
    )
    (root, _, _, diagnostics), count = await run_prefix(tmp_path, [response(probe())], [result])
    assert count == 1 and diagnostics["termination"] == "ordering_judge_error"
    assert not diagnostics["evaluation_completed"] and (root / "prefix-01.error.json").exists()


@pytest.mark.asyncio
async def test_fixture_gap_is_not_converted_to_a_scored_pass(tmp_path):
    (_, _, _, diagnostics), count = await run_prefix(
        tmp_path, [response(probe(), call("Read", {"file_path": "/missing"}, "missing"))], []
    )
    assert count == 1 and diagnostics["termination"] == "fixture_gap"
    assert not diagnostics["evaluation_completed"]


def test_true_verdict_requires_verified_result_not_historical_action_or_uri_mention():
    assert ordering_outcome(fields(), []) is None
    assert ordering_outcome(fields(), [{"kind": "sibling_probe", "result": json.dumps(BUNDLE["sibling"])}]) is None


@pytest.mark.asyncio
async def test_ordering_mode_refuses_to_run_without_a_verifier(tmp_path):
    with pytest.raises(ValueError, match="shared-verifier callback"):
        await execute(
            ReplayConfig(
                package_dir=SOURCE / "assets", artifacts_dir=tmp_path, evaluation_boundary="ordering_prefix_v1"
            ),
            PARAMS,
            model="test",
            secret="",
            request_api=None,
        )


def test_resume_cannot_silently_relabel_an_old_evaluation(tmp_path):
    config = ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir=tmp_path)
    old = config.model_dump()
    old.pop("evaluation_boundary")
    assert replay_app.resume_settings_changes(old, config) == {}
    with pytest.raises(ValueError, match="evaluation boundary"):
        replay_app.resume_settings_changes(
            old, config.model_copy(update={"evaluation_boundary": "ordering_prefix_v1"})
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_action", ["probe", "read"])
async def test_gym_saved_regrade_uses_shared_verifier_and_never_calls_policy_network(
    tmp_path, monkeypatch, saved_action
):
    async def original_provider(payload, timeout):
        return 200, response(
            probe()
            if saved_action == "probe"
            else call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})
        )

    original_config = ReplayConfig(
        package_dir=SOURCE / "assets",
        artifacts_dir=tmp_path / "original",
        max_new_turns=1,
        coverage_profile="expanded-v4",
    )
    original, _, _, _ = await execute(original_config, PARAMS, model="test", secret="", request_api=original_provider)
    before = {p: p.read_bytes() for p in original.rglob("*") if p.is_file()}
    posts = []

    async def post(**kwargs):
        posts.append(kwargs)
        assert kwargs["url_path"] == "/verify"
        assert kwargs["json"]["verification_input"]["artifacts"]["final_reply"] == {"exists": False}
        return kwargs["json"]

    async def success(result):
        pass

    async def decoded(result):
        return result | fields()

    async def forbidden_network(*args, **kwargs):
        pytest.fail("Saved regrading must never call the live actor")

    monkeypatch.setattr(replay_app, "raise_for_status", success)
    monkeypatch.setattr(replay_app, "get_response_json", decoded)
    monkeypatch.setattr("nemo_gym.client.http_client.request", forbidden_network)
    config = replay_app.ReplayAgentConfig.model_construct(
        num_workers=1,
        model="test",
        inference_api_key=SecretStr("prefix-test-token-unique-001"),
        resources_server=SimpleNamespace(name="partner_if"),
        replay=original_config.model_copy(
            update={
                "artifacts_dir": tmp_path / "regraded",
                "evaluation_boundary": "ordering_prefix_v1",
                "max_new_turns": 2,
            }
        ),
        regrade_saved_prefix=original,
        regrade_saved_prefix_sha256=prefix_digest(original),
    )
    agent = replay_app.ReplayAgent.model_construct(config=config, server_client=SimpleNamespace(post=post))
    result = await agent.run(
        Request({"type": "http", "headers": []}), replay_app.ReplayRunRequest.model_validate(TASK)
    )
    assert len(posts) == 1
    assert result.evaluation_completed is (saved_action == "probe")
    assert result.mask_sample is (saved_action != "probe")
    assert result.reward == float(saved_action == "probe")
    artifact = json.loads((Path(result.replay_artifacts_dir) / "saved-regrade.json").read_text())
    assert artifact["new_actor_requests"] == 0 and artifact["consumed_saved_policy_calls"] == 1
    assert all(p.read_bytes() == data for p, data in before.items())
    if saved_action == "probe":
        audit = await audit_attempt(
            Path(result.replay_artifacts_dir), tmp_path / "audit-app-normalized", SOURCE / "assets"
        )
        assert audit["source_unchanged"] and audit["calls_reconstructed"] == 1
        assert audit["new_actor_requests"] == 0 and not audit["judge_called"]
