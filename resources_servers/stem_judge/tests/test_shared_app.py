# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared grading profiles consume ordinary rows and only the last boxed answer."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, mark

from resources_servers.stem_judge.contracts import hash_text
from resources_servers.stem_judge.tests.test_app import _msg, _request, _response, _split
from resources_servers.stem_judge.tests.test_contract_app import SERVER_DIR, contract, judge, server


def shared_request(answer: str = r"\boxed{2}", *, expected: str = "2", **overrides):
    response = _split(answer)
    response.status = "completed"
    row = {
        "uuid": "q1",
        "question": "What is one plus one?",
        "metadata": {
            "domain": "physics",
            "source": "synthetic_source",
            "reference_sha256": hash_text(expected),
            "version_sha256": "synthetic-selected-version",
        },
        **overrides,
    }
    return _request(response, expected=expected, **row)


@mark.asyncio
async def test_semantic_profile_accepts_row_without_per_row_contract(monkeypatch):
    call, _ = judge(monkeypatch)
    body = shared_request()
    assert body.task_contract is None
    result = await server(grading_profile="semantic").verify(body)
    assert result.verdict == "correct" and result.correctness is True
    assert result.evaluation_completed and not result.mask_sample
    call.assert_awaited_once()


@mark.asyncio
@mark.parametrize("verdict", ["correct", "incorrect", "unresolved"])
async def test_completed_judge_generation_preserves_correctness_and_masking(
    monkeypatch: MonkeyPatch, verdict: str
) -> None:
    _, response = judge(
        monkeypatch,
        text=json.dumps(
            {
                "verdict": verdict,
                "reason_code": "reference_conflict",
                "evidence": "Reference uses a different parameter.",
            }
            if verdict == "unresolved"
            else {"verdict": verdict, "reason_code": "reviewed", "evidence": "Compared the requested quantity."}
        ),
    )
    assert response.status == "completed"

    result = await server(grading_profile="semantic").verify(shared_request())

    unresolved = verdict == "unresolved"
    assert result.verdict == verdict
    assert result.correctness is (None if unresolved else verdict == "correct")
    assert result.reward == float(verdict == "correct")
    assert result.evaluation_completed is not unresolved
    assert result.mask_sample is unresolved
    if unresolved:
        assert result.grading_status == "judge_unresolved"
        assert "reference_conflict" in result.failure_reason
        assert "different parameter" in result.failure_reason
        assert result.model_dump()["instance_config"]["mask_sample"] is True
        assert result.model_dump()["_ng_failure_class"] == "judge_failed"
    else:
        assert result.failure_reason is None
        assert not result.model_dump().get("instance_config", {}).get("mask_sample", False)


@mark.asyncio
@mark.parametrize("template", [None, "shared_answer_v1.txt"])
async def test_shared_prompt_default_and_replay_override_are_recorded(
    monkeypatch: MonkeyPatch, template: str | None
) -> None:
    call, _ = judge(monkeypatch)
    path = SERVER_DIR / "prompt_templates" / (template or "shared_answer_v2.txt")
    overrides = {"shared_prompt_template_fpath": str(path)} if template else {}
    result = await server(grading_profile="semantic", **overrides).verify(shared_request())
    system = call.await_args.kwargs["json"].input[0].content

    assert system == path.read_text().strip()
    assert result.grading_provenance["verifier_revision"] == "stem-shared-v3"
    payload = call.await_args.kwargs["json"].input[-1].content
    assert result.grading_provenance["prompt_sha256"] == hash_text(system + "\n" + payload)


@mark.asyncio
@mark.parametrize("status, excluded", [("completed", False), ("incomplete", True)])
async def test_disabled_judge_deadline_preserves_transport_and_outcome(status: str, excluded: bool) -> None:
    resource = server(
        grading_profile="semantic",
        judge_endpoint_max_concurrency=None,
        judge_timeout_seconds=None,
        judge_max_attempts=1,
        judge_responses_create_params={"input": [], "max_output_tokens": 65536},
    )
    response = _response(_msg(json.dumps({"verdict": "correct", "reason_code": "equal", "evidence": "Same value."})))
    response.status = status
    http_response = MagicMock(ok=True, status=200)
    http_response.read = AsyncMock(return_value=response.model_dump_json().encode())
    resource.server_client.post = AsyncMock(return_value=http_response)

    result = await resource.verify(shared_request())

    resource.server_client.post.assert_awaited_once()
    request_kwargs = resource.server_client.post.await_args.kwargs
    assert request_kwargs["timeout"].total is None
    assert request_kwargs["json"].max_output_tokens == 65536
    assert result.grading_provenance["judge_timeout_seconds"] is None
    assert result.mask_sample is excluded
    assert result.evaluation_completed is not excluded
    if excluded:
        assert result.verdict == "unresolved" and result.correctness is None
        assert result.grading_status == "judge_incomplete"
        assert result.failure_kind == "judge_failed"
    else:
        assert result.verdict == "correct" and result.correctness is True
        assert result.reward == 1


@mark.asyncio
async def test_semantic_judge_sees_only_last_box_even_if_earlier_value_matches(monkeypatch):
    call, _ = judge(monkeypatch, verdict="incorrect")
    result = await server(grading_profile="semantic").verify(shared_request(r"Work: \boxed{2}. Final: \boxed{3}"))
    payload = json.loads(call.await_args.kwargs["json"].input[-1].content)
    assert payload["candidate"] == "3"
    assert payload["reference"] == "2"
    assert result.verdict == "incorrect" and result.reward == 0
    assert result.evaluation_completed and not result.mask_sample


@mark.asyncio
@mark.parametrize("answer", ["2", r"\boxed{2} then \boxed{", r"\boxed{2} then \boxed{}", r"\boxed{2} then \boxed"])
async def test_missing_or_invalid_last_box_never_reuses_earlier_answer(monkeypatch, answer):
    call, _ = judge(monkeypatch)
    result = await server(grading_profile="semantic").verify(shared_request(answer))
    assert result.verdict == "incorrect" and result.reward == 0
    assert result.grading_status == "no_boxed"
    call.assert_not_awaited()


@mark.asyncio
@mark.parametrize(
    "answer, selected",
    [(r"\boxed {\frac{1}{2}}", r"\frac{1}{2}"), (r"\boxed{\{1,2\}}", r"\{1,2\}")],
)
async def test_shared_extraction_preserves_nested_and_escaped_braces(monkeypatch, answer, selected):
    call, _ = judge(monkeypatch)
    await server(grading_profile="semantic").verify(shared_request(answer))
    payload = json.loads(call.await_args.kwargs["json"].input[-1].content)
    assert payload["candidate"] == selected


@mark.asyncio
async def test_shared_judge_gets_original_question_and_effective_policy_input(monkeypatch):
    call, _ = judge(monkeypatch)
    body = shared_request(question="Original scientific question.")
    result = await server(grading_profile="semantic").verify(body)
    payload = json.loads(call.await_args.kwargs["json"].input[-1].content)
    assert payload["question"] == "Original scientific question."
    assert payload["policy_input"] == body.responses_create_params.model_dump(mode="json")
    assert result.grading_provenance["candidate_sha256"] == hash_text(r"\boxed{2}")
    assert result.grading_provenance["judged_candidate_sha256"] == hash_text("2")


@mark.asyncio
async def test_shared_domain_and_source_instructions_are_judge_system_instructions(monkeypatch):
    call, _ = judge(monkeypatch)
    resource = server(
        grading_profile="semantic",
        judge_domain_instructions={"physics": "Keep physical direction significant."},
        judge_source_instructions={"synthetic_source": "Preserve the source's stated option mapping."},
    )
    await resource.verify(shared_request())
    system = call.await_args.kwargs["json"].input[0].content
    assert "Keep physical direction significant." in system
    assert "Preserve the source's stated option mapping." in system


@mark.asyncio
async def test_shared_profile_rejects_per_row_contract_override(monkeypatch):
    call, _ = judge(monkeypatch)
    result = await server(grading_profile="semantic").verify(shared_request(task_contract=contract()))
    assert result.verdict == "unresolved" and result.mask_sample
    call.assert_not_awaited()


@mark.asyncio
async def test_shared_profile_does_not_grade_rubric_rows(monkeypatch):
    call, _ = judge(monkeypatch)
    result = await server(grading_profile="semantic").verify(
        shared_request(rubric=[{"id": "r1", "points": 1, "criterion": "Give a proof."}])
    )
    assert result.verdict == "unresolved" and result.mask_sample
    call.assert_not_awaited()


@mark.asyncio
@mark.parametrize("text", ["Judgement: yes", '{"verdict":"correct","verdict":"incorrect"}'])
async def test_shared_profile_keeps_malformed_judgment_unresolved(monkeypatch, text):
    judge(monkeypatch, text=text)
    result = await server(grading_profile="semantic").verify(shared_request())
    assert result.verdict == "unresolved" and result.correctness is None
    assert result.mask_sample and not result.evaluation_completed
    assert result.model_dump()["instance_config"]["mask_sample"]
    assert result.model_dump()["_ng_failure_class"] == "judge_failed"


@mark.asyncio
async def test_shared_profile_rejects_truncated_judge_response(monkeypatch):
    judge(monkeypatch, status="incomplete")
    result = await server(grading_profile="semantic").verify(shared_request())
    assert result.verdict == "unresolved" and result.grading_status == "judge_incomplete"
    assert result.mask_sample and not result.evaluation_completed


@mark.asyncio
async def test_math_profile_accepts_library_match_without_judge(monkeypatch):
    call, _ = judge(monkeypatch)
    resource = server(grading_profile="math")
    library = AsyncMock(return_value=True)
    resource._math_library = SimpleNamespace(verify=library)
    result = await resource.verify(shared_request(r"Intermediate \boxed{999}. Final \boxed{2}"))
    library.assert_awaited_once_with("2", "2")
    call.assert_not_awaited()
    assert result.verdict == "correct" and result.reward == 1
    assert result.evaluation_completed and not result.mask_sample


@mark.asyncio
async def test_math_profile_falls_back_on_same_selected_box(monkeypatch):
    call, _ = judge(monkeypatch, verdict="incorrect")
    resource = server(grading_profile="math")
    library = AsyncMock(return_value=False)
    resource._math_library = SimpleNamespace(verify=library)
    result = await resource.verify(shared_request(r"Intermediate \boxed{2}. Final \boxed{3}"))
    library.assert_awaited_once_with("2", "3")
    payload = json.loads(call.await_args.kwargs["json"].input[-1].content)
    assert payload["candidate"] == "3"
    assert result.verdict == "incorrect" and not result.mask_sample


@mark.asyncio
async def test_math_profile_missing_dependency_does_not_become_incorrect(monkeypatch):
    call, _ = judge(monkeypatch)
    resource = server(grading_profile="math")
    resource._math_library = SimpleNamespace(verify=AsyncMock(side_effect=ImportError("math-verify unavailable")))
    result = await resource.verify(shared_request())
    assert result.verdict == "unresolved" and result.mask_sample
    assert not result.evaluation_completed
    call.assert_not_awaited()


@mark.asyncio
@mark.parametrize("profile", ["math", "semantic"])
@mark.parametrize(
    "metadata",
    [
        {"review_group": "construction_or_rubric_review"},
        {"review_group": "competitive_coding_native"},
        {"review_group": "scientific_coding_native"},
        {"review_group": "noncoding_native_tests_unresolved"},
        {"review_group": "unknown_review_group"},
        {"reference_kind": "tests_only"},
        {"reference_kind": "native_grader"},
        {"declared_methods": ["EXACT_VALUE", "RUBRIC_ASSESSMENT"]},
        {"declared_methods": ["CONSTRUCTION_VALIDATION"]},
        {"declared_methods": ["NATIVE_CODE_TESTS"]},
    ],
)
async def test_shared_profile_rejects_known_unsupported_tasks_before_library_match(monkeypatch, profile, metadata):
    call, _ = judge(monkeypatch)
    resource = server(grading_profile=profile)
    library = AsyncMock(return_value=True)
    resource._math_library = SimpleNamespace(verify=library)
    body = shared_request()
    body.metadata.update(metadata)
    result = await resource.verify(body)
    assert result.verdict == "unresolved" and result.correctness is None
    assert result.grading_status == "unsupported_shared_grading_task"
    assert result.mask_sample and not result.evaluation_completed
    library.assert_not_awaited()
    call.assert_not_awaited()


@mark.asyncio
async def test_answer_comparison_metadata_keeps_math_profile_available(monkeypatch):
    call, _ = judge(monkeypatch)
    resource = server(grading_profile="math")
    library = AsyncMock(return_value=True)
    resource._math_library = SimpleNamespace(verify=library)
    body = shared_request()
    body.metadata.update(
        review_group="answer_comparison_review",
        reference_kind="answer",
        declared_methods=["EXACT_VALUE", "SYMBOLIC_EQUIVALENCE"],
        problem_sha256=hash_text(body.question),
    )
    result = await resource.verify(body)
    assert result.verdict == "correct" and not result.mask_sample
    library.assert_awaited_once_with("2", "2")
    call.assert_not_awaited()


@mark.asyncio
@mark.parametrize("profile", ["math", "semantic"])
async def test_shared_profile_rejects_mismatched_original_question_hash(monkeypatch, profile):
    call, _ = judge(monkeypatch)
    resource = server(grading_profile=profile)
    library = AsyncMock(return_value=True)
    resource._math_library = SimpleNamespace(verify=library)
    body = shared_request()
    body.metadata["problem_sha256"] = hash_text("A different question.")
    result = await resource.verify(body)
    assert result.verdict == "unresolved" and result.grading_status == "question_hash_mismatch"
    assert result.mask_sample and not result.evaluation_completed
    library.assert_not_awaited()
    call.assert_not_awaited()


@mark.asyncio
@mark.parametrize(
    "config_overrides, metadata",
    [
        ({"judge_domain_instructions": {"physics": "Require the requested physical direction."}}, {}),
        ({"judge_source_instructions": {"synthetic_source": "Require every requested labeled component."}}, {}),
        ({}, {"policy_overrides_applied": True}),
    ],
)
async def test_math_library_match_cannot_bypass_additional_instructions(monkeypatch, config_overrides, metadata):
    call, _ = judge(monkeypatch, verdict="incorrect")
    resource = server(grading_profile="math", **config_overrides)
    library = AsyncMock(return_value=True)
    resource._math_library = SimpleNamespace(verify=library)
    body = shared_request()
    body.metadata.update(metadata)
    result = await resource.verify(body)
    library.assert_awaited_once_with("2", "2")
    call.assert_awaited_once()
    assert result.grading_provenance["math_library_match"] is True
    assert result.verdict == "incorrect" and not result.mask_sample
