# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from resources_servers.partner_if.app import VerifyRequest
from resources_servers.partner_if.grading_contract import GradingContract
from resources_servers.partner_if.llm import MAX_JUDGE_INPUT_CHARS
from resources_servers.partner_if.native_evidence import LiveAgentCapture
from resources_servers.partner_if.tests.test_app import CUTOFF, response, server


def shared_request():
    contract = GradingContract(
        task_id="smoke",
        origin="synthetic",
        source="authored",
        rubric_version="shared-v1",
        audit_signoff=True,
        data_kind="synthetic_smoke",
        rubrics=[
            {
                "id": "deterministic",
                "constraint": "The answer says HELLO.",
                "verifier": {"id": "text_contains_v1", "params": {"source": "answer", "literal": "HELLO"}},
            },
            {"id": "semantic", "constraint": "The answer is a greeting.", "verifier": {"id": "llm_judge_v1"}},
        ],
    )
    return VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Say HELLO."}]},
            "verifier_metadata": {"partner_if": contract.model_dump()},
            "response": response("HELLO").model_dump(),
        }
    )


async def test_normal_text_resource_dispatches_deterministic_and_llm_items_together():
    body = shared_request()
    judgment = {
        "items": [
            {
                "id": "semantic",
                "passed": True,
                "reason": "Greeting",
                "evidence": "HELLO",
                "references": [{"source": "answer", "start_line": 1, "end_line": 1}],
            }
        ]
    }
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(judgment)),
    ) as call:
        result = await server().verify(body)
    payload = json.loads(call.call_args.kwargs["json"]["input"][1]["content"])
    assert [r["id"] for r in payload["rubrics"]] == ["semantic"]
    assert result.grading_result.dimensions["if"].points_earned == 2
    assert result.grading_result.dimensions["if"].partial_score == 1
    assert result.grading_result.dimensions["if"].binary_score == 1
    assert result.reward == 1 and not result.mask_sample
    assert result.judge_protocol == "partner-if-evidence-v5.2"
    assert result.grading_result.items[1].quotes[0].quote == "HELLO"


async def test_invalid_shared_judge_output_preserves_receipts_and_remains_unscorable():
    body = shared_request()
    raw = response('{"items": "not a verdict"}')
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock, return_value=raw):
        result = await server().verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"
    assert result.judge_request is not None
    assert result.judge_response == raw
    assert result.grading_result.status == "unscorable"
    assert result.grading_result.dimensions["if"].partial_score is None


async def test_completion_protocol_error_keeps_independent_if_score_and_original_receipt():
    from resources_servers.partner_if.tests.test_shared_judge import verdict

    body = shared_request()
    contract = body.verifier_metadata.partner_if
    contract.rubrics.append(contract.rubrics[1].model_copy(update={"id": "content", "dimension": "completion"}))
    reply = verdict(id="semantic")
    reply["items"].append(reply["items"][0] | {"id": "content", "references": []})
    raw = response(json.dumps(reply))
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock, return_value=raw):
        result = await server().verify(body)
    assert not result.mask_sample and result.reward == 1
    assert result.if_result.passed == result.if_result.total == 2
    assert result.grading_result.dimensions["completion"].binary_score is None
    assert result.judge_response == raw
    assert "does not gate reward" in result.grading_result.items[-1].reason


@pytest.mark.parametrize("file_reply", [None, {"references": []}], ids=["omitted", "invalid"])
async def test_a_prerequisite_of_completion_alone_cannot_mask_the_if_verdicts(file_reply):
    from resources_servers.partner_if.tests.test_shared_judge import verdict

    body = shared_request()
    contract = body.verifier_metadata.partner_if
    # A judged prerequisite keeps the default IF dimension, but only the completion rubric depends on it.
    contract.prerequisites.append(contract.rubrics[1].model_copy(update={"id": "file"}))
    contract.rubrics.append(
        contract.rubrics[1].model_copy(update={"id": "content", "dimension": "completion", "depends_on": ["file"]})
    )
    reply = verdict(id="semantic")
    reply["items"].append(reply["items"][0] | {"id": "content"})
    if file_reply is not None:
        reply["items"].append(reply["items"][0] | {"id": "file"} | file_reply)
    raw = response(json.dumps(reply))
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock, return_value=raw):
        result = await server().verify(body)
    assert not result.mask_sample and result.reward == 1
    assert result.if_result.passed == result.if_result.total == 2
    prerequisite = next(item for item in result.grading_result.prerequisites if item.id == "file")
    assert prerequisite.raw_passed is None and "does not gate reward" in prerequisite.reason
    assert result.grading_result.dimensions["completion"].binary_score is None


async def test_judge_abstention_is_a_protocol_error_not_a_model_failure():
    body = shared_request()
    raw = response(
        json.dumps(
            {
                "items": [
                    {
                        "id": "semantic",
                        "passed": None,
                        "reason": "Needs review",
                        "evidence": "Ambiguous greeting",
                        "references": [],
                    }
                ]
            }
        )
    )
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock, return_value=raw):
        result = await server().verify(body)
    semantic = next(item for item in result.grading_result.items if item.id == "semantic")
    assert semantic.reason.startswith("Invalid shared IF judge response:")
    assert result.mask_sample and result.grading_result.status == "unscorable"
    assert result.grading_result.dimensions["if"].binary_score is None
    assert result.judge_response == raw


def live_request(**capture):
    body = shared_request()
    body.verifier_metadata.partner_if.evidence_mode = "live_agent"
    body.verifier_metadata.partner_if.rubrics[1].verifier.params["artifacts"] = ["submission"]
    body.verification_input = LiveAgentCapture.model_validate(
        {
            "input": [{"role": "user", "content": "Say HELLO.", "type": "message"}],
            "documents": {"skill": "Greet briefly."},
            "artifacts": {"submission": {"exists": True, "text": "HELLO"}},
        }
        | capture
    )
    return body


async def test_live_agent_row_is_graded_on_the_evidence_its_agent_captured():
    from resources_servers.partner_if.tests.test_shared_judge import verdict

    body = live_request()
    payload = body.model_dump(mode="json") | {"verification_input": body.verification_input.model_dump(mode="json")}
    reply = verdict(id="semantic", references=[{"source": "artifact:submission", "start_line": 1, "end_line": 1}])
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(reply)),
    ) as call:
        row = TestClient(server(accept_verification_input=True).setup_webserver()).post("/verify", json=payload).json()
    sources = json.loads(call.call_args.kwargs["json"]["input"][1]["content"])["evidence_sources"]
    assert sources["document:skill"]["lines"][0]["text"] == "Greet briefly."
    assert sources["artifact:submission"]["lines"][0]["text"] == "HELLO"
    assert (row["reward"], row["mask_sample"], row["if_result"]["status"]) == (1.0, False, "scored")
    assert "verification_input" not in row


@pytest.mark.parametrize(
    "body,reason",
    [
        (shared_request, "needs the evidence its agent captured"),
        (lambda: live_request(input=[{"role": "user", "content": "Say GOODBYE.", "type": "message"}]), "frozen row"),
    ],
    ids=["no_capture", "other_input"],
)
@pytest.mark.parametrize("cutoff", [False, True])
async def test_live_agent_row_without_matching_capture_is_masked(body, reason, cutoff):
    body = body()
    body.verifier_metadata.partner_if.evidence_mode = "live_agent"
    if cutoff:
        # A capture fault is masked even in a reply that the output-token cap cut off.
        body.response = response(**CUTOFF)
    result = await server(accept_verification_input=True).verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"
    assert result.failure_kind == "partner_if:unscorable" and reason in result.failure_reason


@pytest.mark.parametrize(
    "overrides,row",
    [
        # Harbor reports each run it grades as completed, including an OpenHands run stopped at its iteration cap.
        ({}, (1, False, None)),
        (CUTOFF, (0, False, "partner_if:truncated")),
        ({"status": "failed"}, (0, True, "partner_if:unscorable")),
        ({"status": None}, (0, True, "partner_if:unscorable")),
    ],
    ids=["completed", "cut_off", "failed", "no_status"],
)
async def test_a_live_agent_reply_is_graded_only_when_it_finished(overrides, row):
    from resources_servers.partner_if.tests.test_shared_judge import verdict

    body = live_request()
    body.response = response(**overrides)
    reply = verdict(id="semantic", references=[{"source": "artifact:submission", "start_line": 1, "end_line": 1}])
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(reply)),
    ) as call:
        result = await server(accept_verification_input=True).verify(body)
    assert (result.reward, result.mask_sample, result.failure_kind) == row
    assert call.called == (row[2] is None)


async def test_the_judge_input_holds_only_the_artifacts_the_judged_items_need():
    from resources_servers.partner_if.tests.test_shared_judge import verdict

    build_log = {"exists": True, "text": "ok\n" * 1000}
    body = live_request(artifacts={"submission": {"exists": True, "text": "HELLO"}, "build_log": build_log})
    reply = verdict(id="semantic", references=[{"source": "artifact:submission", "start_line": 1, "end_line": 1}])
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(reply)),
    ) as call:
        # The build log alone would put the judge's input over this limit.
        result = await server(accept_verification_input=True, judge_max_input_chars=5_000).verify(body)
    sources = json.loads(call.call_args.kwargs["json"]["input"][1]["content"])["evidence_sources"]
    assert "artifact:submission" in sources and "artifact:build_log" not in sources
    assert (result.reward, result.mask_sample) == (1, False)


@pytest.mark.parametrize(
    "limit,judged,row",
    [
        (MAX_JUDGE_INPUT_CHARS, [["semantic"]], (1, False, None)),
        # The IF item's own input is over this limit too, so nothing is sent.
        (500, [], (0, True, "partner_if:evidence_too_large")),
    ],
    ids=["if_items_fit", "if_items_too_large"],
)
async def test_a_completion_items_artifact_cannot_push_the_if_items_over_the_judge_limit(limit, judged, row):
    from resources_servers.partner_if.tests.test_shared_judge import verdict

    # Within the artifact limit, but its many lines put the shared judge input over the default limit.
    build_log = {"exists": True, "text": "ok\n" * 200_000}
    body = live_request(artifacts={"submission": {"exists": True, "text": "HELLO"}, "build_log": build_log})
    rubrics = body.verifier_metadata.partner_if.rubrics
    needs_log = rubrics[1].verifier.model_copy(update={"params": {"artifacts": ["build_log"]}})
    rubrics.append(rubrics[1].model_copy(update={"id": "built", "dimension": "completion", "verifier": needs_log}))
    reply = verdict(id="semantic", references=[{"source": "artifact:submission", "start_line": 1, "end_line": 1}])
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(reply)),
    ) as call:
        result = await server(accept_verification_input=True, judge_max_input_chars=limit).verify(body)
    sent = [json.loads(c.kwargs["json"]["input"][1]["content"]) for c in call.call_args_list]
    # Judged alone, the IF item is sent only its own artifact.
    assert [[r["id"] for r in payload["rubrics"]] for payload in sent] == judged
    assert not any("artifact:build_log" in payload["evidence_sources"] for payload in sent)
    assert (result.reward, result.mask_sample, result.failure_kind) == row
    built = next(item for item in result.grading_result.items if item.id == "built")
    assert (built.raw_passed, built.failure_kind) == (None, "partner_if:evidence_too_large")
    assert result.grading_result.dimensions["completion"].binary_score is None


@pytest.mark.parametrize("accept", [False, True])
def test_only_a_server_that_accepts_captures_grades_live_agent_rows_and_reverify_refuses_it(accept):
    body = live_request()
    payload = body.model_dump(mode="json") | {"verification_input": body.verification_input.model_dump(mode="json")}
    client = TestClient(server(accept_verification_input=accept).setup_webserver())
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps({"items": []})),
    ):
        status = client.post("/verify", json=payload).status_code
    # Rows never carry the capture, so `gym eval reverify` refuses a server that grades it without --force.
    assert (status, client.get("/reverify_mode").json()) == ((200, "unsupported") if accept else (422, "stateless"))
