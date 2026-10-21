# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from resources_servers.partner_if.app import VerifyRequest
from resources_servers.partner_if.grading_contract import GradingContract
from resources_servers.partner_if.native_evidence import LiveAgentCapture
from resources_servers.partner_if.tests.test_app import response, server


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
    assert "Invalid completion judgment" in result.grading_result.items[-1].reason


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
async def test_live_agent_row_without_matching_capture_is_masked(body, reason):
    body = body()
    body.verifier_metadata.partner_if.evidence_mode = "live_agent"
    result = await server(accept_verification_input=True).verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"
    assert result.failure_kind == "partner_if:unscorable" and reason in result.failure_reason


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
