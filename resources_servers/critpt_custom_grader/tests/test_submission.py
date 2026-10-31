# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission extraction from synthetic saved agent trajectories, without source execution."""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.core import AGENT_LIMIT, MODEL_REQUEST_ERROR, MODEL_TRANSPORT_ERROR, PROVIDER_UNAVAILABLE
from resources_servers.critpt_custom_grader.submission import (
    FinishRequest,
    Submission,
    extract_finish_source,
    finish_acknowledgement,
)


SOURCE = "def solve(value):\n    return value + 1"


def _finish(answer=SOURCE, *, call_id="finish-1"):
    arguments = {"reason": "done", "derivation": "Add one to the input.", "answer": answer}
    return [
        {
            "type": "function_call",
            "name": "finish",
            "arguments": json.dumps(arguments),
            "call_id": call_id,
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(
                {"protocol": "critpt-finish-v1", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest()}
            ),
        },
    ]


def _response(output=None, *, status="completed", metadata=None, **extra):
    return {
        "status": status,
        "output": [] if output is None else output,
        "metadata": metadata,
        **extra,
    }


def _outcome(kind, *, status_code=None):
    return {AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=kind, status_code=status_code).model_dump_json()}


def test_acknowledgement_binds_exact_answer_bytes_and_contains_no_answer_text():
    body = FinishRequest(reason="done", derivation="example", answer="abc")
    acknowledgement = finish_acknowledgement(body)
    assert json.loads(acknowledgement) == {
        "protocol": "critpt-finish-v1",
        "answer_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }
    assert finish_acknowledgement(body.model_copy(update={"reason": "checked"})) == acknowledgement
    assert finish_acknowledgement(body.model_copy(update={"answer": "abc\n"})) != acknowledgement


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": 1},
        {"derivation": None},
        {"answer": False},
        {"answer": [SOURCE]},
        {"answer": "\ud800"},
        {"extra": "ignored?"},
    ],
)
def test_finish_arguments_are_strict_and_do_not_ignore_extra_fields(changes):
    with pytest.raises(ValidationError):
        FinishRequest.model_validate({"reason": "done", "derivation": "checked", "answer": SOURCE, **changes})


def test_an_acknowledged_finish_provides_only_the_submitted_answer():
    output = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "wrong code"}]},
        {"type": "function_call", "name": "code_exec", "arguments": '{"cmd":"print(7)"}', "call_id": "run-1"},
        {"type": "function_call_output", "call_id": "run-1", "output": "7"},
        *_finish(),
    ]
    result = extract_finish_source(_response(output))
    assert result == Submission(source=SOURCE, category="candidate_ready", scored=False)
    with pytest.raises(FrozenInstanceError):
        result.source = "changed"


@pytest.mark.parametrize(
    "answer, source",
    [
        ("  " + SOURCE + "\n", SOURCE),
        ("Draft:\n```python\nwrong\n```\nFinal:\n```python\n" + SOURCE + "\n```", SOURCE),
        ("```python\nwrong\n```\n```\n" + SOURCE + "\n```", SOURCE),
        ("```python\n" + SOURCE, "```python\n" + SOURCE),
    ],
)
def test_only_the_last_python_or_bare_fence_in_the_finish_answer_is_extracted(answer, source):
    assert extract_finish_source(_response(_finish(answer))).source == source


@pytest.mark.parametrize("language", ["text", "json", "python3"])
@pytest.mark.parametrize("accepted_language", ["python", ""])
def test_unsupported_fence_before_the_answer_cannot_be_reinterpreted_as_a_bare_fence(language, accepted_language):
    answer = f"```{language}\ndraft\n```\n\n```{accepted_language}\n{SOURCE}\n```"
    assert extract_finish_source(_response(_finish(answer))) == Submission(SOURCE, "candidate_ready", False)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_unsupported_fences_between_or_after_python_answers_do_not_change_the_last_answer(newline):
    answer = (
        f"```python\nwrong\n```\n\n```text\ndraft\n```\n\n```python\n{SOURCE}\n```\n\n```json\n{{}}\n```"
    ).replace("\n", newline)
    assert extract_finish_source(_response(_finish(answer))) == Submission(
        SOURCE.replace("\n", newline), "candidate_ready", False
    )


def test_only_unsupported_fences_preserve_the_raw_answer_fallback():
    answer = "```text\ndraft\n```\n\n```json\n{}\n```"
    assert extract_finish_source(_response(_finish(answer))) == Submission(answer, "candidate_ready", False)


@pytest.mark.parametrize("answer", ["", " \n\t", "```python\n\n```"])
def test_an_acknowledged_empty_answer_is_a_scored_no_answer(answer):
    assert extract_finish_source(_response(_finish(answer))) == Submission(None, "candidate_no_answer", True)


def test_a_well_formed_call_without_a_tool_acknowledgement_is_not_a_submission():
    assert extract_finish_source(_response(_finish()[:1])) == Submission(None, "candidate_no_answer", True)


@pytest.mark.parametrize("reply", ["Answer submitted.", "Tool arguments are not valid", '{"detail":"rejected"}'])
def test_a_tool_reply_without_the_submission_protocol_does_not_acknowledge_an_answer(reply):
    output = _finish()
    output[1]["output"] = reply
    assert extract_finish_source(_response(output)) == Submission(None, "candidate_no_answer", True)


def test_a_rejected_finish_can_be_followed_by_one_successful_finish():
    rejected = _finish(call_id="rejected")
    rejected[0]["arguments"] = '{"reason":"done"}'
    rejected[1]["output"] = "Tool arguments are not valid"
    assert extract_finish_source(_response([*rejected, *_finish()])).source == SOURCE


@pytest.mark.parametrize(
    "arguments",
    [
        "{",
        "[]",
        '{"reason":"done","derivation":"checked","answer":1}',
        '{"reason":"done","derivation":"checked","answer":"first","answer":"second"}',
    ],
)
def test_a_claimed_acknowledgement_for_invalid_arguments_is_invalid_evidence(arguments):
    output = _finish()
    output[0]["arguments"] = arguments
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


@pytest.mark.parametrize(
    "acknowledgement",
    [
        '{"protocol":"critpt-finish-v1","answer_sha256":"wrong"}',
        '{"protocol":"critpt-finish-v1"}',
        '{"protocol":"critpt-finish-v1","protocol":"critpt-finish-v1","answer_sha256":"wrong"}',
    ],
)
def test_a_malformed_or_mismatched_acknowledgement_is_invalid_evidence(acknowledgement):
    output = _finish()
    output[1]["output"] = acknowledgement
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


def test_reordering_a_tool_result_before_its_call_does_not_make_a_submission():
    output = _finish()
    assert extract_finish_source(_response(output[::-1])) == Submission(None, "submission_invalid", False)


def test_an_acknowledgement_with_an_unmatched_call_id_is_invalid_evidence():
    output = _finish()
    output[1]["call_id"] = "another-call"
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


@pytest.mark.parametrize("duplicate_index", [0, 1])
def test_duplicate_call_or_result_ids_cannot_be_paired_by_last_value(duplicate_index):
    output = _finish()
    output.insert(duplicate_index, deepcopy(output[duplicate_index]))
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


@pytest.mark.parametrize("answer", [SOURCE, "def solve(value):\n    return 0"])
def test_multiple_acknowledged_submissions_are_ambiguous_even_if_their_answers_match(answer):
    output = [*_finish(), *_finish(answer, call_id="finish-2")]
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


def test_tool_activity_after_an_acknowledged_finish_invalidates_terminal_evidence():
    output = [
        *_finish(),
        {"type": "function_call", "name": "code_exec", "arguments": '{"cmd":"true"}', "call_id": "late"},
        {"type": "function_call_output", "call_id": "late", "output": ""},
    ]
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


def test_results_for_other_calls_in_the_finishing_response_are_allowed():
    call, result = _finish()
    output = [
        call,
        {"type": "function_call", "name": "code_exec", "arguments": '{"cmd":"true"}', "call_id": "check"},
        result,
        {"type": "function_call_output", "call_id": "check", "output": ""},
    ]
    assert extract_finish_source(_response(output)).source == SOURCE


@pytest.mark.parametrize("field", ["status", "output"])
def test_an_incomplete_acknowledgement_is_not_a_completed_submission(field):
    output = _finish()
    if field == "status":
        output[0]["status"] = "incomplete"
    else:
        output[1]["status"] = "incomplete"
    assert extract_finish_source(_response(output)) == Submission(None, "submission_invalid", False)


@pytest.mark.parametrize(
    "response",
    [
        _response(),
        _response([{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "No"}]}]),
        _response(status="incomplete", metadata={"stop_reason": "max_turns"}),
        _response(status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
        _response(status="failed", metadata={**_outcome(AGENT_LIMIT), "stop_reason": "max_turns"}),
    ],
)
def test_completed_no_answers_and_explicit_response_budget_exhaustion_score_zero(response):
    assert extract_finish_source(response) == Submission(None, "candidate_no_answer", True)


def test_explicit_content_filter_without_failure_metadata_is_a_scored_no_answer():
    response = _response(status="incomplete", incomplete_details={"reason": "content_filter"})
    assert extract_finish_source(response) == Submission(None, "candidate_no_answer", True)


@pytest.mark.parametrize(
    "metadata",
    [
        _outcome(MODEL_TRANSPORT_ERROR),
        _outcome(PROVIDER_UNAVAILABLE),
        _outcome(MODEL_REQUEST_ERROR, status_code=400),
        _outcome(MODEL_REQUEST_ERROR, status_code=422),
    ],
)
def test_content_filter_does_not_override_explicit_failure_metadata(metadata):
    response = _response(status="incomplete", metadata=metadata, incomplete_details={"reason": "content_filter"})
    assert extract_finish_source(response) == Submission(None, "response_incomplete", False)


@pytest.mark.parametrize(
    "response",
    [
        _response(status="incomplete"),
        _response(status="incomplete", metadata={"stop_reason": "cancelled"}),
        _response(status="incomplete", metadata={"stop_reason": "wall_time"}),
        _response(status="failed"),
        _response(status="failed", metadata=_outcome(AGENT_LIMIT)),
        _response(status="failed", metadata={**_outcome(AGENT_LIMIT), "stop_reason": "wall_time"}),
        _response(status="failed", metadata=_outcome(MODEL_TRANSPORT_ERROR)),
        _response(status="failed", metadata=_outcome(PROVIDER_UNAVAILABLE)),
        _response(status="failed", metadata=_outcome(MODEL_REQUEST_ERROR, status_code=400)),
        _response(status="failed", metadata=_outcome(MODEL_REQUEST_ERROR, status_code=422)),
        _response(metadata=_outcome(MODEL_TRANSPORT_ERROR)),
        _response(metadata={"stop_reason": "max_turns"}, error={"code": "server_error", "message": "unavailable"}),
    ],
)
def test_failures_and_unexplained_or_wall_time_incompletion_are_unscorable(response):
    assert extract_finish_source(response) == Submission(None, "response_incomplete", False)


@pytest.mark.parametrize("raw", ["{", "null", "{}", '{"failure_kind":"misspelled"}', {"failure_kind": "agent_limit"}])
def test_malformed_outcome_metadata_is_never_reinterpreted_as_no_answer(raw):
    response = _response(status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: raw})
    assert extract_finish_source(response) == Submission(None, "submission_invalid", False)


def test_a_provider_failure_overrides_an_earlier_acknowledged_answer():
    response = _response(_finish(), status="failed", metadata=_outcome(MODEL_TRANSPORT_ERROR))
    assert extract_finish_source(response) == Submission(None, "response_incomplete", False)


@pytest.mark.parametrize("stop_reason", ["cancelled", "wall_time"])
def test_interruption_evidence_overrides_an_earlier_acknowledged_answer(stop_reason):
    response = _response(_finish(), metadata={"stop_reason": stop_reason})
    assert extract_finish_source(response) == Submission(None, "response_incomplete", False)


def test_a_limit_cannot_be_used_to_accept_a_contradictory_acknowledged_finish():
    response = _response(_finish(), status="incomplete", metadata={"stop_reason": "max_turns"})
    assert extract_finish_source(response) == Submission(None, "submission_invalid", False)
