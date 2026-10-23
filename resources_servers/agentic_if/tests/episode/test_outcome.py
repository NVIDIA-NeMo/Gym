# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How an episode ended, and which failure its verify result reports: execution health stays apart from reward."""

import json

import pytest

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from resources_servers.agentic_if.episode.outcome import Verdict, classify, execution_status, setup_unverified
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.tests.support import finished_session, response


COMPLETE = SessionEvidence(status="complete", steps=[])
INCOMPLETE = SessionEvidence(status="incomplete", steps=[], issues=["unfinished_tool:root:a:c9"])
PASSED = {"reward": 1.0, "evaluation_completed": True}


@pytest.mark.parametrize(
    "status, finish, error, evidence, expected",
    [
        ("completed", "stop", None, COMPLETE, "completed"),
        ("completed", "length", None, COMPLETE, "length"),
        ("completed", "stop", {"name": "APIError"}, COMPLETE, "error"),
        ("completed", "stop", None, INCOMPLETE, "error"),
        ("incomplete", "length", None, COMPLETE, "length"),
        ("incomplete", "content-filter", None, COMPLETE, "completed"),
        ("incomplete", "stop", {"name": "ContextOverflowError"}, COMPLETE, "error"),
        ("incomplete", "tool-calls", None, INCOMPLETE, "timeout"),
        ("failed", "stop", None, COMPLETE, "error"),
    ],
)
def test_execution_status(status, finish, error, evidence, expected):
    root = finished_session(finish=finish, error=error)
    assert execution_status(response(status=status), root, evidence) == expected


def test_an_episode_without_a_root_export_is_an_error():
    assert execution_status(response(), None, COMPLETE) == "error"


def verdict(**options):
    arguments = {
        "response": response(),
        "root_export": finished_session(),
        "evidence": COMPLETE,
        "execution": "completed",
        "backend_result": PASSED,
        "backend_error": None,
        "collection_error": None,
    } | options
    return classify(**arguments)


def test_a_measured_episode_reports_the_task_reward_and_no_failure():
    assert verdict() == Verdict(reward=1.0, mask_sample=False, evaluation_completed=True)
    assert verdict(execution="length") == Verdict(reward=1.0, mask_sample=False, evaluation_completed=True)
    assert Verdict(reward=1.0, mask_sample=False, evaluation_completed=True).fields() == {
        "reward": 1.0,
        "mask_sample": False,
        "evaluation_completed": True,
    }


@pytest.mark.parametrize(
    "options, kind, completed",
    [
        ({"collection_error": "sandbox gone", "backend_error": "also failed"}, "session_lost", False),
        ({"backend_error": "ClientResponseError: 503"}, "verifier_error", False),
        ({"backend_result": None}, "verifier_error", False),
        ({"backend_result": {"evaluation_completed": True}}, "verifier_error", False),
        ({"backend_result": {"reward": 0.0, "evaluation_completed": False}}, "verifier_error", False),
        (
            {
                "backend_result": {
                    "reward": 0.0,
                    "mask_sample": True,
                    "failure_kind": "scale_swe:x",
                    "evaluation_completed": False,
                }
            },
            "scale_swe:x",
            False,
        ),
        ({"evidence": INCOMPLETE}, "agentic_if:evidence_incomplete", False),
        ({"execution": "error"}, "agent_run_error", True),
    ],
)
def test_an_unmeasured_episode_is_masked_with_its_kind_and_reason(options, kind, completed):
    result = verdict(**options)
    assert result.mask_sample is True and result.failure_kind == kind and result.failure_reason
    assert result.evaluation_completed is completed


def test_the_agent_s_reported_kind_names_a_harness_error():
    reported = response(
        status="failed",
        metadata={AGENT_OUTCOME_METADATA_KEY: json.dumps({"failure_kind": "agent_invalid_output"})},
        error={"code": "server_error", "message": "OpenCode output parse failed"},
    )
    result = verdict(response=reported, execution="error")
    assert (result.failure_kind, result.failure_reason) == ("agent_invalid_output", "OpenCode output parse failed")
    malformed = response(status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: "{"})
    assert verdict(response=malformed, execution="error").failure_kind == "agent_run_error"


@pytest.mark.parametrize("evidence", [COMPLETE, INCOMPLETE])
def test_limit_stops_are_scored_and_named_even_with_an_interrupted_tail(evidence):
    timeout = verdict(execution="timeout", evidence=evidence)
    assert (timeout.mask_sample, timeout.evaluation_completed, timeout.failure_kind) == (False, True, "agent_timeout")
    overflow = finished_session(error={"name": "ContextOverflowError", "data": {"message": "prompt too long"}})
    context = verdict(
        execution="error", root_export=overflow, evidence=evidence, response=response(status="incomplete")
    )
    assert (context.mask_sample, context.failure_kind, context.failure_reason) == (
        False,
        "model_context_limit",
        "prompt too long",
    )


@pytest.mark.parametrize(
    "issue",
    [
        "hook_receipt_error:missing live hook load receipt",
        "bundle_gap:boundary_load_receipt_missing",
        "boundary_system_mismatch",
        "system_prompt_not_presented:ses_root:a1",
        "literal_task_not_presented:ses_root",
    ],
)
def test_a_limit_stop_without_the_plugins_receipts_or_the_row_as_written_is_not_a_measurement(issue):
    unverified = SessionEvidence(status="incomplete", steps=[], issues=["unfinished_tool:root:a:c9", issue])
    assert setup_unverified(unverified) and not setup_unverified(INCOMPLETE)
    result = verdict(execution="timeout", evidence=unverified)
    assert result.mask_sample is True and result.failure_kind == "agentic_if:evidence_incomplete"
    overflow = finished_session(error={"name": "ContextOverflowError"})
    context = verdict(execution="error", root_export=overflow, evidence=unverified)
    assert context.mask_sample is True and context.failure_kind == "agentic_if:evidence_incomplete"
