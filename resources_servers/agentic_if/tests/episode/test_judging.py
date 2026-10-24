# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge calls of one grade: each reply's ID and token usage, or why a call got no reply, and their totals."""

import hashlib

from aiohttp import ClientConnectionError

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.agentic_if.episode.judging import JudgeLedger, JudgeUsage, judge_usage, vote_key
from resources_servers.agentic_if.if_grading.contracts import CheckResult, IFResult
from resources_servers.agentic_if.tests.support import JUDGE_USAGE, http_error, judge_reply


def failure(cause):
    try:
        raise JudgeError("the judge call failed") from cause
    except JudgeError as error:
        return error


def reply(response_id, usage=None):
    return NeMoGymResponse.model_validate(
        judge_reply("judge", '{"answer": "yes"}', response_id=response_id, usage=usage)
    )


def test_the_ledger_keeps_each_reply_s_id_and_usage_and_why_a_call_got_none():
    ledger = JudgeLedger()
    ledger.replied("first question", reply("resp-1", JUDGE_USAGE))
    ledger.replied("first question", reply("resp-2"))
    ledger.failed("second question", failure(http_error(503)))
    ledger.failed("third question", failure(ClientConnectionError("refused")))
    ledger.failed("fourth question", failure(ValueError("not a Responses object")))
    calls = ledger.calls()
    assert [(call.response_id, call.total_tokens, call.error) for call in calls] == [
        ("resp-1", 1500, None),
        ("resp-2", None, None),
        (None, None, "ClientResponseError (HTTP 503)"),
        (None, None, "ClientConnectionError"),
        (None, None, "ValueError"),
    ]
    assert calls[0].prompt_sha256 == calls[1].prompt_sha256 == hashlib.sha256(b"first question").hexdigest()
    assert (calls[0].input_tokens, calls[0].output_tokens, calls[0].reasoning_tokens) == (1200, 300, 120)
    # A verify result reports the reply ID, the model that answered and the tokens only.
    assert calls[0].summary().model_dump() == {
        "response_id": "resp-1",
        "model": "judge",
        "input_tokens": 1200,
        "output_tokens": 300,
        "reasoning_tokens": 120,
        "total_tokens": 1500,
    }
    assert judge_usage(calls) == JudgeUsage(
        calls=5,
        failed_calls=3,
        calls_without_usage=1,
        input_tokens=1200,
        output_tokens=300,
        reasoning_tokens=120,
        total_tokens=1500,
    )
    assert judge_usage([]) == JudgeUsage()


def errored(question):
    """A check whose one vote on ``question`` got no verdict, as ``if_grading`` reports it."""
    vote = {
        "status": "error",
        "explanation": "No judge verdict after 1 attempt(s)",
        "metadata": {"vote": 0, "key": vote_key("judge-hash", question, 0)},
    }
    return CheckResult(
        id=question,
        status="error",
        explanation="No judge verdict",
        details={"units": [{"metadata": {"votes": [vote]}}]},
    )


def test_only_a_question_no_call_reached_makes_an_outage():
    ledger = JudgeLedger()
    # The judge answered, but its reply could not be read: a graded check error, not an outage.
    ledger.failed("unreadable", failure(ValueError("not a Responses object")))
    assert ledger.outage(IFResult(checks=[errored("unreadable")], if_all=None, if_frac=None), "judge-hash") is None
    ledger.failed("unreachable", failure(http_error(503)))
    both = IFResult(checks=[errored("unreadable"), errored("unreachable")], if_all=None, if_frac=None)
    assert ledger.outage(both, "judge-hash") == "check unreachable: No judge verdict after 1 attempt(s)"
