# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classification of sampling failures, keyed to errors observed in production.

Every string below was taken from a real OSWorld benchmark run (releases b7 to
b10p on lyris/polyphe, Gym a4471b59) rather than invented, so a future change
to the matcher is measured against what the stack actually emits.
"""

from __future__ import annotations

import pytest

from responses_api_agents.osworld_agent.adapter_agents import (
    MODEL_FAILURE_KINDS,
    MODEL_FAILURE_OUTCOMES,
    classify_model_failure,
)

# (observed error text, model_call_completed, expected kind)
OBSERVED_FAILURES = [
    # -- server rejected the request outright (vLLM 400) -------------------
    (
        "Error code: 400 - {'object': 'error', 'message': \"Input length (64782) exceeds "
        "model's maximum context length (64000).\", 'type': 'BadRequestError'}",
        False,
        "context_overflow",
    ),
    (
        "Input length (64151) exceeds model's maximum context length (64000).",
        False,
        "context_overflow",
    ),
    (
        "Error code: 400 - This model's maximum context length is 64000 tokens. However, "
        "you requested 64296 tokens. Please reduce the length of the messages.",
        False,
        "context_overflow",
    ),
    # -- transport gave out before a response existed (LB 502) -------------
    ("ClientOSError: [Errno 104] Connection reset by peer", False, "transport_error"),
    (
        "ClientOSError: [Errno None] Can not write request body for "
        "http://lyris0152:8000/v1/chat/completions",
        False,
        "transport_error",
    ),
    ("ServerDisconnectedError: Server disconnected", False, "transport_error"),
    ("ClientConnectorError: Cannot connect to host lyris0180:8000", False, "transport_error"),
    ("ClientOSError: [Errno 32] Broken pipe", False, "transport_error"),
    # -- sampler hit its token budget --------------------------------------
    (
        "Model response did not finish cleanly: finish_reason='length'",
        True,
        "output_truncated",
    ),
    # -- model returned nothing --------------------------------------------
    ("model response has no content", True, "empty_response"),
    # -- model stopped mid-construct; finish_reason was 'stop' -------------
    ("<Error>: unsupported or unterminated Code fence", True, "unparseable"),
    (
        "Invalid Python action: unterminated string literal (detected at line 1) (line 1, offset 54)",
        True,
        "unparseable",
    ),
    (
        "Invalid Python action: unterminated triple-quoted string literal "
        "(detected at line 1) (line 1, offset 53)",
        True,
        "unparseable",
    ),
    (
        "Invalid Python action: unexpected character after line continuation character "
        "(line 1, offset 22)",
        True,
        "unparseable",
    ),
    (
        "Invalid Python action: leading zeros in decimal integer literals are not permitted",
        True,
        "unparseable",
    ),
    ("<Error>: no explicit ## Code section found", True, "unparseable"),
]


@pytest.mark.parametrize(("text", "completed", "expected"), OBSERVED_FAILURES)
def test_observed_failures_are_classified(text: str, completed: bool, expected: str) -> None:
    assert classify_model_failure(ValueError(text), model_call_completed=completed) == expected


def test_every_kind_has_a_distinct_reported_outcome() -> None:
    """The whole point: a 4xx and a malformed sample must not share a name."""

    assert set(MODEL_FAILURE_OUTCOMES) == set(MODEL_FAILURE_KINDS)
    assert len(set(MODEL_FAILURE_OUTCOMES.values())) == len(MODEL_FAILURE_KINDS)


def test_the_four_shapes_that_used_to_collapse_into_one_bucket_now_differ() -> None:
    kinds = {
        classify_model_failure(ValueError(text), model_call_completed=completed)
        for text, completed, _ in OBSERVED_FAILURES
    }
    outcomes = {MODEL_FAILURE_OUTCOMES[kind] for kind in kinds}
    assert len(outcomes) == 5, "all five production failure shapes must remain distinguishable"


def test_a_rejected_request_is_context_overflow_even_if_the_call_completed() -> None:
    """Some transports surface a 4xx body instead of raising before the response."""

    assert (
        classify_model_failure(
            ValueError("maximum context length is 64000 tokens"), model_call_completed=True
        )
        == "context_overflow"
    )


def test_an_unknown_error_falls_back_to_unparseable_not_to_overflow() -> None:
    """Fail into the retryable bucket, never into the one that shrinks the prompt."""

    assert (
        classify_model_failure(ValueError("something nobody has seen"), model_call_completed=True)
        == "unparseable"
    )
    assert (
        classify_model_failure(ValueError("something nobody has seen"), model_call_completed=False)
        == "transport_error"
    )


def test_classification_reads_the_exception_type_as_well_as_its_message() -> None:
    assert (
        classify_model_failure(ConnectionError("policy endpoint unreachable"), model_call_completed=False)
        == "transport_error"
    )
    assert (
        classify_model_failure(TimeoutError(""), model_call_completed=False) == "transport_error"
    )
