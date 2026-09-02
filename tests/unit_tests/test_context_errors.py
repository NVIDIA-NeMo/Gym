# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nemo_gym.context_errors import error_text, is_context_overflow_error


@pytest.mark.parametrize(
    "value",
    [
        "This model's maximum context length is 8192 tokens",
        "max_completion_tokens cannot be greater than max_model_len",
        b'{"error":{"message":"prompt is too long"}}',
        {"message": "input exceeded the context window"},
    ],
)
def test_context_overflow_text_is_detected(value) -> None:
    assert is_context_overflow_error(value)


def test_exception_response_content_is_searched() -> None:
    exc = RuntimeError("400, message='Bad Request'")
    exc.response_content = b'{"error":{"message":"maximum context length is 512 tokens"}}'

    assert "maximum context length" in error_text(exc)
    assert is_context_overflow_error(exc)


def test_exception_body_is_searched() -> None:
    exc = SimpleNamespace(body={"message": "`inputs` tokens + `max_new_tokens` exceeded"})
    assert is_context_overflow_error(exc)


def test_unrelated_error_is_not_detected() -> None:
    assert not is_context_overflow_error(RuntimeError("connection refused"))
