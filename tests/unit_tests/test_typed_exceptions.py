# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed failures carry observed facts only: a registered kind where one fits, the call, and the hops."""

import pickle
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError
from multidict import CIMultiDict

from nemo_gym.episode_types import EpisodeFailure
from nemo_gym.judge import JudgeError, JudgeRateLimited
from nemo_gym.sandbox.providers.base import SandboxCreateError
from nemo_gym.typed_exceptions import (
    FAILURE_EVENT_HEADER,
    TypedException,
    encode_header,
    failure_path,
    find_typed_exception,
    innermost_http_status,
)


class _ProbeFailure(TypedException):
    failure_kind = "test:probe"


def test_a_kind_must_be_registered_or_namespaced() -> None:
    with pytest.raises(TypeError, match="neither registered nor namespaced"):
        type("Bad", (TypedException,), {"failure_kind": "judge_exploded"})
    assert type("Fine", (TypedException,), {"failure_kind": "judge_failed"}).failure_kind == "judge_failed"


def test_typed_failures_keep_the_types_their_handlers_catch() -> None:
    judge = JudgeRateLimited(message="ClientResponseError: 429", operation="judge.request")
    create = SandboxCreateError("no capacity")
    assert isinstance(judge, JudgeError) and isinstance(create, RuntimeError)
    assert judge.failure_kind == "judge_rate_limited" and create.failure_kind == "provider_unavailable"
    assert str(judge) == "ClientResponseError: 429" and create.args == ("no capacity",)
    with pytest.raises(ValueError, match="must not be empty"):
        _ProbeFailure(message="", operation="x")


def test_the_path_follows_the_exception_chain_by_type_never_by_message() -> None:
    try:
        try:
            raise ClientResponseError(request_info=SimpleNamespace(real_url="x"), history=(), status=429)
        except ClientResponseError as err:
            raise JudgeRateLimited(message="Connection refused (any text)", operation="judge.request") from err
    except JudgeRateLimited as error:
        path = failure_path(error)
    assert path == [
        {"operation": "judge.request", "exception_type": "nemo_gym.judge.JudgeRateLimited"},
        {"exception_type": "aiohttp.client_exceptions.ClientResponseError", "http_status": 429},
    ]


def test_an_inner_servers_path_continues_the_path() -> None:
    inner = EpisodeFailure(
        failure_reason="Unhandled ClientResponseError in judge_model",
        terminal=False,
        failure_path=[
            {"exception_type": "ClientResponseError", "http_status": 401},
            {"exception_type": "ExceptionGroup"},
        ],
    )
    reply = ClientResponseError(
        request_info=SimpleNamespace(real_url="judge-model"),
        history=(),
        status=500,
        headers=CIMultiDict({FAILURE_EVENT_HEADER: encode_header(inner)}),
    )
    assert failure_path(reply) == [
        {"exception_type": "aiohttp.client_exceptions.ClientResponseError", "http_status": 500},
        {"exception_type": "ClientResponseError", "http_status": 401},
        {"exception_type": "ExceptionGroup"},
    ]
    assert innermost_http_status(reply) == 401


def test_the_status_of_an_http_error_raised_while_handling_another_is_kept() -> None:
    try:
        try:
            raise ExceptionGroup("task group", [ValueError("x")])
        except ExceptionGroup:
            raise ClientResponseError(request_info=SimpleNamespace(real_url="provider"), history=(), status=401)
    except ClientResponseError as error:
        assert innermost_http_status(error) == 401
        assert failure_path(error)[-1] == {"exception_type": "ExceptionGroup"}


def test_typed_failures_survive_pickling_with_their_fields() -> None:
    failure = JudgeRateLimited(message="ClientResponseError: 429", operation="judge.request", stage="verification")
    copy = pickle.loads(pickle.dumps(failure))
    assert type(copy) is JudgeRateLimited and vars(copy) == vars(failure) and str(copy) == str(failure)
    create = pickle.loads(pickle.dumps(SandboxCreateError("no capacity")))
    assert create.args == ("no capacity",) and create.operation == "sandbox.create"


def test_a_failure_re_raised_without_its_fields_resolves_to_the_original() -> None:
    """Ray re-raises a task's failure as a derived class built without __init__, with the original on .cause."""
    original = SandboxCreateError("no capacity")
    derived = type("RayTaskError(SandboxCreateError)", (SandboxCreateError,), {})
    rewrapped = Exception.__new__(derived)
    rewrapped.cause = original
    assert find_typed_exception(rewrapped) is original
    assert failure_path(rewrapped)[0] == {
        "operation": "sandbox.create",
        "exception_type": "nemo_gym.sandbox.providers.base.SandboxCreateError",
    }
    assert find_typed_exception(Exception.__new__(derived)) is None  # no original: no fields to trust
    assert find_typed_exception(RuntimeError("plain")) is None and find_typed_exception(None) is None
