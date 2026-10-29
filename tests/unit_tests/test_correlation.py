# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The correlation prefix grammar: ``resolve`` reads it, ``encode`` builds it."""

import itertools

import pytest

from nemo_gym.correlation import (
    CORRELATION_STATE_KEY,
    CorrelationContext,
    CorrelationPathError,
    agent_kind_of,
    encode,
    parse_path,
    remember,
    resolve,
    validate_agent_instance,
)


TRACEPARENT = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


def _scope(path: str) -> dict:
    return {"type": "http", "path": path}


@pytest.mark.parametrize(
    "agent_instance, token_capture, trace_parent",
    list(itertools.product([None, "judge", "debater.1"], [False, True], [None, TRACEPARENT])),
)
def test_every_combination_of_segments_round_trips(agent_instance, token_capture, trace_parent):
    context = CorrelationContext(
        capture_key="7-0-a1", agent_instance=agent_instance, token_capture=token_capture, trace_parent=trace_parent
    )
    path = encode(context) + "/v1/responses"

    assert parse_path(path) == (context, "/v1/responses")
    assert encode(parse_path(path)[0]) + "/v1/responses" == path


def test_segments_appear_in_their_fixed_order():
    context = CorrelationContext(capture_key="k", agent_instance="user", token_capture=True, trace_parent=TRACEPARENT)

    assert encode(context) == f"/ng-rollout/k/ng-agent-instance/user/training-token-capture/ng-trace/{TRACEPARENT}"


def test_a_trace_only_path_carries_no_capture_key():
    context, rest = parse_path(f"/ng-trace/{TRACEPARENT}/v1/models")

    assert context == CorrelationContext(trace_parent=TRACEPARENT)
    assert rest == "/v1/models"
    assert encode(context) == f"/ng-trace/{TRACEPARENT}"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/responses",
        "/training-token-capture/control/rollouts/retire",
        "/ng-other/x/v1/responses",
        "/ng-rollout/k",
        "/",
        "",
    ],
)
def test_paths_without_a_prefix_or_without_a_rest_resolve_to_none(path):
    assert parse_path(path) is None


def test_a_bare_prefix_is_read_when_allowed():
    assert parse_path("/ng-rollout/k/ng-agent-instance/judge", allow_empty_rest=True) == (
        CorrelationContext(capture_key="k", agent_instance="judge"),
        "",
    )


@pytest.mark.parametrize(
    "path, reason",
    [
        ("/ng-rollout/k/training-token-capture/ng-agent-instance/judge/v1/responses", "out of order"),
        (f"/ng-rollout/k/ng-trace/{TRACEPARENT}/training-token-capture/v1/responses", "out of order"),
        (f"/ng-trace/{TRACEPARENT}/ng-agent-instance/judge/v1/responses", "out of order"),
        ("/ng-rollout/k/ng-agent-instance/judge/ng-agent-instance/user/v1/responses", "out of order"),
        ("/ng-rollout/k/ng-rollout/j/v1/responses", "out of order"),
        ("/ng-agent-instance/judge/v1/responses", "without a preceding capture key"),
        ("/ng-rollout//v1/responses", "without a value"),
        ("/ng-rollout/k/ng-agent-instance//v1/responses", "without a value"),
        ("/ng-rollout/k/ng-agent-instance", "without a value"),
        ("/ng-rollout/k/ng-agent-instance/v1/responses", "invalid agent instance"),
        ("/ng-rollout/k/ng-agent-instance/Judge/v1/responses", "invalid agent instance"),
        ("/ng-rollout/k/ng-agent-instance/judge.1.2/v1/responses", "invalid agent instance"),
        ("/ng-trace/not-a-traceparent/v1/responses", "invalid traceparent"),
    ],
)
def test_a_malformed_prefix_raises_one_typed_error(path, reason):
    with pytest.raises(CorrelationPathError, match=reason) as error:
        parse_path(path)
    assert error.value.path == path


@pytest.mark.parametrize("name", ["assistant", "judge", "user", "tool_simulation", "debater.1", "verify.3f2a", "a-b"])
def test_valid_agent_instance_names(name):
    assert validate_agent_instance(name) == name


@pytest.mark.parametrize(
    "name",
    ["", "Judge", "1judge", "judge.", ".1", "a.b.c", "judge 1", "judge/1", "ng-rollout", "ng-agent-instance"]
    + ["training-token-capture", "ng-trace", "v1"],
)
def test_invalid_and_reserved_agent_instance_names(name):
    with pytest.raises(ValueError, match="invalid agent instance name"):
        validate_agent_instance(name)


def test_the_agent_kind_is_the_part_before_the_dot():
    assert agent_kind_of("debater.1") == "debater"
    assert agent_kind_of("judge") == "judge"
    assert CorrelationContext(capture_key="k", agent_instance="verify.0a").agent_kind == "verify"
    assert CorrelationContext(capture_key="k").agent_kind is None


@pytest.mark.parametrize(
    "fields",
    [{"agent_instance": "judge"}, {"token_capture": True}, {"capture_key": ""}, {"capture_key": "a/b"}],
)
def test_contexts_that_cannot_be_encoded_are_refused(fields):
    with pytest.raises(ValueError):
        CorrelationContext(**fields)


def test_resolve_reads_a_path_once_and_shares_the_result_through_state(monkeypatch):
    import nemo_gym.correlation as correlation

    calls = []
    original = correlation.parse_path

    def counting(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(correlation, "parse_path", counting)
    scope = _scope("/ng-rollout/k/ng-agent-instance/judge/v1/responses")
    inner_scope = {**scope}

    first = resolve(scope)
    # An inner middleware's scope copy shares the state dict.
    inner_scope["state"] = scope["state"]
    second = resolve(inner_scope)

    assert first == second == (CorrelationContext(capture_key="k", agent_instance="judge"), "/v1/responses")
    assert calls == ["/ng-rollout/k/ng-agent-instance/judge/v1/responses"]


def test_resolve_reads_again_when_the_path_changed_without_remember():
    scope = _scope("/ng-rollout/k/v1/responses")
    resolve(scope)
    scope["path"] = "/ng-rollout/other/v1/responses"

    assert resolve(scope)[0].capture_key == "other"


def test_remember_stores_the_resolution_of_a_rewritten_path():
    scope = _scope(f"/ng-rollout/k/ng-trace/{TRACEPARENT}/v1/responses")
    resolve(scope)
    rewritten = {**scope, "path": "/ng-rollout/k/v1/responses"}
    remember(rewritten, (CorrelationContext(capture_key="k"), "/v1/responses"))

    assert scope["state"][CORRELATION_STATE_KEY].path == "/ng-rollout/k/v1/responses"
    assert resolve(rewritten) == (CorrelationContext(capture_key="k"), "/v1/responses")


def test_resolve_ignores_non_http_scopes():
    assert resolve({"type": "lifespan"}) is None


def test_the_checkpoint_admission_reader_still_finds_the_capture_key_before_a_name():
    from nemo_gym._checkpoint.model import _ROLLOUT_PREFIX

    match = _ROLLOUT_PREFIX.match("/ng-rollout/r-a1/ng-agent-instance/judge/training-token-capture/v1/responses")

    assert match is not None and match.group("capture_key") == "r-a1"
