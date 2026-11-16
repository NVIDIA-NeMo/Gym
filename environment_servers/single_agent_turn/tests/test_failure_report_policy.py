# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy validation at the real environment boundary; its server client is a labelled protocol stub."""

import pytest

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core import (
    MODEL_POLICY_RESPONSE_TIMEOUT,
    FailureBoundary,
    NgFailureEvent,
    SandboxFailureDetails,
    failure_report_for,
)
from tests.helpers.single_agent_turn import _agent_response, _environment_server, _request, _Response


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "omitted",
        "wrong_id",
        "wrong_kind",
        "wrong_operation",
        "missing_upload",
        "connect",
        "mixed",
        "duplicate_attempt",
        "foreign_operation",
    ],
)
async def test_unproven_timeout_never_counts_as_zero_or_verifies(damage):
    environment, client = _environment_server()
    client.global_config_dict.agent.responses_api_agents.test.final_on_model_timeout = True
    model_event = {
        "event_id": "model-attempt",
        "boundary": "model",
        "failure_kind": "model_request_error",
        "failure_reason": MODEL_RESPONSE_TIMEOUT_REASON,
        "operation": "model_request",
        "operation_id": "model-operation",
        "attempt_id": "model-operation-1",
        "details": {"request_upload_complete": True},
    }
    agent_event = {
        "event_id": "agent-timeout",
        "boundary": "agent",
        "failure_kind": MODEL_POLICY_RESPONSE_TIMEOUT,
        "operation": "agent.policy_request",
        "operation_id": "model-operation",
        "cause_id": "model-attempt",
    }
    report = {"failure_id": "agent-timeout", "events": {"model-attempt": model_event, "agent-timeout": agent_event}}
    if damage == "omitted":
        report["omitted"] = True
    elif damage == "wrong_id":
        report["failure_id"] = "not-present"
    elif damage == "wrong_kind":
        agent_event["failure_kind"] = "model_request_error"
    elif damage == "wrong_operation":
        agent_event["operation"] = "unrelated"
    elif damage == "missing_upload":
        model_event.pop("details")
    elif damage == "connect":
        model_event["details"]["request_upload_complete"] = False
    elif damage == "mixed":
        report["events"]["previous"] = dict(
            model_event,
            event_id="previous",
            attempt_id="model-operation-0",
            failure_reason="connection failed",
            details={"request_upload_complete": False},
        )
    elif damage == "duplicate_attempt":
        report["events"]["previous"] = dict(model_event, event_id="previous")
    elif damage == "foreign_operation":
        model_event["operation_id"] = "other-operation"
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        output=[],
        error={"code": "server_error", "message": "scripted timeout"},
        metadata={
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT).model_dump_json()
        },
    )
    if damage != "missing":
        response["failure_report"] = report
    client.responses[2] = _Response(response)
    del client.responses[4]
    result = await environment.run_request(_request())
    assert result.result is None
    assert result.failure.terminal is True
    assert not any(path == "/verify" for _, path, _ in client.calls)
    assert result.failure.partial_response.model_dump(mode="json") == response
    assert result.failure.failure_kind is None


@pytest.mark.parametrize(
    "case,retryable",
    [
        ("connector", True),
        ("handshake_timeout", True),
        ("upload_only", False),
        ("wrong_type", False),
        ("read_timeout", False),
        ("unknown_upload", False),
        ("after_upload", False),
        ("earlier_uploaded", False),
        ("unidentified", False),
    ],
)
async def test_only_proven_unconnected_selected_operation_retains_retry(case, retryable):
    environment, client = _environment_server()
    event = {
        "event_id": "selected",
        "boundary": "model",
        "failure_kind": "model_request_error",
        "operation": "model_request",
        "operation_id": "operation",
        "attempt_id": "attempt-1",
        "details": {"request_upload_complete": False},
        "foreign_error": {"exception_type": "aiohttp.client_exceptions.ClientConnectorError", "message": ""},
    }
    events = {"selected": event}
    if case == "upload_only":
        event.pop("foreign_error")
    elif case == "handshake_timeout":
        event["foreign_error"]["exception_type"] = "aiohttp.client_exceptions.ConnectionTimeoutError"
    elif case == "read_timeout":
        event["foreign_error"]["exception_type"] = "aiohttp.client_exceptions.SocketTimeoutError"
    elif case == "wrong_type":
        event["foreign_error"]["exception_type"] = "TimeoutError"
    elif case == "unknown_upload":
        event.pop("details")
    elif case == "after_upload":
        event["details"]["request_upload_complete"] = True
    elif case == "earlier_uploaded":
        events["earlier"] = dict(
            event, event_id="earlier", attempt_id="attempt-0", details={"request_upload_complete": True}
        )
    elif case == "unidentified":
        event.pop("attempt_id")
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        output=[],
        error={"code": "server_error", "message": "scripted transport"},
        metadata={
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                failure_kind="model_request_error", status_code=500
            ).model_dump_json()
        },
        failure_report={"failure_id": "selected", "events": events},
    )
    client.responses[2] = _Response(response)
    del client.responses[4]
    result = await environment.run_request(_request())
    assert result.result is None and result.failure.terminal is not retryable
    assert result.failure.failure_kind == "model_request_error"
    actual = result.failure_report.events["selected"].model_dump(mode="json", exclude_none=True)
    assert {key: actual[key] for key in event} == event
    assert result.failure_report.events[result.failure_report.failure_id].cause_id == "selected"
    assert not any(path == "/verify" for _, path, _ in client.calls)


@pytest.mark.parametrize(
    "policy,action",
    [
        ({}, "count_as_zero"),
        ({"by_status_code": {401: "fail"}}, "count_as_zero"),
        ({"by_failure_kind": {"agent_invalid_output": "verify"}}, "count_as_zero"),
        ({"by_failure_kind": {"model_policy_response_timeout": "fail"}}, "fail"),
        ({"by_failure_kind": {"model_policy_response_timeout": "retry"}}, "retry"),
        ({"default_action": "fail"}, "fail"),
        ({"default_action": "retry"}, "retry"),
        (
            {
                "classification": {"by_failure_kind": {"model_policy_response_timeout": "evaluation"}},
                "by_failure_class": {"evaluation": "retry"},
                "default_action": "fail",
            },
            "retry",
        ),
    ],
)
@pytest.mark.parametrize("status", ["failed", "incomplete"])
async def test_only_matching_environment_selector_overrides_agent_timeout_rule(policy, action, status):
    environment, client = _environment_server()
    client.global_config_dict.agent.responses_api_agents.test.final_on_model_timeout = True
    environment.config.agent_outcome_policy = environment.config.agent_outcome_policy.model_validate(policy)
    model = {
        "event_id": "model",
        "boundary": "model",
        "failure_kind": "model_request_error",
        "failure_reason": MODEL_RESPONSE_TIMEOUT_REASON,
        "operation": "model_request",
        "operation_id": "operation",
        "attempt_id": "attempt",
        "details": {"request_upload_complete": True},
    }
    agent = {
        "event_id": "agent",
        "boundary": "agent",
        "failure_kind": MODEL_POLICY_RESPONSE_TIMEOUT,
        "operation": "agent.policy_request",
        "operation_id": "operation",
        "cause_id": "model",
    }
    response = _agent_response().model_dump(mode="json")
    response.update(
        status=status,
        output=[],
        error={"code": "server_error", "message": "scripted timeout"},
        metadata={
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT).model_dump_json()
        },
        failure_report={"failure_id": "agent", "events": {"agent": agent, "model": model}},
    )
    client.responses[2] = _Response(response)
    del client.responses[4]
    result = await environment.run_request(_request())
    assert not any(path == "/verify" for _, path, _ in client.calls)
    assert {"agent", "model"} <= result.failure_report.events.keys()
    if action == "count_as_zero":
        assert result.failure is None and result.result.reward == 0
        assert result.result.model_extra["_ng_failure_counts_as_zero"] is True
        assert result.failure_report.failure_id == "agent"
        # JSON/checkpoint roundtrip retains precisely the original selected operation.
        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored.failure_report == result.failure_report
        assert restored.result.response.model_dump(mode="json") == response
    else:
        assert result.result is None
        assert result.failure.terminal is (action == "fail")
        assert result.failure.failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT


async def test_report_without_selected_failure_follows_outcome_policy():
    # A plain provider 400 carries no rebuildable failure, so the agent reports nothing selected.
    environment, client = _environment_server()
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        output=[],
        error={"code": "server_error", "message": "scripted bad request"},
        metadata={
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                failure_kind="model_request_error", status_code=400
            ).model_dump_json()
        },
        failure_report={"failure_id": None, "events": {}},
    )
    client.responses[2] = _Response(response)
    result = await environment.run_request(_request())
    assert result.failure is None and result.result is not None
    (verify,) = [kwargs["json"] for _, path, kwargs in client.calls if path == "/verify"]
    assert verify["response"]["status"] == "failed"
    assert verify["response"]["failure_report"] == {"failure_id": None, "events": {}}


RETRYABLE_PROVIDER_KINDS = [
    "provider_unavailable",
    "provider_io_timeout",
    "provider_status_unavailable",
    "provider_command_outcome_unknown",
    "provider_protocol_error",
    "provider_request_failed",
    "provider_internal_error",
    "provider_pty_unavailable",
]


def _provider_failure_response(kind, *, http_status):
    """The failed response a sandbox agent returns: its own event caused by the provider's event, as the agent builds it."""
    provider = NgFailureEvent(
        boundary="sandbox",
        failure_kind=kind,
        failure_reason="Sandbox provider request failed",
        http_status=http_status,
        details=SandboxFailureDetails(provider="p", sandbox_id="s-1"),
        operation="sandbox.exec",
        stage="agent",
    )
    reported = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=kind,
        failure_reason=provider.failure_reason,
        operation="agent.run",
        stage="agent",
    )
    reported.__cause__ = provider
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        output=[],
        error={"code": "server_error", "message": provider.failure_reason},
        metadata={AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=kind).model_dump_json()},
        failure_report=failure_report_for(reported).model_dump(mode="json"),
    )
    return provider, response


@pytest.mark.parametrize("kind", RETRYABLE_PROVIDER_KINDS)
@pytest.mark.parametrize("http_status", [None, 404])
@pytest.mark.parametrize("configured", [None, "fail"])
async def test_provider_failure_from_an_agent_follows_the_configured_kind_rule_not_the_cause_status(
    kind, http_status, configured
):
    """Transient provider kinds retry the episode by default; a configured kind rule decides instead, and the cause stays."""
    environment, client = _environment_server()
    policy = {"by_failure_kind": {kind: configured}} if configured else {}
    terminal = configured == "fail"
    environment.config.agent_outcome_policy = environment.config.agent_outcome_policy.model_validate(policy)
    provider, response = _provider_failure_response(kind, http_status=http_status)
    client.responses[2] = _Response(response)
    del client.responses[4]
    result = await environment.run_request(_request())
    assert result.result is None and result.failure.terminal is terminal
    assert result.failure.failure_kind == kind
    kept = result.failure_report.events[provider.event_id]
    assert (kept.failure_kind, kept.http_status, kept.operation) == (kind, http_status, "sandbox.exec")
    assert kept.details == provider.details
    chain, event_id = [], result.failure_report.failure_id
    while event_id is not None:
        chain.append(event_id)
        event_id = result.failure_report.events[event_id].cause_id
    assert chain[-1] == provider.event_id and len(chain) >= 2
    assert not any(path == "/verify" for _, path, _ in client.calls)


@pytest.mark.parametrize("omitted", [False, True])
async def test_session_verification_forwards_canonical_observations_without_changing_success(omitted):
    from nemo_gym.core import (
        PROVIDER_IO_TIMEOUT,
        PROVIDER_OOM_KILLED,
        FailureReport,
        NgFailureEvent,
        RequestFailureLog,
        SandboxFailureDetails,
        observe_failure,
    )

    environment, client = _environment_server()
    cause = NgFailureEvent(boundary="sandbox", failure_kind=PROVIDER_IO_TIMEOUT, failure_reason="Read failed")
    failure = NgFailureEvent(
        boundary="sandbox",
        failure_kind=PROVIDER_OOM_KILLED,
        failure_reason="Current workload OOMKilled",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="sandbox-1"),
    )
    failure.__cause__ = cause
    with RequestFailureLog.capture() as log:
        if omitted:
            log.MAX_EVENTS = 1
        observe_failure(failure)
        result = await environment.run_request(_request())

    assert result.failure is None
    assert result.result.reward == 1.0
    [verify] = [kwargs["json"] for _, path, kwargs in client.calls if path == "/verify"]
    report = FailureReport.model_validate(verify["failure_report"])
    assert report.failure_id is None
    assert report.omitted is omitted
    if omitted:
        assert failure.event_id not in report.events
    else:
        assert report.events[failure.event_id].cause_id == cause.event_id
        assert report.events[failure.event_id].details.sandbox_id == "sandbox-1"


@pytest.mark.parametrize("omitted", [False, True])
async def test_checkpoint_restores_failure_reference_and_report_before_verification(omitted):
    from nemo_gym.core import (
        PROVIDER_OOM_KILLED,
        FailureReport,
        NgFailureEvent,
        RequestFailureLog,
        SandboxFailureDetails,
        observe_failure,
    )
    from nemo_gym.server_runtime import EnvironmentParticipant, EpisodeRecord, task_digest
    from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation

    failure = NgFailureEvent(
        boundary="sandbox",
        failure_kind=PROVIDER_OOM_KILLED,
        failure_reason="Current workload OOMKilled",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="sandbox-1"),
    )
    observation = SandboxObservation(role="agent", stage="agent", outcome="oom", failure_id=failure.event_id)
    with RequestFailureLog.capture() as original:
        observe_failure(failure)
        report = original.report.model_copy(update={"omitted": omitted})
    response = _agent_response().model_copy(
        update={
            "failure_report": report.model_dump(mode="json"),
            "metadata": {NG_AGENT_SANDBOX_OBSERVATION_KEY: observation.model_dump_json()},
        }
    )
    environment, client = _environment_server()
    request = _request()
    record = EpisodeRecord(
        episode_id=request.episode_id.model_copy(update={"attempt": request.episode_id.attempt - 1}),
        task_digest=task_digest(request.task.model_dump(mode="json")),
        boundary={
            "next": "verify",
            "handles": {"resources_session_id": "resources-session", "agent_session_id": "agent-session"},
            "response": response.model_dump(mode="json"),
        },
    )
    participant = EnvironmentParticipant(mode="exported")
    participant.restore_records([EpisodeRecord.model_validate_json(record.model_dump_json())])
    environment._checkpoint = participant
    del client.responses[:4]
    with RequestFailureLog.capture() as restored:
        assert not restored.report.events
        result = await environment.run_request(request)
        assert failure.event_id in restored.report.events
    assert result.failure is None
    [verify] = [kwargs["json"] for _, path, kwargs in client.calls if path == "/verify"]
    forwarded = FailureReport.model_validate(verify["failure_report"])
    reference = SandboxObservation.model_validate_json(
        verify["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY]
    )
    assert forwarded.failure_id is None
    assert forwarded.omitted is omitted
    assert reference.failure_id == failure.event_id
    assert forwarded.events[reference.failure_id].details.sandbox_id == "sandbox-1"
    assert [path for _, path, _ in client.calls] == ["/verify", "/close_session"]


@pytest.mark.parametrize("action", ["verify", "fail", "retry"])
async def test_confirmed_sandbox_oom_uses_environment_outcome_policy(action):
    from nemo_gym.agents import AgentOutcomePolicyConfig
    from nemo_gym.core import PROVIDER_OOM_KILLED, FailureReport, NgFailureEvent, SandboxFailureDetails
    from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation

    failure = NgFailureEvent(
        boundary="sandbox",
        failure_kind=PROVIDER_OOM_KILLED,
        failure_reason="Current workload OOMKilled",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="sandbox-1"),
    )
    observation = SandboxObservation(
        role="agent",
        stage="agent",
        outcome="oom",
        failure_id=failure.event_id,
    )
    report = FailureReport(events={failure.event_id: failure.event})
    response = _agent_response().model_copy(
        update={
            "status": "failed",
            "output": [],
            "failure_report": report.model_dump(mode="json"),
            "metadata": {
                AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=failure.failure_kind).model_dump_json(),
                NG_AGENT_SANDBOX_OBSERVATION_KEY: observation.model_dump_json(),
            },
        }
    )
    environment, client = _environment_server()
    if action != "verify":
        environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(
            by_failure_kind={PROVIDER_OOM_KILLED: action}
        )
        del client.responses[4]
    client.responses[2] = _Response(response.model_dump(mode="json"))
    result = await environment.run_request(_request())
    verified = any(path == "/verify" for _, path, _ in client.calls)
    assert verified is (action == "verify")
    if action == "verify":
        assert result.failure is None
        assert result.result.reward == 1.0  # The verifier's own verdict is retained.
    else:
        assert result.result is None
        assert result.failure.terminal is (action == "fail")
        partial = result.failure.partial_response
        assert partial.failure_report["events"][failure.event_id]["failure_kind"] == PROVIDER_OOM_KILLED
        saved = SandboxObservation.model_validate_json(partial.metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY])
        assert saved.failure_id == failure.event_id


@pytest.mark.parametrize("action,terminal", [("retry", False), ("fail", True)])
async def test_environment_dependency_config_controls_seed_failure(action, terminal):
    from nemo_gym.client import DependencyOutcomePolicyConfig

    environment, client = _environment_server()
    environment.config.dependency_outcome_policy = DependencyOutcomePolicyConfig(
        by_failure_kind={"transport_timeout": action}
    )
    client.fail_path = "/seed_session"
    result = await environment.run_request(_request())
    assert result.result is None
    assert result.failure.terminal is terminal
    assert result.failure.stage == "seed"
    assert result.failure_report.events
    assert not any(path.endswith("/v1/responses") for _, path, _ in client.calls)
