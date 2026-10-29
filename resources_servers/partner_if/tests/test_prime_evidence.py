# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.partner_if.evidence import EvidenceUnavailable
from resources_servers.partner_if.prime_app import PrimeIFConfig, PrimeIFResourcesServer, delivered_report_evidence


def capture():
    return {
        "version": 1,
        "termination": "completed",
        "events": [
            {"type": "tool_execution_start", "toolName": "ipython", "toolCallId": "call-1"},
            {
                "type": "tool_execution_end",
                "toolName": "ipython",
                "toolCallId": "call-1",
                "result": {
                    "details": {
                        "status": "ok",
                        "sentAgentMessages": [
                            {"id": "receipt-1", "receiverRole": "parent", "deliveryStatus": "delivered"}
                        ],
                    }
                },
            },
        ],
        "deliveries": [
            {
                "id": "receipt-1",
                "source": "agent_message",
                "deliveryStatus": "delivered",
                "message": "Task-specific findings",
            }
        ],
    }


def test_delivery_is_corroborated_and_drafts_are_excluded():
    value = capture()
    value["events"].append(
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "UNSENT DRAFT"}]},
        }
    )
    bundle = delivered_report_evidence(value, [{"role": "user", "content": "Do the task"}])
    assert "verified_parent_send" in bundle.artifact_text("parent_send")
    assert "Task-specific findings" in bundle.artifact_text("parent_reports")
    assert "UNSENT DRAFT" not in bundle.artifact_text("parent_reports")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(deliveries=[]),
        lambda c: c["events"].pop(0),
        lambda c: c["events"][1].update(isError=True),
        lambda c: c["events"][1]["result"].update(isError=True),
        lambda c: c["events"][1]["result"]["details"].update(status="error"),
        lambda c: c["events"][1]["result"]["details"]["sentAgentMessages"][0].update(receiverRole="peer"),
        lambda c: c["events"][1]["result"]["details"]["sentAgentMessages"][0].update(deliveryStatus="failed"),
        lambda c: c["deliveries"][0].update(id="unrelated"),
        lambda c: c["deliveries"][0].update(source="assistant_text"),
    ],
)
def test_unverified_sends_do_not_count(mutation):
    value = capture()
    mutation(value)
    assert delivered_report_evidence(value, []).artifact_text("parent_send") is None


@pytest.mark.parametrize("termination", ["completed", "turn_limit"])
def test_no_delivery_with_complete_capture_is_valid_failure(termination):
    value = capture() | {"termination": termination, "events": [], "deliveries": []}
    assert delivered_report_evidence(value, []).artifact_text("parent_reports") is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(version=2),
        lambda c: c.pop("events"),
        lambda c: c.update(deliveries=None),
        lambda c: c.update(termination="error", events=[], deliveries=[]),
        lambda c: c.update(
            events=[{"type": "message_end", "message": {"role": "assistant", "stopReason": "error"}}], deliveries=[]
        ),
        lambda c: c["deliveries"][0].update(message=None),
    ],
)
def test_infrastructure_or_capture_gap_is_not_a_model_failure(mutation):
    value = capture()
    mutation(value)
    with pytest.raises(EvidenceUnavailable):
        delivered_report_evidence(value, [])


def test_later_api_error_does_not_undo_delivered_report():
    value = capture()
    value["events"].append({"type": "message_end", "message": {"role": "assistant", "stopReason": "error"}})
    assert delivered_report_evidence(value, []).artifact_text("parent_send")


def server(**overrides):
    return PrimeIFResourcesServer(
        config=PrimeIFConfig(
            name="prime_resources",
            host="127.0.0.1",
            port=8100,
            entrypoint="prime_app.py",
            sandbox_provider="sandbox",
            sandbox_config={},
            repository_url="https://example.org/repository.git",
            repository_commit=overrides.pop("repository_commit", "a" * 40),
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            **overrides,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed():
    return ResourcesSeedSessionRequest(
        resources_session_id="resource-session",
        episode_id=EpisodeId.from_capture_key("0-0"),
        task_id={"taskset": "test", "task_id": "one"},
        task_data={},
    )


@pytest.mark.asyncio
async def test_resource_seed_retry_close_and_fencing():
    instance, body = server(), seed()
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"id": "sandbox-one"}

    async def setup(spec, callback):
        await callback(sandbox)

    sandbox.start_with_setup.side_effect = setup
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    request = Request({"type": "http", "session": {}})
    with (
        patch("resources_servers.partner_if.prime_app.AsyncSandbox", return_value=sandbox),
        patch(
            "resources_servers.partner_if.prime_app.get_global_config_dict", return_value={"sandbox": {"local": {}}}
        ),
    ):
        first = await instance.seed_session(request, body)
        assert await instance.seed_session(request, body) == first
        sandbox.start_with_setup.assert_awaited_once()
        changed = body.model_copy(update={"task_data": {"changed": True}})
        with pytest.raises(HTTPException, match="different task"):
            await instance.seed_session(request, changed)
        close = ResourcesCloseSessionRequest(
            resources_session_id=body.resources_session_id, episode_id=body.episode_id
        )
        await instance.close_resources_session(close)
        await instance.close_resources_session(close)
        sandbox.stop.assert_awaited_once()
        with pytest.raises(HTTPException, match="closed"):
            await instance.seed_session(request, body)


@pytest.mark.asyncio
async def test_failed_stop_retains_handle_for_close_retry():
    instance, body = server(), seed()
    sandbox = AsyncMock()
    sandbox.stop.side_effect = [RuntimeError("transport"), None]
    instance._sandboxes[body.resources_session_id] = sandbox
    instance._seeds[body.resources_session_id] = body
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    with pytest.raises(RuntimeError):
        await instance.close_resources_session(close)
    assert body.resources_session_id in instance._sandboxes
    await instance.close_resources_session(close)
    assert body.resources_session_id not in instance._sandboxes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,expected_reward,expected_mask", [("delivery", 1, False), ("absent", 0, False), ("capture_error", 0, True)]
)
async def test_resource_verification_uses_shared_grading_and_masks_capture_errors(
    kind, expected_reward, expected_mask
):
    import json

    from resources_servers.partner_if.app import VerifyRequest
    from resources_servers.partner_if.tests.test_app import response

    instance = server()
    contract = {
        "task_id": "synthetic-parent-report",
        "origin": "synthetic",
        "source": "authored fixture",
        "rubric_version": "test-v1",
        "audit_signoff": True,
        "data_kind": "synthetic_smoke",
        "evidence_mode": "live_agent",
        "prerequisites": [
            {
                "id": "delivered",
                "constraint": "Send through IPython",
                "verifier": {
                    "id": "text_contains_v1",
                    "params": {"source": "artifact:parent_send", "literal": "verified_parent_send"},
                },
            }
        ],
        "rubrics": [
            {
                "id": "content",
                "constraint": "Report findings",
                "dimension": "if",
                "depends_on": ["delivered"],
                "verifier": {
                    "id": "text_contains_v1",
                    "params": {"source": "artifact:parent_reports", "literal": "Task-specific findings"},
                },
            }
        ],
    }
    body = VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Send your findings"}]},
            "response": response().model_dump(),
            "verifier_metadata": {"partner_if": contract},
        }
    )
    seeded = seed().model_copy(
        update={"task_data": {"verifier_metadata": body.verifier_metadata.model_dump(mode="json")}}
    )
    instance._seeds[seeded.resources_session_id] = seeded
    instance._sandboxes[seeded.resources_session_id] = AsyncMock()
    value = capture()
    if kind == "absent":
        value.update(events=[], deliveries=[])
    text = "not JSON" if kind == "capture_error" else json.dumps(value)
    with patch("resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=text)):
        result = await instance.verify(
            Request({"type": "http", "session": {"session_id": seeded.resources_session_id}}), body
        )
    assert result.reward == expected_reward
    assert result.mask_sample == expected_mask
    instance.server_client.post.assert_not_called()


@pytest.mark.parametrize(
    "overrides", [{"repository_commit": "main"}, {"capture_directory": "relative"}, {"num_workers": 2}]
)
def test_invalid_repository_or_process_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        server(**overrides)


@pytest.mark.asyncio
async def test_shutdown_stops_owned_sandboxes():
    instance = server()
    first, second = AsyncMock(), AsyncMock()
    instance._sandboxes.update(first=first, second=second)
    app = instance.setup_webserver()
    async with app.router.lifespan_context(app):
        pass
    first.stop.assert_awaited_once()
    second.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_repository_setup_stops_the_sandbox():
    instance, body = server(), seed()
    sandbox = AsyncMock()

    async def setup(spec, callback):
        await callback(sandbox)

    sandbox.start_with_setup.side_effect = setup
    sandbox.exec.return_value = SandboxExecResult(return_code=1, stderr="checkout failed", stdout="")
    with (
        patch("resources_servers.partner_if.prime_app.AsyncSandbox", return_value=sandbox),
        patch(
            "resources_servers.partner_if.prime_app.get_global_config_dict", return_value={"sandbox": {"local": {}}}
        ),
    ):
        with pytest.raises(RuntimeError, match="Repository setup"):
            await instance.seed_session(Request({"type": "http", "session": {}}), body)
    sandbox.stop.assert_awaited_once()
    assert body.resources_session_id not in instance._sandboxes


@pytest.mark.asyncio
async def test_failed_setup_handle_must_stop_before_replacement():
    instance, body = server(), seed()
    previous = AsyncMock()
    previous.stop.side_effect = RuntimeError("stop pending")
    instance._sandboxes[body.resources_session_id] = previous
    with patch("resources_servers.partner_if.prime_app.AsyncSandbox") as allocate:
        with pytest.raises(RuntimeError, match="stop pending"):
            await instance.seed_session(Request({"type": "http", "session": {}}), body)
        allocate.assert_not_called()
    assert instance._sandboxes[body.resources_session_id] is previous


@pytest.mark.asyncio
async def test_close_checks_episode_and_retirement_reclaims_the_sandbox():
    instance, body = server(), seed()
    sandbox = AsyncMock()
    instance._seeds[body.resources_session_id] = body
    instance._sandboxes[body.resources_session_id] = sandbox
    with pytest.raises(HTTPException, match="another episode"):
        await instance.close_resources_session(
            ResourcesCloseSessionRequest(
                resources_session_id=body.resources_session_id, episode_id=EpisodeId.from_capture_key("1-0")
            )
        )
    await instance.retire_session_state(body.resources_session_id)
    sandbox.stop.assert_awaited_once()
    await instance.retire_session_state(body.resources_session_id)


@pytest.mark.parametrize(
    "value",
    [None, [], {"version": 1, "events": [42], "deliveries": []}, {"version": 1, "events": [], "deliveries": [None]}],
)
def test_malformed_capture_shapes_are_infrastructure_gaps(value):
    with pytest.raises(EvidenceUnavailable):
        delivered_report_evidence(value, [])
