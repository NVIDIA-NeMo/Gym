# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.partner_if.evidence import EvidenceUnavailable
from resources_servers.partner_if.prime_app import (
    TASK_PYTHON_DIRECTORY,
    TASK_REQUIREMENTS_PATH,
    PrimeIFConfig,
    PrimeIFResourcesServer,
    delivered_report_evidence,
)


LENDING = "nemo_gym.server_runtime.sandbox_lending"


@pytest.fixture(autouse=True)
def lending_config():
    with patch(f"{LENDING}.get_global_config_dict", return_value={"sandbox": {"local": {}}}):
        yield


def sandbox_for_test():
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"id": "sandbox-one"}
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    return sandbox


async def lend_for_test(instance, body, sandbox):
    """Seed through the real lifecycle without provisioning a remote provider."""
    with (
        patch(f"{LENDING}.AsyncSandbox", return_value=sandbox),
        patch(f"{LENDING}.get_global_config_dict", return_value={"sandbox": {"local": {}}}),
    ):
        return await instance.seed_session(Request({"type": "http", "session": {}}), body)


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
    sandbox = sandbox_for_test()
    request = Request({"type": "http", "session": {}})
    with patch(f"{LENDING}.AsyncSandbox", return_value=sandbox):
        first = await instance.seed_session(request, body)
        assert await instance.seed_session(request, body) == first
        sandbox.start.assert_awaited_once()
        assert sandbox.exec.await_count == 2
        sandbox.serialize.assert_awaited_with(scope="operate")
        assert first.sandbox_access.workdir == instance.config.workdir
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
    sandbox = sandbox_for_test()
    await lend_for_test(instance, body, sandbox)
    sandbox.stop.side_effect = [RuntimeError("transport"), None]
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    with pytest.raises(RuntimeError):
        await instance.close_resources_session(close)
    assert instance._sandboxes.get(body.resources_session_id) is sandbox
    assert instance.session_identity(body.resources_session_id) == (body.episode_id, body.task_id)
    await instance.close_resources_session(close)
    assert instance._sandboxes.get(body.resources_session_id) is None
    assert instance.session_identity(body.resources_session_id) is None
    assert not instance.has_session_state(body.resources_session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,expected_reward,expected_mask", [("delivery", 1, False), ("absent", 0, False), ("capture_error", 0, True)]
)
async def test_resource_verification_uses_shared_grading_and_masks_capture_errors(
    kind, expected_reward, expected_mask
):
    import json

    instance = server()
    body = verification_request()
    seeded = seed().model_copy(
        update={"task_data": {"verifier_metadata": body.verifier_metadata.model_dump(mode="json")}}
    )
    await lend_for_test(instance, seeded, sandbox_for_test())
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
    first, second = sandbox_for_test(), sandbox_for_test()
    for identifier, sandbox in (("first", first), ("second", second)):
        await lend_for_test(instance, seed().model_copy(update={"resources_session_id": identifier}), sandbox)
    app = instance.setup_webserver()
    async with app.router.lifespan_context(app):
        pass
    first.stop.assert_awaited_once()
    second.stop.assert_awaited_once()
    assert instance._sandboxes.get("first") is None and instance._sandboxes.get("second") is None
    with pytest.raises(HTTPException, match="shutting down"):
        await instance.seed_session(Request({"type": "http", "session": {}}), seed())


@pytest.mark.asyncio
async def test_failed_repository_setup_stops_the_sandbox():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    sandbox.exec.return_value = SandboxExecResult(return_code=1, stderr="checkout failed", stdout="")
    with patch(f"{LENDING}.AsyncSandbox", return_value=sandbox):
        with pytest.raises(RuntimeError, match="Repository setup"):
            await instance.seed_session(Request({"type": "http", "session": {}}), body)
    sandbox.stop.assert_awaited_once()
    assert instance._sandboxes.get(body.resources_session_id) is None
    assert instance.session_identity(body.resources_session_id) is None


@pytest.mark.asyncio
async def test_failed_setup_handle_must_stop_before_replacement():
    instance, body = server(), seed()
    previous = sandbox_for_test()
    previous.exec.return_value = SandboxExecResult(return_code=1, stderr="checkout failed", stdout="")
    previous.stop.side_effect = RuntimeError("stop pending")
    request = Request({"type": "http", "session": {}})
    with patch(f"{LENDING}.AsyncSandbox", return_value=previous) as allocate:
        with pytest.raises(RuntimeError, match="Repository setup"):
            await instance.seed_session(request, body)
        with pytest.raises(RuntimeError, match="stop pending"):
            await instance.seed_session(request, body)
        allocate.assert_called_once()
    assert previous.stop.await_count == 2
    assert instance._sandboxes.get(body.resources_session_id) is previous
    assert not instance.has_session_state(body.resources_session_id)
    assert instance.session_identity(body.resources_session_id) == (body.episode_id, body.task_id)


@pytest.mark.asyncio
async def test_partial_repository_is_not_reused_after_failed_cleanup():
    instance, body = server(), seed()
    first, replacement = sandbox_for_test(), sandbox_for_test()
    first.exec.return_value = SandboxExecResult(return_code=1, stderr="checkout failed", stdout="")
    first.stop.side_effect = [RuntimeError("stop pending"), None]
    request = Request({"type": "http", "session": {}})
    with patch(f"{LENDING}.AsyncSandbox", side_effect=[first, replacement]) as allocate:
        with pytest.raises(RuntimeError, match="Repository setup"):
            await instance.seed_session(request, body)
        assert instance._sandboxes.get(body.resources_session_id) is first
        assert not instance.has_session_state(body.resources_session_id)
        with pytest.raises(HTTPException, match="different task"):
            await instance.seed_session(request, body.model_copy(update={"task_data": {"other": True}}))
        await instance.seed_session(request, body)
        assert allocate.call_count == 2
    assert first.stop.await_count == 2
    assert instance._sandboxes.get(body.resources_session_id) is replacement
    assert replacement.exec.await_count == 2
    assert instance.has_session_state(body.resources_session_id)


@pytest.mark.asyncio
async def test_close_checks_episode_and_retirement_reclaims_the_sandbox():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    await lend_for_test(instance, body, sandbox)
    with pytest.raises(HTTPException, match="episode_id"):
        await instance.close_resources_session(
            ResourcesCloseSessionRequest(
                resources_session_id=body.resources_session_id, episode_id=EpisodeId.from_capture_key("1-0")
            )
        )
    await instance.retire_session_state(body.resources_session_id)
    sandbox.stop.assert_awaited_once()
    await instance.retire_session_state(body.resources_session_id)
    assert instance.session_identity(body.resources_session_id) is None
    assert not instance.has_session_state(body.resources_session_id)
    # The base retirement hook leaves restoration to the checkpoint router, without an extra close fence.
    replacement = sandbox_for_test()
    restored = body.model_copy(update={"episode_id": body.episode_id.model_copy(update={"attempt": 1})})
    await lend_for_test(instance, restored, replacement)
    assert instance.session_identity(body.resources_session_id) == (restored.episode_id, restored.task_id)
    replacement.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_seed_and_close_share_the_repository_lock():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def setup(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(proceed.wait(), timeout=3)
        return SandboxExecResult(return_code=0, stdout="", stderr="")

    sandbox.exec.side_effect = setup
    request = Request({"type": "http", "session": {}})
    with patch(f"{LENDING}.AsyncSandbox", return_value=sandbox):
        first = asyncio.create_task(instance.seed_session(request, body))
        await asyncio.wait_for(entered.wait(), timeout=3)
        repeated = asyncio.create_task(instance.seed_session(request, body))
        close = asyncio.create_task(
            instance.close_resources_session(
                ResourcesCloseSessionRequest(
                    resources_session_id=body.resources_session_id, episode_id=body.episode_id
                )
            )
        )
        await asyncio.sleep(0)
        sandbox.stop.assert_not_awaited()
        proceed.set()
        seeded, again, _ = await asyncio.wait_for(asyncio.gather(first, repeated, close), timeout=3)
    assert seeded == again
    sandbox.start.assert_awaited_once()
    assert sandbox.exec.await_count == 2
    sandbox.stop.assert_awaited_once()
    assert instance._sandboxes.get(body.resources_session_id) is None


@pytest.mark.asyncio
async def test_shutdown_waits_for_repository_setup_before_releasing():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def setup(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(proceed.wait(), timeout=3)
        return SandboxExecResult(return_code=0, stdout="", stderr="")

    sandbox.exec.side_effect = setup
    app = instance.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    with patch(f"{LENDING}.AsyncSandbox", return_value=sandbox):
        pending = asyncio.create_task(instance.seed_session(Request({"type": "http", "session": {}}), body))
        await asyncio.wait_for(entered.wait(), timeout=3)
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0)
        sandbox.stop.assert_not_awaited()
        proceed.set()
        await asyncio.wait_for(asyncio.gather(pending, shutdown), timeout=3)
    sandbox.stop.assert_awaited_once()
    assert instance._sandboxes.get(body.resources_session_id) is None


@pytest.mark.asyncio
async def test_unknown_close_fences_a_future_seed():
    instance, body = server(), seed()
    await instance.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    )
    with patch(f"{LENDING}.AsyncSandbox") as allocate:
        with pytest.raises(HTTPException, match="closed"):
            await instance.seed_session(Request({"type": "http", "session": {}}), body)
    allocate.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [None, [], {"version": 1, "events": [42], "deliveries": []}, {"version": 1, "events": [], "deliveries": [None]}],
)
def test_malformed_capture_shapes_are_infrastructure_gaps(value):
    with pytest.raises(EvidenceUnavailable):
        delivered_report_evidence(value, [])


@pytest.mark.parametrize(
    "package",
    [
        "pytz",
        "pytz>=2026.2",
        "pytz==2026.*",
        "pytz===2026.2",
        "pytz==2026.2,!=2026.1",
        "pytz[extra]==2026.2",
        "pytz==2026.2; python_version > '3.0'",
        "pytz @ https://example.invalid/pytz.whl",
        "pytz==2026.2\n--extra-index-url https://example.invalid",
    ],
)
def test_task_python_dependencies_require_exact_plain_version_pins(package):
    with pytest.raises(ValueError, match="exact name==version"):
        server(task_python_packages=[package], task_python_package_hashes={package: "a" * 64})


@pytest.mark.parametrize(
    "packages,hashes",
    [
        (["pytz==2026.2"], {}),
        ([], {"pytz==2026.2": "a" * 64}),
        (["pytz==2026.2"], {"pytz==2026.1": "a" * 64}),
        (["pytz==2026.2"], {"pytz==2026.2": "not-a-hash"}),
        (["pytz==2026.2"], {"pytz==2026.2": "A" * 64}),
        (["pytz==2026.2", "Pytz==2026.2"], {"pytz==2026.2": "a" * 64, "Pytz==2026.2": "a" * 64}),
    ],
)
def test_task_python_dependencies_require_matching_wheel_hashes(packages, hashes):
    with pytest.raises(ValueError):
        server(task_python_packages=packages, task_python_package_hashes=hashes)


@pytest.mark.asyncio
async def test_task_python_wheels_are_hash_bound_and_only_added_to_the_task_access():
    package, digest = "pytz==2026.2", "a" * 64
    instance = server(task_python_packages=[package], task_python_package_hashes={package: digest})
    sandbox, body = sandbox_for_test(), seed()
    request = Request({"type": "http", "session": {}})
    with (
        patch(f"{LENDING}.AsyncSandbox", return_value=sandbox),
        patch("resources_servers.partner_if.prime_app.upload_text", new_callable=AsyncMock) as upload,
    ):
        first = await instance.seed_session(request, body)
        assert await instance.seed_session(request, body) == first
    upload.assert_awaited_once_with(sandbox, path=TASK_REQUIREMENTS_PATH, text=f"{package} --hash=sha256:{digest}\n")
    assert first.sandbox_access.env == {"PYTHONPATH": TASK_PYTHON_DIRECTORY}
    assert sandbox.exec.await_count == 3
    command = sandbox.exec.await_args_list[-1].args[0]
    assert "--require-hashes --no-deps --only-binary=:all:" in command
    assert "--platform any --implementation py --abi none --python-version 3.13" in command
    assert f"--target {TASK_PYTHON_DIRECTORY} -r {TASK_REQUIREMENTS_PATH}" in command
    assert "kernel/bin" not in command


@pytest.mark.asyncio
async def test_failed_task_python_setup_releases_its_sandbox():
    package = "pytz==2026.2"
    instance = server(task_python_packages=[package], task_python_package_hashes={package: "a" * 64})
    sandbox, body = sandbox_for_test(), seed()
    sandbox.exec.side_effect = [
        SandboxExecResult(return_code=0, stdout="", stderr=""),
        SandboxExecResult(return_code=0, stdout="", stderr=""),
        SandboxExecResult(return_code=1, stdout="", stderr="wheel hash mismatch"),
    ]
    with (
        patch(f"{LENDING}.AsyncSandbox", return_value=sandbox),
        patch("resources_servers.partner_if.prime_app.upload_text", new_callable=AsyncMock),
    ):
        with pytest.raises(RuntimeError, match="Task Python setup failed"):
            await instance.seed_session(Request({"type": "http", "session": {}}), body)
    sandbox.stop.assert_awaited_once()
    assert instance._sandboxes.get(body.resources_session_id) is None
    assert instance.session_identity(body.resources_session_id) is None


def verification_request():
    """A signed synthetic contract that grades corroborated delivery without a model."""
    from resources_servers.partner_if.app import VerifyRequest
    from resources_servers.partner_if.tests.test_app import response

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
    return VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Send your findings"}]},
            "response": response().model_dump(),
            "verifier_metadata": {"partner_if": contract},
        }
    )


def verification_seed(body):
    return seed().model_copy(
        update={"task_data": {"verifier_metadata": body.verifier_metadata.model_dump(mode="json")}}
    )


def test_prime_sessions_follow_gyms_http_close_contract():
    instance = server()
    body = verification_request()
    with patch(f"{LENDING}.AsyncSandbox", side_effect=lambda *args, **kwargs: sandbox_for_test()):
        check_resources_session_contract(
            instance.setup_webserver(),
            verification_seed(body),
            keeps_state=True,
            holds_state=instance.has_session_state,
            verify_body=body.model_dump(mode="json"),
        )


def test_invalid_typed_contract_and_cookie_seed_are_rejected_before_allocating():
    instance = server()
    body = seed().model_copy(update={"task_data": {"verifier_metadata": {"partner_if": {}}}})
    with patch(f"{LENDING}.AsyncSandbox") as allocate:
        client = TestClient(instance.setup_webserver(), raise_server_exceptions=False)
        assert client.post("/seed_session", json=body.model_dump(mode="json")).status_code == 422
        assert client.post("/seed_session", json={}).status_code == 422
    allocate.assert_not_called()


@pytest.mark.asyncio
async def test_queued_seeds_cannot_replace_the_winning_tasks_inputs():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    request = Request({"type": "http", "session": {}})
    changed = body.model_copy(update={"task_data": {"changed": True}})
    with patch(f"{LENDING}.AsyncSandbox", return_value=sandbox):
        async with instance.session_lock(body.resources_session_id):
            first = asyncio.create_task(instance.seed_session(request, body))
            second = asyncio.create_task(instance.seed_session(request, changed))
            await asyncio.sleep(0)
        assert (await first).resources_session_id == body.resources_session_id
        with pytest.raises(HTTPException, match="different task"):
            await second
    sandbox.start.assert_awaited_once()
    assert instance.has_session_state(body.resources_session_id)


@pytest.mark.asyncio
async def test_cancelled_setup_retains_failed_cleanup_for_the_owning_close():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    entered = asyncio.Event()

    async def blocked_setup(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    sandbox.exec.side_effect = blocked_setup
    sandbox.stop.side_effect = [RuntimeError("stop pending"), None]
    with patch(f"{LENDING}.AsyncSandbox", return_value=sandbox):
        pending = asyncio.create_task(instance.seed_session(Request({"type": "http", "session": {}}), body))
        await asyncio.wait_for(entered.wait(), timeout=3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert instance.session_identity(body.resources_session_id) == (body.episode_id, body.task_id)
    assert instance._sandboxes.get(body.resources_session_id) is sandbox
    with pytest.raises(HTTPException, match="episode_id"):
        await instance.close_resources_session(
            ResourcesCloseSessionRequest(
                resources_session_id=body.resources_session_id, episode_id=EpisodeId.from_capture_key("other")
            )
        )
    await instance.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    )
    assert sandbox.stop.await_count == 2
    assert instance.session_identity(body.resources_session_id) is None
    assert instance._sandboxes.get(body.resources_session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_verification_releases_the_sandbox_and_retains_bound_report_evidence(cleanup_fails):
    import json

    instance, body = server(), verification_request()
    seeded = verification_seed(body)
    sandbox = sandbox_for_test()
    await lend_for_test(instance, seeded, sandbox)
    if cleanup_fails:
        sandbox.stop.side_effect = [RuntimeError("stop pending"), None]
    request = Request({"type": "http", "session": {"session_id": seeded.resources_session_id}})
    with patch(
        "resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=json.dumps(capture()))
    ) as read:
        first = await instance.verify(request, body)
        assert first.reward == 1 and not first.mask_sample
        assert instance.session_release_pending(seeded.resources_session_id) is cleanup_fails
        repeated = await instance.verify(request, body)
        assert repeated.reward == 1 and not repeated.mask_sample
        read.assert_awaited_once()
    assert instance._sandboxes.get(seeded.resources_session_id) is None
    assert not instance.has_session_state(seeded.resources_session_id)
    assert not instance.session_release_pending(seeded.resources_session_id)
    assert instance.session_is_verified(seeded.resources_session_id)
    assert sandbox.stop.await_count == (2 if cleanup_fails else 1)
    with pytest.raises(HTTPException, match="verified"):
        await instance.seed_session(request, seeded)
    await instance.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id=seeded.resources_session_id, episode_id=seeded.episode_id)
    )
    lost = await instance.verify(request, body)
    assert lost.mask_sample and not lost.evaluation_completed
    assert lost.failure_kind == "session_lost"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["contract", "input"])
async def test_retained_delivery_cannot_be_rebound_to_other_verification_inputs(changed_field):
    import json

    instance, body = server(), verification_request()
    seeded = verification_seed(body)
    await lend_for_test(instance, seeded, sandbox_for_test())
    request = Request({"type": "http", "session": {"session_id": seeded.resources_session_id}})
    with patch("resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=json.dumps(capture()))):
        assert (await instance.verify(request, body)).reward == 1
    changed = body.model_copy(deep=True)
    if changed_field == "contract":
        changed.verifier_metadata.partner_if.task_id = "another-task"
    else:
        changed.responses_create_params.input = "Another task"
    result = await instance.verify(request, changed)
    assert result.mask_sample and not result.evaluation_completed
    assert result.if_result.status == "unscorable"
    assert (await instance.verify(request, body)).reward == 1


@pytest.mark.asyncio
async def test_grading_exception_releases_state_and_retains_delivery_for_retry():
    import json

    instance, body = server(), verification_request()
    seeded, sandbox = verification_seed(body), sandbox_for_test()
    await lend_for_test(instance, seeded, sandbox)
    request = Request({"type": "http", "session": {"session_id": seeded.resources_session_id}})
    with (
        patch(
            "resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=json.dumps(capture()))
        ) as read,
        patch.object(type(instance), "grade_response", AsyncMock(side_effect=RuntimeError("judge unavailable"))),
    ):
        with pytest.raises(RuntimeError, match="judge unavailable"):
            await instance.verify(request, body)
        read.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
    assert not instance.has_session_state(seeded.resources_session_id)
    assert (await instance.verify(request, body)).reward == 1


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_verification_before_releasing():
    import json

    instance, body = server(), verification_request()
    seeded, sandbox = verification_seed(body), sandbox_for_test()
    await lend_for_test(instance, seeded, sandbox)
    request = Request({"type": "http", "session": {"session_id": seeded.resources_session_id}})
    entered, proceed = asyncio.Event(), asyncio.Event()
    original_grade = instance.grade_response

    async def blocked_grade(_server, *args, **kwargs):
        entered.set()
        await asyncio.wait_for(proceed.wait(), timeout=3)
        return await original_grade(*args, **kwargs)

    with (
        patch("resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=json.dumps(capture()))),
        patch.object(type(instance), "grade_response", blocked_grade),
    ):
        verification = asyncio.create_task(instance.verify(request, body))
        await asyncio.wait_for(entered.wait(), timeout=3)
        closing = asyncio.create_task(
            instance.close_resources_session(
                ResourcesCloseSessionRequest(
                    resources_session_id=seeded.resources_session_id, episode_id=seeded.episode_id
                )
            )
        )
        await asyncio.sleep(0)
        sandbox.stop.assert_not_awaited()
        proceed.set()
        result, _ = await asyncio.wait_for(asyncio.gather(verification, closing), timeout=3)
    assert result.reward == 1 and not result.mask_sample
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_seed_cannot_replace_the_actors_input():
    instance, body = server(), verification_request()
    seeded = verification_seed(body).model_copy(
        update={"responses_create_params": body.responses_create_params.model_dump(mode="json", exclude_unset=True)}
    )
    sandbox = sandbox_for_test()
    await lend_for_test(instance, seeded, sandbox)
    changed = seeded.model_copy(update={"responses_create_params": {"input": "A different task"}})
    with patch(f"{LENDING}.AsyncSandbox") as allocate:
        with pytest.raises(HTTPException, match="different task"):
            await instance.seed_session(Request({"type": "http", "session": {}}), changed)
    allocate.assert_not_called()
    assert instance.has_session_state(seeded.resources_session_id)


@pytest.mark.asyncio
async def test_first_verification_cannot_replace_the_seeded_actors_input():
    import json

    instance, body = server(), verification_request()
    seeded = verification_seed(body).model_copy(
        update={"responses_create_params": body.responses_create_params.model_dump(mode="json", exclude_unset=True)}
    )
    await lend_for_test(instance, seeded, sandbox_for_test())
    request = Request({"type": "http", "session": {"session_id": seeded.resources_session_id}})
    changed = body.model_copy(deep=True)
    changed.responses_create_params.input = "A different task"
    with patch(
        "resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=json.dumps(capture()))
    ) as read:
        rejected = await instance.verify(request, changed)
        assert rejected.mask_sample and not rejected.evaluation_completed
        assert (await instance.verify(request, body)).reward == 1
        read.assert_awaited_once()
