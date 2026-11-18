# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
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


async def lend_for_test(instance, body, sandbox, *, prepared=True):
    """Admit a sandbox through the real lender without provisioning a remote provider."""
    with (
        patch(f"{LENDING}.AsyncSandbox", return_value=sandbox),
        patch(f"{LENDING}.get_global_config_dict", return_value={"sandbox": {"local": {}}}),
    ):
        await instance._sandboxes.lend(body.resources_session_id, body.task_id)
    instance._seeds[body.resources_session_id] = body.model_copy(deep=True)
    if prepared:
        instance._prepared.add(body.resources_session_id)


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
    assert body.resources_session_id in instance._seeds
    await instance.close_resources_session(close)
    assert instance._sandboxes.get(body.resources_session_id) is None
    assert body.resources_session_id not in instance._seeds
    assert body.resources_session_id not in instance._prepared


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
    assert body.resources_session_id not in instance._seeds


@pytest.mark.asyncio
async def test_failed_setup_handle_must_stop_before_replacement():
    instance, body = server(), seed()
    previous = sandbox_for_test()
    await lend_for_test(instance, body, previous, prepared=False)
    previous.stop.side_effect = RuntimeError("stop pending")
    with patch(f"{LENDING}.AsyncSandbox") as allocate:
        with pytest.raises(RuntimeError, match="stop pending"):
            await instance.seed_session(Request({"type": "http", "session": {}}), body)
        allocate.assert_not_called()
    previous.stop.assert_awaited_once()
    assert instance._sandboxes.get(body.resources_session_id) is previous
    assert body.resources_session_id not in instance._prepared


@pytest.mark.asyncio
async def test_partial_repository_is_not_reused_after_failed_cleanup():
    instance, body = server(), seed()
    first, replacement = sandbox_for_test(), sandbox_for_test()
    first.exec.return_value = SandboxExecResult(return_code=1, stderr="checkout failed", stdout="")
    first.stop.side_effect = [RuntimeError("stop pending"), None]
    request = Request({"type": "http", "session": {}})
    with patch(f"{LENDING}.AsyncSandbox", side_effect=[first, replacement]) as allocate:
        with pytest.raises(RuntimeError, match="stop pending"):
            await instance.seed_session(request, body)
        assert instance._sandboxes.get(body.resources_session_id) is first
        assert body.resources_session_id not in instance._prepared
        with pytest.raises(HTTPException, match="different task"):
            await instance.seed_session(request, body.model_copy(update={"task_data": {"other": True}}))
        await instance.seed_session(request, body)
        assert allocate.call_count == 2
    assert first.stop.await_count == 2
    assert instance._sandboxes.get(body.resources_session_id) is replacement
    assert replacement.exec.await_count == 2
    assert body.resources_session_id in instance._prepared


@pytest.mark.asyncio
async def test_close_checks_episode_and_retirement_reclaims_the_sandbox():
    instance, body = server(), seed()
    sandbox = sandbox_for_test()
    await lend_for_test(instance, body, sandbox)
    with pytest.raises(HTTPException, match="another episode"):
        await instance.close_resources_session(
            ResourcesCloseSessionRequest(
                resources_session_id=body.resources_session_id, episode_id=EpisodeId.from_capture_key("1-0")
            )
        )
    await instance.retire_session_state(body.resources_session_id)
    sandbox.stop.assert_awaited_once()
    await instance.retire_session_state(body.resources_session_id)
    with pytest.raises(HTTPException, match="closed"):
        await instance.seed_session(Request({"type": "http", "session": {}}), body)


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
    assert body.resources_session_id not in instance._seeds
