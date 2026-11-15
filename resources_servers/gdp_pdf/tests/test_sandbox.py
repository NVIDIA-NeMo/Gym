# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    check_resources_session_contract,
)
from resources_servers.gdp_pdf import sandbox_app as module
from resources_servers.gdp_pdf.tests.test_app import _verify_request, config  # noqa: F401


# LentSandboxes looks up the sandbox class and the global config in its own module.
LENDING = "nemo_gym.server_runtime.sandbox_lending"


def _use_provider(monkeypatch, provider):
    global_config = {"sandbox": provider}
    monkeypatch.setattr(module, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(f"{LENDING}.get_global_config_dict", lambda: global_config)


@pytest.fixture
def setup(config, tmp_path, monkeypatch):  # noqa: F811
    component = tmp_path / "resources_servers/gdp_pdf"
    (component / "data").mkdir(parents=True)
    pdf = component / "data/source.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(module, "PARENT_DIR", tmp_path)
    monkeypatch.chdir(component)  # Gym starts resources from their component directory.
    server = module.GdpPdfSandboxResourcesServer(
        config=module.GdpPdfSandboxConfig(**config.model_dump(), sandbox_provider="sandbox", sandbox_image="tools"),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "test"}
    monkeypatch.setattr(f"{LENDING}.AsyncSandbox", lambda _: sandbox)
    _use_provider(monkeypatch, {"docker": {"create": {"network": "isolated"}}})
    monkeypatch.setattr(module.subprocess, "run", MagicMock(return_value=MagicMock(stdout="true\n")))
    body = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id={"rollout_id": "run", "attempt": 0},
        task_id={"taskset": "train", "task_id": "task"},
        task_data={"verifier_metadata": {"source_pdf": "source.pdf", "criteria": [{"criterion": "hidden"}]}},
    )
    return server, sandbox, body, pdf


def test_session_contract_and_pdf_delivery(setup):
    server, sandbox, body, pdf = setup
    check_resources_session_contract(
        server.setup_webserver(),
        body,
        keeps_state=True,
        holds_state=lambda sid: server._sandboxes.get(sid) is not None,
    )
    specs = [call.args[0] for call in sandbox.start.await_args_list]
    assert specs
    for spec in specs:
        assert (spec.image, spec.workdir) == ("tools", module.WORKDIR)
        # Only the source document is delivered: no rubric or reference answer.
        assert spec.files == {module.DOCUMENT_PATH: pdf.read_bytes()}


@pytest.mark.parametrize(
    "status,tool_tail,incomplete",
    [
        ("completed", False, False),
        (None, False, False),
        ("failed", False, False),
        ("incomplete", False, False),
        ("incomplete", True, False),
        ("completed", False, True),
    ],
)
async def test_only_completed_answers_are_judged(setup, status, tool_tail, incomplete):
    server, _, _, _ = setup
    body = _verify_request("The cap is $5M.")
    payload = body.model_dump()
    payload["response"]["status"] = status
    if incomplete:
        payload["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    if tool_tail:
        payload["response"]["output"].append(
            {"type": "function_call", "name": "bash", "arguments": "{}", "call_id": "pending"}
        )
    body = module.GdpPdfVerifyRequest.model_validate(payload)
    server._judge_criterion = AsyncMock(return_value=module.CriterionEvaluation(criterion="cap", passed=True))
    result = await server.verify(body)
    complete = status in (None, "completed") and not incomplete
    assert result.response == body.response
    assert result.reward == result.all_pass == float(complete)
    assert result.num_criteria == 1
    assert result.criterion_evaluations[0].passed == complete
    assert server._judge_criterion.await_count == int(complete)


@pytest.mark.parametrize("source", ["../outside.pdf", "link.pdf", "/etc/passwd", "missing.pdf", ""])
async def test_invalid_pdf_never_allocates(setup, source):
    server, sandbox, body, pdf = setup
    outside = pdf.parent.parent / "outside.pdf"
    outside.write_bytes(pdf.read_bytes())
    (pdf.parent / "link.pdf").symlink_to(outside)
    body.task_data["verifier_metadata"]["source_pdf"] = source
    with pytest.raises(ResourcesSeedRejectedError):
        await server.seed_session(Request({"type": "http", "session": {}}), body)
    sandbox.start.assert_not_awaited()


@pytest.mark.parametrize(
    "provider,internal,options",
    [
        ({"docker": {}}, "true", {}),
        ({"apptainer": {}}, "true", {}),
        ({"docker": {"create": {"network": "public"}}}, "false", {}),
        ({"docker": {"create": {"network": "internal", "extra_run_args": ["--privileged"]}}}, "true", {}),
        ({"docker": {"create": {"network": "internal"}}}, "true", {"volumes": ["/tmp:/tmp"]}),
        ({"docker": {"create": {"network": "internal"}}}, "true", {"run_args": ["--network", "host"]}),
    ],
)
async def test_unsafe_sandbox_requires_recorded_opt_out(setup, monkeypatch, provider, internal, options):
    server, sandbox, body, _ = setup
    _use_provider(monkeypatch, provider)
    module.subprocess.run.return_value.stdout = internal
    server.config.sandbox_config.provider_options = options
    request = Request({"type": "http", "session": {}})
    with pytest.raises(ResourcesSeedRejectedError):
        await server.seed_session(request, body)
    sandbox.start.assert_not_awaited()
    server.config.allow_unisolated_sandbox = True
    await server.seed_session(request, body)
    server._judge_criterion = AsyncMock(return_value=module.CriterionEvaluation(criterion="cap", passed=True))
    result = await server.verify(_verify_request("The cap is $5M."))
    assert result.model_dump()["allow_unisolated_sandbox"] is True


async def test_network_inspection_failure_never_allocates(setup):
    server, sandbox, body, _ = setup
    module.subprocess.run.side_effect = module.subprocess.TimeoutExpired("docker", 10)
    with pytest.raises(ResourcesSeedRejectedError, match="Could not verify"):
        await server.seed_session(Request({"type": "http", "session": {}}), body)
    sandbox.start.assert_not_awaited()


async def test_failed_start_and_close_retain_retryable_cleanup(setup):
    server, sandbox, body, pdf = setup
    server.config.documents_base_dir = str(pdf.parent)  # Absolute roots work too.
    request = Request({"type": "http", "session": {}})
    sandbox.start.side_effect = RuntimeError("start failed")
    with pytest.raises(RuntimeError, match="start failed"):
        await server.seed_session(request, body)
    sandbox.stop.assert_awaited_once()
    assert not server._seeds
    sandbox.start.side_effect = None
    await server.seed_session(request, body)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.close_resources_session(close)
    assert "session" in server._seeds
    await server.close_resources_session(close)
    assert not server._seeds
    assert server._sandboxes.get("session") is None


@pytest.mark.parametrize("release", ["close", "retire", "shutdown"])
async def test_sandbox_of_a_failed_seed_is_stopped_later(setup, release):
    server, sandbox, body, _ = setup
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    sandbox.start.side_effect = RuntimeError("start failed")
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="start failed"):
        await server.seed_session(Request({"type": "http", "session": {}}), body)
    assert not server._seeds
    if release == "close":
        await server.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
        )
    elif release == "retire":
        await server.retire_session_state("session")
    assert sandbox.stop.await_count == (1 if release == "shutdown" else 2)
    await lifespan.__aexit__(None, None, None)
    assert sandbox.stop.await_count == 2


async def test_shutdown_waits_for_session_operation(setup):
    server, sandbox, body, _ = setup
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    await server.seed_session(Request({"type": "http", "session": {}}), body)
    async with server._locks.locked(body.resources_session_id):
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0.05)
        sandbox.stop.assert_not_awaited()
    await asyncio.wait_for(shutdown, timeout=1)
    sandbox.stop.assert_awaited_once()
    assert server._sandboxes.get(body.resources_session_id) is None


async def test_cancelled_seed_stops_its_sandbox_and_holds_nothing(setup):
    server, sandbox, body, _ = setup
    started = asyncio.Event()

    async def start(spec):
        started.set()
        await asyncio.Event().wait()

    sandbox.start.side_effect = start
    seeding = asyncio.create_task(server.seed_session(Request({"type": "http", "session": {}}), body))
    await started.wait()
    seeding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await seeding
    sandbox.stop.assert_awaited_once()
    assert not server._seeds


@pytest.mark.parametrize("blocked", ["network inspection", "sandbox start"])
async def test_shutdown_waits_for_a_first_seed_in_progress(setup, blocked):
    server, sandbox, body, _ = setup
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    entered, proceed = threading.Event(), threading.Event()

    def inspect(*args, **kwargs):
        entered.set()
        proceed.wait(5)
        return MagicMock(stdout="true\n")

    async def start(spec):
        entered.set()
        await asyncio.to_thread(proceed.wait, 5)

    if blocked == "network inspection":
        module.subprocess.run.side_effect = inspect
    else:
        sandbox.start.side_effect = start
    seeding = asyncio.create_task(server.seed_session(Request({"type": "http", "session": {}}), body))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0.05)
        assert not shutdown.done()
        sandbox.stop.assert_not_awaited()
    finally:
        proceed.set()
    await asyncio.wait_for(asyncio.gather(seeding, shutdown), timeout=5)
    # The sandbox that finished starting during shutdown is stopped, not left running.
    sandbox.start.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
    assert server._sandboxes.get(body.resources_session_id) is None


async def test_retirement_releases_sandbox_and_retries_failed_stop(setup):
    server, sandbox, body, _ = setup
    request = Request({"type": "http", "session": {}})
    await server.seed_session(request, body)
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.retire_session_state(body.resources_session_id)
    assert body.resources_session_id in server._seeds
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(request, body)
    await server.retire_session_state(body.resources_session_id)
    await server.retire_session_state(body.resources_session_id)
    assert not server._seeds
    assert sandbox.stop.await_count == 2
