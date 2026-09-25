# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.apex_agent import prebuilt_world_entrypoint
from responses_api_agents.apex_agent.app import (
    NG_FAILURE_CLASS_KEY,
    NG_FAILURE_TERMINAL_KEY,
    ApexAgent,
    ApexAgentConfig,
    ApexAgentRunRequest,
    PolicyEgressRelay,
    instruction_from_input,
    load_prebuilt_runner_source,
    load_runner_source,
)
from responses_api_agents.apex_agent.sandbox_entrypoint import _discover_gateway_url, _patch_code_mcp_cancellation_race


def _body() -> ApexAgentRunRequest:
    return ApexAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Do the work"}]},
            "task_id": "task-1",
            "world_id": "world-1",
            "verifier_metadata": {
                "rubric": [{"criteria": "secret rubric"}],
                "gold_response": "secret gold",
            },
        }
    )


def _agent(
    *,
    image: str = "registry.example/archipelago@sha256:1234",
    auto_build: bool = False,
    supports_vision: bool = True,
    prebuilt_world_manifest: str | None = None,
    sandbox_provider: dict | None = None,
    policy_egress_relay: str = "auto",
) -> ApexAgent:
    config = ApexAgentConfig(
        host="0.0.0.0",
        port=8080,
        name="apex_agent",
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        concurrency=4,
        timeout=3600,
        image=image,
        image_build={
            "enabled": auto_build,
            "source_repo": "https://github.com/Mercor-Intelligence/archipelago.git",
            "source_revision": "0cb5c476c219a9df637e0bd37fb86b2361f4ab89",  # pragma: allowlist secret
            "source_root": None,
            "source_github_token": None,
            "dockerfile": "environment/Dockerfile",
            "docker_tag": "nemo-gym-archipelago:test",
            "timeout": 60,
        },
        sandbox_provider=sandbox_provider or {"apptainer": {}},
        sandbox_spec={},
        edgar_user_agent=None,
        max_turns=200,
        max_output_tokens=32_768,
        supports_vision=supports_vision,
        temperature=1.0,
        top_p=1.0,
        max_snapshot_bytes=None,
        max_world_bytes=None,
        artifact_output_dir=None,
        prebuilt_world_manifest=prebuilt_world_manifest,
        policy_egress_relay=policy_egress_relay,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model_name": "moonshotai/Kimi-K3"}
    agent = ApexAgent(config=config, server_client=client)
    agent._model_base_url = lambda _body: "http://model/v1"
    return agent


def test_instruction_from_input_only_uses_user_messages() -> None:
    body = _body()
    assert instruction_from_input(body.responses_create_params) == "Do the work"


def test_timeout_failure_is_routed_as_retryable_infra() -> None:
    result = _agent()._failure(
        _body(),
        "sandbox Stirrup rollout exited: direct apptainer command timed out after 12600s",
        return_code=125,
        failure_class="timeout_exceeded",
    )
    payload = result.model_dump()

    assert payload["reward"] == 0.0
    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert "_ng_failure_terminal" not in payload
    assert "_ng_no_persist" not in payload


def test_non_timeout_failure_is_not_misclassified_as_timeout() -> None:
    payload = _agent()._failure(_body(), "sandbox Stirrup rollout exited: command failed", return_code=1).model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "apex_error"
    assert payload[NG_FAILURE_CLASS_KEY] != "timeout_exceeded"


def test_failure_preserves_partial_trajectory_and_usage() -> None:
    trajectory = [{"role": "assistant", "content": "work in progress"}]
    payload = (
        _agent()
        ._failure(
            _body(),
            "sandbox failed",
            partial_result={
                "trajectory": trajectory,
                "completion_status": "error",
                "n_input_tokens": 12,
                "n_output_tokens": 7,
                "n_reasoning_tokens": 3,
            },
        )
        .model_dump()
    )

    assert payload["apex_trajectory"] == trajectory
    assert payload["apex_completion_status"] == "error"
    assert payload["response"]["apex_trajectory"] == trajectory
    assert payload["response"]["usage"]["input_tokens"] == 12
    assert payload["response"]["usage"]["output_tokens"] == 7


def test_terminal_failure_is_marked_for_sidecar_without_retry() -> None:
    payload = (
        _agent()
        ._failure(
            _body(),
            "maximum turns reached",
            failure_class="agent_incomplete",
            failure_terminal=True,
        )
        .model_dump()
    )

    assert payload[NG_FAILURE_CLASS_KEY] == "agent_incomplete"
    assert payload[NG_FAILURE_TERMINAL_KEY] is True


async def test_run_classifies_sandbox_timeout_for_retry(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent()
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent._stirrup_archive = tmp_path / "stirrup-runtime.tar.gz"
    seed_response = MagicMock(cookies={})
    agent.server_client.post = AsyncMock(return_value=seed_response)
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())

    class FakeSandbox:
        def __init__(self) -> None:
            self._exec_results = [
                MagicMock(return_code=0),
                MagicMock(return_code=0),
                MagicMock(
                    return_code=125,
                    stderr="direct apptainer command timed out after 12600s",
                    stdout=None,
                    error_type="timeout",
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, *_args) -> None:
            return None

        async def download(self, _source: str, destination: Path) -> None:
            destination.write_text(
                json.dumps(
                    {
                        "trajectory": [{"role": "assistant", "content": "partial"}],
                        "completion_status": "running",
                        "n_input_tokens": 4,
                        "n_output_tokens": 2,
                    }
                ),
                encoding="utf-8",
            )

        async def exec(self, *_args, **_kwargs):
            return self._exec_results.pop(0)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", MagicMock(return_value=FakeSandbox()))

    result = await agent.run(MagicMock(cookies={}), _body())
    payload = result.model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert "_ng_failure_terminal" not in payload
    assert "_ng_no_persist" not in payload
    assert payload["apex_trajectory"] == [{"role": "assistant", "content": "partial"}]


def test_run_request_preserves_task_input_files() -> None:
    payload = _body().model_dump()
    payload["task_input_files"] = "snap_0123456789abcdef0123456789abcdef"

    body = ApexAgentRunRequest.model_validate(payload)

    assert body.task_input_files == "snap_0123456789abcdef0123456789abcdef"


def _prebuilt_body(world_id: str = "world_0123456789abcdef0123456789abcdef") -> ApexAgentRunRequest:
    payload = _body().model_dump()
    payload.update(
        {
            "runtime_mode": "prebuilt_world",
            "world_id": world_id,
            "task_slug": "accounting-example-b1-01234567",
        }
    )
    return ApexAgentRunRequest.model_validate(payload)


def _prebuilt_agent(tmp_path: Path) -> tuple[ApexAgent, Path]:
    cache_root = tmp_path / "images"
    cache_root.mkdir()
    world_id = "world_0123456789abcdef0123456789abcdef"
    image = cache_root / f"{world_id}.sif"
    image.write_bytes(b"SIF_MAGIC")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "image_cache_root": str(cache_root),
                "worlds": {world_id: {"runtime_image": str(image)}},
            }
        ),
        encoding="utf-8",
    )
    return _agent(prebuilt_world_manifest=str(manifest)), image


def test_prebuilt_world_selects_only_manifest_image(tmp_path: Path) -> None:
    agent, image = _prebuilt_agent(tmp_path)
    body = _prebuilt_body()

    assert agent._prebuilt_image(body) == str(image)
    spec = agent._sandbox_spec(body, "Do the work", image=str(image), prebuilt_world=True)
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert spec.image == str(image)
    assert runner["task_slug"] == "accounting-example-b1-01234567"
    assert runner["startup_timeout_seconds"] == 1800
    assert spec.files["/app/apex-gym/sandbox_entrypoint.py"] == load_prebuilt_runner_source()
    assert "secret rubric" not in json.dumps(runner)


def test_prebuilt_world_rejects_missing_slug_and_unknown_world(tmp_path: Path) -> None:
    agent, _ = _prebuilt_agent(tmp_path)
    missing_slug = _prebuilt_body().model_copy(update={"task_slug": None})
    unknown_world = _prebuilt_body("world_ffffffffffffffffffffffffffffffff")

    try:
        agent._prebuilt_image(missing_slug)
    except ValueError as exc:
        assert "missing task_slug" in str(exc)
    else:
        raise AssertionError("missing task_slug was accepted")

    try:
        agent._prebuilt_image(unknown_world)
    except ValueError as exc:
        assert "trusted prebuilt-world manifest" in str(exc)
    else:
        raise AssertionError("unknown world was accepted")


async def test_prebuilt_world_config_errors_are_terminal(tmp_path: Path) -> None:
    agent, image = _prebuilt_agent(tmp_path)
    agent._ensure_runtime_setup = AsyncMock()
    request = MagicMock(cookies={})
    missing_slug = _prebuilt_body().model_copy(update={"task_slug": None})
    unknown_world = _prebuilt_body("world_ffffffffffffffffffffffffffffffff")

    for body, expected_error in (
        (missing_slug, "missing task_slug"),
        (unknown_world, "absent from the trusted prebuilt-world manifest"),
    ):
        payload = (await agent.run(request, body)).model_dump()
        assert payload[NG_FAILURE_CLASS_KEY] == "prebuilt_world_config_error"
        assert payload[NG_FAILURE_TERMINAL_KEY] is True
        assert expected_error in payload["apex_error"]

    image.unlink()
    payload = (await agent.run(request, _prebuilt_body())).model_dump()
    assert payload[NG_FAILURE_CLASS_KEY] == "prebuilt_world_config_error"
    assert payload[NG_FAILURE_TERMINAL_KEY] is True
    assert "prebuilt-world image is missing" in payload["apex_error"]
    agent._ensure_runtime_setup.assert_not_awaited()


def test_prebuilt_manifest_rejects_image_outside_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "images"
    cache_root.mkdir()
    world_id = "world_0123456789abcdef0123456789abcdef"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "image_cache_root": str(cache_root),
                "worlds": {world_id: {"runtime_image": str(tmp_path / f"{world_id}.sif")}},
            }
        ),
        encoding="utf-8",
    )

    try:
        _agent(prebuilt_world_manifest=str(manifest))
    except ValueError as exc:
        assert "untrusted prebuilt-world image path" in str(exc)
    else:
        raise AssertionError("image outside the trusted cache was accepted")


def test_sandbox_config_never_contains_verifier_secrets() -> None:
    body = _body()
    spec = _agent()._sandbox_spec(body, "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    serialized = json.dumps(runner)

    assert runner["instruction"] == "Do the work"
    assert runner["policy_model"] == "moonshotai/Kimi-K3"
    assert runner["max_turns"] == 200
    assert runner["max_output_tokens"] == 32_768
    assert runner["supports_vision"] is True
    assert "tokenizer_path" not in runner
    assert "context_window_tokens" not in runner
    assert "max_tool_output_tokens" not in runner
    assert "secret rubric" not in serialized
    assert "secret gold" not in serialized
    assert "CODE_EXEC_RUN_AS_USER" not in spec.env
    assert "/app/apex-gym/stirrup_runtime.py" in spec.files
    assert "FOUNDRY_LOCAL_ROOT" not in spec.env


def test_sandbox_config_propagates_text_only_model_capability() -> None:
    spec = _agent(supports_vision=False)._sandbox_spec(_body(), "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert runner["supports_vision"] is False


def test_sandbox_runner_uses_archipelago_gateway_and_stirrup() -> None:
    source = load_runner_source()

    assert "_patch_code_mcp_cancellation_race()" in source
    assert "configure_gateway(" in source
    assert '"0"' in source
    assert "_discover_gateway_url(" in source
    assert "run_stirrup_rollout(" in source
    assert "checkpoint_path=PARTIAL_RESULT_PATH" in source
    assert 'write_snapshot(OUTPUT / "initial.zip")' in source
    assert "overlay_task_files(task_files_zip, scratch)" in source
    assert source.index("overlay_task_files(task_files_zip, scratch)") < source.index(
        'write_snapshot(OUTPUT / "initial.zip")'
    )
    assert "stdout=gateway_log" in source
    assert "stderr=asyncio.subprocess.STDOUT" in source
    assert "stdout=asyncio.subprocess.PIPE" not in source


def test_prebuilt_snapshot_converts_official_environment_export(tmp_path: Path) -> None:
    source = tmp_path / "snapshot.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, content in (("filesystem/answer.docx", b"document"), (".apps_data/mail.json", b"{}")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    snapshot = tmp_path / "snapshot.zip"
    manifest = prebuilt_world_entrypoint.snapshot_tar_to_zip(source, snapshot)

    assert manifest == ["filesystem/answer.docx", ".apps_data/mail.json"]
    with zipfile.ZipFile(snapshot) as archive:
        assert sorted(archive.namelist()) == [".apps_data/mail.json", "filesystem/answer.docx"]


async def test_gateway_url_uses_uvicorn_dynamic_port(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text("INFO: Uvicorn running on http://127.0.0.1:43127 (Press CTRL+C to quit)\n")
    process = MagicMock(returncode=None)

    assert await _discover_gateway_url(process, log_path) == "http://127.0.0.1:43127"


def test_code_mcp_cancellation_patch_is_idempotent(tmp_path: Path) -> None:
    session_path = tmp_path / "code/.venv/lib/python3.13/site-packages/mcp/shared/session.py"
    session_path.parent.mkdir(parents=True)
    session_path.write_text(
        "async def respond(self, response):\n"
        '        assert not self._completed, "Request already responded to"\n'
        "        await self._send(response)\n",
        encoding="utf-8",
    )

    _patch_code_mcp_cancellation_race(tmp_path)
    patched = session_path.read_text(encoding="utf-8")
    _patch_code_mcp_cancellation_race(tmp_path)

    assert "        if self._completed:\n            return\n" in patched
    assert session_path.read_text(encoding="utf-8") == patched


async def test_runtime_setup_resolves_image_before_stirrup(monkeypatch: MonkeyPatch) -> None:
    agent = _agent()
    events: list[str] = []
    monkeypatch.setattr(
        "responses_api_agents.apex_agent.app.resolve_image",
        MagicMock(side_effect=lambda **_kwargs: events.append("image") or "archipelago.sif"),
    )

    async def _build(_image: str) -> Path:
        events.append("runtime")
        return Path("/tmp/stirrup-runtime.tar.gz")

    agent._build_stirrup_archive = AsyncMock(side_effect=_build)

    async def _inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)

    await agent._ensure_runtime_setup()

    assert events == ["image", "runtime"]


def test_incomplete_rollout_snapshots_are_saved_without_grading(tmp_path: Path) -> None:
    agent = _agent()
    agent.config.artifact_output_dir = str(tmp_path / "saved")
    body = _body()

    output_dir = agent._persist_ungraded_snapshots(
        body,
        {"completion_status": "max_turns"},
        b"initial",
        b"final",
    )

    assert output_dir is not None
    assert (output_dir / "initial_snapshot.zip").read_bytes() == b"initial"
    assert (output_dir / "final_snapshot.zip").read_bytes() == b"final"
    assert json.loads((output_dir / "rollout.json").read_text())["completion_status"] == "max_turns"


async def _idle_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Accept a connection, hold it open until the peer goes away, then close it."""
    await reader.read()
    writer.close()


def test_policy_egress_relay_follows_private_network_namespace() -> None:
    private = {"apptainer": {"create": {"extra_start_args": ["--net", "--network=none", "--fakeroot"]}}}

    assert _agent(sandbox_provider=private)._policy_egress_relay_enabled() is True
    assert _agent()._policy_egress_relay_enabled() is False
    assert _agent(policy_egress_relay="always")._policy_egress_relay_enabled() is True
    assert _agent(sandbox_provider=private, policy_egress_relay="never")._policy_egress_relay_enabled() is False


def test_sandbox_config_binds_egress_socket_only_when_relaying(tmp_path: Path) -> None:
    relay = SimpleNamespace(socket_dir=tmp_path, socket_name="policy.sock")
    spec = _agent()._sandbox_spec(_body(), "Do the work", relay=relay)
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert spec.provider_options["binds"] == [f"{tmp_path}:/egress"]
    assert runner["model_egress_socket"] == "/egress/policy.sock"
    assert runner["model_base_url"] == "http://model/v1"

    plain = _agent()._sandbox_spec(_body(), "Do the work")

    assert "binds" not in plain.provider_options
    assert "model_egress_socket" not in json.loads(plain.files["/app/apex-gym/runner_config.json"])

    agent = _agent()
    agent.config.sandbox_spec = {"provider_options": {"binds": "/data:/data:ro"}}
    spec = agent._sandbox_spec(_body(), "Do the work", relay=relay)

    assert spec.provider_options["binds"] == ["/data:/data:ro", f"{tmp_path}:/egress"]


async def test_policy_egress_relay_forwards_to_model_server() -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"echo:" + await reader.read(1024))
        await writer.drain()
        writer.close()

    upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]
    relay = await PolicyEgressRelay.start(f"http://127.0.0.1:{port}/v1")
    try:
        reader, writer = await asyncio.open_unix_connection(str(relay.socket_dir / relay.socket_name))
        writer.write(b"ping")
        await writer.drain()

        assert await reader.read(1024) == b"echo:ping"

        writer.close()
    finally:
        await relay.close()
        upstream.close()

    assert not relay.socket_dir.exists()


async def test_policy_egress_relay_rejects_non_http_model_urls() -> None:
    try:
        await PolicyEgressRelay.start("https://model/v1")
    except ValueError as exc:
        assert "http" in str(exc)
    else:
        raise AssertionError("https model URLs must be rejected")


async def test_policy_egress_relay_closes_with_idle_keepalive_connection() -> None:
    upstream = await asyncio.start_server(_idle_peer, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]
    relay = await PolicyEgressRelay.start(f"http://127.0.0.1:{port}/v1")
    reader, writer = await asyncio.open_unix_connection(str(relay.socket_dir / relay.socket_name))
    writer.write(b"GET /v1/models HTTP/1.1\r\n\r\n")
    await writer.drain()
    await asyncio.sleep(0.05)  # the bridge is now parked on both idle sockets
    try:
        await asyncio.wait_for(relay.close(), timeout=3)
    finally:
        writer.close()
        upstream.close()

    assert not relay.socket_dir.exists()


async def test_run_wires_and_tears_down_the_relay(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent(policy_egress_relay="always")
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())
    model = await asyncio.start_server(_idle_peer, "127.0.0.1", 0)
    agent._model_base_url = lambda _body: f"http://127.0.0.1:{model.sockets[0].getsockname()[1]}/v1"
    seen: dict[str, object] = {}

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, *_args) -> None:
            return None

        async def download(self, _source: str, destination: Path) -> None:
            destination.write_text("{}", encoding="utf-8")

        def __init__(self) -> None:
            self._exec_results = [
                MagicMock(return_code=0),
                MagicMock(return_code=0),
                MagicMock(return_code=1, stderr="boom", stdout=None, error_type=None),
            ]

        async def exec(self, *_args, **_kwargs):
            return self._exec_results.pop(0)

    def fake_sandbox(_provider, spec):
        seen["spec"] = spec
        return FakeSandbox()

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", fake_sandbox)
    try:
        result = await agent.run(MagicMock(cookies={}), _body())
    finally:
        model.close()

    spec = seen["spec"]
    bind = spec.provider_options["binds"][0]
    socket_dir = Path(bind.split(":")[0])
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert bind.endswith(":/egress")
    assert runner["model_egress_socket"] == "/egress/policy.sock"
    assert result.model_dump()[NG_FAILURE_CLASS_KEY] == "sandbox_error"
    assert not socket_dir.exists()
