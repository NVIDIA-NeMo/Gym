# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
import tempfile
import zipfile
from inspect import getsource
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from responses_api_agents.apex_agent import stirrup_runtime


def test_stirrup_repackages_image_tool_responses_for_chat_completions() -> None:
    source = getsource(stirrup_runtime.run_stirrup_rollout)

    assert "text_only_tool_responses=True" in source


def test_text_only_model_replaces_tool_images_and_preserves_other_content() -> None:
    class FakeImage:
        pass

    other_block = object()
    content = ["extracted text", FakeImage(), other_block, FakeImage()]

    result = stirrup_runtime.replace_tool_images_for_text_only_model(
        content,
        supports_vision=False,
        image_content_type=FakeImage,
    )

    assert result == [
        "extracted text",
        other_block,
        "[2 image(s) not shown: model does not support vision]",
    ]


def test_vision_model_keeps_tool_images_unchanged() -> None:
    class FakeImage:
        pass

    content = ["extracted text", FakeImage()]

    assert (
        stirrup_runtime.replace_tool_images_for_text_only_model(
            content,
            supports_vision=True,
            image_content_type=FakeImage,
        )
        is content
    )


def test_partial_result_checkpoint_preserves_completed_turns_and_usage(tmp_path: Path) -> None:
    class FakeMessage:
        def __init__(self, role: str, content: str, *, input_tokens: int = 0, answer_tokens: int = 0) -> None:
            self.role = role
            self.content = content
            self.token_usage = SimpleNamespace(input=input_tokens, answer=answer_tokens, reasoning=0)

        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"role": self.role, "content": self.content}

    session = SimpleNamespace(
        _current_run_state=SimpleNamespace(
            full_msg_history=[[FakeMessage("user", "task")]],
            msgs=[FakeMessage("assistant", "partial answer", input_tokens=11, answer_tokens=5)],
        )
    )
    destination = tmp_path / "partial_result.json"

    assert stirrup_runtime.write_partial_result_checkpoint(session, destination)
    checkpoint = json.loads(destination.read_text(encoding="utf-8"))

    assert checkpoint["completed"] is False
    assert checkpoint["completion_status"] == "running"
    assert checkpoint["trajectory"] == [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "partial answer"},
    ]
    assert checkpoint["n_input_tokens"] == 11
    assert checkpoint["n_output_tokens"] == 5


def test_partial_result_checkpoint_waits_for_stirrup_state(tmp_path: Path) -> None:
    destination = tmp_path / "partial_result.json"

    assert not stirrup_runtime.write_partial_result_checkpoint(SimpleNamespace(), destination)
    assert not destination.exists()


def test_tool_output_uses_head_and_tail_excerpt() -> None:
    text = "H" * 20_000 + "removed" * 11_000 + "T" * 5_000

    result = stirrup_runtime.truncate_tool_text(text)

    assert result.startswith("H" * 20_000)
    assert result.endswith("T" * 5_000)
    assert "characters truncated" in result


def test_tool_output_within_estimated_token_budget_is_unchanged() -> None:
    text = "x" * (
        stirrup_runtime.TOOL_OUTPUT_TOKEN_BUDGET * stirrup_runtime.TOOL_OUTPUT_ESTIMATED_CHARACTERS_PER_TOKEN
    )

    assert stirrup_runtime.truncate_tool_text(text) == text


def test_mcp_call_arguments_omit_optional_nulls() -> None:
    class Params(BaseModel):
        query: str
        case_sensitive: bool | None = None
        max_results: int | None = None

    result = stirrup_runtime.mcp_call_arguments(Params(query="tariffs"))

    assert result == {"query": "tariffs"}


def test_world_restore_and_snapshot_preserve_subsystems(monkeypatch, tmp_path: Path) -> None:
    filesystem = tmp_path / "filesystem"
    apps = tmp_path / ".apps_data"
    monkeypatch.setattr(stirrup_runtime, "FILESYSTEM_ROOT", filesystem)
    monkeypatch.setattr(stirrup_runtime, "APPS_DATA_ROOT", apps)
    world = tmp_path / "world.zip"
    with zipfile.ZipFile(world, "w") as archive:
        archive.writestr("filesystem/input.txt", "input")
        archive.writestr(".apps_data/mail/state.json", "{}")

    stirrup_runtime.populate_world(world, tmp_path / "scratch")
    manifest = stirrup_runtime.write_snapshot(tmp_path / "snapshot.zip")

    assert (filesystem / "input.txt").read_text() == "input"
    assert (apps / "mail/state.json").read_text() == "{}"
    assert manifest == ["filesystem/input.txt", ".apps_data/mail/state.json"]


def test_task_files_overlay_world_before_initial_snapshot(monkeypatch, tmp_path: Path) -> None:
    filesystem = tmp_path / "filesystem"
    apps = tmp_path / ".apps_data"
    monkeypatch.setattr(stirrup_runtime, "FILESYSTEM_ROOT", filesystem)
    monkeypatch.setattr(stirrup_runtime, "APPS_DATA_ROOT", apps)
    world = tmp_path / "world.zip"
    with zipfile.ZipFile(world, "w") as archive:
        archive.writestr("filesystem/shared.txt", "world")
        archive.writestr(".apps_data/mail/state.json", "{}")
    task_files = tmp_path / "task_files.zip"
    with zipfile.ZipFile(task_files, "w") as archive:
        archive.writestr("filesystem/source.docx", "task input")
        archive.writestr("filesystem/shared.txt", "task override")

    stirrup_runtime.populate_world(world, tmp_path / "scratch")
    stirrup_runtime.overlay_task_files(task_files, tmp_path / "scratch")
    manifest = stirrup_runtime.write_snapshot(tmp_path / "snapshot.zip")

    assert (filesystem / "source.docx").read_text() == "task input"
    assert (filesystem / "shared.txt").read_text() == "task override"
    assert (apps / "mail/state.json").read_text() == "{}"
    assert "filesystem/source.docx" in manifest


def test_gateway_config_runs_packaged_servers_and_offline_edgar(monkeypatch, tmp_path: Path) -> None:
    mcp_root = tmp_path / "mcp_servers"
    monkeypatch.setattr(stirrup_runtime, "MCP_ROOT", mcp_root)
    monkeypatch.setattr(stirrup_runtime, "FILESYSTEM_ROOT", tmp_path / "filesystem")
    monkeypatch.setattr(stirrup_runtime, "APPS_DATA_ROOT", tmp_path / ".apps_data")
    specs = [*stirrup_runtime._STANDARD_SERVERS.values(), ("edgar_sec", "edgar_sec", "unused")]
    for component, server_dir, _ in specs:
        (mcp_root / component / ".venv/bin").mkdir(parents=True, exist_ok=True)
        (mcp_root / component / ".venv/bin/python3").touch()
        (mcp_root / component / "mcp_servers" / server_dir).mkdir(parents=True, exist_ok=True)

    config = stirrup_runtime.gateway_config(["edgar"], "Apex test@example.com")

    assert set(config["mcpServers"]) == {*stirrup_runtime._STANDARD_SERVERS, "edgar"}
    edgar_env = config["mcpServers"]["edgar"]["env"]
    assert edgar_env["EDGAR_OFFLINE_MODE"] == "true"
    assert edgar_env["INTERNET_ENABLED"] == "false"
    assert edgar_env["EDGAR_USER_AGENT"] == "Apex test@example.com"


async def _idle_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Accept a connection, hold it open until the peer goes away, then close it."""
    await reader.read()
    writer.close()


def test_rewrite_model_base_url_keeps_path_and_rejects_https() -> None:
    assert stirrup_runtime.rewrite_model_base_url("http://10.1.2.3:8000/v1", 4242) == "http://127.0.0.1:4242/v1"
    with pytest.raises(ValueError):
        stirrup_runtime.rewrite_model_base_url("https://10.1.2.3/v1", 4242)


async def test_policy_endpoint_without_socket_uses_model_url_directly() -> None:
    async with stirrup_runtime.policy_endpoint({"model_base_url": "http://model/v1"}) as base_url:
        assert base_url == "http://model/v1"


async def test_policy_endpoint_relays_http_through_unix_socket() -> None:
    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(4096)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\npong")
        await writer.drain()
        writer.close()

    model = await asyncio.start_server(respond, "127.0.0.1", 0)
    model_port = model.sockets[0].getsockname()[1]
    # AF_UNIX paths are capped at 107 bytes; pytest's tmp_path can exceed that.
    socket_dir = tempfile.mkdtemp(prefix="egr-", dir="/tmp")
    socket_path = f"{socket_dir}/policy.sock"
    host_side = await stirrup_runtime.serve_unix_to_tcp(socket_path, "127.0.0.1", model_port)
    config = {"model_base_url": f"http://10.0.0.1:{model_port}/v1", "model_egress_socket": socket_path}
    try:
        async with stirrup_runtime.policy_endpoint(config) as base_url:
            assert base_url.startswith("http://127.0.0.1:")
            assert base_url.endswith("/v1")
            port = int(base_url.removeprefix("http://127.0.0.1:").split("/", 1)[0])
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n")
            await writer.drain()

            assert b"pong" in await reader.read(4096)

            writer.close()
    finally:
        await host_side.close()
        model.close()
        shutil.rmtree(socket_dir, ignore_errors=True)


async def test_policy_endpoint_exits_promptly_with_idle_keepalive_connection() -> None:
    socket_dir = tempfile.mkdtemp(prefix="egr-", dir="/tmp")
    socket_path = f"{socket_dir}/policy.sock"
    host_side = await asyncio.start_unix_server(_idle_peer, path=socket_path)
    config = {"model_base_url": "http://10.0.0.1:8000/v1", "model_egress_socket": socket_path}
    writer = None
    try:
        async with asyncio.timeout(5):
            async with stirrup_runtime.policy_endpoint(config) as base_url:
                port = int(base_url.removeprefix("http://127.0.0.1:").split("/", 1)[0])
                _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"GET /v1/models HTTP/1.1\r\n\r\n")
                await writer.drain()
                await asyncio.sleep(0.05)  # idle keep-alive connection stays bridged
    finally:
        if writer is not None:
            writer.close()
        host_side.close()
        shutil.rmtree(socket_dir, ignore_errors=True)


async def test_policy_endpoint_closes_listener_when_body_raises() -> None:
    socket_dir = tempfile.mkdtemp(prefix="egr-", dir="/tmp")
    socket_path = f"{socket_dir}/policy.sock"
    host_side = await asyncio.start_unix_server(_idle_peer, path=socket_path)
    config = {"model_base_url": "http://10.0.0.1:8000/v1", "model_egress_socket": socket_path}
    port = None
    try:
        with pytest.raises(RuntimeError):
            async with stirrup_runtime.policy_endpoint(config) as base_url:
                port = int(base_url.removeprefix("http://127.0.0.1:").split("/", 1)[0])
                raise RuntimeError("rollout failed")
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", port)
    finally:
        host_side.close()
        shutil.rmtree(socket_dir, ignore_errors=True)
