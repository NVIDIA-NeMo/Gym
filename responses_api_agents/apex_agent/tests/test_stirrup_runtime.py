# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zipfile
from inspect import getsource
from pathlib import Path

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
