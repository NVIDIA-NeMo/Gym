# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import shutil
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from resources_servers.terminal_bench_4 import task as module
from resources_servers.terminal_bench_4.task import (
    Artifact,
    PackageLoader,
    Task,
    TaskSettings,
    content_hash,
    resolve_env,
)


def package(path):
    path.mkdir(parents=True)
    (path / "environment").mkdir()
    (path / "environment/Dockerfile").write_text("FROM public")
    (path / "instruction.md").write_text("<!-- canary marker -->\n\nDo the task.\n")
    (path / "task.toml").write_text(
        '[task]\nname="terminal-bench/test"\n[environment]\ndocker_image="agent"\n[verifier.environment]\ndocker_image="verifier"\n'
    )
    return path


def settings():
    return {"environment": {"docker_image": "agent"}, "verifier": {"environment": {"docker_image": "verifier"}}}


def test_content_hash_records_and_ignored_files(tmp_path):
    path = package(tmp_path / "package")
    records = []
    for name in ["environment/Dockerfile", "instruction.md", "task.toml"]:
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        records.append(f"{name}\0{digest}\n")
    expected = hashlib.sha256("".join(records).encode()).hexdigest()
    assert content_hash(path) == expected
    (path / "environment/__pycache__").mkdir()
    (path / "environment/__pycache__/code.pyc").write_bytes(b"ignored")
    assert content_hash(path) == expected
    (path / "environment/Dockerfile").write_text("FROM changed")
    assert content_hash(path) != expected


async def test_warm_cache_revalidated_and_instruction_stripped(tmp_path):
    path = package(tmp_path / "source")
    ref = "sha256:" + content_hash(path)
    dest = tmp_path / "terminal-bench/test" / ref[7:]
    dest.parent.mkdir(parents=True)
    path.rename(dest)
    loader = PackageLoader(tmp_path)
    loaded = await loader.load("terminal-bench/test", ref)
    assert loaded.instruction == "Do the task.\n"
    (dest / "environment/Dockerfile").write_text("FROM tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        await loader.load("terminal-bench/test", ref)


@pytest.mark.parametrize("name,ref", [("terminal-bench/test", "latest"), ("../test", "sha256:" + "a" * 64)])
async def test_loader_rejects_unpinned_and_unsafe_names(tmp_path, name, ref):
    with pytest.raises(ValueError):
        await PackageLoader(tmp_path).load(name, ref)


class Response:
    def __init__(self, *, data=None, blob=None):
        self.data = data
        self.blob = blob
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return self.data

    async def iter_chunked(self, size):
        yield self.blob


@pytest.mark.parametrize("failure", [None, "identity", "content", "layout"])
async def test_cold_cache_atomic_download_and_shared_load(tmp_path, monkeypatch, failure):
    source = package(tmp_path / "source")
    ref = "sha256:" + content_hash(source)
    if failure == "content":
        (source / "instruction.md").write_text("changed")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for p in source.rglob("*"):
            if p.is_file():
                tar.add(p, arcname=("wrong/" if failure == "layout" else "") + str(p.relative_to(source)))
    call = AsyncMock(
        side_effect=[
            Response(
                data={
                    "content_hash": "0" * 64 if failure == "identity" else ref[7:],
                    "archive_path": "public/archive.tar.gz",
                }
            ),
            Response(blob=stream.getvalue()),
        ]
    )
    monkeypatch.setattr(module, "request", call)
    loader = PackageLoader(tmp_path / "cache")
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            await loader.load("terminal-bench/test", ref)
        assert not (loader.root / "terminal-bench/test" / ref[7:]).exists()
    else:
        a, b = await asyncio.gather(*(loader.load("terminal-bench/test", ref) for _ in range(2)))
        assert a.path == b.path and a.instruction == "Do the task.\n"
        assert call.await_count == 2
        assert call.await_args_list[0].kwargs["json"]["p_ref"] == ref
    assert not list(loader.root.rglob(".tb4-download-*"))


@pytest.mark.parametrize("winner", [True, False])
async def test_cache_promotion_race_or_filesystem_failure(tmp_path, monkeypatch, winner):
    source = package(tmp_path / "source")
    ref = "sha256:" + content_hash(source)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for path in source.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(source)))
    monkeypatch.setattr(
        module,
        "request",
        AsyncMock(
            side_effect=[
                Response(data={"content_hash": ref[7:], "archive_path": "public/archive.tar.gz"}),
                Response(blob=stream.getvalue()),
            ]
        ),
    )

    def promote(path, target):
        if winner:
            shutil.copytree(path, target)
            raise FileExistsError("Another process promoted this pin")
        raise PermissionError("Cache filesystem is read-only")

    monkeypatch.setattr(Path, "rename", promote)
    loader = PackageLoader(tmp_path / "cache")
    if winner:
        loaded = await loader.load("terminal-bench/test", ref)
        assert content_hash(loaded.path) == ref[7:]
    else:
        with pytest.raises(PermissionError):
            await loader.load("terminal-bench/test", ref)
    assert not list(loader.root.rglob(".tb4-download-*"))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("environment", "tpu", {}),
        ("environment", "os", "windows"),
        ("environment", "unknown_execution_field", 1),
        ("environment", "network_mode", "allowlist"),
        ("agent", "network_mode", "no-network"),
        ("verifier", "environment_mode", "shared"),
        ("verifier", "timeout_sec", float("inf")),
    ],
)
def test_unsupported_execution_fields_fail_before_allocation(section, key, value):
    data = settings()
    data.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        TaskSettings.model_validate(data)


def test_legacy_network_resource_fields_and_separate_inheritance():
    data = settings()
    data["environment"].update(memory="1G", storage="1.5G", allow_internet=False)
    data["verifier"] = {"environment_mode": "separate"}
    cfg = TaskSettings.model_validate(data)
    assert cfg.environment.memory_mb == 1024 and cfg.environment.storage_mb == 1536
    assert cfg.verifier_environment.network_mode == "no-network"
    cfg.verifier_environment.env["local"] = "x"
    assert cfg.environment.env == {}
    data["environment"]["memory_mb"] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        TaskSettings.model_validate(data)


def test_mcp_skills_and_multistep_rejection():
    data = settings()
    data["environment"].update(
        mcp_servers=[{"name": "tools", "transport": "http", "url": "http://tools/mcp"}], skills_dir="/app/skills"
    )
    cfg = TaskSettings.model_validate(data)
    assert cfg.environment.mcp_servers[0].transport == "streamable-http"
    assert cfg.environment.skills_dir == "/app/skills"
    data["steps"] = [{"name": "step"}]
    with pytest.raises(ValueError, match="single-step"):
        TaskSettings.model_validate(data)


@pytest.mark.parametrize(
    "value",
    [
        {"source": "../bad"},
        {"source": "/ok", "destination": "/bad"},
        {"source": "/ok", "destination": "manifest.json"},
        {"source": "relative", "service": "db"},
    ],
)
def test_artifact_paths_are_contained(value):
    with pytest.raises(ValueError):
        Artifact.model_validate(value)


def test_destinations_and_conventional_directory_are_distinct():
    data = settings()
    data["artifacts"] = [{"source": "/app/output", "destination": "saved", "exclude": ["*.tmp"]}, "/app/file"]
    cfg = TaskSettings.model_validate(data)
    assert cfg.collected_artifacts[0].source == "/logs/artifacts"
    assert cfg.collected_artifacts[1].host_path == Path("saved")
    assert cfg.collected_artifacts[1].source == "/app/output"
    assert cfg.collected_artifacts[2].host_path == Path("app/file")


def test_env_templates_match_reference(monkeypatch):
    monkeypatch.setenv("TB4_TEST_VALUE", "actual")
    assert resolve_env(
        {"a": "${TB4_TEST_VALUE}", "b": "${TB4_MISSING:-default}", "c": "prefix-${TB4_TEST_VALUE}"}
    ) == {
        "a": "actual",
        "b": "default",
        "c": "prefix-${TB4_TEST_VALUE}",
    }
    monkeypatch.delenv("TB4_MISSING", raising=False)
    with pytest.raises(ValueError):
        resolve_env({"a": "${TB4_MISSING}"})


def test_links_cannot_escape_package(tmp_path):
    path = package(tmp_path / "package")
    (path / "environment/link").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("secret")
    with pytest.raises(ValueError, match="escapes"):
        Task.read(path, "terminal-bench/test", "sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="escapes"):
        content_hash(path)
