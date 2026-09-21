# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.terminal_bench_4.collection import collect
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.verifier import (
    RewardFileEmptyError,
    RewardFileNotFoundError,
    VerifierOutputParseError,
    VerifierTimeoutError,
    parse_reward,
    restore,
    run_verifier,
)


class FilesystemSandbox:
    """Exercise real tar and shell transfers against isolated temporary trees."""

    def __init__(self, root):
        self.root = root
        self.commands = []
        for name in ["app", "logs/artifacts", "logs/agent", "logs/verifier", "tests", "tmp", "evidence"]:
            (root / name).mkdir(parents=True, exist_ok=True)

    def path(self, path):
        return self.root / path.lstrip("/")

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        # Only replace the controlled virtual container roots used by fixtures.
        translated = re.sub(
            r"/(app|logs|tests|tmp|evidence)(?=/|\s|[\x27\x22;]|$)", lambda m: str(self.root / m[1]), command
        )
        process = await asyncio.create_subprocess_shell(
            translated, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return SimpleNamespace(
            return_code=process.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def upload(self, source, target):
        self.path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.path(target))

    async def download(self, source, target):
        shutil.copy2(self.path(source), target)


def environment(tmp_path, cfg):
    main = FilesystemSandbox(tmp_path / "main")
    sidecar = FilesystemSandbox(tmp_path / "sidecar")
    env = SimpleNamespace(main=main, task=SimpleNamespace(config=cfg), stopped=False, stop_main=None)

    def sandbox(service=None):
        return main if service in (None, "main") else sidecar

    async def execute(command, service=None, timeout_sec=None, user=None, env=None):
        if service not in (None, "main"):
            assert instance.stopped, "sidecar evidence was collected before main stopped"
        return await sandbox(service).exec(command, timeout_s=timeout_sec, user=user, env=env)

    async def stop_main():
        instance.stopped = True

    instance = env
    env.sandbox = sandbox
    env.exec = execute
    env.stop_main = stop_main
    return env, main, sidecar


async def test_known_positive_main_sidecar_and_conventional_artifacts_restore(tmp_path):
    cfg = TaskSettings.model_validate(
        {
            "environment": {"docker_image": "agent"},
            "verifier": {
                "environment": {"docker_image": "verifier"},
                "collect": [
                    {"command": "echo main-hook > /app/output/hook.txt"},
                    {"service": "db", "command": "echo sidecar-hook > /evidence/hook.txt"},
                    {"service": "db", "command": "exit 9"},
                ],
            },
            "artifacts": [
                {"source": "/app/output", "destination": "saved", "exclude": ["*.tmp"]},
                {"source": "/evidence", "service": "db"},
                "/app/optional.dat",
            ],
        }
    )
    env, main, sidecar = environment(tmp_path / "agent", cfg)
    main.path("/app/output").mkdir()
    main.path("/app/output/answer.txt").write_text("42")
    main.path("/app/output/ignored.tmp").write_text("exclude me")
    main.path("/logs/artifacts/report.txt").write_text("conventional")
    sidecar.path("/evidence/state.txt").write_text("trusted-sidecar")
    diagnostics = []
    manifest = await collect(env, tmp_path / "artifacts", diagnostics)
    assert (tmp_path / "artifacts/saved/answer.txt").read_text() == "42"
    assert not (tmp_path / "artifacts/saved/ignored.tmp").exists()
    assert (tmp_path / "artifacts/evidence/state.txt").read_text() == "trusted-sidecar"
    assert any(x["status"] == "failed" and x["source"] == "/app/optional.dat" for x in manifest)
    assert any(x.get("return_code") == 9 for x in diagnostics)
    verifier, target, _ = environment(tmp_path / "verifier", cfg)
    target.path("/app/output").mkdir()
    target.path("/app/output/stale.txt").write_text("must be removed")
    await restore(verifier, tmp_path / "artifacts")
    assert target.path("/app/output/answer.txt").read_text() == "42"
    assert target.path("/app/output/hook.txt").read_text().strip() == "main-hook"
    assert target.path("/evidence/hook.txt").read_text().strip() == "sidecar-hook"
    assert target.path("/logs/artifacts/report.txt").read_text() == "conventional"
    assert not target.path("/app/output/stale.txt").exists()
    assert not target.path("/saved").exists()
    # Baked-in tests validate bytes from every transfer channel. The test file
    # is immutable input; native verification must never upload task-side tests.
    script = target.path("/tests/test.sh")
    script.write_text(
        "#!/bin/sh\n"
        + "\n".join(
            [
                f'test "$(cat {target.path("/app/output/answer.txt")})" = 42 &&',
                f'test "$(cat {target.path("/evidence/state.txt")})" = trusted-sidecar &&',
                f'test "$(cat {target.path("/logs/artifacts/report.txt")})" = conventional &&',
                f"echo 1 > {target.path('/logs/verifier/reward.txt')}",
            ]
        )
        + "\n"
    )
    result = await run_verifier(verifier, tmp_path / "result", diagnostics)
    assert result == {"rewards": {"reward": 1.0}}
    assert not list(main.root.glob("tmp/.nemo-gym-*"))
    assert not list(target.root.glob("tmp/.nemo-gym-*"))


async def test_failed_convention_probe_preserves_directory_for_restore(tmp_path):
    cfg = TaskSettings.model_validate(
        {"environment": {"docker_image": "agent"}, "verifier": {"environment": {"docker_image": "verifier"}}}
    )
    env, main, _ = environment(tmp_path / "agent", cfg)
    artifacts = tmp_path / "artifacts"
    # This is the host layout created by prepare_session, matching the reference.
    (artifacts / "logs/artifacts").mkdir(parents=True)
    execute = env.exec

    async def failed_probe(command, **kwargs):
        if command == "test -d /logs/artifacts":
            return SimpleNamespace(return_code=1, stderr="operation not permitted")
        return await execute(command, **kwargs)

    async def sdk_file_download(source, target):
        # The provider's file API returns empty bytes when asked for a directory.
        target.write_bytes(b"")

    env.exec = failed_probe
    main.download = sdk_file_download
    manifest = await collect(env, artifacts, [])
    assert manifest[0]["status"] == "failed"
    assert (artifacts / "logs/artifacts").is_dir()
    verifier, target, _ = environment(tmp_path / "verifier", cfg)
    await restore(verifier, artifacts)
    assert target.path("/logs/artifacts").is_dir()


async def test_colliding_host_destinations_keep_first(tmp_path):
    cfg = TaskSettings.model_validate(
        {
            "environment": {"docker_image": "agent"},
            "verifier": {"environment": {"docker_image": "verifier"}},
            "artifacts": [
                {"source": "/app/one.txt", "destination": "same.txt"},
                {"source": "/app/two.txt", "destination": "same.txt"},
            ],
        }
    )
    env, main, _ = environment(tmp_path / "agent", cfg)
    main.path("/app/one.txt").write_text("first")
    main.path("/app/two.txt").write_text("second")
    records = await collect(env, tmp_path / "artifacts", [])
    assert (tmp_path / "artifacts/same.txt").read_text() == "first"
    assert records[-1]["status"] == "skipped"


@pytest.mark.parametrize("text,expected", [("0", 0), ("1", 1), ("0.5", 0.5), ("-1", -1)])
def test_text_reward(tmp_path, text, expected):
    (tmp_path / "reward.txt").write_text(text)
    assert parse_reward(tmp_path) == {"reward": expected}


@pytest.mark.parametrize("text", ["NaN", "Infinity", "not a number"])
def test_malformed_text_reward(tmp_path, text):
    (tmp_path / "reward.txt").write_text(text)
    with pytest.raises(VerifierOutputParseError):
        parse_reward(tmp_path)


@pytest.mark.parametrize("text", ['{"reward":NaN}', '{"reward":1e309}', '{"reward":"1"}', "[1]", "1", "{broken"])
def test_malformed_json_has_precedence_over_valid_text(tmp_path, text):
    (tmp_path / "reward.txt").write_text("1")
    (tmp_path / "reward.json").write_text(text)
    with pytest.raises(VerifierOutputParseError):
        parse_reward(tmp_path)


def test_zero_json_precedence_empty_and_missing(tmp_path):
    with pytest.raises(RewardFileNotFoundError):
        parse_reward(tmp_path)
    (tmp_path / "reward.txt").write_text("1")
    (tmp_path / "reward.json").write_text("")
    with pytest.raises(RewardFileEmptyError):
        parse_reward(tmp_path)
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 0}))
    assert parse_reward(tmp_path) == {"reward": 0}


async def test_verifier_nonzero_exit_with_official_zero_is_completed(tmp_path):
    cfg = TaskSettings.model_validate(
        {"environment": {"docker_image": "a"}, "verifier": {"environment": {"docker_image": "v"}}}
    )
    env, main, _ = environment(tmp_path / "verifier", cfg)
    main.path("/tests/test.sh").write_text(f"#!/bin/sh\necho 0 > {main.path('/logs/verifier/reward.txt')}\nexit 4\n")
    diagnostics = []
    assert await run_verifier(env, tmp_path / "result", diagnostics) == {"rewards": {"reward": 0}}
    assert diagnostics[0]["return_code"] == 4


async def test_verifier_timeout_does_not_promote_partial_reward(tmp_path, monkeypatch):
    cfg = TaskSettings.model_validate(
        {"environment": {"docker_image": "a"}, "verifier": {"timeout_sec": 0.01, "environment": {"docker_image": "v"}}}
    )

    async def execute(*args, **kwargs):
        await asyncio.Event().wait()

    env = SimpleNamespace(task=SimpleNamespace(config=cfg), exec=execute, main=None)
    from resources_servers.terminal_bench_4 import verifier

    download = AsyncMock()
    monkeypatch.setattr(verifier, "download_dir", download)
    with pytest.raises(VerifierTimeoutError):
        await run_verifier(env, tmp_path, [])
    download.assert_awaited_once()


async def test_transfer_fallbacks_preserve_file_bytes_and_empty_directories(tmp_path):
    from resources_servers.terminal_bench_4.transfers import download_dir, upload_dir

    box = FilesystemSandbox(tmp_path / "sandbox")
    local = tmp_path / "source"
    (local / "empty").mkdir(parents=True)
    (local / "file.txt").write_bytes(b"\x00payload\xff")
    execute = box.exec

    async def without_tar(command, **kwargs):
        if "tar -xzf" in command or command.startswith("tar -czf"):
            return SimpleNamespace(return_code=1, stdout="", stderr="tar unavailable")
        result = await execute(command, **kwargs)
        if command.startswith("find "):
            result.stdout = result.stdout.replace(str(box.root), "")
        return result

    box.exec = without_tar
    await upload_dir(box, local, "/app/output")
    assert box.path("/app/output/file.txt").read_bytes() == b"\x00payload\xff"
    assert box.path("/app/output/empty").is_dir()
    await download_dir(box, "/app/output", tmp_path / "copied")
    assert (tmp_path / "copied/file.txt").read_bytes() == b"\x00payload\xff"
    with pytest.raises(RuntimeError, match="archive"):
        await download_dir(box, "/app/output", tmp_path / "excluded", exclude=["*.tmp"])
    with pytest.raises(RuntimeError, match="list"):
        await download_dir(box, "/app/missing", tmp_path / "missing")


async def test_restore_file_and_best_effort_collection_errors(tmp_path):
    cfg = TaskSettings.model_validate(
        {
            "environment": {"docker_image": "a"},
            "verifier": {"environment": {"docker_image": "v"}, "collect": [{"command": "failing-hook"}]},
            "artifacts": ["/app/output.txt", {"source": "/evidence/state.txt", "service": "db"}],
        }
    )
    env, main, sidecar = environment(tmp_path / "agent", cfg)
    main.path("/app/output.txt").write_text("file-data")
    sidecar.path("/evidence/state.txt").write_text("sidecar-data")
    original = env.exec

    async def execute(command, **kwargs):
        if command == "failing-hook" or command.startswith("test -d /app"):
            raise RuntimeError("optional hook/probe failed")
        return await original(command, **kwargs)

    async def stop():
        env.stopped = True
        raise RuntimeError("main stop failed")

    env.exec = execute
    env.stop_main = stop
    diagnostics = []
    records = await collect(env, tmp_path / "artifacts", diagnostics)
    assert records[1]["type"] == "file" and records[1]["status"] == "ok"
    assert {d["operation"] for d in diagnostics} == {"collect_hook", "stop_main"}
    verifier, box, _ = environment(tmp_path / "verifier", cfg)
    await restore(verifier, tmp_path / "artifacts")
    assert box.path("/app/output.txt").read_text() == "file-data"
    assert box.path("/evidence/state.txt").read_text() == "sidecar-data"
