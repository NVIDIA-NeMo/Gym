# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from shlex import quote
from types import SimpleNamespace

import pytest

from environment_servers.swe_together import runtime


def manifest(tmp_path):
    artifacts = {}
    for name in ["python", "opencode", "rg", "claude"]:
        source = tmp_path / (name + ".tar.gz" if name == "python" else name)
        if name == "python":
            with tarfile.open(source, "w:gz") as archive:
                content = b"#!/bin/sh\necho Python\n"
                entry = tarfile.TarInfo("python/bin/python3")
                entry.size = len(content)
                entry.mode = 0o755
                archive.addfile(entry, io.BytesIO(content))
        else:
            source.write_bytes(name.encode())
        artifacts[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        if name == "python":
            artifacts[name]["format"] = "tar.gz"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "artifacts": artifacts}))
    return path


def test_runtime_hashes_are_checked_before_provider_allocation(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    (tmp_path / "python.tar.gz").write_text("changed")
    monkeypatch.setattr(runtime, "create_provider", lambda _: pytest.fail("provider allocated before validation"))
    with pytest.raises(ValueError, match="SHA-256 mismatch: python"):
        runtime.RuntimeProvisioner(provider_config={}, manifest_path=path, helper_image="helper")


@pytest.fixture
def provisioner(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    root.mkdir()
    monkeypatch.setattr(runtime, "STORAGE_ROOT", str(root))

    class Provider:
        def shared_volume_options(self, source, target, *, read_only=False):
            return {"volumes": [{"subPath": source, "mountPath": target, "readOnly": read_only}]}

        def shared_volume_metadata(self):
            return {"placement": "shared"}

    monkeypatch.setattr(runtime, "create_provider", lambda _: Provider())

    class Helper:
        def __init__(self, provider):
            self.uploads = 0
            self.stopped = False

        async def start(self, spec):
            if self.stopped:
                raise RuntimeError("Helper is closed")
            self.spec = spec

        async def exec(self, command, **kwargs):
            command = command.replace("python3 ", quote(sys.executable) + " ", 1)
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return SimpleNamespace(
                return_code=process.returncode, error_type=None, stdout=stdout.decode(), stderr=stderr.decode()
            )

        async def upload(self, source, target):
            self.uploads += 1
            shutil.copyfile(source, target)

        async def stop(self):
            self.stopped = True

    monkeypatch.setattr(runtime, "AsyncSandbox", Helper)
    return runtime.RuntimeProvisioner(provider_config={}, manifest_path=manifest(tmp_path), helper_image="helper")


def max_cache_links(root: Path) -> int:
    return max(path.lstat().st_nlink for path in (root / runtime.CACHE).rglob("*") if path.is_file())


@pytest.mark.asyncio
async def test_cache_upload_once_and_direct_read_only_mounts(provisioner):
    root = Path(runtime.STORAGE_ROOT)
    first = await provisioner.prepare("one")
    second = await provisioner.prepare("two")
    assert first != second and provisioner.helper.uploads == 4
    assert provisioner.helper.spec.provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
    artifacts = provisioner.manifest["artifacts"]
    for session, bundle in [("one", first), ("two", second)]:
        candidate = provisioner.mount_options(session, "candidate")["volumes"]
        judge = provisioner.mount_options(session, "judge")["volumes"]
        assert candidate[:3] == [
            {
                "subPath": "gym-swet-runtime-cache/" + artifacts["python"]["sha256"] + ".python/python",
                "mountPath": "/opt/gym-runtime/python",
                "readOnly": True,
            },
            {
                "subPath": "gym-swet-runtime-cache/" + artifacts["opencode"]["sha256"],
                "mountPath": "/opt/gym-runtime/opencode",
                "readOnly": True,
            },
            {
                "subPath": "gym-swet-runtime-cache/" + artifacts["rg"]["sha256"],
                "mountPath": "/opt/gym-runtime/rg",
                "readOnly": True,
            },
        ]
        # The judge never sees the candidate's tools or a writable submission.
        assert judge[:1] == candidate[:1]
        assert candidate[3] == {
            "subPath": bundle + "/submission-candidate",
            "mountPath": "/opt/gym-submission",
            "readOnly": False,
        }
        assert judge[1:] == [
            {"subPath": bundle + "/submission-judge", "mountPath": "/opt/gym-submission", "readOnly": True}
        ]
        assert (root / candidate[0]["subPath"] / "bin/python3").is_file()
        assert (root / candidate[1]["subPath"]).read_bytes() == b"opencode"
        assert set(path.name for path in (root / bundle).iterdir()) == {"submission-candidate", "submission-judge"}
    assert max_cache_links(root) == 1
    await provisioner.release("one")
    assert not (root / first).exists() and (root / second).exists()
    with pytest.raises(RuntimeError, match="remain owned"):
        await provisioner.close()
    await provisioner.release("two")
    await provisioner.close()
    assert provisioner.helper.stopped and (root / "gym-swet-runtime-cache").is_dir()
    assert [path.name for path in root.iterdir()] == ["gym-swet-runtime-cache"]


@pytest.mark.asyncio
async def test_repeated_concurrent_prepare_and_release_add_no_cache_links(provisioner):
    root = Path(runtime.STORAGE_ROOT)
    for round in range(3):
        sessions = [f"{round}-{index}" for index in range(12)]
        await asyncio.gather(*(provisioner.prepare(session) for session in sessions))
        assert len(list((root / ("gym-swet-runtime-" + provisioner._owner)).iterdir())) == 12
        assert max_cache_links(root) == 1
        await asyncio.gather(*(provisioner.release(session) for session in sessions))
    assert provisioner._bundles == {} and max_cache_links(root) == 1
    assert [path.name for path in root.iterdir()] == ["gym-swet-runtime-cache"]
    await provisioner.close()


@pytest.mark.asyncio
async def test_interrupted_driver_leaves_no_cache_links_for_the_next_driver(provisioner, tmp_path):
    root = Path(runtime.STORAGE_ROOT)
    abandoned = [await provisioner.prepare(f"interrupted-{index}") for index in range(4)]
    # Bundles from earlier Gym releases hold hard links to every cached runtime file.
    cached = next((root / runtime.CACHE).rglob("python3"))
    earlier = root / ("gym-swet-runtime-" + "e" * 32)
    (earlier / "bundle/candidate").mkdir(parents=True)
    os.link(cached, earlier / "bundle/candidate/python3")

    restarted = runtime.RuntimeProvisioner(
        provider_config={}, manifest_path=tmp_path / "manifest.json", helper_image="helper"
    )
    for round in range(3):
        sessions = [f"{round}-{index}" for index in range(8)]
        await asyncio.gather(*(restarted.prepare(session) for session in sessions))
        await asyncio.gather(*(restarted.release(session) for session in sessions))
    # Nothing deletes another owner's directories; their links neither grow nor block preparation.
    assert all((root / bundle).is_dir() for bundle in abandoned) and earlier.is_dir()
    assert cached.stat().st_nlink == 2 and max_cache_links(root) == 2
    assert not (root / ("gym-swet-runtime-" + restarted._owner)).exists()
    # Only the worker is uploaded: the restarted driver reuses the verified cache.
    assert restarted.helper.uploads == 1
    await restarted.close()


@pytest.mark.asyncio
async def test_failed_worker_keeps_traceback_from_background_output(provisioner, monkeypatch):
    await provisioner.prepare("one")
    execute = provisioner.helper.exec

    async def background_exec(command, **kwargs):
        # OpenSandbox background commands return combined output in stdout.
        result = await execute(command, **kwargs)
        return SimpleNamespace(
            return_code=result.return_code, error_type=None, stdout=result.stderr, stderr="exit status 1"
        )

    monkeypatch.setattr(provisioner.helper, "exec", background_exec)
    monkeypatch.setattr(runtime, "uuid4", lambda: SimpleNamespace(hex=provisioner._bundles["one"].split("/")[1]))
    with pytest.raises(RuntimeError) as failure:
        await provisioner.prepare("two")
    message = str(failure.value)
    assert message.startswith("Trusted runtime preparation failed: prepare exited 1 (process): exit status 1\n")
    assert "FileExistsError: [Errno 17] File exists" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_fails", [False, True])
async def test_failed_startup_retries_with_fresh_helper_and_provider(provisioner, monkeypatch, stop_fails):
    helper, provider = provisioner.helper, provisioner.provider
    cache = provisioner._cache
    stop = helper.stop

    async def fail_cache_once(name, artifact):
        monkeypatch.setattr(provisioner, "_cache", cache)
        raise RuntimeError("Transient cache failure")

    async def fail_stop_once():
        monkeypatch.setattr(helper, "stop", stop)
        raise RuntimeError("Transient stop failure")

    monkeypatch.setattr(provisioner, "_cache", fail_cache_once)
    if stop_fails:
        monkeypatch.setattr(helper, "stop", fail_stop_once)
    with pytest.raises(RuntimeError, match="Transient stop failure" if stop_fails else "Transient cache failure"):
        await provisioner.prepare("one")
    if stop_fails:
        assert provisioner.helper is helper and provisioner.provider is provider
        assert provisioner._start_task.done() and not helper.stopped
        with pytest.raises(RuntimeError, match="Transient cache failure"):
            await provisioner.prepare("one")
    assert helper.stopped
    assert provisioner.helper is not helper and provisioner.provider is not provider
    await provisioner.prepare("two")
    await provisioner.release("two")
    await provisioner.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_preserves_shared_startup(provisioner, monkeypatch):
    entered, resume = asyncio.Event(), asyncio.Event()
    cache = provisioner._cache
    helper = provisioner.helper

    async def delayed_cache(name, artifact):
        entered.set()
        await resume.wait()
        await cache(name, artifact)

    monkeypatch.setattr(provisioner, "_cache", delayed_cache)
    waiter = asyncio.create_task(provisioner.prepare("one"))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not provisioner._start_task.done() and not helper.stopped
    resume.set()
    await provisioner.prepare("two")
    assert provisioner.helper is helper and helper.uploads == 4
    await provisioner.release("two")
    await provisioner.close()


@pytest.mark.asyncio
async def test_concurrent_release_and_prepare_are_serialized(provisioner, monkeypatch):
    first, second = await provisioner.prepare("one"), await provisioner.prepare("two")
    execute = provisioner._exec
    active = 0

    async def exclusive_exec(action, *args):
        nonlocal active
        active += 1
        try:
            assert active == 1, "Runtime bundle filesystem operations overlap"
            await asyncio.sleep(0)
            return await execute(action, *args)
        finally:
            active -= 1

    monkeypatch.setattr(provisioner, "_exec", exclusive_exec)
    await asyncio.gather(provisioner.release("one"), provisioner.release("two"), provisioner.prepare("three"))
    root = Path(runtime.STORAGE_ROOT)
    assert not (root / first).exists() and not (root / second).exists()
    assert set(provisioner._bundles) == {"three"}
    assert (root / provisioner._bundles["three"]).is_dir()
    await provisioner.release("three")
    await provisioner.close()


@pytest.mark.asyncio
async def test_failed_release_retains_bundle_for_retry(provisioner, monkeypatch):
    relative = await provisioner.prepare("one")
    execute = provisioner._exec

    async def fail_release(action, *args):
        raise RuntimeError("Transient deletion failure")

    monkeypatch.setattr(provisioner, "_exec", fail_release)
    with pytest.raises(RuntimeError, match="Transient deletion failure"):
        await provisioner.release("one")
    assert provisioner._bundles == {"one": relative}
    with pytest.raises(RuntimeError, match="remain owned"):
        await provisioner.close()
    monkeypatch.setattr(provisioner, "_exec", execute)
    await provisioner.release("one")
    await provisioner.close()
