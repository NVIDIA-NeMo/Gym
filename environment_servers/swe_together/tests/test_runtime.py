# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import io
import json
import shutil
import sys
import tarfile
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


@pytest.mark.asyncio
async def test_cache_upload_once_and_private_role_mounts(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    root.mkdir()
    monkeypatch.setattr(runtime, "STORAGE_ROOT", str(root))

    class Provider:
        def shared_volume_options(self, source, target, *, read_only=False):
            return {"volumes": [{"subPath": source, "mountPath": target, "readOnly": read_only}]}

        def shared_volume_metadata(self):
            return {"placement": "shared"}

    provider = Provider()
    monkeypatch.setattr(runtime, "create_provider", lambda _: provider)

    class Helper:
        def __init__(self, provider):
            self.uploads = 0
            self.stopped = False

        async def start(self, spec):
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
    provisioner = runtime.RuntimeProvisioner(
        provider_config={}, manifest_path=manifest(tmp_path), helper_image="helper"
    )
    first = await provisioner.prepare("one")
    second = await provisioner.prepare("two")
    assert first != second and provisioner.helper.uploads == 2
    assert provisioner.helper.spec.provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
    for session in ["one", "two"]:
        candidate = provisioner.mount_options(session, "candidate")["volumes"][0]
        judge = provisioner.mount_options(session, "judge")["volumes"][0]
        assert candidate["readOnly"] and judge["readOnly"]
        assert candidate["mountPath"] == judge["mountPath"] == "/opt/gym-runtime"
        candidate_root, judge_root = root / candidate["subPath"], root / judge["subPath"]
        assert set(path.name for path in candidate_root.iterdir()) == {"owner", "python"}
        assert set(path.name for path in judge_root.iterdir()) == {"owner", "python"}
        assert (candidate_root / "python/bin/python3").is_file()
    await provisioner.release("one")
    assert not (root / first).exists() and (root / second).exists()
    with pytest.raises(RuntimeError, match="remain owned"):
        await provisioner.close()
    await provisioner.release("two")
    await provisioner.close()
    assert provisioner.helper.stopped and (root / "gym-swet-runtime-cache").is_dir()
