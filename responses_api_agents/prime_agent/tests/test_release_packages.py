# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime release staging preserves code, rewrites pinned release dependencies and refuses remote ones."""

import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from responses_api_agents.prime_agent.sandbox import _localize_release_package


# Written out rather than imported, so a stale constant in sandbox.py cannot match itself.
REMOTE_AI = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/releases/v0.7.0/prime-agent-ai-0.7.0.tgz"


def release_archive(path: Path, manifest: dict, files: dict[str, bytes] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in {"package/package.json": json.dumps(manifest).encode(), **(files or {})}.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(content), 0o755
            archive.addfile(member, io.BytesIO(content))
    return path


def test_local_release_dependencies_preserve_package_contents(tmp_path: Path):
    code = b"export const value = 42;\n"
    manifest = {"name": "prime-agent-core", "dependencies": {"@earendil-works/pi-ai": REMOTE_AI, "chalk": "^5.5.0"}}
    source = release_archive(tmp_path / "original.tgz", manifest, {"package/dist/index.js": code})
    target = tmp_path / "uploaded.tgz"
    local = "file:/tmp/session with spaces/prime-agent-ai.tgz"
    original = source.read_bytes()
    checksum = _localize_release_package(source, target, {REMOTE_AI: local})
    assert source.read_bytes() == original
    assert checksum == hashlib.sha256(target.read_bytes()).hexdigest()
    with tarfile.open(target) as archive:
        staged = json.load(archive.extractfile("package/package.json"))
        assert staged == {
            "name": "prime-agent-core",
            "dependencies": {"@earendil-works/pi-ai": local, "chalk": "^5.5.0"},
        }
        assert archive.extractfile("package/dist/index.js").read() == code
        assert archive.getmember("package/dist/index.js").mode == 0o755


@pytest.mark.parametrize("field", ["dependencies", "optionalDependencies", "peerDependencies"])
def test_remote_release_dependency_left_in_place_is_rejected(tmp_path: Path, field: str):
    # A version bump whose manifest URLs no longer match must fail on the agent server, not fetch in the sandbox.
    source = release_archive(tmp_path / "prime-agent.tgz", {"name": "prime-agent", field: {"pi-ai": REMOTE_AI}})
    with pytest.raises(ValueError, match=re.escape(f"prime-agent.tgz: {field} pi-ai still points at {REMOTE_AI}")):
        _localize_release_package(source, tmp_path / "target.tgz", {})


def test_missing_release_manifest_is_rejected(tmp_path: Path):
    source = tmp_path / "empty.tgz"
    with tarfile.open(source, "w:gz"):
        pass
    with pytest.raises(ValueError, match="empty.tgz: no package/package.json"):
        _localize_release_package(source, tmp_path / "target.tgz", {})


@pytest.mark.skipif(
    sys.platform != "linux"
    or platform.libc_ver()[0] != "glibc"
    or any(shutil.which(tool) is None for tool in ["bash", "flock", "sha256sum", "getconf"]),
    reason="The installer needs glibc Linux with bash, flock, sha256sum and getconf",
)
def test_installer_rejects_corrupt_release_dependency_before_install(tmp_path: Path):
    from responses_api_agents.prime_agent import sandbox

    asset = tmp_path / "asset"
    asset.write_bytes(b"uploaded asset")
    checksum = hashlib.sha256(asset.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime"
    # The production argument shape: Node, package, Python, uv, then three dependency pairs, the last one corrupt.
    command = [
        "bash",
        str(Path(sandbox.__file__).with_name("install_prime_runtime.sh")),
        str(runtime),
        "0.7.0",
        *[str(asset), checksum] * 3,
        str(asset),
        *[str(asset), checksum] * 2,
        str(asset),
        "0" * 64,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    # Node, package, Python and the first two dependencies pass; the last dependency fails before tar or npm runs.
    assert result.stdout.count(f"{asset}: OK") == 5
    assert f"{asset}: FAILED" in result.stdout
    assert not runtime.exists()
    assert not list(tmp_path.glob("runtime.install.*"))


@pytest.mark.parametrize("cache_matches_recipe", [True, False])
def test_installer_reuses_only_the_matching_runtime_recipe(tmp_path: Path, cache_matches_recipe: bool) -> None:
    from responses_api_agents.prime_agent import sandbox

    runtime, session = tmp_path / "runtime", tmp_path / "session"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "kernel/bin").mkdir(parents=True)
    session.mkdir()
    # The real installer executes these health checks before deciding whether to reuse the runtime.
    node, python = runtime / "bin/node", runtime / "kernel/bin/python3"
    node.write_text("#!/bin/sh\nprintf '%s\\n' 0.7.0\n")
    python.write_text("#!/bin/sh\nexit 0\n")
    node.chmod(0o755)
    python.chmod(0o755)
    profile = "1" * 64 + "\n"
    (session / "runtime-profile.txt").write_text(profile)
    (runtime / "ready").write_text(profile if cache_matches_recipe else "")
    installer = session / "install_prime_runtime.sh"
    shutil.copy2(Path(sandbox.__file__).with_name("install_prime_runtime.sh"), installer)
    asset = tmp_path / "asset"
    asset.write_bytes(b"invalid replacement archives must not be needed for a compatible cache")
    checksum = hashlib.sha256(asset.read_bytes()).hexdigest()
    result = subprocess.run(
        ["bash", str(installer), str(runtime), "0.7.0", *[str(asset), checksum] * 3, str(asset)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    # A matching recipe reuses the checked runtime. An empty marker from the old installer must rebuild and fail
    # on these intentionally invalid archives (or the unsupported host), rather than silently accepting that cache.
    assert (result.returncode == 0) is cache_matches_recipe, result.stdout + result.stderr
