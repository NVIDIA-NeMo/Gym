# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime release staging preserves code and rewrites direct and transitive pinned dependencies."""

import hashlib
import io
import json
import tarfile
import subprocess
import shutil
from pathlib import Path

import pytest

from responses_api_agents.prime_agent.sandbox import PRIME_DEPENDENCY_URL, _localize_release_package


@pytest.mark.parametrize("package", ["prime-agent", "prime-agent-core"])
def test_local_release_dependencies_preserve_package_contents(tmp_path: Path, package: str):
    source, target = tmp_path / "original.tgz", tmp_path / "uploaded.tgz"
    remote = f"{PRIME_DEPENDENCY_URL}/prime-agent-ai-0.7.0.tgz"
    local = "file:/tmp/session with spaces/prime-agent-ai.tgz"
    manifest = {"name": package, "dependencies": {"@earendil-works/pi-ai": remote, "chalk": "^5.5.0"}}
    code = b"export const value = 42;\n"
    with tarfile.open(source, "w:gz") as archive:
        for name, content in {"package/package.json": json.dumps(manifest).encode(), "package/dist/index.js": code}.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(content), 0o755
            archive.addfile(member, io.BytesIO(content))
    original = source.read_bytes()
    checksum = _localize_release_package(source, target, {remote: local})
    assert source.read_bytes() == original
    assert checksum == hashlib.sha256(target.read_bytes()).hexdigest()
    with tarfile.open(target) as archive:
        staged = json.load(archive.extractfile("package/package.json"))
        assert staged == {"name": package, "dependencies": {"@earendil-works/pi-ai": local, "chalk": "^5.5.0"}}
        assert archive.extractfile("package/dist/index.js").read() == code
        assert archive.getmember("package/dist/index.js").mode == 0o755


def test_missing_release_manifest_is_rejected(tmp_path: Path):
    source = tmp_path / "empty.tgz"
    with tarfile.open(source, "w:gz"):
        pass
    with pytest.raises(ValueError, match="no package/package.json"):
        _localize_release_package(source, tmp_path / "target.tgz", {})


@pytest.mark.skipif(any(shutil.which(tool) is None for tool in ["bash", "flock", "sha256sum", "getconf"]),
                    reason="Linux installer tools are required")
def test_installer_rejects_corrupt_release_dependency_before_install(tmp_path: Path):
    from responses_api_agents.prime_agent import sandbox

    asset = tmp_path / "asset"
    asset.write_bytes(b"uploaded asset")
    checksum = hashlib.sha256(asset.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime"
    command = ["bash", str(Path(sandbox.__file__).with_name("install_prime_runtime.sh")), str(runtime), "0.7.0",
               str(asset), checksum, str(asset), checksum, str(asset), checksum, str(asset), str(asset), "0" * 64]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert f"{asset}: FAILED" in result.stdout
    assert not runtime.exists()
    assert not list(tmp_path.glob("runtime.install.*"))
