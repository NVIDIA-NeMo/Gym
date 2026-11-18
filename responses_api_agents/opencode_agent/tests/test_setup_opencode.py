# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from unittest.mock import patch

import pytest

from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode, installed_opencode_version


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "expected"),
    [
        ("1.17.11\n", "", 0, "1.17.11"),
        ("", "1.17.11\n", 0, "1.17.11"),
        ("startup notice\n1.17.11\n", "", 0, "1.17.11"),
        ("", "", 0, None),
        (" \n\t", "", 0, None),
        ("1.17.11", "", 1, None),
        ("1.17.10", "", 0, "1.17.10"),
    ],
)
def test_installed_version_handles_unavailable_output(stdout, stderr, returncode, expected) -> None:
    result = subprocess.CompletedProcess(["opencode", "--version"], returncode, stdout, stderr)
    with (
        patch("responses_api_agents.opencode_agent.setup_opencode.shutil.which", return_value="/bin/opencode"),
        patch("responses_api_agents.opencode_agent.setup_opencode.subprocess.run", return_value=result),
    ):
        assert installed_opencode_version() == expected
        if expected == "1.17.11":
            ensure_opencode("1.17.11")
        else:
            with pytest.raises(RuntimeError, match="Expected OpenCode 1.17.11"):
                ensure_opencode("1.17.11")


def test_installed_version_timeout_is_unavailable() -> None:
    with (
        patch("responses_api_agents.opencode_agent.setup_opencode.shutil.which", return_value="/bin/opencode"),
        patch(
            "responses_api_agents.opencode_agent.setup_opencode.subprocess.run",
            side_effect=subprocess.TimeoutExpired("opencode --version", 60),
        ),
    ):
        assert installed_opencode_version() is None


def test_opencode_install_uses_shared_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.harness_conformance.ci import install_runtime

    from responses_api_agents.opencode_agent import setup_opencode
    from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
    from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION
    from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgentConfig

    versions = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup_opencode, "_npm_install", lambda npm, version: versions.append(version))
    install_runtime("opencode")
    assert versions == [OPENCODE_VERSION]
    assert OpenCodeAgentConfig.model_fields["opencode_version"].default == OPENCODE_VERSION
    assert OpenCodeSandboxedAgentConfig.model_fields["opencode_version"].default == OPENCODE_VERSION


class _Release:
    """A release download that serves ``body`` in chunks, or keeps serving chunks with ``endless``."""

    def __init__(self, body: bytes, *, endless: bool = False) -> None:
        self.body = body
        self.endless = endless

    def __enter__(self) -> "_Release":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        if self.endless:
            return b"x" * 16
        chunk, self.body = self.body[:size], self.body[size:]
        return chunk


@pytest.fixture
def opencode2_release(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A Linux host whose OpenCode 2.9.9 release is a tarball holding a stand-in binary; returns its bytes."""
    import hashlib
    import io
    import tarfile

    from responses_api_agents.opencode_agent import setup_opencode

    binary = b'#!/bin/sh\necho "opencode v2.9.9"\n'
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        info = tarfile.TarInfo("opencode")
        info.size = len(binary)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(binary))
    body = archive.getvalue()
    monkeypatch.setattr(setup_opencode, "_OPENCODE2_PREFIX", tmp_path / "runtime")
    monkeypatch.setattr(setup_opencode.sys, "platform", "linux")
    monkeypatch.setattr(setup_opencode, "_linux_release_arch", lambda: "x64-baseline")
    monkeypatch.setattr(
        setup_opencode, "OPENCODE_RELEASE_SHA256", {"2.9.9": {"x64-baseline": hashlib.sha256(body).hexdigest()}}
    )
    return body


def test_opencode_2_release_is_installed_once_with_bounded_reads(opencode2_release, monkeypatch, tmp_path) -> None:
    from responses_api_agents.opencode_agent import setup_opencode

    opened = []

    def urlopen(url, *, timeout):
        opened.append((url, timeout))
        return _Release(opencode2_release)

    monkeypatch.setattr(setup_opencode.urllib.request, "urlopen", urlopen)
    binary = setup_opencode.ensure_opencode2("2.9.9")
    assert binary == str(tmp_path / "runtime" / "2.9.9" / "opencode")
    assert opened == [
        (
            "https://github.com/anomalyco/opencode/releases/download/v2.9.9/opencode-linux-x64-baseline.tar.gz",
            setup_opencode._DOWNLOAD_READ_TIMEOUT_S,
        )
    ]
    # A second call finds the checked binary and downloads nothing.
    assert setup_opencode.ensure_opencode2("2.9.9") == binary
    assert len(opened) == 1


def test_a_stalled_opencode_2_download_fails_and_releases_the_lock(opencode2_release, monkeypatch, tmp_path) -> None:
    from responses_api_agents.opencode_agent import setup_opencode

    monkeypatch.setattr(setup_opencode, "_DOWNLOAD_TIMEOUT_S", 0)
    monkeypatch.setattr(setup_opencode.urllib.request, "urlopen", lambda url, *, timeout: _Release(b"", endless=True))
    with pytest.raises(TimeoutError, match="longer than 0 s"):
        setup_opencode.ensure_opencode2("2.9.9")
    assert not (tmp_path / "runtime" / "2.9.9" / "opencode").exists()
    # The lock was released: a later call downloads again and installs.
    monkeypatch.setattr(setup_opencode, "_DOWNLOAD_TIMEOUT_S", 600)
    monkeypatch.setattr(setup_opencode.urllib.request, "urlopen", lambda url, *, timeout: _Release(opencode2_release))
    assert setup_opencode.ensure_opencode2("2.9.9") == str(tmp_path / "runtime" / "2.9.9" / "opencode")


def test_an_opencode_2_download_with_another_digest_is_not_installed(opencode2_release, monkeypatch, tmp_path) -> None:
    from responses_api_agents.opencode_agent import setup_opencode

    monkeypatch.setattr(
        setup_opencode.urllib.request, "urlopen", lambda url, *, timeout: _Release(opencode2_release + b"tampered")
    )
    with pytest.raises(RuntimeError, match="not the pinned"):
        setup_opencode.ensure_opencode2("2.9.9")
    assert not (tmp_path / "runtime" / "2.9.9" / "opencode").exists()
