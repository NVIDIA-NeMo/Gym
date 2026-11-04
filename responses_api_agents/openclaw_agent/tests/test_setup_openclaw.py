# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from responses_api_agents.openclaw_agent import setup_openclaw


@pytest.fixture(autouse=True)
def _no_version_overrides(monkeypatch):
    """Keep expectations independent of the developer's shell."""
    monkeypatch.delenv(setup_openclaw.OPENCLAW_VERSION_ENV, raising=False)
    monkeypatch.delenv(setup_openclaw.NODE_VERSION_ENV, raising=False)


# Every (sys.platform, platform.machine()) pair we claim to support, mapped to the
# archive nodejs.org actually publishes. Verified against
# https://nodejs.org/dist/v24.21.0/SHASUMS256.txt.
SUPPORTED = [
    ("linux", "x86_64", "node-v24.21.0-linux-x64.tar.xz"),
    ("linux", "aarch64", "node-v24.21.0-linux-arm64.tar.xz"),
    ("darwin", "x86_64", "node-v24.21.0-darwin-x64.tar.xz"),
    ("darwin", "arm64", "node-v24.21.0-darwin-arm64.tar.xz"),
    ("win32", "AMD64", "node-v24.21.0-win-x64.zip"),
    ("win32", "ARM64", "node-v24.21.0-win-arm64.zip"),
]


@pytest.fixture
def fake_platform(monkeypatch):
    """Pretend to run on an arbitrary (sys.platform, machine) pair."""

    def _set(sys_platform: str, machine: str) -> None:
        monkeypatch.setattr(setup_openclaw.sys, "platform", sys_platform)
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: machine)
        # On Windows the module asks _windows_machine(), not platform.machine(),
        # because platform.machine() is unreliable under ARM64 x64 emulation.
        monkeypatch.setattr(setup_openclaw, "_windows_machine", lambda: machine)

    return _set


class _FakeWin32Function:
    """Callable Win32 export that accepts ctypes signature attributes."""

    def __init__(self, implementation):
        self._implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._implementation(*args)


class _FakeKernel32:
    """kernel32 double whose IsWow64Process2 reports a fixed native machine."""

    def __init__(self, native_machine: int, result: int = 1):
        self.current_process_handle = object()
        self.received_handle = None
        self.GetCurrentProcess = _FakeWin32Function(lambda: self.current_process_handle)

        def _is_wow64_process2(handle, process_machine_ref, native_machine_ref):
            self.received_handle = handle
            native_machine_ref._obj.value = native_machine
            return result

        self.IsWow64Process2 = _FakeWin32Function(_is_wow64_process2)


class _FakeWindll:
    """Stands in for ``ctypes.windll`` exposing a fake kernel32."""

    def __init__(self, kernel32):
        self.kernel32 = kernel32


class _Kernel32WithoutIsWow64Process2:
    """kernel32 double for pre-Windows-10 kernels: no IsWow64Process2 export."""

    def __init__(self):
        self.GetCurrentProcess = _FakeWin32Function(lambda: object())


def _patch_kernel32(monkeypatch, kernel32) -> None:
    """Point ``setup_openclaw.ctypes.windll.kernel32`` at a test double.

    ``ctypes.windll`` only exists on Windows; give non-Windows interpreters a
    stand-in attribute so the same test runs on both.
    """
    monkeypatch.setattr(setup_openclaw.ctypes, "windll", _FakeWindll(kernel32), raising=False)


class TestNodeDistUrl:
    @pytest.mark.parametrize(("sys_platform", "machine", "archive"), SUPPORTED)
    def test_builds_url_published_by_nodejs_org(self, fake_platform, sys_platform, machine, archive):
        fake_platform(sys_platform, machine)
        assert setup_openclaw._node_dist_url("24.21.0") == f"https://nodejs.org/dist/v24.21.0/{archive}"

    def test_windows_uses_zip_and_others_use_tar_xz(self, fake_platform):
        fake_platform("win32", "AMD64")
        assert setup_openclaw._node_dist_url("24.21.0").endswith(".zip")
        fake_platform("linux", "x86_64")
        assert setup_openclaw._node_dist_url("24.21.0").endswith(".tar.xz")

    def test_machine_spelling_is_case_insensitive(self, fake_platform):
        fake_platform("linux", "X86_64")
        assert "-linux-x64." in setup_openclaw._node_dist_url("24.21.0")

    def test_version_is_interpolated(self, fake_platform):
        fake_platform("linux", "x86_64")
        assert setup_openclaw._node_dist_url("26.1.0") == (
            "https://nodejs.org/dist/v26.1.0/node-v26.1.0-linux-x64.tar.xz"
        )

    def test_unsupported_os_raises_actionable_error(self, fake_platform):
        fake_platform("freebsd14", "x86_64")
        with pytest.raises(RuntimeError, match="freebsd14"):
            setup_openclaw._node_dist_url("24.21.0")

    def test_unsupported_arch_raises_actionable_error(self, fake_platform):
        fake_platform("linux", "riscv64")
        with pytest.raises(RuntimeError, match="riscv64"):
            setup_openclaw._node_dist_url("24.21.0")


class TestWindowsMachine:
    """platform.machine() on Windows reports the *interpreter's* architecture.

    An x64 python.exe running under emulation on ARM64 silicon gets "AMD64",
    and CPython 3.13+ flips between WMI truth (ARM64) and the emulated env var
    (AMD64) depending on whether the WMI service answers in time. The module
    therefore must not trust platform.machine() for the download choice.
    """

    def test_prefers_iswow64process2_over_platform_machine(self, monkeypatch):
        """The native machine and documented current-process handle drive detection."""
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        kernel32 = _FakeKernel32(native_machine=setup_openclaw._IMAGE_FILE_MACHINE_ARM64)
        _patch_kernel32(monkeypatch, kernel32)
        assert setup_openclaw._windows_machine() == "ARM64"
        assert kernel32.received_handle is kernel32.current_process_handle
        assert kernel32.GetCurrentProcess.argtypes == []
        assert kernel32.GetCurrentProcess.restype is setup_openclaw.wintypes.HANDLE
        assert kernel32.IsWow64Process2.argtypes == [
            setup_openclaw.wintypes.HANDLE,
            setup_openclaw.ctypes.POINTER(setup_openclaw.wintypes.USHORT),
            setup_openclaw.ctypes.POINTER(setup_openclaw.wintypes.USHORT),
        ]
        assert kernel32.IsWow64Process2.restype is setup_openclaw.wintypes.BOOL

    def test_recognised_amd64_native_machine(self, monkeypatch):
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "ARM64")
        _patch_kernel32(monkeypatch, _FakeKernel32(native_machine=setup_openclaw._IMAGE_FILE_MACHINE_AMD64))
        assert setup_openclaw._windows_machine() == "AMD64"

    def test_unrecognised_machine_code_raises_actionable_error(self, monkeypatch):
        """A valid-but-unmapped kernel answer is authoritative truth we cannot
        translate; falling back to platform.machine() (the lying primitive) would
        be worse than surfacing the gap, so the caller sees the actionable error."""
        # sys.platform only (not the fake_platform fixture, which would replace
        # _windows_machine and bypass the kernel answer entirely).
        monkeypatch.setattr(setup_openclaw.sys, "platform", "win32")
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        _patch_kernel32(monkeypatch, _FakeKernel32(native_machine=0xFFFF))
        with pytest.raises(RuntimeError, match="architecture"):
            setup_openclaw._node_platform()

    def test_api_absent_falls_back_to_platform_machine(self, monkeypatch):
        """Pre-Windows-10 kernels have no IsWow64Process2; behave as before."""
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        _patch_kernel32(monkeypatch, _Kernel32WithoutIsWow64Process2())
        assert setup_openclaw._windows_machine() == "AMD64"

    def test_api_failure_falls_back_to_platform_machine(self, monkeypatch):
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        _patch_kernel32(
            monkeypatch,
            _FakeKernel32(native_machine=setup_openclaw._IMAGE_FILE_MACHINE_ARM64, result=0),
        )
        assert setup_openclaw._windows_machine() == "AMD64"

    def test_emulated_interpreter_gets_native_download(self, fake_platform):
        """End-to-end trap: x64-emulated interpreter on an ARM64 host must pick
        the arm64 zip, not the x64 zip its own image would suggest."""
        fake_platform("win32", "ARM64")
        assert setup_openclaw._node_dist_url("24.21.0").endswith("node-v24.21.0-win-arm64.zip")


class TestNodeBinDir:
    def test_windows_launchers_sit_at_the_root(self, fake_platform):
        """The win-x64 zip has node.exe/npm.cmd at the top level, with no bin/."""
        fake_platform("win32", "AMD64")
        assert setup_openclaw._node_bin_dir(Path("/prefix")) == Path("/prefix")

    @pytest.mark.parametrize(("sys_platform", "machine"), [("linux", "x86_64"), ("darwin", "arm64")])
    def test_posix_uses_bin_subdirectory(self, fake_platform, sys_platform, machine):
        fake_platform(sys_platform, machine)
        assert setup_openclaw._node_bin_dir(Path("/prefix")) == Path("/prefix/bin")

    def test_architecture_without_download_does_not_raise(self, fake_platform):
        """An existing npm on linux/ppc64le needs the bin layout, not a download."""
        fake_platform("linux", "ppc64le")
        assert setup_openclaw._node_bin_dir(Path("/prefix")) == Path("/prefix/bin")


class TestExtractNodeArchive:
    def test_extracts_tar_xz(self, tmp_path):
        payload = tmp_path / "node-v24.21.0-linux-x64" / "bin"
        payload.mkdir(parents=True)
        (payload / "node").write_text("#!/bin/sh\n")
        archive = tmp_path / "node.tar.xz"
        with tarfile.open(archive, "w:xz") as tf:
            tf.add(payload.parent, arcname="node-v24.21.0-linux-x64")

        dest = tmp_path / "dest"
        dest.mkdir()
        setup_openclaw._extract_node_archive(archive, dest)

        assert (dest / "node-v24.21.0-linux-x64" / "bin" / "node").is_file()

    def test_extracts_zip(self, tmp_path):
        archive = tmp_path / "node.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("node-v24.21.0-win-x64/node.exe", "binary")

        dest = tmp_path / "dest"
        dest.mkdir()
        setup_openclaw._extract_node_archive(archive, dest)

        assert (dest / "node-v24.21.0-win-x64" / "node.exe").read_text() == "binary"


class TestFlattenExtractedNode:
    def test_hoists_payload_into_prefix(self, tmp_path):
        nested = tmp_path / "node-v24.21.0-darwin-arm64" / "bin"
        nested.mkdir(parents=True)
        (nested / "node").write_text("x")

        setup_openclaw._flatten_extracted_node(tmp_path)

        assert (tmp_path / "bin" / "node").is_file()
        assert not (tmp_path / "node-v24.21.0-darwin-arm64").exists()

    def test_raises_when_archive_layout_is_unexpected(self, tmp_path):
        (tmp_path / "unrelated").mkdir()
        with pytest.raises(RuntimeError, match="node-\\*"):
            setup_openclaw._flatten_extracted_node(tmp_path)


class TestResolveVersions:
    def test_config_value_is_used_when_env_is_unset(self, monkeypatch):
        monkeypatch.delenv(setup_openclaw.OPENCLAW_VERSION_ENV, raising=False)
        assert setup_openclaw.resolve_openclaw_version("2026.6.11") == "2026.6.11"

    def test_defaults_when_nothing_is_supplied(self, monkeypatch):
        monkeypatch.delenv(setup_openclaw.OPENCLAW_VERSION_ENV, raising=False)
        monkeypatch.delenv(setup_openclaw.NODE_VERSION_ENV, raising=False)
        assert setup_openclaw.resolve_openclaw_version(None) == setup_openclaw.DEFAULT_OPENCLAW_VERSION
        assert setup_openclaw.resolve_node_version() == setup_openclaw.DEFAULT_NODE_VERSION

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv(setup_openclaw.OPENCLAW_VERSION_ENV, "2026.9.5")
        monkeypatch.setenv(setup_openclaw.NODE_VERSION_ENV, "26.1.0")
        assert setup_openclaw.resolve_openclaw_version("2026.6.11") == "2026.9.5"
        assert setup_openclaw.resolve_node_version() == "26.1.0"

    def test_empty_env_falls_back_to_config(self, monkeypatch):
        monkeypatch.setenv(setup_openclaw.OPENCLAW_VERSION_ENV, "")
        assert setup_openclaw.resolve_openclaw_version("2026.6.11") == "2026.6.11"

    @pytest.mark.parametrize("spec", ["^2026.9.0", ">=2026.9.0", "2026.9", "latest", "<2026.9.4"])
    def test_ranges_are_rejected(self, spec):
        with pytest.raises(ValueError, match="exact version"):
            setup_openclaw.resolve_openclaw_version(spec)

    def test_prerelease_is_an_exact_version(self):
        assert setup_openclaw.resolve_openclaw_version("2026.9.4-beta.1") == "2026.9.4-beta.1"

    def test_default_node_satisfies_every_pinned_engine_range(self):
        """The default Node must satisfy every release in OPENCLAW_ENGINES_NODE."""
        for engines in setup_openclaw.OPENCLAW_ENGINES_NODE.values():
            assert setup_openclaw._satisfies_range(setup_openclaw.DEFAULT_NODE_VERSION, engines)


class TestSatisfiesRange:
    """The npm engine-range matcher, against operands openclaw actually publishes."""

    @pytest.mark.parametrize(
        ("node_version", "expected"),
        [
            ("24.21.0", True),  # the default pin
            ("24.16.0", True),  # inclusive lower bound
            ("24.15.9", False),  # just below the 24-line bound
            ("25.9.0", False),  # between the two alternatives (25 has no build)
            ("22.15.0", False),  # the stale cache case
            ("26.1.0", True),  # second alternative's lower bound
            ("26.0.9", False),
        ],
    )
    def test_openclaw_engine_range(self, node_version, expected):
        assert (
            setup_openclaw._satisfies_range(node_version, setup_openclaw.OPENCLAW_ENGINES_NODE["2026.9.4"]) is expected
        )

    def test_old_release_range(self):
        """openclaw 2026.6.11 declared '>=22.19.0' — 24.21.0 must still satisfy it."""
        assert setup_openclaw._satisfies_range("24.21.0", ">=22.19.0")
        assert not setup_openclaw._satisfies_range("22.15.0", ">=22.19.0")

    def test_leading_v_is_accepted(self):
        assert setup_openclaw._satisfies_range("v24.21.0", ">=22.19.0")

    def test_prerelease_suffix_is_ignored(self):
        assert setup_openclaw._satisfies_range("24.21.0-rc.1", ">=22.19.0")

    def test_wildcard_operand(self):
        assert setup_openclaw._satisfies_range("22.9.0", ">=22.x")
        assert not setup_openclaw._satisfies_range("21.9.0", ">=22.x")


class _FakeExecutables:
    """Stands in for shutil.which plus version probes of the found binaries.

    ``install(cmd, path, version)`` puts a binary at *path* whose ``--version``
    prints *version* (or exits non-zero when *version* is ``None``, i.e. an
    unusable binary). ``install_missing(cmd)`` hides it entirely. ``views``
    maps an ``openclaw@<version>`` spec to the ``npm view ... engines.node``
    answer; a missing spec makes ``npm view`` fail.
    """

    def __init__(self, monkeypatch):
        self._paths: dict[str, str | None] = {}
        self._versions: dict[str, str] = {}
        self.views: dict[str, str] = {}
        monkeypatch.setattr(setup_openclaw.shutil, "which", self._which)
        monkeypatch.setattr(setup_openclaw.subprocess, "run", self._run)

    def _which(self, cmd, path=None):
        found = self._paths.get(cmd)
        if not found:
            return None
        # A single-directory lookup only sees binaries inside it; a full PATH
        # search sees every registered binary.
        if path is not None and os.pathsep not in str(path) and not found.startswith(str(path)):
            return None
        return found

    def _run(self, cmd, **kwargs):
        if cmd[1:2] == ["view"]:
            answer = self.views.get(cmd[2])
            return subprocess.CompletedProcess(cmd, 0 if answer else 1, stdout=f"{answer or ''}\n", stderr="")
        version = self._versions.get(cmd[0])
        if version is None:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="cannot execute")
        stdout = f"v{version}\n" if cmd[0].endswith("/node") else f"{version}\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def install(self, cmd: str, path: str, version: str | None) -> None:
        """Register a binary at *path* whose ``--version`` prints *version*.

        ``version=None`` simulates an unusable binary (non-zero exit).
        """
        self._paths[cmd] = path
        if version is not None:
            self._versions[path] = version

    def install_missing(self, cmd: str) -> None:
        """Hide *cmd* from ``which`` entirely."""
        self._paths[cmd] = None


@pytest.fixture
def provisioning(monkeypatch, tmp_path):
    """Fake ``_install_node_locally`` that records calls.

    The local ``node``/``npm`` become visible only when provisioning actually
    runs, so a test can tell reuse from provisioning.
    """
    calls: list[str] = []
    local_bin = tmp_path / "local" / "bin"

    def install(fake: _FakeExecutables) -> list[str]:
        def _install_node_locally(node_version):
            calls.append(node_version)
            fake.install("node", str(local_bin / "node"), node_version)
            fake.install("npm", str(local_bin / "npm"), "11.0.0")
            return local_bin

        monkeypatch.setattr(setup_openclaw, "_install_node_locally", _install_node_locally)
        monkeypatch.setattr(setup_openclaw, "_prepend_path", lambda _dir: None)
        return calls

    install.local_npm = str(local_bin / "npm")
    return install


class TestEnsureNpmValidatesNode:
    """System npm must not be reused against a missing, broken or incompatible node."""

    def test_incompatible_system_node_is_bypassed(self, monkeypatch, tmp_path, provisioning):
        """Node 25 is outside both range alternatives; system npm must be bypassed."""
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "node"), "25.9.0")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.9.4", None) == provisioning.local_npm
        assert calls == [setup_openclaw.DEFAULT_NODE_VERSION]

    def test_compatible_system_node_is_reused(self, monkeypatch, tmp_path, provisioning):
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "node"), "24.21.0")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.9.4", None) == "/usr/bin/npm"
        assert calls == []

    def test_system_node_that_fails_to_run_is_bypassed(self, monkeypatch, tmp_path, provisioning):
        """A broken system node reports no version; that is not evidence of compatibility."""
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "node"), None)
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.9.4", None) == provisioning.local_npm
        assert calls == [setup_openclaw.DEFAULT_NODE_VERSION]

    def test_npm_without_any_node_is_bypassed(self, monkeypatch, provisioning):
        fake = _FakeExecutables(monkeypatch)
        fake.install_missing("node")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.9.4", None) == provisioning.local_npm
        assert len(calls) == 1

    def test_no_system_node_and_no_system_npm_provisions_locally(self, monkeypatch, provisioning):
        fake = _FakeExecutables(monkeypatch)
        fake.install_missing("node")
        fake.install_missing("npm")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.9.4", None) == provisioning.local_npm
        assert len(calls) == 1


class TestEngineRangePerRelease:
    """The Node requirement comes from the requested openclaw release, not the newest pin."""

    def test_older_pin_accepts_node_22_19(self, monkeypatch, tmp_path, provisioning):
        """openclaw 2026.6.11 declares '>=22.19.0'; no download is needed."""
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "node"), "22.19.0")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.6.11", None) == "/usr/bin/npm"
        assert calls == []

    def test_newer_pin_rejects_node_22_19(self, monkeypatch, tmp_path, provisioning):
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "node"), "22.19.0")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        calls = provisioning(fake)

        assert setup_openclaw._ensure_npm("2026.9.4", None) == provisioning.local_npm
        assert len(calls) == 1

    def test_unknown_release_asks_npm_view(self, monkeypatch, tmp_path, provisioning):
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "node"), "24.21.0")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        fake.views["openclaw@2026.10.1"] = ">=26.1.0"
        calls = provisioning(fake)

        assert setup_openclaw._engines_node("2026.10.1", None) == ">=26.1.0"
        assert setup_openclaw._ensure_npm("2026.10.1", None) == provisioning.local_npm
        assert len(calls) == 1

    def test_unknown_release_without_registry_uses_newest_known_range(self, monkeypatch):
        fake = _FakeExecutables(monkeypatch)
        fake.install("npm", "/usr/bin/npm", "10.9.0")

        assert setup_openclaw._engines_node("2026.10.1", None) == setup_openclaw._NEWEST_KNOWN_ENGINES_NODE

    def test_incompatible_configured_runtime_raises(self, monkeypatch, tmp_path, provisioning):
        """node_bin_dir precedes any private toolchain at rollout time, so provisioning cannot help."""
        fake = _FakeExecutables(monkeypatch)
        fake.install("node", str(tmp_path / "deps" / "node"), "20.11.0")
        fake.install("npm", str(tmp_path / "deps" / "npm"), "10.9.0")
        calls = provisioning(fake)

        with pytest.raises(RuntimeError, match="node_bin_dir"):
            setup_openclaw._ensure_npm("2026.6.11", str(tmp_path / "deps"))
        assert calls == []


class TestInstallNodeLocallyValidatesCache:
    """A cached toolchain must be replaced when incompatible."""

    def _write_cached_node(self, prefix, version_script):
        bin_dir = prefix / "bin"
        bin_dir.mkdir(parents=True)
        node = bin_dir / "node"
        node.write_text(version_script)
        node.chmod(0o755)
        return node

    def test_matching_cache_is_reused(self, monkeypatch, tmp_path):
        requested = setup_openclaw.DEFAULT_NODE_VERSION
        self._write_cached_node(tmp_path, f"#!/bin/sh\necho v{requested}\n")
        monkeypatch.setattr(setup_openclaw, "_LOCAL_PREFIX", tmp_path)

        assert setup_openclaw._install_node_locally(requested) == tmp_path / "bin"
        # Nothing was re-downloaded: the cache survived intact.
        assert (tmp_path / "bin" / "node").read_text().startswith("#!/bin/sh")

    def test_stale_cache_is_replaced(self, monkeypatch, tmp_path):
        """A cache from an older pin (22.15.0) must not survive a 24.21.0 request."""
        self._write_cached_node(tmp_path, "#!/bin/sh\necho v22.15.0\n")
        monkeypatch.setattr(setup_openclaw, "_LOCAL_PREFIX", tmp_path)

        downloaded: list[str] = []

        def fake_download(url, dest):
            downloaded.append(url)
            # Write an archive whose layout _flatten_extracted_node accepts.
            with tarfile.open(dest, "w:xz") as tf:
                info = tarfile.TarInfo(f"node-v{setup_openclaw.DEFAULT_NODE_VERSION}-linux-x64/bin/node")
                payload = b"#!/bin/sh\n"
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))

        monkeypatch.setattr(setup_openclaw, "_download_node_archive", fake_download)

        setup_openclaw._install_node_locally(setup_openclaw.DEFAULT_NODE_VERSION)

        assert downloaded, "the stale cache must be wiped and re-provisioned"
        assert (tmp_path / "bin" / "node").read_bytes() == b"#!/bin/sh\n"

    def test_corrupt_cache_is_replaced(self, monkeypatch, tmp_path):
        """A cache whose node exits non-zero (e.g. wrong-arch binary) is unusable."""
        self._write_cached_node(tmp_path, "#!/bin/sh\nexit 1\n")
        monkeypatch.setattr(setup_openclaw, "_LOCAL_PREFIX", tmp_path)
        monkeypatch.setattr(setup_openclaw, "_download_node_archive", lambda url, dest: None)

        # The download/extract pair is faked end-to-end: after "extraction" the
        # prefix holds a working node of the requested version.
        def fake_extract(archive, prefix):
            nested = prefix / f"node-v{setup_openclaw.DEFAULT_NODE_VERSION}-linux-x64"
            (nested / "bin").mkdir(parents=True, exist_ok=True)
            fresh = nested / "bin" / "node"
            fresh.write_text("#!/bin/sh\necho v24.21.0\n")
            fresh.chmod(0o755)

        monkeypatch.setattr(setup_openclaw, "_extract_node_archive", fake_extract)

        setup_openclaw._install_node_locally(setup_openclaw.DEFAULT_NODE_VERSION)

        # The prefix was wiped and re-provisioned: the launcher at bin/node now
        # comes from the fresh extraction, not the corrupt original.
        assert (tmp_path / "bin" / "node").read_text() == "#!/bin/sh\necho v24.21.0\n"

    def test_missing_cache_is_provisioned(self, monkeypatch, tmp_path):
        monkeypatch.setattr(setup_openclaw, "_LOCAL_PREFIX", tmp_path)
        monkeypatch.setattr(setup_openclaw, "_download_node_archive", lambda url, dest: None)

        def fake_extract(archive, prefix):
            nested = prefix / f"node-v{setup_openclaw.DEFAULT_NODE_VERSION}-linux-x64"
            (nested / "bin").mkdir(parents=True)
            fresh = nested / "bin" / "node"
            fresh.write_text("#!/bin/sh\necho v24.21.0\n")
            fresh.chmod(0o755)

        monkeypatch.setattr(setup_openclaw, "_extract_node_archive", fake_extract)

        assert setup_openclaw._install_node_locally(setup_openclaw.DEFAULT_NODE_VERSION) == tmp_path / "bin"


class TestEnsureOpenclawRespectsRequestedVersion:
    """An existing install must not shadow a version override."""

    def _fake_openclaw(self, monkeypatch, reported: str | None):
        fake = _FakeExecutables(monkeypatch)
        fake.install("openclaw", "/usr/local/bin/openclaw", reported)
        fake.install("node", "/usr/bin/node", "24.21.0")
        fake.install("npm", "/usr/bin/npm", "10.9.0")
        return fake

    def _record_install(self, monkeypatch, fake: _FakeExecutables) -> list[str]:
        """Fake npm install that makes the selected launcher report the installed version."""
        installed: list[str] = []

        def fake_install(npm, version, node_bin_dir):
            installed.append(version)
            fake.install("openclaw", "/usr/local/bin/openclaw", version)

        monkeypatch.setattr(setup_openclaw, "_npm_install", fake_install)
        monkeypatch.setattr(setup_openclaw, "_adopt_npm_global_bin", lambda npm, node_bin_dir: False)
        return installed

    def test_matching_existing_install_is_kept(self, monkeypatch):
        fake = self._fake_openclaw(monkeypatch, "2026.9.4")
        installed = self._record_install(monkeypatch, fake)

        setup_openclaw.ensure_openclaw("2026.9.4")
        assert installed == []

    def test_version_override_reinstalls_other_release(self, monkeypatch):
        """An override must take effect even when openclaw is already installed."""
        fake = self._fake_openclaw(monkeypatch, "2026.9.4")
        installed = self._record_install(monkeypatch, fake)

        setup_openclaw.ensure_openclaw()
        assert installed == [setup_openclaw.DEFAULT_OPENCLAW_VERSION]

    def test_env_override_beats_existing_install(self, monkeypatch):
        fake = self._fake_openclaw(monkeypatch, "2026.9.4")
        installed = self._record_install(monkeypatch, fake)
        monkeypatch.setenv(setup_openclaw.OPENCLAW_VERSION_ENV, "2026.6.11")

        setup_openclaw.ensure_openclaw("2026.9.4")
        assert installed == ["2026.6.11"]

    def test_unreporting_launcher_is_reinstalled(self, monkeypatch):
        """A shim whose node cannot run reports no version — not acceptable."""
        fake = self._fake_openclaw(monkeypatch, None)
        installed = self._record_install(monkeypatch, fake)

        setup_openclaw.ensure_openclaw()
        assert installed == [setup_openclaw.DEFAULT_OPENCLAW_VERSION]

    def test_prerelease_is_not_the_stable_release(self, monkeypatch):
        fake = self._fake_openclaw(monkeypatch, "2026.9.4-beta.1")
        installed = self._record_install(monkeypatch, fake)

        setup_openclaw.ensure_openclaw("2026.9.4")
        assert installed == ["2026.9.4"]


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell launchers")
class TestRealPathSelection:
    """Exercise real ``shutil.which``/``subprocess`` lookup with executable fixtures."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(setup_openclaw, "_USER_LOCAL_BIN", tmp_path / "no-local-bin")
        monkeypatch.setattr(setup_openclaw, "_LOCAL_PREFIX", tmp_path / "no-local-prefix")

    def _fake_npm(self, tools: Path, prefix: Path, installed_version: str | None) -> None:
        """npm whose ``install`` writes ``openclaw`` into *prefix*/bin (or nothing)."""
        install = (
            f'mkdir -p "{prefix}/bin" && printf \'#!/bin/sh\\necho {installed_version}\\n\' > "{prefix}/bin/openclaw"'
            f' && chmod +x "{prefix}/bin/openclaw"'
            if installed_version
            else "true"
        )
        _script(
            tools / "npm",
            f'case "$1" in prefix) echo "{prefix}";; install) {install};; *) exit 1;; esac',
        )
        _script(tools / "node", "echo v24.21.0")

    def test_new_install_shadows_older_launcher_on_path(self, monkeypatch, tmp_path):
        _script(tmp_path / "old" / "bin" / "openclaw", "echo 2026.6.11")
        self._fake_npm(tmp_path / "tools", tmp_path / "prefix", "2026.9.4")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "old" / "bin"), str(tmp_path / "tools"), "/usr/bin:/bin"])
        )

        setup_openclaw.ensure_openclaw("2026.9.4")

        selected = shutil.which("openclaw")
        assert selected == str(tmp_path / "prefix" / "bin" / "openclaw")
        assert subprocess.run([selected], capture_output=True, text=True).stdout.strip() == "2026.9.4"

    def test_install_that_leaves_old_launcher_selected_fails(self, monkeypatch, tmp_path):
        _script(tmp_path / "old" / "bin" / "openclaw", "echo 2026.6.11")
        self._fake_npm(tmp_path / "tools", tmp_path / "prefix", None)
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "old" / "bin"), str(tmp_path / "tools"), "/usr/bin:/bin"])
        )

        with pytest.raises(RuntimeError, match="reports 2026.6.11"):
            setup_openclaw.ensure_openclaw("2026.9.4")

    def test_configured_runtime_is_used_for_probe(self, monkeypatch, tmp_path):
        """anyterminal layout: task Node 20 first on PATH, bundled runtime appended.

        The bundled launcher refuses Node 20, like the real one. With
        ``node_bin_dir`` set, setup must see the bundled launcher working, return
        without installing, and leave PATH alone (the runner's order is deliberate).
        """
        _script(tmp_path / "task" / "bin" / "node", "echo v20.11.0")
        deps = tmp_path / "deps" / "bin"
        _script(deps / "node", "echo v22.19.0")
        _script(
            deps / "openclaw",
            'case "$(node --version)" in v22.*) echo 2026.6.11;; *) echo "node too old" >&2; exit 1;; esac',
        )
        path = os.pathsep.join([str(tmp_path / "task" / "bin"), "/usr/bin:/bin", str(deps)])
        monkeypatch.setenv("PATH", path)
        read_only = tmp_path / "no-local-prefix"
        read_only.mkdir(mode=0o555)

        def fail_install(*args, **kwargs):
            raise AssertionError("setup must not install when the configured runtime is compatible")

        monkeypatch.setattr(setup_openclaw, "_npm_install", fail_install)
        monkeypatch.setattr(setup_openclaw, "_install_node_locally", fail_install)

        setup_openclaw.ensure_openclaw("2026.6.11", node_bin_dir=str(deps))

        assert os.environ["PATH"] == path
