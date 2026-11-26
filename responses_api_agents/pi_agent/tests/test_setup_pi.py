# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several worker processes share one local Pi install."""

import io
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from responses_api_agents.pi_agent import setup_pi as setup


def _executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with a system npm whose global prefix is not on PATH, and no pi yet."""
    monkeypatch.setattr(setup, "_LOCAL_PREFIX", tmp_path / ".pi_node")
    monkeypatch.setattr(setup, "_INSTALL_LOCK", tmp_path / ".pi_node.lock", raising=False)
    monkeypatch.setattr(setup, "_INSTALL_RECORD", tmp_path / ".pi_node.installed", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _executable(tmp_path / "system" / "npm", f"echo {tmp_path / 'npm-global'}\n")
    monkeypatch.setenv("PATH", str(tmp_path / "system"))

    def npm_install(npm_bin: str, version: str | None) -> None:
        with (tmp_path / "installs").open("a") as log:
            log.write(f"{os.getpid()}\n")
        # Long enough for the other worker to reach the install while this one is still in it.
        time.sleep(0.5)
        _executable(tmp_path / "npm-global" / "bin" / "pi", "")

    monkeypatch.setattr(setup, "_npm_install", npm_install)
    return tmp_path


def _first_local_call(results: "multiprocessing.Queue[str]") -> None:
    try:
        setup.ensure_pi()
        results.put(shutil.which("pi") or "not on PATH")
    except Exception as error:
        results.put(repr(error))


def test_concurrent_first_calls_in_two_workers_install_once(host: Path) -> None:
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    workers = [context.Process(target=_first_local_call, args=(results,)) for _ in range(2)]
    for worker in workers:
        worker.start()
    found = [results.get(timeout=60) for _ in workers]
    for worker in workers:
        worker.join(timeout=60)

    # Both workers can run the tool, and only one of them installed it.
    assert found == [str(host / "npm-global" / "bin" / "pi")] * 2
    assert len((host / "installs").read_text().splitlines()) == 1


def test_an_interrupted_node_toolchain_is_unpacked_again(host: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = host / ".pi_node"
    # An extraction that stopped before it finished: node is there, npm is not.
    _executable(prefix / "bin" / "node", "")

    def download(url: str, destination: Path) -> None:
        with tarfile.open(destination, "w:xz") as archive:
            for name in ("node", "npm"):
                info = tarfile.TarInfo(f"node-v1-linux-x64/bin/{name}")
                info.size = len(b"#!/bin/sh\n")
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(b"#!/bin/sh\n"))

    monkeypatch.setattr(setup.urllib.request, "urlretrieve", download)

    assert setup._install_node_locally() == prefix / "bin"
    assert (prefix / "bin" / "npm").is_file()

    # A finished toolchain is reused without another download.
    monkeypatch.setattr(setup.urllib.request, "urlretrieve", lambda url, destination: pytest.fail("downloaded again"))
    assert setup._install_node_locally() == prefix / "bin"


def test_local_setup_rejects_a_stale_pi_on_path(host: Path) -> None:
    _executable(host / "system" / "pi", "echo 0.80.2")
    with pytest.raises(RuntimeError, match=r"Expected Pi 1\.5\.2.*Got: 0\.80\.2"):
        setup.ensure_pi("1.5.2")
    # The stale CLI is not replaced: the operator upgrades it.
    assert not (host / "installs").exists()


def test_local_setup_accepts_the_pinned_pi_on_path_without_installing(host: Path) -> None:
    _executable(host / "system" / "pi", "echo 1.5.2")
    path = os.environ["PATH"]
    setup.ensure_pi("1.5.2")
    assert os.environ["PATH"] == path
    assert not (host / "installs").exists()


def test_local_setup_checks_a_pi_found_in_local_bin(host: Path) -> None:
    _executable(host / "home" / ".local" / "bin" / "pi", "echo 0.80.2")
    with pytest.raises(RuntimeError, match=r"Expected Pi 1\.5\.2 from pi \(.*/\.local/bin/pi\).*Got: 0\.80\.2"):
        setup.ensure_pi("1.5.2")


def test_local_setup_reports_what_a_failing_version_command_printed(host: Path) -> None:
    _executable(host / "system" / "pi", "echo 1.5.2\necho 'cannot load' >&2\nexit 1")
    with pytest.raises(RuntimeError, match=r"Got: 1\.5\.2 \(exit code 1\); stderr: cannot load"):
        setup.ensure_pi("1.5.2")


def test_local_setup_turns_a_hung_version_command_into_the_setup_error(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _executable(host / "system" / "pi", f"exec {sys.executable} -c 'import time; time.sleep(5)'")
    run = subprocess.run
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: run(*args, **{**kwargs, "timeout": 0.5}))
    with pytest.raises(RuntimeError, match=r"Expected Pi 1\.5\.2 .*Got: .*timed out"):
        setup.ensure_pi("1.5.2")


@pytest.mark.parametrize("version", [None, "latest", "^1.5.0"])
def test_local_setup_checks_only_an_exact_pin(host: Path, version: str | None) -> None:
    # An unpinned or ranged install takes whatever npm resolves, so there is no single version to compare.
    _executable(host / "system" / "pi", f"touch {host / 'ran'}\nexit 1")
    setup.ensure_pi(version)
    assert not (host / "ran").exists()


def test_local_setup_checks_the_configured_command_and_environment(host: Path) -> None:
    # A stale pi on the worker's PATH does not matter when the run uses another launcher or PATH.
    _executable(host / "system" / "pi", "echo 0.80.2")
    _executable(host / "pinned" / "pi", "echo 1.5.2")
    setup.ensure_pi("1.5.2", command=[str(host / "pinned" / "pi")])
    setup.ensure_pi("1.5.2", env={"PATH": str(host / "pinned")})
    with pytest.raises(RuntimeError, match=r"Got: 0\.80\.2"):
        setup.ensure_pi("1.5.2", command=["pi"])


@pytest.mark.parametrize(("installed", "accepted"), [("1.5.2", True), ("1.5.1", False)])
def test_a_fresh_install_is_checked_against_the_pin(
    host: Path, monkeypatch: pytest.MonkeyPatch, installed: str, accepted: bool
) -> None:
    def npm_install(npm_bin: str, version: str | None) -> None:
        assert version == "1.5.2"
        _executable(host / "npm-global" / "bin" / "pi", f"echo {installed}")

    monkeypatch.setattr(setup, "_npm_install", npm_install)
    if accepted:
        setup.ensure_pi("1.5.2")
        record = json.loads((host / ".pi_node.installed").read_text())
        assert record == {"bin_dir": str(host / "npm-global" / "bin"), "version": "1.5.2"}
    else:
        with pytest.raises(RuntimeError, match=r"Expected Pi 1\.5\.2 .*Got: 1\.5\.1"):
            setup.ensure_pi("1.5.2")
