# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the block_target_v2 guard: identifier derivation, wrapper behaviour (run under bash with a stub in place of the real binary) and egress policy."""

import os
import shutil
import stat
import subprocess

import pytest

from resources_servers.nl2repobench import guard_v2


BCRYPT = guard_v2.derive_identifiers("pyca_bcrypt_pr-1", ["git clone --quiet https://github.com/pyca/bcrypt /tmp/_reference_repo"])
SIX = guard_v2.derive_identifiers("benjaminp_six_pr-1", ["git clone https://github.com/benjaminp/six /tmp/_reference_repo"])

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_derive_identifiers_uses_clone_url():
    idents, slugs = BCRYPT
    assert "bcrypt" in idents and slugs == ["pyca/bcrypt"]
    idents, slugs = guard_v2.derive_identifiers("python-foo_pr3", ["git clone https://github.com/o/python-foo x"])
    assert "foo" in idents and "pythonfoo" in idents


def test_derive_identifiers_falls_back_to_task_id_without_clone_url():
    idents, slugs = guard_v2.derive_identifiers("six", ["pip install -e ."])
    assert "six" in idents and slugs == []


def test_network_policy_denies_source_hosts_and_allows_by_default():
    policy = guard_v2.network_policy_for_v2({})
    assert policy["defaultAction"] == "allow"
    targets = {rule["target"] for rule in policy["egress"] if rule["action"] == "deny"}
    assert {"github.com", "*.githubusercontent.com", "gitlab.com"} <= targets


@pytest.fixture()
def run_wrapper(tmp_path):
    stub = tmp_path / "stub"
    stub.write_text('#!/bin/bash\necho "REAL-RAN $*"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    def run(kind, tool, args, ident_slug=BCRYPT, stdin=None):
        wrapper = tmp_path / tool
        wrapper.write_text(guard_v2.render_wrapper(kind, tool, str(stub), *ident_slug))
        wrapper.chmod(0o755)
        proc = subprocess.run([str(wrapper), *args], input=stdin, capture_output=True, text=True, timeout=30, stdin=None if stdin is not None else subprocess.DEVNULL)
        return proc.returncode != 0 and "nl2repobench guard v2" in proc.stderr, proc

    return run


@pytest.mark.parametrize(
    "kind,tool,args,blocked",
    [
        ("pip", "pip", ["install", "requests"], False),
        ("pip", "pip", ["install", "bcrypt"], True),
        ("pip", "pip", ["install", "bcrypt==4.1.2"], True),
        ("pip", "pip", ["download", "pyca-bcrypt"], True),
        ("pip", "pip", ["install", "-e", "/workspace/bcrypt"], False),
        ("pip", "pip", ["install", "-e", "."], False),
        ("pip", "pip", ["install", "git+https://github.com/pyca/bcrypt.git"], True),
        ("pip", "pip", ["--index-url", "https://x.example/simple", "install", "requests"], True),
        ("pip", "pip", ["config", "set", "global.index-url", "http://evil"], True),
        ("pip", "pip", ["list"], False),
        ("git", "git", ["clone", "https://github.com/pyca/bcrypt", "/tmp/x"], True),
        ("git", "git", ["clone", "https://github.com/other/thing", "/tmp/x"], False),
        ("git", "git", ["add", "bcrypt/"], False),
        ("fetcher", "curl", ["-sL", "https://raw.githubusercontent.com/pyca/bcrypt/main/x.py"], True),
        ("fetcher", "curl", ["-sL", "https://pypi.org/pypi/bcrypt/json"], True),
        ("fetcher", "curl", ["-sL", "https://pypi.org/pypi/requests/json"], False),
        ("uv", "uv", ["pip", "install", "bcrypt"], True),
        ("uv", "uv", ["pip", "install", "requests"], False),
        ("fetcher", "apt-get", ["install", "-y", "python3-bcrypt"], True),
        ("fetcher", "apt-get", ["install", "-y", "build-essential"], False),
        ("interp", "python", ["-m", "pip", "install", "bcrypt"], True),
        ("interp", "python", ["-mpip", "install", "bcrypt"], True),
        ("interp", "python", ["-m", "pip", "install", "requests"], False),
        ("interp", "python", ["-m", "venv", "/tmp/v"], True),
        ("interp", "python", ["-m", "ensurepip"], True),
        ("interp", "python", ["-m", "pytest", "tests/"], False),
        ("interp", "python", ["-c", "import urllib.request; urllib.request.urlopen('https://github.com/pyca/bcrypt/archive/main.tar.gz')"], True),
        ("interp", "python", ["-c", "import urllib.request; urllib.request.urlopen('https://pypi.org/pypi/bcrypt/json')"], True),
        ("interp", "python", ["-c", "import urllib.request; urllib.request.urlopen('https://github.com/other/thing')"], False),
        ("interp", "python", ["-c", "import bcrypt; print(bcrypt.__file__)"], False),
        ("interp", "python", ["--version"], False),
    ],
)
def test_wrapper_blocks_target_fetches_only(run_wrapper, kind, tool, args, blocked):
    got, proc = run_wrapper(kind, tool, args)
    assert got == blocked, (proc.stdout, proc.stderr)
    if not blocked:
        assert "REAL-RAN" in proc.stdout


def test_short_identifiers_need_an_exact_match(run_wrapper):
    assert run_wrapper("pip", "pip", ["install", "six"], SIX)[0] is True
    assert run_wrapper("pip", "pip", ["install", "six==1.16"], SIX)[0] is True
    assert run_wrapper("pip", "pip", ["install", "sixpack"], SIX)[0] is False


def test_python_heredoc_fetching_the_target_is_blocked(run_wrapper):
    code = "import urllib.request\nurllib.request.urlopen('https://codeload.github.com/pyca/bcrypt/tar.gz/main')\n"
    assert run_wrapper("interp", "python", ["-"], stdin=code)[0] is True
    assert run_wrapper("interp", "python", ["-"], stdin="print('hi')\n")[0] is False


def test_requirements_file_naming_the_target_is_blocked(run_wrapper, tmp_path):
    req = tmp_path / "reqs.txt"
    req.write_text("requests\nbcrypt==4.0\n")
    assert run_wrapper("pip", "pip", ["install", "-r", str(req)])[0] is True
