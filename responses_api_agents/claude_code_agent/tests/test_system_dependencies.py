# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Harbor's dependency script under real shells, with stand-in package managers on a bare PATH."""

import shlex
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.claude_code_agent.system_dependencies import (
    CLAUDE_CODE_DEPENDENCIES,
    PACKAGE_MANAGERS,
    PackageSpec,
    system_dependencies_command,
)


SHELLS = [shell for shell in ("/bin/sh", "/bin/bash", "/bin/dash") if Path(shell).exists()]
ALWAYS_AND_DUPLICATE = (
    *CLAUDE_CODE_DEPENDENCIES,
    PackageSpec(
        name="ca_certificates",
        commands=(),
        packages={manager: ("ca-certificates",) for manager in PACKAGE_MANAGERS},
        always_install=True,
    ),
    PackageSpec(
        name="node_again", commands=("node",), packages={manager: ("nodejs",) for manager in PACKAGE_MANAGERS}
    ),
)


def test_script_carries_harbors_checks_packages_and_commands():
    command = system_dependencies_command()
    for expected in (
        "if command -v curl >/dev/null 2>&1; then",
        "if command -v node >/dev/null 2>&1; then",
        "if command -v pgrep >/dev/null 2>&1; then",
        "for harbor_deps_candidate in apt-get dnf yum apk; do",
        "apt-get:nodejs) harbor_deps_add=nodejs ;;",
        "dnf:procps) harbor_deps_add=procps-ng ;;",
        "yum:procps) harbor_deps_add=procps-ng ;;",
        "apk:procps) harbor_deps_add=procps ;;",
        'apt-get) harbor_deps_install="apt-get update && apt-get install -y $harbor_deps_packages" ;;',
        'apk) harbor_deps_install="apk add --no-cache $harbor_deps_packages" ;;',
        "ensure_system_dependencies(curl, bash, nodejs, npm, procps)",
    ):
        assert expected in command, expected


@pytest.mark.skipif(not SHELLS, reason="no POSIX shell")
@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "specs, present, managers, fail_with, calls, exit_code, log",
    [
        pytest.param(
            CLAUDE_CODE_DEPENDENCIES,
            ["curl", "bash"],
            ["apt-get"],
            None,
            ["apt-get noninteractive update", "apt-get noninteractive install -y nodejs npm procps"],
            0,
            "install with apt-get (DEBIAN_FRONTEND=noninteractive): apt-get update && apt-get install -y nodejs npm procps",
            id="apt_missing_node_tools",
        ),
        pytest.param(
            CLAUDE_CODE_DEPENDENCIES,
            [],
            ["dnf", "yum"],
            None,
            ["dnf - install -y curl bash nodejs npm procps-ng"],
            0,
            None,
            id="dnf_before_yum",
        ),
        pytest.param(
            CLAUDE_CODE_DEPENDENCIES,
            ["bash"],
            ["apk"],
            None,
            ["apk - add --no-cache curl nodejs npm procps"],
            0,
            None,
            id="apk",
        ),
        pytest.param(
            CLAUDE_CODE_DEPENDENCIES,
            ["curl", "bash", "node", "npm", "pgrep"],
            ["apt-get"],
            None,
            [],
            0,
            "nothing to install",
            id="all_present",
        ),
        pytest.param(
            CLAUDE_CODE_DEPENDENCIES,
            ["curl"],
            [],
            None,
            [],
            0,
            "warning: No supported package manager found; cannot ensure system dependencies: "
            "curl, bash, nodejs, npm, procps",
            id="no_package_manager",
        ),
        pytest.param(
            CLAUDE_CODE_DEPENDENCIES,
            [],
            ["apt-get"],
            "update",
            ["apt-get noninteractive update"],
            100,
            "install exit status 100",
            id="apt_update_fails",
        ),
        pytest.param(
            ALWAYS_AND_DUPLICATE,
            ["curl", "bash", "npm", "pgrep"],
            ["apk"],
            None,
            ["apk - add --no-cache nodejs ca-certificates"],
            0,
            "ca_certificates: always installed",
            id="always_install_and_dedup",
        ),
    ],
)
def test_script_per_package_manager(tmp_path, shell, specs, present, managers, fail_with, calls, exit_code, log):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "calls"
    for name in present:
        (bin_dir / name).write_text("#!/bin/sh\nexit 0\n")
        (bin_dir / name).chmod(0o755)
    for manager in managers:
        stub = f'#!/bin/sh\nprintf \'%s %s %s\\n\' {manager} "${{DEBIAN_FRONTEND:--}}" "$*" >> {shlex.quote(str(record))}\n'
        if fail_with:
            stub += f'if [ "$1" = {fail_with} ]; then exit 100; fi\n'
        (bin_dir / manager).write_text(stub + "exit 0\n")
        (bin_dir / manager).chmod(0o755)
    result = subprocess.run(
        [shell, "-c", system_dependencies_command(specs)],
        env={"PATH": str(bin_dir)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    got = record.read_text().splitlines() if record.exists() else []
    assert got == calls, result.stdout
    if log is not None:
        assert f"harbor_deps: {log}" in result.stdout.splitlines()
