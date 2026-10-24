# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable dependency pins follow canonical component metadata, not removed requirements files."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = (
    ("hermes_agent/scripts/hermes_agent_deps.sh", "hermes_agent", "hermes-agent", "HERMES_SPEC"),
    ("anyswe_agent/setup_scripts/hermes_agent_deps.sh", "hermes_agent", "hermes-agent", "HERMES_SPEC"),
    ("cvdp_agent/setup_scripts/hermes_agent_deps.sh", "hermes_agent", "hermes-agent", "HERMES_SPEC"),
    ("terminus_2_agent/scripts/terminus_2_agent_deps.sh", "terminus_2_agent", "harbor", "HARBOR_SPEC"),
)


@pytest.mark.parametrize("script,agent,package,variable", SCRIPTS)
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("ambient_python", ["supported", "unavailable", "no_tomllib"])
def test_portable_pin_uses_project_or_explicit_override(
    tmp_path, script, agent, package, variable, override, ambient_python
):
    source = ROOT / "responses_api_agents" / script
    run_dir = tmp_path / "script"
    run_dir.mkdir()
    target = run_dir / source.name
    target.write_bytes(source.read_bytes())
    helper = run_dir / "_portable_python.sh"
    helper.write_text(
        'install_portable_python() { echo python >> "$SETUP_LOG"; }\n'
        'install_nemo_gym_deps() { echo gym >> "$SETUP_LOG"; }\n'
    )
    deps = tmp_path / "deps"
    (deps / "bin").mkdir(parents=True)
    python = deps / "bin/python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    python.chmod(0o755)
    project = tmp_path / "root/responses_api_agents" / agent
    project.mkdir(parents=True)
    metadata = (ROOT / "responses_api_agents" / agent / "pyproject.toml").read_bytes()
    expected = next(
        value.removeprefix(f"{package} @ ")
        for value in tomllib.loads(metadata.decode())["project"]["dependencies"]
        if value.startswith(f"{package} @ ")
    )
    env = os.environ.copy()
    for name in ("HERMES_SPEC", "HARBOR_SPEC"):
        env.pop(name, None)
    if override:
        # Experiments must remain usable even when canonical metadata is unavailable.
        expected = "git+https://example.com/experimental.git@explicit-pin"
        env[variable] = expected
    else:
        (project / "pyproject.toml").write_bytes(metadata)
    env.update(
        NEMO_GYM_ROOT=str(tmp_path / "root"),
        DEPS_DIR=str(deps),
        PORTABLE_PYTHON_SH=str(helper),
        SETUP_LOG=str(tmp_path / "setup.log"),
        CALL_LOG=str(tmp_path / "calls.log"),
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
    )
    if ambient_python != "supported":
        # Bootstrap must work before its portable interpreter exists, even when
        # the task image has no Python or has Python older than tomllib.
        tools = tmp_path / "host-tools"
        tools.mkdir()
        host_python = tools / "python3"
        message = "python3: command not found" if ambient_python == "unavailable" else "No module named tomllib"
        host_python.write_text(f"#!/bin/sh\necho '{message}' >&2\nexit 127\n")
        host_python.chmod(0o755)
        env["PATH"] = f"{tools}:{env['PATH']}"
    result = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (project / "requirements.txt").exists()
    assert (tmp_path / "setup.log").read_text().splitlines() == ["python", "gym"]
    calls = (tmp_path / "calls.log").read_text().splitlines()
    install_calls = [line for line in calls if line.startswith("-m pip install ")]
    if package == "hermes-agent":
        assert install_calls == [
            f"-m pip install --force-reinstall --no-deps {expected}",
            f"-m pip install {expected}",
        ]
    else:
        assert install_calls == [f"-m pip install {expected}"]
    assert calls[-1].startswith("-c ")
