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
SCRIPTS = (("terminus_2_agent/scripts/terminus_2_agent_deps.sh", "terminus_2_agent", "harbor", "HARBOR_SPEC"),)


@pytest.mark.parametrize("script,agent,package,variable", SCRIPTS)
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("separator", ["@", " @ ", "\t@\t", " @", "@ "])
@pytest.mark.parametrize("ambient_python", ["supported", "unavailable", "no_tomllib"])
def test_portable_pin_uses_project_or_explicit_override(
    tmp_path, script, agent, package, variable, override, ambient_python, separator
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
    requirement = next(
        value
        for value in tomllib.loads(metadata.decode())["project"]["dependencies"]
        if value.partition("@")[0].strip() == package
    )
    expected = requirement.partition("@")[2].strip()
    # PEP 508 allows whitespace on either side of @, including none. Exercise
    # every script before a portable interpreter or tomllib is available.
    metadata = metadata.replace(requirement.encode(), f"{package}{separator}{expected}".encode())
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
    assert install_calls == [f"-m pip install {expected}"]
    assert calls[-1].startswith("-c ")


@pytest.mark.parametrize(
    "script",
    [
        "hermes_agent/scripts/hermes_agent_deps.sh",
        "anyswe_agent/setup_scripts/hermes_agent_deps.sh",
        "cvdp_agent/setup_scripts/hermes_agent_deps.sh",
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_hermes_portable_builder_uses_target_and_optional_source(tmp_path, script, override):
    source = ROOT / "responses_api_agents" / script
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    log = tmp_path / "calls"
    (tmp_path / "_portable_python.sh").write_text(
        'install_portable_python() { echo python >> "$CALL_LOG"; }\n'
        'install_nemo_gym_deps() { echo gym >> "$CALL_LOG"; }\n'
    )
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CALL_LOG"\n')
    uv.chmod(0o755)
    arch = "aarch64-unknown-linux-gnu"
    env = {
        **os.environ,
        "NEMO_GYM_ROOT": str(ROOT),
        "DEPS_DIR": str(tmp_path / "deps"),
        "ARCH": arch,
        "CALL_LOG": str(log),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }
    env.pop("HERMES_SPEC", None)
    extra = []
    if override:
        env["HERMES_SPEC"] = "git+https://example.com/experiment.git@explicit-pin"
        extra = ["--source", env["HERMES_SPEC"]]
    subprocess.run(["bash", str(target)], env=env, check=True)
    assert log.read_text().splitlines() == [
        "python",
        "gym",
        "run",
        "--isolated",
        "--no-dev",
        "--locked",
        "--directory",
        str(ROOT),
        "python",
        "-m",
        "responses_api_agents.hermes_agent.scripts.prepare_portable",
        "--prefix",
        env["DEPS_DIR"],
        "--target",
        arch,
        *extra,
    ]
