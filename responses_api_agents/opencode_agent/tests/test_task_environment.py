# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task shell directories survive removal of OpenCode's private CLI state."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from responses_api_agents.opencode_agent import sandbox_runner


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the OpenCode plugin")
@pytest.mark.parametrize("custom_xdg", [False, True])
def test_shell_environment_keeps_task_files_after_session_close(tmp_path: Path, custom_xdg: bool) -> None:
    session = tmp_path / "session"
    session.mkdir()
    home = tmp_path / "task-home"
    home.mkdir()
    plugin = session / "task_environment.mjs"
    shutil.copyfile(Path(sandbox_runner.__file__).with_name(plugin.name), plugin)
    # Use a subprocess-local environment: no task directories come from the agent-server host.
    env = {key: value for key, value in os.environ.items() if not key.startswith("XDG_")}
    env["HOME"] = str(home)
    if custom_xdg:
        env["XDG_CONFIG_HOME"] = str(tmp_path / "task-config")
    cli_env = {
        "HOME": str(session / "home"),
        "XDG_CONFIG_HOME": str(session / "config"),
        "XDG_DATA_HOME": str(session / "data"),
    }
    script = """
import { TaskEnvironment } from "./task_environment.mjs";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
const hooks = await TaskEnvironment();
const output = {env: {UNRELATED: "retained"}};
await hooks["shell.env"]({}, output);
// The CLI's private storage remains isolated; only shell env is changed.
mkdirSync(process.env.XDG_DATA_HOME + "/opencode", {recursive: true});
writeFileSync(process.env.XDG_DATA_HOME + "/opencode/cli-state", "private");
execFileSync("/bin/sh", ["-c", 'mkdir -p "$XDG_CONFIG_HOME"; printf task > "$HOME/home-marker"; printf config > "$XDG_CONFIG_HOME/config-marker"'],
  {env: {...process.env, ...output.env}});
console.log(JSON.stringify({cli: process.env.XDG_CONFIG_HOME, cliHome: process.env.HOME, tool: output.env}));
"""
    (session / "probe.mjs").write_text(script)
    params = {
        "directory": str(session),
        "command": [shutil.which("node"), str(session / "probe.mjs")],
        "cwd": str(session),
        "env": cli_env,
        "prompt": "",
    }
    input_path = session / "input.json"
    input_path.write_text(json.dumps(params))
    result = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, str(input_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads((session / "stdout.jsonl").read_text())
    assert observed["cli"] == str(session / "config")
    assert observed["cliHome"] == str(session / "home")
    assert observed["tool"]["UNRELATED"] == "retained"
    assert observed["tool"]["HOME"] == str(home)
    assert (session / "data/opencode/cli-state").read_text() == "private"
    task_config = tmp_path / "task-config" if custom_xdg else home / ".config"
    assert observed["tool"]["XDG_CONFIG_HOME"] == str(task_config)
    assert observed["tool"]["XDG_DATA_HOME"] == str(home / ".local/share")
    shutil.rmtree(session)
    assert (home / "home-marker").read_text() == "task"
    assert (task_config / "config-marker").read_text() == "config"
