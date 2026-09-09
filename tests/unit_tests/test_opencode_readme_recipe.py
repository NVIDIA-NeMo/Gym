# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for the copyable OpenCode first-run recipe; no services are started."""

import re
import shlex
import textwrap
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "responses_api_agents/opencode_sandboxed_agent/README.md"


def _commands():
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(), flags=re.DOTALL)
    return [
        shlex.split(line, comments=True)
        for block in blocks
        for line in block.replace("\\\n", " ").splitlines()
        if line.startswith("gym ")
    ]


def _flag(command, flag):
    assert flag in command, f"Missing {flag} in {command}"
    return command[command.index(flag) + 1]


def test_readme_prepares_and_runs_one_real_task():
    commands = _commands()
    prepare = next((c for c in commands if c[:3] == ["gym", "eval", "prepare"]), None)
    rollout = next((c for c in commands if c[:3] == ["gym", "eval", "run"]), None)
    assert prepare is not None, "First run must prepare the dataset before consuming it"
    assert rollout is not None, "Use the real agent /run path, with persisted rollout artifacts"
    start = next(c for c in commands if c[:3] == ["gym", "env", "start"])
    assert commands.index(prepare) < commands.index(start) < commands.index(rollout)
    assert "--no-serve" in rollout
    for flag in ("--limit", "--num-repeats", "--concurrency"):
        assert _flag(rollout, flag) == "1"
    assert _flag(rollout, "--output").endswith(".jsonl")


def test_readme_agent_and_input_match_resolved_benchmark_recipe(monkeypatch):
    monkeypatch.chdir(ROOT)
    commands = _commands()
    prepare = next((c for c in commands if c[:3] == ["gym", "eval", "prepare"]), None)
    assert prepare is not None, "A canonical benchmark recipe must prepare the input"
    recipe_path = _flag(prepare, "--config")
    start = next(c for c in commands if c[:3] == ["gym", "env", "start"])
    assert recipe_path in start
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([recipe_path])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    rollout = next(c for c in commands if c[:3] == ["gym", "eval", "run"])
    agent = config[_flag(rollout, "--agent")].responses_api_agents.opencode_sandboxed_agent
    resolved_agent = OmegaConf.to_container(agent, resolve=True, throw_on_missing=True)
    resources = resolved_agent["resources_server"]
    assert resources["type"] == "resources_servers"
    assert config[resources["name"]].resources_servers.swebench.entrypoint == "app.py"
    dataset = next(d for d in resolved_agent["datasets"] if d["jsonl_fpath"] == _flag(rollout, "--input"))
    assert (ROOT / dataset["prepare_script"]).is_file()


def test_offline_binary_override_targets_the_documented_agent():
    rollout = next(c for c in _commands() if c[:3] == ["gym", "eval", "run"])
    agent_name = _flag(rollout, "--agent")
    blocks = re.findall(r"^[ \t]*```yaml\n(.*?)^[ \t]*```", README.read_text(), flags=re.DOTALL | re.MULTILINE)
    overrides = [
        OmegaConf.create(textwrap.dedent(block)) for block in blocks if "remote_opencode_binary_path:" in block
    ]
    assert overrides, "Pre-staged binaries need a copyable override for the actual agent instance"
    for override in overrides:
        assert set(override) == {agent_name}, "Offline overrides must not configure a different, unused agent"
        settings = override[agent_name].responses_api_agents.opencode_sandboxed_agent
        assert settings.remote_opencode_binary_path.startswith("/")
        assert settings.remote_opencode_install_script_path.startswith("/")
