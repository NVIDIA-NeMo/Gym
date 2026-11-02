# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component selection and rules of the Environment Server compliance suite."""

import sys

import pytest
from scripts.environment_server_compliance import ci
from scripts.environment_server_compliance.ci import all_components, select_components
from scripts.environment_server_compliance.rules import (
    AGENT_SERVER,
    RESOURCES_SERVER,
    _base_class,
    findings,
    inspect_component,
)


def write(root, path, text):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)


def test_a_changed_file_selects_its_existing_component_only(tmp_path):
    for component in ("responses_api_agents/pi_agent", "resources_servers/math", "responses_api_agents/kept"):
        write(tmp_path, f"{component}/app.py", "")
    paths = [
        "responses_api_agents/pi_agent/app.py",
        "responses_api_agents/pi_agent/configs/pi.yaml",
        "resources_servers/math/data/example.jsonl",
        "responses_api_agents/deleted/app.py",
        "nemo_gym/base_responses_api_agent.py",
        ".github/workflows/cicd-main.yml",
    ]
    assert select_components(paths, root=tmp_path) == ["responses_api_agents/pi_agent"]
    assert all_components(tmp_path) == ["responses_api_agents/kept", "responses_api_agents/pi_agent"]


def test_run_defined_by_an_agent_is_reported_at_its_definition():
    from nemo_gym.agents import SimpleResponsesAPIAgent

    class Agent(SimpleResponsesAPIAgent):
        async def run(self, body):
            pass

    class InheritingAgent(Agent):
        pass

    path = "responses_api_agents/example/app.py"
    [finding] = findings(path, [Agent, InheritingAgent], AGENT_SERVER)
    assert finding["rule"] == "A1" and finding["path"] == path
    assert finding["line"] == Agent.run.__code__.co_firstlineno
    assert findings(path, [Agent], RESOURCES_SERVER) == []


def test_an_unknown_component_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown component kind: 'tools'"):
        _base_class("tools")


def test_only_agent_classes_defined_in_app_py_are_inspected(tmp_path, monkeypatch):
    component = "responses_api_agents/compliance_probe"
    write(
        tmp_path,
        f"{component}/base.py",
        "from nemo_gym.agents import SimpleResponsesAPIAgent\n"
        "\n"
        "\n"
        "class Imported(SimpleResponsesAPIAgent):\n"
        "    async def run(self, body):\n"
        "        pass\n",
    )
    write(
        tmp_path,
        f"{component}/app.py",
        "from nemo_gym.agents import SimpleResponsesAPIAgent\n"
        "from responses_api_agents.compliance_probe.base import Imported\n"
        "\n"
        "\n"
        "class Helper:\n"
        "    def run(self):\n"
        "        pass\n"
        "\n"
        "\n"
        "class Own(SimpleResponsesAPIAgent):\n"
        "    async def run(self, body):\n"
        "        pass\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        results = inspect_component(component)
    finally:
        for name in [name for name in sys.modules if name.startswith("responses_api_agents.compliance_probe")]:
            del sys.modules[name]
    assert [(f["rule"], f["path"], f["line"]) for f in results] == [("A1", f"{component}/app.py", 11)]


def test_the_inspection_can_import_modules_next_to_the_apps_entry_point(monkeypatch, tmp_path):
    from nemo_gym.config import global_config
    from nemo_gym.orchestration import server_venv

    class Done:
        returncode = 1

        def wait(self):
            return 0

    environments = []
    monkeypatch.setattr(global_config, "get_global_config_dict", lambda config: {})
    monkeypatch.setattr(server_venv, "setup_env_command", lambda *args: "true")
    monkeypatch.setattr(server_venv, "run_command", lambda *args, **kwargs: Done())
    monkeypatch.setattr(server_venv, "get_venv_path", lambda directory, config: tmp_path)
    monkeypatch.setattr(ci.subprocess, "run", lambda command, **kwargs: environments.append(kwargs["env"]) or Done())
    assert ci.inspect_component("responses_api_agents/sibling_imports") is None
    paths = environments[0]["PYTHONPATH"].split(ci.os.pathsep)
    assert paths == [str(ci.ROOT), str(ci.ROOT / "responses_api_agents/sibling_imports")]


@pytest.mark.parametrize("flag,expected", [(None, ci.COMPARE_BRANCH), *((mode, mode) for mode in ci.COMPARE_MODES)])
def test_select_passes_the_comparison_mode_to_the_diff(monkeypatch, tmp_path, flag, expected):
    calls = []
    monkeypatch.setattr(
        ci, "changed_paths", lambda base, *, root, subject, compare: calls.append((base, compare)) or []
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr("sys.argv", ["ci", "select", "--base", "abc", *(["--compare", flag] if flag else [])])
    ci.main()
    assert calls == [("abc", expected)]


def test_check_prints_a_warning_per_finding_with_its_message_escaped(monkeypatch, capsys):
    finding = {"rule": "A1", "path": "a/app.py", "line": 3, "message": "50% done"}
    monkeypatch.setattr(ci, "inspect_component", lambda component: [finding, {**finding, "line": 7}])
    assert ci.check("responses_api_agents/legacy") == 2
    assert capsys.readouterr().out.splitlines() == [
        "::warning file=a/app.py,line=3,title=A1::50%25 done",
        "::warning file=a/app.py,line=7,title=A1::50%25 done",
        "2 finding(s).",
    ]


def test_check_reports_a_component_it_could_not_inspect(monkeypatch, capsys):
    monkeypatch.setattr(ci, "inspect_component", lambda component: None)
    assert ci.check("responses_api_agents/broken") == 0
    assert capsys.readouterr().out.splitlines() == [
        "::warning title=Environment Server compliance unavailable::"
        "responses_api_agents/broken: could not install or import the component."
    ]


@pytest.mark.parametrize("path", ["responses_api_agents/pi_agent/", "./responses_api_agents/pi_agent"])
def test_component_path_from_the_command_line_is_normalized(monkeypatch, path):
    checked = []
    monkeypatch.setattr(ci, "check", checked.append)
    monkeypatch.setattr("sys.argv", ["ci", "check", path])
    ci.main()
    assert checked == ["responses_api_agents/pi_agent"]
