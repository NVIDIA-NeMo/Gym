# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component selection and rules of the Environment Server compliance suite."""

import sys

import pytest
from scripts.environment_server_compliance import ci
from scripts.environment_server_compliance.ci import all_components, select_components
from scripts.environment_server_compliance.component import _base_class, default_agent_config
from scripts.environment_server_compliance.engine import (
    AGENT_SERVER,
    RESOURCES_SERVER,
    Rule,
    Target,
    Unavailable,
    evaluate,
    findings,
    validate,
)
from scripts.environment_server_compliance.inspection import inspect_component
from scripts.environment_server_compliance.rules import rules_for


def write(root, path, text):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)


def _filter_findings(f: list[dict], rule: str) -> list[dict]:
    return [finding for finding in f if finding["rule"] == rule]


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
    from nemo_gym.agents.base_responses_api_agent import SimpleResponsesAPIAgent

    class Agent(SimpleResponsesAPIAgent):
        async def run(self, body):
            pass

    class InheritingAgent(Agent):
        pass

    path = "responses_api_agents/example/app.py"
    [finding] = _filter_findings(findings(path, [Agent, InheritingAgent], rules_for(AGENT_SERVER)), "A1")
    assert finding["path"] == path
    assert finding["line"] == Agent.run.__code__.co_firstlineno
    assert findings(path, [Agent], rules_for(RESOURCES_SERVER)) == []


def test_an_agent_without_session_seed_and_close_is_reported_at_its_class():
    from nemo_gym.agents.base_responses_api_agent import SimpleResponsesAPIAgent

    class Legacy(SimpleResponsesAPIAgent):
        pass

    class SeedOnly(SimpleResponsesAPIAgent):
        async def _seed_agent_session_state(self, body):
            pass

    class Sessions(SimpleResponsesAPIAgent):
        async def _seed_agent_session_state(self, body):
            pass

        async def _close_agent_session_state(self, state):
            pass

    class InheritingSessions(Sessions):
        pass

    path = "responses_api_agents/example/app.py"
    results = _filter_findings(
        findings(path, [Legacy, SeedOnly, Sessions, InheritingSessions], rules_for(AGENT_SERVER)), "A2"
    )
    assert [(f["path"], f["line"]) for f in results] == [
        (path, Legacy.__firstlineno__),
        (path, SeedOnly.__firstlineno__),
    ]


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
        "    async def responses(self, body):\n"
        "        pass\n"
        "\n"
        "    async def run(self, body):\n"
        "        pass\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        results = inspect_component(component)
    finally:
        for name in [name for name in sys.modules if name.startswith("responses_api_agents.compliance_probe")]:
            del sys.modules[name]
    assert [(f["rule"], f["path"], f["line"]) for f in results] == [
        ("A1", f"{component}/app.py", 14),
        ("A2", f"{component}/app.py", 10),
    ]


def _inspect_written(tmp_path, monkeypatch, component, files):
    for name, text in files.items():
        write(tmp_path, f"{component}/{name}", text)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    try:
        return inspect_component(component)
    finally:
        for name in [name for name in sys.modules if name.startswith(component.replace("/", "."))]:
            del sys.modules[name]


RUNNING_AGENT = (
    "from nemo_gym.agents import SimpleResponsesAPIAgent\n"
    "\n"
    "\n"
    "class Own(SimpleResponsesAPIAgent):\n"
    "    async def responses(self, body):\n"
    "        pass\n"
    "\n"
    "    async def run(self, body):\n"
    "        pass\n"
)


def test_the_entry_point_named_by_the_default_config_is_inspected(tmp_path, monkeypatch):
    component = "responses_api_agents/compliance_probe"
    results = _inspect_written(
        tmp_path,
        monkeypatch,
        component,
        {
            "app.py": "ABSTRACT_BASE_ONLY = True\n",
            "main.py": RUNNING_AGENT,
            "configs/compliance_probe.yaml": (
                "probe:\n  responses_api_agents:\n    own:\n      entrypoint: main.py\n"
            ),
        },
    )
    assert [f["path"] for f in results if f["rule"] == "A1"] == [f"{component}/main.py"]


@pytest.mark.parametrize("entrypoint", ["???", "missing.py"])
def test_app_py_is_inspected_when_the_default_config_names_no_existing_entry_point(tmp_path, monkeypatch, entrypoint):
    component = "responses_api_agents/compliance_probe"
    results = _inspect_written(
        tmp_path,
        monkeypatch,
        component,
        {
            "app.py": RUNNING_AGENT,
            "configs/compliance_probe.yaml": (
                f"probe:\n  responses_api_agents:\n    own:\n      entrypoint: {entrypoint}\n"
            ),
        },
    )
    assert {f["path"] for f in results} == {f"{component}/app.py"}


def test_abstract_classes_are_not_inspected(tmp_path, monkeypatch):
    component = "responses_api_agents/compliance_probe"
    results = _inspect_written(
        tmp_path,
        monkeypatch,
        component,
        {
            "app.py": (
                "from abc import abstractmethod\n"
                "from nemo_gym.agents import SimpleResponsesAPIAgent\n"
                "\n"
                "\n"
                "class Base(SimpleResponsesAPIAgent):\n"
                "    async def responses(self, body):\n"
                "        pass\n"
                "\n"
                "    async def run(self, body):\n"
                "        pass\n"
                "\n"
                "    @abstractmethod\n"
                "    def build(self):\n"
                "        pass\n"
                "\n"
                "\n"
                "class Concrete(Base):\n"
                "    def build(self):\n"
                "        pass\n"
                "\n"
                "    async def run(self, body):\n"
                "        pass\n"
            )
        },
    )
    assert [f["line"] for f in results if f["rule"] == "A1"] == [21]


def test_the_default_agent_config_is_the_first_agent_instance_with_placeholders(tmp_path):
    component = tmp_path / "example"
    write(
        component,
        "configs/example.yaml",
        "other:\n  resources_servers:\n    server:\n      entrypoint: app.py\n"
        "agents:\n  responses_api_agents:\n    first:\n      entrypoint: app.py\n      label: ${policy_model_name}\n",
    )
    assert default_agent_config(component) == {"name": "agents", "entrypoint": "app.py", "label": "probe"}


def test_a_component_without_an_agent_instance_in_its_default_config_has_no_agent_config(tmp_path):
    component = tmp_path / "example"
    with pytest.raises(Unavailable, match="example.yaml does not exist"):
        default_agent_config(component)
    write(
        component,
        "configs/example.yaml",
        "unrelated:\n  resources_servers:\n    server:\n      entrypoint: app.py\n",
    )
    with pytest.raises(Unavailable, match="defines no agent instance"):
        default_agent_config(component)


def test_the_inspection_can_import_modules_next_to_the_apps_entry_point(monkeypatch, tmp_path):
    from nemo_gym.config import global_config
    from nemo_gym.server_runtime.supervisor import server_venv

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


def _rule(id, check, *requires):
    return Rule(id, AGENT_SERVER, id, check, requires)


def _outcomes(rules, target=None):
    return {o.rule.id: (o.status, o.reason) for o in evaluate(rules, target or Target(object))}


def test_a_rule_is_judged_by_its_check_whatever_the_check_does():
    results = _outcomes((_rule("S1", lambda target: None), _rule("S2", lambda target: target.server)))
    assert results == {"S1": ("held", None), "S2": ("violated", None)}


def test_a_rule_is_skipped_unless_every_rule_it_requires_held():
    judged = []

    def check(target):
        judged.append(1)

    results = _outcomes(
        (
            _rule("H", lambda target: None),
            _rule("V", lambda target: target.server),
            _rule("ON_H", check, "H"),
            _rule("ON_V", check, "H", "V"),
            _rule("ON_SKIPPED", check, "ON_V"),
        )
    )
    assert results["ON_H"] == ("held", None)
    assert results["ON_V"] == results["ON_SKIPPED"] == ("skipped", None)
    assert judged == [1]


def test_a_rule_that_cannot_be_judged_is_skipped_with_its_reason_and_so_are_its_dependents():
    def unavailable(target):
        raise Unavailable("cannot run it")

    results = _outcomes((_rule("U", unavailable), _rule("D", lambda target: target.server, "U")))
    assert results == {"U": ("skipped", "cannot run it"), "D": ("skipped", None)}


def test_rules_share_a_fact_that_is_produced_once_even_when_it_fails():
    produced = []

    def produce():
        produced.append(1)
        raise Unavailable("no")

    def check(target):
        target.once("fact", produce)

    results = _outcomes((_rule("A", check), _rule("B", check)))
    assert results == {"A": ("skipped", "no"), "B": ("skipped", "no")}
    assert produced == [1]


def test_rules_must_require_earlier_rules_of_their_own_kind():
    nothing = lambda target: None  # noqa: E731
    with pytest.raises(ValueError, match="requires 'B'"):
        validate((_rule("A", nothing, "B"), _rule("B", nothing)))
    with pytest.raises(ValueError, match="requires 'A'"):
        validate((_rule("A", nothing), Rule("B", RESOURCES_SERVER, "B", nothing, ("A",))))
    with pytest.raises(ValueError, match="Duplicate rule id"):
        validate((_rule("A", nothing), _rule("A", nothing)))
