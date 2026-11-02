# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component selection and rules of the Environment Server compliance suite."""

import json
import subprocess
import sys

import pytest
from scripts.check_suites.github_actions import read_ci_result, write_ci_result
from scripts.environment_server_compliance import checks, ci
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
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

    class Agent(SimpleResponsesAPIAgent):
        async def run(self, body):
            pass

    class InheritingAgent(Agent):
        pass

    path = "responses_api_agents/example/app.py"
    [finding] = _filter_findings(findings(path, [Agent, InheritingAgent], AGENT_SERVER), "A1")
    assert finding["path"] == path
    assert finding["line"] == Agent.run.__code__.co_firstlineno
    assert findings(path, [Agent], RESOURCES_SERVER) == []


def test_an_agent_without_session_seed_and_close_is_reported_at_its_class():
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

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
    results = _filter_findings(findings(path, [Legacy, SeedOnly, Sessions, InheritingSessions], AGENT_SERVER), "A2")
    assert [(f["path"], f["line"]) for f in results] == [
        (path, Legacy.__firstlineno__),
        (path, SeedOnly.__firstlineno__),
    ]


def test_an_agent_overriding_aggregate_metrics_is_reported_at_the_override():
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

    class Overriding(SimpleResponsesAPIAgent):
        async def aggregate_metrics(self, body):
            pass

    class Inheriting(Overriding):
        pass

    class Default(SimpleResponsesAPIAgent):
        pass

    path = "responses_api_agents/example/app.py"
    [finding] = _filter_findings(findings(path, [Overriding, Inheriting, Default], AGENT_SERVER), "A3")
    assert finding["path"] == path
    assert finding["line"] == Overriding.aggregate_metrics.__code__.co_firstlineno


def test_an_unknown_component_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown component kind: 'tools'"):
        _base_class("tools")


def test_only_agent_classes_defined_in_app_py_are_inspected(tmp_path, monkeypatch):
    component = "responses_api_agents/compliance_probe"
    write(
        tmp_path,
        f"{component}/base.py",
        "from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent\n"
        "\n"
        "\n"
        "class Imported(SimpleResponsesAPIAgent):\n"
        "    async def run(self, body):\n"
        "        pass\n",
    )
    write(
        tmp_path,
        f"{component}/app.py",
        "from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent\n"
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
    assert [(f["rule"], f["path"], f["line"]) for f in results] == [
        ("A1", f"{component}/app.py", 11),
        ("A2", f"{component}/app.py", 10),
    ]


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
    monkeypatch.setattr(ci, "check", lambda component, *, output: checked.append(component))
    monkeypatch.setattr("sys.argv", ["ci", "check", path])
    ci.main()
    assert checked == ["responses_api_agents/pi_agent"]


def _write_result(artifacts, component, status, findings=()):
    name = ci.subject(component)
    write_ci_result(artifacts / name, subject=name, status=status)
    (artifacts / name / ci.FINDINGS_FILE).write_text(json.dumps(list(findings)))


def test_check_writes_the_findings_and_result_of_a_component(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha")
    finding = {"rule": "A1", "path": "responses_api_agents/legacy/app.py", "line": 3, "message": "m"}
    results = {"responses_api_agents/legacy": [finding], "responses_api_agents/broken": None}
    monkeypatch.setattr(ci, "inspect_component", lambda component: results[component])
    for component, status, findings in [
        ("responses_api_agents/legacy", "failed", [finding]),
        ("responses_api_agents/broken", "unavailable", []),
    ]:
        output = tmp_path / ci.subject(component)
        ci.check(component, output=output)
        assert read_ci_result(output / "ci-result.json", subject=ci.subject(component), sha="sha")["status"] == status
        assert json.loads((output / ci.FINDINGS_FILE).read_text()) == findings


def test_the_check_summarizes_every_component_from_this_commits_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha")
    finding = {"rule": "A1", "path": "responses_api_agents/legacy/app.py", "line": 3, "message": "Do | not"}
    _write_result(tmp_path, "responses_api_agents/clean", "passed")
    _write_result(tmp_path, "responses_api_agents/legacy", "failed", [finding, {**finding, "rule": "A3", "line": 9}])
    monkeypatch.setenv("GITHUB_SHA", "other-sha")
    _write_result(tmp_path, "responses_api_agents/stale", "passed")
    components = [f"responses_api_agents/{name}" for name in ("clean", "legacy", "stale", "missing")]
    check = checks.compliance_check(components, artifacts=tmp_path, sha="sha")
    assert check["conclusion"] == "failure"
    assert check["output"]["title"] == "1 compliant, 1 with findings, 2 unavailable"
    summary = check["output"]["summary"]
    assert "| responses_api_agents/legacy | with findings | A1, A3 | See workflow artifacts/logs |" in summary
    assert "| responses_api_agents/stale | unavailable | - |" in summary
    assert "- `responses_api_agents/legacy/app.py:3` **A1**: Do &#124; not" in summary
    assert "- `responses_api_agents/legacy/app.py:9` **A3**: Do &#124; not" in summary


def test_the_check_succeeds_when_nothing_is_changed_or_every_component_is_compliant(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha")
    _write_result(tmp_path, "responses_api_agents/clean", "passed")
    check = checks.compliance_check(["responses_api_agents/clean"], artifacts=tmp_path, sha="sha")
    assert (check["conclusion"], check["output"]["title"]) == (
        "success",
        "1 compliant, 0 with findings, 0 unavailable",
    )
    check = checks.compliance_check([], artifacts=tmp_path, sha="sha")
    assert (check["conclusion"], check["output"]["title"]) == ("success", "No changed components")


def test_the_check_fails_when_component_selection_failed(tmp_path):
    check = checks.compliance_check(None, artifacts=tmp_path, sha="sha")
    assert (check["conclusion"], check["output"]["title"]) == ("failure", "Could not select components")


def test_the_reporter_imports_without_gym_or_site_packages():
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from scripts.environment_server_compliance import checks; "
            "assert not any(name.startswith(('nemo_gym', 'psutil')) for name in sys.modules)",
        ],
        check=True,
    )
