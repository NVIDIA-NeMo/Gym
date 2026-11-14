# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component selection and rules of the Environment Server compliance suite."""

import importlib
import json
import subprocess
import sys

import pytest
from scripts.check_suites.github_actions import read_ci_result, write_ci_result
from scripts.environment_server_compliance import checks, ci, probe, rules
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


@pytest.mark.parametrize(
    "path",
    [
        "scripts/environment_server_compliance/rules.py",
        "scripts/check_suites/github_actions.py",
        ".github/workflows/environment-server-compliance.yml",
    ],
)
def test_a_change_to_the_suite_selects_every_component(tmp_path, path):
    for component in ("responses_api_agents/pi_agent", "responses_api_agents/codex_agent"):
        write(tmp_path, f"{component}/app.py", "")
    assert select_components([path, "README.md"], root=tmp_path) == [
        "responses_api_agents/codex_agent",
        "responses_api_agents/pi_agent",
    ]


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


def test_an_agent_overriding_aggregate_metrics_is_reported_at_the_override():
    from nemo_gym.agents.base_responses_api_agent import SimpleResponsesAPIAgent

    class Overriding(SimpleResponsesAPIAgent):
        async def aggregate_metrics(self, body):
            pass

    class Inheriting(Overriding):
        pass

    class Default(SimpleResponsesAPIAgent):
        pass

    path = "responses_api_agents/example/app.py"
    [finding] = _filter_findings(findings(path, [Overriding, Inheriting, Default], rules_for(AGENT_SERVER)), "A3")
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
    monkeypatch.setattr(sys, "path", list(sys.path))
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


@pytest.mark.parametrize("root_main", [{"main.py": "NOT_AN_AGENT = True\n"}, {}])
def test_an_entry_point_in_a_subdirectory_is_inspected_at_its_own_path(tmp_path, monkeypatch, root_main):
    component = "responses_api_agents/compliance_probe"
    # The inspection subprocess has the component root on its path; its sibling of the same name must not win.
    monkeypatch.syspath_prepend(str(tmp_path / component))
    try:
        results = _inspect_written(
            tmp_path,
            monkeypatch,
            component,
            {
                **root_main,
                "compliance_sibling.py": "raise ImportError('the component root shadowed the entry point directory')\n",
                "impl/compliance_sibling.py": "VALUE = 1\n",
                "impl/main.py": "from compliance_sibling import VALUE\n" + RUNNING_AGENT,
                "configs/compliance_probe.yaml": (
                    "probe:\n  responses_api_agents:\n    own:\n      entrypoint: impl/main.py\n"
                ),
            },
        )
    finally:
        sys.modules.pop("compliance_sibling", None)
    assert [(f["rule"], f["path"], f["line"]) for f in results] == [
        ("A1", f"{component}/impl/main.py", 9),
        ("A2", f"{component}/impl/main.py", 5),
    ]


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
    commands = []
    monkeypatch.setattr(
        ci.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or environments.append(kwargs["env"]) or Done(),
    )
    assert ci.inspect_component("responses_api_agents/sibling_imports") is None
    assert commands[0][1:3] == ["-m", "scripts.environment_server_compliance.inspection"]
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


def test_the_reporter_reads_evidence_of_the_tested_merge_and_posts_to_the_event_head(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTED_SHA", "merge-sha")
    _write_result(tmp_path / "compliance-evidence", "responses_api_agents/clean", "passed")
    for key, value in {
        "GITHUB_SHA": "head-sha",
        "GITHUB_REPOSITORY": "example/Gym",
        "GITHUB_RUN_ID": "12",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "SELECTED_COMPONENTS": json.dumps(["responses_api_agents/clean"]),
    }.items():
        monkeypatch.setenv(key, value)
    posted = []
    monkeypatch.setattr(checks, "download_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(checks, "artifact_download_urls", lambda *args, **kwargs: {})
    monkeypatch.setattr(checks, "post_check_run", lambda payload, *, repo: posted.append(payload))
    assert checks.main() == 0
    [check] = posted
    assert check["head_sha"] == "head-sha"
    assert check["output"]["title"] == "1 compliant, 0 with findings, 0 unavailable"
    assert "Tested commit: `merge-sha`" in check["output"]["summary"]


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


LENDING_AGENT = """
from dataclasses import dataclass

from fastapi import HTTPException

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config


@dataclass
class State(AgentSessionState):
    sandbox: AsyncSandbox | None


class Config(BaseResponsesAPIAgentConfig):
    label: str = ""
    refuse_the_seed: bool = False
    crash_the_seed: bool = False
    reject_sandbox_access: bool = False
    fail_the_close: bool = False
    stop_the_lent_sandbox: bool = False
    accept_pool_eula: bool = False


class LendingAgent(SimpleResponsesAPIAgent):
    config: Config

    async def responses(self, body=None):
        raise NotImplementedError

    async def run(self, body=None):
        raise NotImplementedError

    async def _seed_agent_session_state(self, body):
        if self.config.crash_the_seed:
            raise KeyError("policy_model")
        if self.config.refuse_the_seed:
            raise HTTPException(422, "refused")
        if self.config.reject_sandbox_access and body.sandbox_access:
            raise HTTPException(422, "Lending Agent does not support sandbox access")
        access = body.sandbox_access
        if access is None:
            return State(request=body, sandbox=None)
        provider = create_provider(
            resolve_provider_config(access.connection.provider_config_ref, get_global_config_dict())
        )
        return State(request=body, sandbox=await AsyncSandbox.connect(access.connection.descriptor, provider=provider))

    async def _close_agent_session_state(self, state):
        if self.config.fail_the_close:
            raise HTTPException(500, "close failed")
        if state.sandbox is None:
            pass
        elif self.config.stop_the_lent_sandbox:
            await state.sandbox.stop()
        else:
            await state.sandbox.disconnect()
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)
"""


@pytest.fixture
def lending_component(tmp_path, monkeypatch):
    """Write a component whose agent borrows the lent sandbox, import it, and give its agent class."""
    from nemo_gym.config.global_config import reset_global_config_dict

    created = []

    def create(**config):
        name = f"compliance_probe_{len(created)}"
        write(tmp_path, f"responses_api_agents/{name}/app.py", LENDING_AGENT)
        lines = "".join(f"      {key}: {str(value).lower()}\n" for key, value in config.items())
        write(
            tmp_path,
            f"responses_api_agents/{name}/configs/{name}.yaml",
            f"{name}:\n  responses_api_agents:\n    lending_agent:\n      entrypoint: app.py\n"
            f"      label: ${{policy_model_name}}\n{lines}",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.chdir(tmp_path)
        created.append(name)
        module = importlib.import_module(f"responses_api_agents.{name}.app")
        return tmp_path / "responses_api_agents" / name, module.LendingAgent

    reset_global_config_dict()
    yield create
    reset_global_config_dict()
    for name in created:
        for module in [module for module in sys.modules if module.startswith(f"responses_api_agents.{name}")]:
            del sys.modules[module]


def test_the_agent_config_is_the_default_configs_agent_instance_with_placeholders(lending_component):
    component, _ = lending_component(stop_the_lent_sandbox=True)
    assert probe.agent_config(component) == {
        "name": component.name,
        "host": "127.0.0.1",
        "port": 1,
        "entrypoint": "app.py",
        "label": "probe",
        "stop_the_lent_sandbox": True,
    }


def test_the_agent_config_takes_the_probe_override_for_what_the_default_config_leaves_unset(tmp_path):
    component = tmp_path / "claude_code_agent"
    write(
        component,
        "configs/claude_code_agent.yaml",
        "claude_code_agent:\n  responses_api_agents:\n    claude_code_agent:\n      claude_code_version: null\n",
    )
    assert probe.agent_config(component)["claude_code_version"] == "2.1.261"


def _pool_agent(tmp_path, **config):
    """A Pool-shaped default config: a component named `pool_agent`, whose agent needs the EULA opt-in."""
    component = tmp_path / "pool_agent"
    lines = "".join(f"      {key}: {str(value).lower()}\n" for key, value in config.items())
    write(component, "configs/pool_agent.yaml", f"pool_agent:\n  responses_api_agents:\n    pool_agent:\n{lines}")
    return component


def test_an_agent_without_its_operator_opt_in_is_not_probed_and_keeps_its_static_findings(
    lending_component, tmp_path, capsys
):
    _, agent = lending_component()
    component = _pool_agent(tmp_path, accept_pool_eula=False)
    results = findings("responses_api_agents/pool_agent/app.py", [agent], rules_for(AGENT_SERVER), component=component)
    assert [f["rule"] for f in results] == ["A1"]
    assert "A4 not applicable: pool_agent needs the operator to set accept_pool_eula" in capsys.readouterr().err


@pytest.mark.parametrize("stop", [False, True])
def test_an_agent_with_its_operator_opt_in_is_probed(lending_component, tmp_path, stop):
    _, agent = lending_component()
    component = _pool_agent(tmp_path, accept_pool_eula=True, stop_the_lent_sandbox=stop)
    results = findings("responses_api_agents/pool_agent/app.py", [agent], rules_for(AGENT_SERVER), component=component)
    assert _filter_findings(results, "A4") == []
    assert len(_filter_findings(results, "A5")) == int(stop)


def test_a_session_that_disconnects_leaves_the_lent_sandbox_running(lending_component):
    component, agent = lending_component()
    assert probe.probe_session(agent, probe.agent_config(component)) == probe.SessionObservation(
        "accepted", "accepted", False
    )


def test_a_session_that_stops_the_lent_sandbox_is_observed(lending_component):
    component, agent = lending_component(stop_the_lent_sandbox=True)
    assert probe.probe_session(agent, probe.agent_config(component)) == probe.SessionObservation(
        "accepted", "accepted", True
    )


def test_an_agent_that_says_it_does_not_support_sandbox_access_is_observed_refusing_it(lending_component):
    component, agent = lending_component(reject_sandbox_access=True)
    assert probe.probe_session(agent, probe.agent_config(component)) == probe.SessionObservation("refused", None, None)


@pytest.mark.parametrize("option", ["refuse_the_seed", "crash_the_seed"])
def test_a_seed_that_fails_is_observed_as_failed(lending_component, option):
    component, agent = lending_component(**{option: True})
    assert probe.probe_session(agent, probe.agent_config(component)) == probe.SessionObservation("failed", None, None)


def test_a_close_that_fails_is_observed_as_failed_and_the_sandbox_is_not_applicable(lending_component):
    component, agent = lending_component(fail_the_close=True)
    assert probe.probe_session(agent, probe.agent_config(component)) == probe.SessionObservation(
        "accepted", "failed", None
    )


def test_a_sandbox_whose_state_after_the_close_cannot_be_read_is_not_judged_stopped(
    lending_component, monkeypatch, capsys
):
    from nemo_gym.sandbox.providers.base import SandboxExecResult

    async def timed_out(self, handle, *args, **kwargs):
        return SandboxExecResult(None, "local command timed out", 125, "timeout")

    component, agent = lending_component()
    monkeypatch.setattr(probe.BorrowableLocalProvider, "exec", timed_out)
    assert probe.probe_session(agent, probe.agent_config(component)) == probe.SessionObservation(
        "accepted", "accepted", None
    )
    results = inspect_component(f"responses_api_agents/{component.name}")
    assert _filter_findings(results, "A4") == _filter_findings(results, "A5") == []
    assert "A5 not applicable: whether the lent sandbox survived the close could not be determined" in (
        capsys.readouterr().err
    )


def test_an_agent_that_stops_the_lent_sandbox_is_reported_at_its_close_hook(lending_component):
    component, agent = lending_component(stop_the_lent_sandbox=True)
    [finding] = _filter_findings(inspect_component(f"responses_api_agents/{component.name}"), "A5")
    assert finding["path"] == f"responses_api_agents/{component.name}/app.py"
    assert finding["line"] == agent._close_agent_session_state.__code__.co_firstlineno


def test_an_agent_that_refuses_the_lent_sandbox_properly_has_no_session_finding(lending_component, capsys):
    component, _ = lending_component(reject_sandbox_access=True)
    results = inspect_component(f"responses_api_agents/{component.name}")
    assert _filter_findings(results, "A4") == _filter_findings(results, "A5") == []
    assert capsys.readouterr().err == ""


def test_an_agent_whose_close_fails_is_noted_and_has_no_session_finding(lending_component, capsys):
    component, _ = lending_component(fail_the_close=True)
    assert _filter_findings(inspect_component(f"responses_api_agents/{component.name}"), "A5") == []
    assert "A5 not applicable: the session close was failed" in capsys.readouterr().err


def test_an_agent_that_leaves_the_lent_sandbox_running_has_no_session_finding(lending_component):
    component, _ = lending_component()
    assert _filter_findings(inspect_component(f"responses_api_agents/{component.name}"), "A5") == []


def test_a_stopped_lent_sandbox_is_located_at_the_own_close_hook_or_else_the_class():
    from nemo_gym.agents.base_responses_api_agent import SimpleResponsesAPIAgent

    class Closing(SimpleResponsesAPIAgent):
        async def _close_agent_session_state(self, state):
            pass

    class Inheriting(Closing):
        pass

    def stopped(agent, *observation):
        target = Target(agent)
        target.once("session", lambda: probe.SessionObservation(*observation))
        return rules.stopped_lent_sandbox(target)

    assert stopped(Closing, "accepted", "accepted", True) is vars(Closing)["_close_agent_session_state"]
    assert stopped(Inheriting, "accepted", "accepted", True) is Inheriting
    assert stopped(Closing, "accepted", "accepted", False) is None
    assert stopped(Closing, "refused", None, None) is None
    with pytest.raises(Unavailable, match="the session close was failed"):
        stopped(Closing, "accepted", "failed", None)
    with pytest.raises(Unavailable, match="could not be determined"):
        stopped(Closing, "accepted", "accepted", None)


def test_a_session_is_not_observed_without_a_component_to_run_it_from():
    with pytest.raises(Unavailable, match="no component directory"):
        probe.observe_session(Target(object))


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


@pytest.mark.parametrize("option", ["refuse_the_seed", "crash_the_seed"])
def test_an_agent_whose_seed_fails_is_reported_at_its_seed_hook_and_its_close_is_not_applicable(
    lending_component, capsys, option
):
    component, agent = lending_component(**{option: True})
    results = inspect_component(f"responses_api_agents/{component.name}")
    [finding] = _filter_findings(results, "A4")
    assert finding["line"] == agent._seed_agent_session_state.__code__.co_firstlineno
    assert _filter_findings(results, "A5") == []
    assert "not applicable" not in capsys.readouterr().err


def test_completed_findings_are_neutral_without_losing_their_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha")
    component = "responses_api_agents/legacy"
    finding = {"rule": "A1", "path": f"{component}/app.py", "line": 3, "message": "Keep | visible"}
    _write_result(tmp_path, component, "failed", [finding])
    _write_result(tmp_path, "responses_api_agents/clean", "passed")
    artifact_url = "https://github.com/example/Gym/actions/runs/123/artifacts/456"

    check = checks.compliance_check(
        [component, "responses_api_agents/clean"],
        artifacts=tmp_path,
        sha="sha",
        artifact_urls={ci.subject(component): artifact_url},
    )

    assert check["conclusion"] == "neutral"
    assert check["output"]["title"] == "1 compliant, 1 with findings, 0 unavailable"
    summary = check["output"]["summary"]
    assert f"| {component} | with findings | A1 | [Download]({artifact_url}) |" in summary
    assert f"- `{component}/app.py:3` **A1**: Keep &#124; visible" in summary
    assert "Tested commit: `sha`" in summary


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "{",
        "{}",
        "[null]",
        '[{"rule":"A1"}]',
        '[{"rule":"","path":"agent/app.py","line":3,"message":"finding"}]',
        '[{"rule":"A1","path":3,"line":3,"message":"finding"}]',
        '[{"rule":"A1","path":"agent/app.py","line":3,"message":""}]',
        '[{"rule":"A1","path":"agent/app.py","line":0,"message":"finding"}]',
        '[{"rule":"A1","path":"agent/app.py","line":true,"message":"finding"}]',
    ],
    ids=[
        "missing",
        "corrupt",
        "not-list",
        "not-record",
        "incomplete",
        "blank-rule",
        "invalid-path",
        "blank-message",
        "zero-line",
        "boolean-line",
    ],
)
def test_invalid_findings_artifacts_cannot_be_reported_as_advisory_success(tmp_path, monkeypatch, contents):
    monkeypatch.setenv("GITHUB_SHA", "sha")
    component = "responses_api_agents/legacy"
    finding = {"rule": "A1", "path": f"{component}/app.py", "line": 3, "message": "finding"}
    _write_result(tmp_path, component, "failed", [finding])
    path = tmp_path / ci.subject(component) / ci.FINDINGS_FILE
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)

    check = checks.compliance_check([component], artifacts=tmp_path, sha="sha")

    assert check["conclusion"] == "failure"
    assert check["output"]["title"] == "0 compliant, 0 with findings, 1 unavailable"
    assert f"| {component} | unavailable | - |" in check["output"]["summary"]


@pytest.mark.parametrize("status,has_findings", [("passed", True), ("failed", False)])
def test_contradictory_component_receipts_are_unavailable(tmp_path, monkeypatch, status, has_findings):
    monkeypatch.setenv("GITHUB_SHA", "sha")
    component = "responses_api_agents/legacy"
    finding = {"rule": "A1", "path": f"{component}/app.py", "line": 3, "message": "finding"}
    _write_result(tmp_path, component, status, [finding] if has_findings else [])

    check = checks.compliance_check([component], artifacts=tmp_path, sha="sha")

    assert check["conclusion"] == "failure"
    assert check["output"]["title"] == "0 compliant, 0 with findings, 1 unavailable"
