# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lifecycle + seam tests for SandboxCliAgent.

The sandbox and capture proxy are fully mocked (CI's base venv has no
boto3/botocore), so these exercise the *harness* logic: wire selection, the
capture proxy transport, in-box install/run/patch/eval sequencing, trajectory
gather, and grading dispatch — asserting real return values and env wiring
rather than padding coverage.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.adapters.capture_store import CaptureStore
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.sandbox_cli_agent import (
    HARBOR_GRADE,
    LaunchPlan,
    SandboxCliAgent,
    SandboxCliAgentConfig,
    SandboxCliAgentRunRequest,
    extract_instruction,
    node_install_command,
    swe_fields,
    swe_instance,
    swebench_report,
    swebench_test_spec,
)


# ──────────────────────────────────────────────────────────────────────
# Fakes / helpers
# ──────────────────────────────────────────────────────────────────────


def _exec(return_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code)


class _FakeSandbox:
    """Stand-in for AsyncSandbox: records exec/download calls, returns canned results."""

    def __init__(self, *, router=None, endpoint="http://box-endpoint", download_error=None):
        self._router = router or (lambda command: _exec(0))
        self.endpoint = endpoint
        self.download_error = download_error
        self.exec_calls: list[SimpleNamespace] = []
        self.downloads: list[tuple[str, str]] = []
        self.started = False
        self.stopped = False
        self.provider = None
        self.spec = None
        self.last_env_var = None

    async def start(self) -> None:
        self.started = True

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SimpleNamespace:
        self.exec_calls.append(SimpleNamespace(command=command, cwd=cwd, env=env, timeout_s=timeout_s, user=user))
        return self._router(command)

    def resolved_endpoint_url(self, env_var):
        self.last_env_var = env_var
        return self.endpoint

    async def download(self, remote, local) -> None:
        self.downloads.append((str(remote), str(local)))
        if self.download_error is not None:
            raise self.download_error

    async def stop(self) -> None:
        self.stopped = True


def _fake_proxy() -> SimpleNamespace:
    return SimpleNamespace(handle=SimpleNamespace(url="http://harness-proxy:9999"), stop=MagicMock())


@contextlib.contextmanager
def _run_env(fake: _FakeSandbox, proxy: SimpleNamespace | None = None):
    """Patch the AsyncSandbox + start_capture_proxy seams used by run()."""
    proxy = proxy if proxy is not None else _fake_proxy()
    start = MagicMock(return_value=proxy)

    def _ctor(provider, spec):
        fake.provider = provider
        fake.spec = spec
        return fake

    with (
        patch("nemo_gym.sandbox_cli_agent.AsyncSandbox", side_effect=_ctor),
        patch("nemo_gym.sandbox_cli_agent.start_capture_proxy", start),
    ):
        yield SimpleNamespace(start=start, proxy=proxy)


def _agent(*, plan=None, fallback=None, captured=None, server_client=None, **config_overrides):
    """A concrete SandboxCliAgent with a configurable launch seam.

    Built via ``model_construct`` so the heavy server plumbing (server_client,
    head config, ...) isn't needed; defaults are still applied to the config.
    """
    sandbox = config_overrides.pop("sandbox", {"ecs_fargate": {}})
    cfg = SandboxCliAgentConfig.model_construct(sandbox=sandbox, name="agent", **config_overrides)
    launch_plan = plan if plan is not None else LaunchPlan(run_command="AGENT_RUN")

    class _Agent(SandboxCliAgent):
        def build_launch(self, **kwargs) -> LaunchPlan:
            if captured is not None:
                captured.update(kwargs)
            return launch_plan

        def parse_stdout(self, stdout):
            return list(fallback or [])

    construct_kwargs = {"config": cfg}
    if server_client is not None:
        construct_kwargs["server_client"] = server_client
    return _Agent.model_construct(**construct_kwargs)


def _body(*, input="solve the task", metadata=None, **extra) -> SandboxCliAgentRunRequest:
    params = NeMoGymResponseCreateParamsNonStreaming(input=input, metadata=metadata)
    return SandboxCliAgentRunRequest(responses_create_params=params, **extra)


def _train_msg(text: str = "hi") -> NeMoGymResponseOutputMessageForTraining:
    return NeMoGymResponseOutputMessageForTraining(
        id="msg-x",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
        prompt_token_ids=[],
        generation_token_ids=[],
        generation_log_probs=[],
    )


@contextlib.contextmanager
def _fake_swebench_modules(*, make_test_spec=None, get_eval_report=None):
    """Inject a fake ``swebench`` package tree so the lazily-imported grading paths run."""
    injected: dict[str, types.ModuleType] = {}

    def _module(name: str, *, is_pkg: bool) -> types.ModuleType:
        module = types.ModuleType(name)
        if is_pkg:
            module.__path__ = []  # mark as a package so submodule imports resolve
        injected[name] = module
        return module

    swebench = _module("swebench", is_pkg=True)
    harness = _module("swebench.harness", is_pkg=True)
    swebench.harness = harness
    if make_test_spec is not None:
        test_spec_pkg = _module("swebench.harness.test_spec", is_pkg=True)
        harness.test_spec = test_spec_pkg
        test_spec_mod = _module("swebench.harness.test_spec.test_spec", is_pkg=False)
        test_spec_pkg.test_spec = test_spec_mod
        test_spec_mod.make_test_spec = make_test_spec
    if get_eval_report is not None:
        grading = _module("swebench.harness.grading", is_pkg=False)
        harness.grading = grading
        grading.get_eval_report = get_eval_report

    with patch.dict(sys.modules, injected):
        yield


# ──────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────


def test_extract_instruction_string_input_has_no_system():
    assert extract_instruction("hello world") == ("hello world", None)


def test_extract_instruction_reads_system_then_last_user():
    items = [
        SimpleNamespace(role="system", content="be brief"),
        SimpleNamespace(role="user", content="first"),
        SimpleNamespace(role="assistant", content="ignored"),
        {"role": "user", "content": "final ask"},
    ]
    assert extract_instruction(items) == ("final ask", "be brief")


def test_extract_instruction_joins_content_parts_dict_and_object():
    items = [{"role": "user", "content": [{"text": "a"}, SimpleNamespace(text="b"), {"no_text": 1}]}]
    user, system = extract_instruction(items)
    assert user == "ab"
    assert system is None


def test_extract_instruction_empty_inputs():
    assert extract_instruction([]) == ("", None)
    assert extract_instruction(None) == ("", None)


def test_node_install_command_bootstraps_static_node_then_npm_installs():
    cmd = node_install_command("@openai/codex", node_version="20.1.0", node_dir="/opt/n")
    assert "npm install -g @openai/codex" in cmd
    assert "nodejs.org/dist/v20.1.0/node-v20.1.0-linux-x64.tar.xz" in cmd
    assert "export PATH=/opt/n/bin:$PATH" in cmd


def test_swe_fields_parses_json_string_and_scalar_test_ids():
    fields = swe_fields({"FAIL_TO_PASS": '["t::a", "t::b"]', "PASS_TO_PASS": "t::c", "test_patch": "TP"})
    assert fields["f2p"] == ["t::a", "t::b"]  # JSON-list string
    assert fields["p2p"] == ["t::c"]  # bare scalar wrapped
    assert fields["test_patch"] == "TP"


def test_swe_fields_ignores_unparseable_instance_dict():
    fields = swe_fields({"instance_dict": "{not valid json", "FAIL_TO_PASS": ["t::a"]})
    assert fields["f2p"] == ["t::a"]  # top-level survives the bad blob


def test_swe_instance_merges_dict_instance_dict():
    inst = swe_instance(
        {"instance_dict": {"repo": "a/b", "version": "1.0", "instance_id": "iid", "FAIL_TO_PASS": ["x"]}}
    )
    assert inst is not None
    assert (inst["repo"], inst["version"], inst["instance_id"]) == ("a/b", "1.0", "iid")
    assert inst["FAIL_TO_PASS"] == ["x"]


def test_swebench_test_spec_none_without_canonical_fields():
    assert swebench_test_spec({"instance_id": "x"}) is None


def test_swebench_test_spec_none_when_swebench_unavailable():
    # full fields, but swebench isn't installed in this env -> lightweight fallback (None)
    md = {"instance_dict": json.dumps({"repo": "a/b", "version": "1.0", "instance_id": "iid"})}
    assert swebench_test_spec(md) is None


def test_swebench_test_spec_uses_make_test_spec_when_available():
    spec = SimpleNamespace(instance_id="iid")
    seen = []

    def make_test_spec(inst):
        seen.append(inst)
        return spec

    md = {"instance_dict": json.dumps({"repo": "a/b", "version": "1.0", "instance_id": "iid"})}
    with _fake_swebench_modules(make_test_spec=make_test_spec):
        assert swebench_test_spec(md) is spec
    assert seen[0]["repo"] == "a/b"


def test_swebench_test_spec_returns_none_when_make_test_spec_raises():
    def boom(inst):
        raise RuntimeError("cannot build")

    md = {"instance_dict": json.dumps({"repo": "a/b", "version": "1.0", "instance_id": "iid"})}
    with _fake_swebench_modules(make_test_spec=boom):
        assert swebench_test_spec(md) is None


def test_swebench_report_parses_official_eval_report():
    report = {
        "iid": {
            "resolved": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["a", "b"], "failure": ["c"]},
                "PASS_TO_PASS": {"success": ["d"], "failure": []},
            },
        }
    }
    seen = {}

    def get_eval_report(*, test_spec, prediction, test_log_path, include_tests_status):
        seen["prediction"] = prediction
        seen["include"] = include_tests_status
        with open(test_log_path) as handle:
            seen["log"] = handle.read()
        return report

    spec = SimpleNamespace(instance_id="iid")
    with _fake_swebench_modules(get_eval_report=get_eval_report):
        reward, fields = swebench_report(spec, "the patch", "raw pytest log")

    assert reward == 1.0
    assert fields["resolved"] is True
    assert fields["fail_to_pass_pass"] == 2 and fields["fail_to_pass_fail"] == 1
    assert fields["pass_to_pass_pass"] == 1 and fields["pass_to_pass_fail"] == 0
    assert seen["prediction"] == {"instance_id": "iid", "model_patch": "the patch"}
    assert seen["include"] is True
    assert seen["log"] == "raw pytest log"


def test_swebench_report_unresolved_returns_zero():
    def get_eval_report(*, test_spec, prediction, test_log_path, include_tests_status):
        return {"iid": {"resolved": False, "tests_status": {}}}

    with _fake_swebench_modules(get_eval_report=get_eval_report):
        reward, fields = swebench_report(SimpleNamespace(instance_id="iid"), "", "log")
    assert reward == 0.0
    assert fields == {"resolved": False}


# ──────────────────────────────────────────────────────────────────────
# Method seams (real method dispatch via a constructed agent)
# ──────────────────────────────────────────────────────────────────────


def test_resolve_base_url_from_model_base_url():
    agent = _agent(model_base_url="http://model-root/")
    assert agent._resolve_base_url() == "http://model-root"  # trailing slash stripped


def test_resolve_base_url_from_model_server_uses_server_client():
    client = SimpleNamespace(
        global_config_dict={"k": "v"},
        _build_server_base_url=lambda cfg: "http://resolved-root",
    )
    agent = _agent(
        model_server=ModelServerRef(type="responses_api_models", name="srv"),
        server_client=client,
    )
    with patch("nemo_gym.sandbox_cli_agent.get_first_server_config_dict", return_value={"host": "h"}) as resolve:
        assert agent._resolve_base_url() == "http://resolved-root"
    resolve.assert_called_once_with({"k": "v"}, "srv")


def test_resolve_image_prefers_metadata_docker_image():
    agent = _agent(image="cfg-img", image_template="tmpl-{instance_id}")
    assert agent._resolve_image({"docker_image": "per/task:img"}) == "per/task:img"


def test_resolve_image_uses_template_with_swebench_tag():
    agent = _agent(image_template="ecr/repo:{instance_id}", image="cfg-img")
    assert agent._resolve_image({"instance_id": "astropy__astropy-12907"}) == "ecr/repo:astropy_1776_astropy-12907"


def test_resolve_image_defaults_to_config_image():
    agent = _agent(image="cfg-img")
    assert agent._resolve_image({}) == "cfg-img"


def test_build_eval_explicit_command_wins():
    assert _agent(eval_command="run-checks.sh")._build_eval({}) == "run-checks.sh"


def test_build_eval_none_without_fail_to_pass():
    assert _agent()._build_eval({"instance_id": "x"}) is None


def test_build_eval_applies_test_patch_and_activates_conda():
    cmd = _agent(eval_conda_env="testbed", apply_test_patch=True)._build_eval(
        {"FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": ["t::b"], "test_patch": "PATCHDATA"}
    )
    assert "base64 -d > /tmp/nemo_gym_test.patch" in cmd
    assert "git apply" in cmd
    assert "conda activate testbed" in cmd
    assert "python -m pytest" in cmd
    assert "t::a" in cmd and "t::b" in cmd


def test_build_eval_skips_conda_when_disabled():
    cmd = _agent(eval_conda_env=None)._build_eval({"FAIL_TO_PASS": ["t::a"]})
    assert "conda activate" not in cmd
    assert "python -m pytest" in cmd


def test_eval_plan_routes_harbor_tests_to_harbor_grade():
    cmd, grade = _agent()._eval_plan({"harbor_tests": json.dumps({"/tests/test.sh": "echo hi"})})
    assert grade is HARBOR_GRADE
    assert "/tests/test.sh" in cmd and "NEMO_GYM_HARBOR_REWARD=" in cmd


def test_eval_plan_uses_swebench_spec_when_available():
    spec = SimpleNamespace(eval_script="#!/bin/bash\n  run the eval  ", instance_id="iid")
    with patch("nemo_gym.sandbox_cli_agent.swebench_test_spec", return_value=spec):
        cmd, grade = _agent()._eval_plan({"instance_id": "iid"})
    assert grade is spec
    assert cmd == "run the eval"  # shebang stripped + trimmed


def test_eval_plan_falls_back_to_membership_builder():
    with patch("nemo_gym.sandbox_cli_agent.swebench_test_spec", return_value=None):
        cmd, grade = _agent()._eval_plan({"FAIL_TO_PASS": ["t::a"]})
    assert grade is None
    assert "python -m pytest" in cmd


def test_harbor_eval_stages_tests_runs_entry_and_echoes_reward():
    cmd = _agent()._harbor_eval({"/tests/test.sh": "echo hi", "/tests/t.py": "x=1"}, entry="/tests/test.sh")
    assert "mkdir -p /tests /logs/verifier" in cmd
    assert "base64 -d > /tests/test.sh" in cmd
    assert "bash /tests/test.sh" in cmd
    assert 'echo "NEMO_GYM_HARBOR_REWARD=' in cmd


def test_gather_prefers_token_id_capture(tmp_path):
    agent = _agent(capture_dir=str(tmp_path / "cap"), model_api="responses")
    CaptureStore(agent.config.capture_dir).record(
        "sX",
        {
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}],
                        "generation_token_ids": [1, 2, 3],
                    }
                ]
            }
        },
    )
    items, rl_ready = agent._gather("sX", "stdout ignored")
    assert rl_ready is True
    assert len(items) == 1


def test_gather_falls_back_to_stdout_without_capture(tmp_path):
    agent = _agent(capture_dir=str(tmp_path / "cap"), model_api="chat", fallback=[_train_msg("x")])
    items, rl_ready = agent._gather("missing-session", "stdout")
    assert rl_ready is False
    assert len(items) == 1


# ──────────────────────────────────────────────────────────────────────
# run() — full lifecycle
# ──────────────────────────────────────────────────────────────────────


async def test_run_proxy_membership_grade_with_install_and_path_prepend(tmp_path):
    def router(command: str):
        if "git diff --cached" in command:
            return _exec(0, stdout="diff --git a/x b/x\n+fix\n")
        if "AGENT_INSTALL" in command:
            return _exec(3, stderr="install boom")
        if "AGENT_RUN" in command:
            return _exec(2, stdout="agent stdout", stderr="agent boom")
        if "pytest" in command:
            return _exec(0, stdout="PASSED t::a\n")
        return _exec(0)

    fake = _FakeSandbox(router=router, endpoint=None)  # no reverse tunnel -> advertise URL used
    captured: dict = {}
    plan = LaunchPlan(
        run_command="AGENT_RUN",
        env={"X": "1"},
        setup_commands=["AGENT_SETUP"],
        install_command="AGENT_INSTALL",
        path_prepend="/opt/nodejs/bin",
    )
    agent = _agent(
        plan=plan,
        fallback=[_train_msg("h")],
        captured=captured,
        sandbox={"opensandbox": {}},
        model_api="responses",
        model_base_url="http://model-root",
        proxy_advertise_url="http://advertise",
        capture_dir=str(tmp_path / "cap"),
    )
    body = _body(input="solve the task", instance_id="iid", FAIL_TO_PASS=["t::a"])

    with _run_env(fake) as env:
        resp = await agent.run(None, body)

    # grading: lightweight membership grader scored the F2P test as resolved
    assert resp.reward == 1.0
    assert resp.patch_exists is True
    assert resp.turns_used == 1  # stdout fallback (1 assistant turn)
    assert resp.response.metadata["resolved"] == "True"
    assert resp.response.metadata["rl_token_ids"] == "false"
    assert resp.response.metadata["run_install_rc"] == "3"
    assert "install boom" in resp.response.metadata["run_install_stderr"]
    assert resp.response.metadata["run_agent_rc"] == "2"
    assert resp.response.metadata["run_eval_rc"] == "0"

    # capture proxy wired with the upstream root + token-id injection, no anthropic translate
    proxy_kwargs = env.start.call_args.kwargs
    assert proxy_kwargs["model_base_url"] == "http://model-root"
    assert proxy_kwargs["translate_anthropic"] is False
    assert proxy_kwargs["inject_extra_body"] == {"return_token_id_information": True}
    assert proxy_kwargs["upstream_api_key"] is None
    assert proxy_kwargs["session_id"].startswith("agent-")

    # the proxy ROOT (not /v1) is registered as an outside endpoint for the box
    assert fake.spec.provider_options == {
        "outside_endpoints": [{"url": "http://harness-proxy:9999", "env_var": "OPENAI_BASE_URL"}]
    }
    # box reaches the proxy via the advertised URL, with the wire /v1 suffix appended once
    assert captured["box_base_url"] == "http://advertise/v1"
    assert captured["prompt"] == "solve the task"
    assert captured["system_prompt"] is None
    assert captured["workdir"] == "/workspace"
    assert captured["config_dir"] == "/workspace/.agent"

    # PATH was prepended before the run command; cleanup happened
    run_call = next(c for c in fake.exec_calls if "AGENT_RUN" in c.command)
    assert "export PATH=/opt/nodejs/bin:$PATH" in run_call.command
    assert run_call.env == {"X": "1"}
    env.proxy.stop.assert_called_once()
    assert fake.started is True and fake.stopped is True


async def test_run_proxy_unreachable_raises_and_cleans_up(tmp_path):
    fake = _FakeSandbox(endpoint=None)  # no reverse tunnel
    agent = _agent(
        sandbox={"opensandbox": {}},
        model_base_url="http://model-root",
        proxy_advertise_url=None,  # and nothing advertised
        capture_dir=str(tmp_path / "cap"),
    )
    body = _body()
    with _run_env(fake) as env:
        with pytest.raises(RuntimeError, match="unreachable from the sandbox"):
            await agent.run(None, body)
    # the finally block still stopped the proxy + sandbox
    env.proxy.stop.assert_called_once()
    assert fake.stopped is True


async def test_run_harbor_grade(tmp_path):
    def router(command: str):
        if "git diff --cached" in command:
            return _exec(0, stdout="patch")
        if "AGENT_RUN" in command:
            return _exec(0, stdout="ran")
        # the harbor eval command
        return _exec(0, stdout="noise\nNEMO_GYM_HARBOR_REWARD=1\n")

    fake = _FakeSandbox(router=router, endpoint="http://box")
    agent = _agent(
        plan=LaunchPlan(run_command="AGENT_RUN"),
        sandbox={"opensandbox": {}},
        model_base_url="http://model-root",
        capture_dir=str(tmp_path / "cap"),
    )
    body = _body(harbor_tests=json.dumps({"/tests/test.sh": "echo hi"}))
    with _run_env(fake):
        resp = await agent.run(None, body)

    assert resp.reward == 1.0
    assert resp.response.metadata["resolved"] == "True"
    eval_call = fake.exec_calls[-1]
    assert "bash /tests/test.sh" in eval_call.command


async def test_run_swebench_spec_grade(tmp_path):
    spec = SimpleNamespace(eval_script="#!/bin/bash\nrun-eval", instance_id="iid")

    def router(command: str):
        if "git diff --cached" in command:
            return _exec(0, stdout="patch text")
        if "AGENT_RUN" in command:
            return _exec(0, stdout="ran")
        if command == "run-eval":
            return _exec(0, stdout="raw eval log")
        return _exec(0)

    fake = _FakeSandbox(router=router, endpoint="http://box")
    agent = _agent(
        plan=LaunchPlan(run_command="AGENT_RUN"),
        sandbox={"opensandbox": {}},
        model_base_url="http://model-root",
        capture_dir=str(tmp_path / "cap"),
    )
    body = _body(instance_id="iid")
    report = MagicMock(return_value=(1.0, {"resolved": True, "fail_to_pass_pass": 5}))
    with _run_env(fake):
        with (
            patch("nemo_gym.sandbox_cli_agent.swebench_test_spec", return_value=spec),
            patch("nemo_gym.sandbox_cli_agent.swebench_report", report),
        ):
            resp = await agent.run(None, body)

    assert resp.reward == 1.0
    assert resp.response.metadata["resolved"] == "True"
    assert resp.response.metadata["fail_to_pass_pass"] == "5"
    # official grader received the captured patch + raw eval output
    spec_arg, patch_arg, output_arg = report.call_args.args
    assert spec_arg is spec
    assert patch_arg == "patch text"
    assert "raw eval log" in output_arg
    # the eval command is the spec's eval_script (shebang stripped)
    assert any(c.command == "run-eval" for c in fake.exec_calls)


async def test_run_captured_trajectory_wins_with_token_ids(tmp_path):
    def router(command: str):
        if "git diff --cached" in command:
            return _exec(0, stdout="d")
        if "AGENT_INSTALL" in command:
            return _exec(0)
        if "AGENT_RUN" in command:
            return _exec(0, stdout="out")
        return _exec(0)

    fake = _FakeSandbox(router=router, endpoint="http://box")  # reverse tunnel resolves
    plan = LaunchPlan(run_command="AGENT_RUN", install_command="AGENT_INSTALL")
    captured: dict = {}
    agent = _agent(
        plan=plan,
        captured=captured,
        sandbox={"opensandbox": {}},
        model_api="responses",
        model_base_url="http://model-root",
        capture_dir=str(tmp_path / "cap"),
    )

    fixed = SimpleNamespace(hex="deadbeefcafebabe0123456789")
    session_id = f"agent-{fixed.hex[:12]}"
    CaptureStore(agent.config.capture_dir).record(
        session_id,
        {
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                        "generation_token_ids": [10, 11, 12],
                    }
                ]
            }
        },
    )

    body = _body(input="go")  # no grading metadata -> no eval
    with _run_env(fake):
        with patch("nemo_gym.sandbox_cli_agent.uuid4", return_value=fixed):
            resp = await agent.run(None, body)

    assert resp.response.metadata["rl_token_ids"] == "true"
    assert resp.response.metadata["session_id"] == session_id
    assert resp.turns_used == 1
    assert resp.reward == 0.0  # nothing to grade
    assert resp.response.metadata["run_install_rc"] == "0"
    # reverse-tunnel endpoint used directly (no advertise needed)
    assert captured["box_base_url"] == "http://box/v1"
