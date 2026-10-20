# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in sandbox-session settings for running OpenCode as a harness started in the task container."""

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.app import _REMAINING_CONTEXT_PLUGIN as REMAINING_CONTEXT_PLUGIN
from responses_api_agents.opencode_agent.app import _sandbox_prepare_command, argv_prompt
from responses_api_agents.opencode_agent.runtime import OBSERVABILITY_PATCH
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import close_body, seed, setup  # noqa: F401


ROLLOUT = "/ng-rollout/opencode-smoke-a2/v1/responses"
PACKAGE = Path(__file__).parents[1]


def configure_direct(agent, **overrides):
    """A direct OpenAI-compatible endpoint instead of a Gym model server."""
    settings = {
        "model_server": None,
        "model_base_url": "http://gateway.example:16100/v1/",
        "model_api_key": "gateway-key",  # pragma: allowlist secret
        "model": "deepseek-flash",
    } | overrides
    for name, value in settings.items():
        setattr(agent.config, name, value)


def run_session(agent, sandbox, body=None, request=None):
    """Seed, activate and close one session; return the response and the runner input."""
    body = body or seed()
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post(ROLLOUT, json=request or {"input": "Fix the code"})
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        closed = client.post("/v1/agent_sessions/close", json=close_body(body.agent_session_id))
        assert closed.status_code == 200, closed.text
    return response, payload, closed.json()


def test_direct_endpoint_session_calls_the_configured_endpoint(setup):
    agent, sandbox = setup
    configure_direct(agent)
    # Gym capture is on, but calls to a direct endpoint never pass through Gym: no correlation plugin.
    agent.server_client.global_config_dict["observability_enabled"] = True
    response, payload, _ = run_session(agent, sandbox)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "deepseek-flash"
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert (config["model"], config["small_model"]) == ("nemo_gym/deepseek-flash", "nemo_gym/deepseek-flash")
    provider = config["provider"]["nemo_gym"]
    assert provider["options"] == {
        "baseURL": "http://gateway.example:16100/v1",
        "apiKey": "gateway-key",  # pragma: allowlist secret
        "timeout": False,
    }
    assert list(provider["models"]) == ["deepseek-flash"]
    assert not any(OBSERVABILITY_PATCH.name in plugin for plugin in config["plugin"])
    assert not any(path.endswith(OBSERVABILITY_PATCH.name) for path in sandbox.files)
    agent.server_client.post.assert_not_called()


def test_direct_endpoint_key_defaults_to_a_placeholder(setup):
    agent, sandbox = setup
    configure_direct(agent, model_api_key=None)
    response, payload, _ = run_session(agent, sandbox)
    assert response.status_code == 200, response.text
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["nemo_gym"]["options"]["apiKey"] == "EMPTY"  # pragma: allowlist secret


@pytest.mark.parametrize("routes", ["neither", "both"])
def test_seed_requires_exactly_one_model_route(setup, routes):
    agent, sandbox = setup
    if routes == "neither":
        agent.config.model_server = None
    else:
        agent.config.model_base_url = "http://gateway.example/v1"
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert "model_server or model_base_url" in response.text
    sandbox.exec.assert_not_awaited()


def test_direct_endpoint_accepts_its_own_model_name_in_requests(setup):
    agent, sandbox = setup
    configure_direct(agent)
    response, _, _ = run_session(agent, sandbox, request={"input": "Fix the code", "model": "deepseek-flash"})
    assert response.status_code == 200, response.text


def test_request_settings_reach_the_session_provider(setup):
    agent, sandbox = setup
    configure_direct(
        agent,
        reasoning_effort="max",
        interleaved_reasoning_field="reasoning",
        output_token_policy="remaining_context",
        declare_model_limits=False,
    )
    response, payload, _ = run_session(agent, sandbox)
    assert response.status_code == 200, response.text
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    # No size: OpenCode compacts only on an overflow error; effort goes on every request.
    assert config["provider"]["nemo_gym"]["models"]["deepseek-flash"] == {
        "interleaved": {"field": "reasoning"},
        "options": {"reasoningEffort": "max"},
    }
    plugin = f"{sandbox.directory}/{REMAINING_CONTEXT_PLUGIN}"
    assert Path(plugin).as_uri() in config["plugin"]
    assert sandbox.files[plugin] == (PACKAGE / REMAINING_CONTEXT_PLUGIN).read_text()


def test_default_request_settings_keep_the_declared_limits(setup):
    agent, sandbox = setup
    response, payload, _ = run_session(agent, sandbox)
    assert response.status_code == 200, response.text
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"] == {
        "interleaved": {"field": "reasoning_content"},
        "limit": {"context": 32000, "input": 32000, "output": 4096},
    }
    assert not any(REMAINING_CONTEXT_PLUGIN in plugin for plugin in config["plugin"])
    assert not any(path.endswith(REMAINING_CONTEXT_PLUGIN) for path in sandbox.files)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute the OpenCode plugin")
def test_remaining_context_plugin_drops_max_tokens_for_the_session_provider_only():
    script = """
import assert from 'node:assert/strict';
const { RemainingContext } = await import(process.argv[1]);
const hooks = await RemainingContext();
for (const [providerID, kept] of [['nemo_gym', false], ['other', true]]) {
  const output = { maxOutputTokens: 32000, temperature: 1 };
  await hooks['chat.params']({ model: { providerID } }, output);
  assert.equal('maxOutputTokens' in output, kept);
  assert.equal(output.temperature, 1);
}
"""
    completed = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script, (PACKAGE / REMAINING_CONTEXT_PLUGIN).as_uri()],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_root_workdir_runs_when_allowed(setup):
    # Task images without a WORKDIR start in "/": OpenCode's runtime and session files stay under /tmp.
    agent, sandbox = setup
    agent.config.allow_root_workdir = True
    sandbox.expected_workdir = "/"
    body = seed()
    body.sandbox_access.workdir = "/"
    response, payload, _ = run_session(agent, sandbox, body=body)
    assert response.status_code == 200, response.text
    assert payload["cwd"] == "/"
    assert sandbox.directory.startswith("/tmp/nemo-gym-opencode-sessions/")
    assert "allow_root=True" in sandbox.exec.await_args_list[0].args[0]


@pytest.mark.parametrize("workdir", ["/tmp", "/tmp/nemo-gym-opencode-sessions"])
async def test_root_workdir_setting_still_rejects_adapter_storage(setup, workdir):
    agent, sandbox = setup
    agent.config.allow_root_workdir = True
    body = seed()
    body.sandbox_access.workdir = workdir
    with pytest.raises(HTTPException) as error:
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    assert error.value.status_code == 422
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("allow_root", [False, True])
def test_resolved_root_workdir_check(tmp_path, allow_root):
    command = shlex.split(
        _sandbox_prepare_command("/", "/tmp/nemo-gym-opencode-sessions/x", "/tmp/rt", allow_root=allow_root)
    )
    script = command[command.index("-I") + 2]
    sessions, runtime = tmp_path / "sessions", tmp_path / "runtime"
    checked = subprocess.run(
        [sys.executable, "-I", "-c", script, "/", str(sessions), str(runtime)], capture_output=True, text=True
    )
    assert (checked.returncode == 0) is allow_root, checked.stderr
    # A storage path that holds the workdir is rejected either way.
    workdir = sessions / "task"
    workdir.mkdir(parents=True)
    held = subprocess.run(
        [sys.executable, "-I", "-c", script, str(workdir), str(sessions), str(runtime)], capture_output=True, text=True
    )
    assert held.returncode != 0
    assert "overlaps the task workdir" in held.stderr


@pytest.mark.parametrize(
    "instruction,sent",
    [
        ('Fix "the" code\nand test it', '"Fix \\"the\\" code\nand test it"'),
        ("one-word", "one-word"),
    ],
)
def test_argv_prompt_quoting_sends_the_message_opencode_builds_from_one_argument(setup, instruction, sent):
    agent, sandbox = setup
    agent.config.argv_prompt_quoting = True
    response, payload, _ = run_session(agent, sandbox, request={"input": instruction})
    assert response.status_code == 200, response.text
    assert payload["prompt"] == sent
    assert argv_prompt(instruction) == sent


def test_prompt_is_sent_as_written_by_default(setup):
    agent, sandbox = setup
    response, payload, _ = run_session(agent, sandbox, request={"input": 'Fix "the" code'})
    assert response.status_code == 200, response.text
    assert payload["prompt"] == 'Fix "the" code'
