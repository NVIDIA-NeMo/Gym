# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose variables the deployment fills, such as a simulated user's model endpoint and key."""

from pathlib import Path

import pytest

from environment_servers.harbor import app as harbor_app
from environment_servers.harbor import sandboxes as harbor_sandboxes
from environment_servers.harbor.app import ComposeVariable
from environment_servers.harbor.tests import test_app
from environment_servers.harbor.tests.conftest import run_episode
from nemo_gym.tasks.harbor import load_task


# The shape of SWE-Interact's sidecar: its model endpoint, key and model come from the operator's shell.
SIDECAR_YAML = """
services:
  main:
    image: org/agent:1
  api:
    image: org/api:1
    environment:
      - LITELLM_API_KEY=${LITELLM_API_KEY}
      - LITELLM_BASE_URL=${LITELLM_BASE_URL}
      - SIM_USER_MODEL=${SIM_USER_MODEL:-}
  kafka:
    image: org/kafka:1
"""


def sidecar_server(tmp_path, monkeypatch, compose_yaml: str = SIDECAR_YAML, **variables):
    # The fixture's class is reached through its module, so pytest does not collect its tests here again.
    server, task, _, _, _ = test_app.TestComposeAndRouting().compose_server(tmp_path, monkeypatch)
    (task.path / "environment" / "docker-compose.yaml").write_text(compose_yaml)
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    server.config.compose_variables = {name: ComposeVariable.model_validate(v) for name, v in variables.items()}
    return server, task


def test_seed_fills_deployment_values_with_a_rollout_prefixed_model_url(tmp_path, monkeypatch):
    server, task = sidecar_server(
        tmp_path,
        monkeypatch,
        LITELLM_BASE_URL={"model_server": "user_model"},
        LITELLM_API_KEY={"env": "USER_MODEL_KEY"},
        SIM_USER_MODEL={"value": "gpt-5.5"},
    )
    monkeypatch.setenv("USER_MODEL_KEY", "sk-test")
    reached = []

    def sandbox_url(name, *, require_reachable=False):
        reached.append((name, require_reachable))
        return "http://10.0.0.5:9000"

    monkeypatch.setattr(harbor_sandboxes, "sandbox_server_url", sandbox_url)
    # The deployment's value sits over the process environment, as an operator's export would.
    monkeypatch.setenv("SIM_USER_MODEL", "from-the-shell")

    document = server._compose_document(task, server._compose_variable_values("r1-a2"))

    # The model server is reached from inside the sandbox, under the episode's rollout prefix and without the
    # training-capture segment: the simulated user's calls are recorded with the rollout, never trained on.
    assert reached == [("user_model", True)]
    assert document["services"]["api"]["environment"] == {
        "LITELLM_API_KEY": "sk-test",
        "LITELLM_BASE_URL": "http://10.0.0.5:9000/ng-rollout/r1-a2",
        "SIM_USER_MODEL": "gpt-5.5",
    }


async def test_an_unset_environment_source_is_a_configuration_error(tmp_path, monkeypatch):
    server, task = sidecar_server(tmp_path, monkeypatch, LITELLM_API_KEY={"env": "SURELY_UNSET_KEY"})
    monkeypatch.delenv("SURELY_UNSET_KEY", raising=False)

    response = await run_episode(server, task)

    assert response.failure.terminal and "SURELY_UNSET_KEY" in response.failure.failure_reason


@pytest.mark.parametrize("source", [{}, {"value": "a", "env": "B"}])
def test_a_compose_variable_takes_exactly_one_source(source):
    with pytest.raises(ValueError, match="exactly one"):
        ComposeVariable.model_validate(source)


async def test_a_document_with_deployment_values_never_lands_in_the_artifacts(tmp_path, monkeypatch):
    server, task = sidecar_server(
        tmp_path,
        monkeypatch,
        LITELLM_API_KEY={"value": "sk-secret"},
        LITELLM_BASE_URL={"value": "http://model"},
    )
    started: list[tuple[Path, str]] = []

    class RecordingCompose:
        def __init__(self, provider, path, *, service_specs, timeout_s):
            self.path = Path(path)
            self.services = {name: object() for name in service_specs}

        async def start(self):
            started.append((self.path, self.path.read_text()))

    monkeypatch.setattr(harbor_sandboxes, "AsyncSandboxCompose", RecordingCompose)

    # The fixture stubs `_create_compose` on the instance; this test needs the real one.
    await harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1")

    ((path, text),) = started
    assert "sk-secret" in text  # the adapter got the value it needs
    assert not path.exists()  # and the file is gone once the group started
    assert not path.is_relative_to(server._artifacts_dir)
    assert "sk-secret" not in "".join(p.read_text() for p in server._artifacts_dir.rglob("*") if p.is_file())
