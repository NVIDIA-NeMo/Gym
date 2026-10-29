# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose variables a task leaves to the deployment, such as a simulated user's model endpoint and key."""

import asyncio
import tempfile
import threading
from pathlib import Path

import pytest

from environment_servers.harbor import app as harbor_app
from environment_servers.harbor.app import ComposeVariable
from environment_servers.harbor.tests import test_app
from environment_servers.harbor.tests.conftest import run_episode
from environment_servers.harbor.users import HarborUserError
from nemo_gym.data import ImageConfigError, load_compose_document, load_task


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


def test_prepare_leaves_environment_only_variables_to_the_deployment(tmp_path, monkeypatch):
    _, task = sidecar_server(tmp_path, monkeypatch)

    document = load_compose_document(task)

    # Images and structure are what prepare needs; the deployment's values are filled at seed.
    assert document["services"]["api"]["image"] == "org/api:1"
    assert document["services"]["api"]["environment"] == {
        "LITELLM_API_KEY": "",
        "LITELLM_BASE_URL": "",
        "SIM_USER_MODEL": "",
    }


def test_prepare_empties_an_undefined_variable_outside_environment_as_compose_does(tmp_path, monkeypatch):
    # Prepare follows docker compose for every unset name (an empty string, with one warning); the seed, with the
    # deployment's values in hand, is where an unset name is refused.
    _, task = sidecar_server(
        tmp_path, monkeypatch, compose_yaml="services:\n  main:\n    image: org/agent:1\n  api:\n    image: ${API}\n"
    )
    assert load_compose_document(task)["services"]["api"]["image"] == ""
    with pytest.raises(ImageConfigError, match="API"):
        load_compose_document(task, operator_variables={})


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

    monkeypatch.setattr("environment_servers.harbor.sandboxes.sandbox_server_url", sandbox_url)

    document = server._compose_document(task, "r1-a2")

    # The model server is reached from inside the sandbox, under the episode's rollout prefix and without the
    # training-capture segment: the simulated user's calls are recorded with the rollout, never trained on.
    assert reached == [("user_model", True)]
    assert document["services"]["api"]["environment"] == {
        "LITELLM_API_KEY": "sk-test",
        "LITELLM_BASE_URL": "http://10.0.0.5:9000/ng-rollout/r1-a2",
        "SIM_USER_MODEL": "gpt-5.5",
    }


async def test_a_variable_the_deployment_leaves_unset_fails_the_seed_terminally(tmp_path, monkeypatch):
    server, task = sidecar_server(tmp_path, monkeypatch, SIM_USER_MODEL={"value": "gpt-5.5"})

    response = await run_episode(server, task)

    assert response.failure.terminal and response.failure.stage == "seed"
    assert "LITELLM_API_KEY" in response.failure.failure_reason


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
        def __init__(self, provider, path, *, service_specs, timeout_s, **adapter_kwargs):
            self.path = Path(path)
            self.services = {name: object() for name in service_specs}

        async def start(self):
            started.append((self.path, self.path.read_text()))

    monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", RecordingCompose)

    # The fixture stubs `_create_compose` on the instance; this test needs the real one.
    await harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1")

    ((path, text),) = started
    assert "sk-secret" in text  # the adapter got the value it needs
    assert not path.exists()  # and the file is gone once the group started
    assert not path.is_relative_to(server._artifacts_dir)
    assert "sk-secret" not in "".join(p.read_text() for p in server._artifacts_dir.rglob("*") if p.is_file())


async def test_the_private_document_is_removed_when_the_start_is_refused_before_any_group(tmp_path, monkeypatch):
    """A refusal between the document's write and the group's start (an `app:app` agent user OpenSandbox cannot run
    as, refused by `_sandbox_spec`) leaves no document with the deployment's values behind: the private folder goes
    with every exit of `_create_compose`, not only with the group's start and its retries."""
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setenv("TMPDIR", str(private))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    server, task = sidecar_server(
        tmp_path, monkeypatch, LITELLM_API_KEY={"value": "sk-secret"}, LITELLM_BASE_URL={"value": "http://model"}
    )
    (task.path / "task.toml").write_text(test_app.COMPOSE_TOML.replace("[agent]\n", '[agent]\nuser = "app:app"\n'))
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    assert task.user == "app:app"
    written: list[Path] = []
    write = server._write_compose_document

    def recording_write(*args, **kwargs):
        document, path = write(*args, **kwargs)
        written.append(path)
        return document, path

    monkeypatch.setattr(server, "_write_compose_document", recording_write)

    # The fixture stubs `_create_compose` on the instance; this test needs the real one.
    with pytest.raises(HarborUserError, match="app:app"):
        await harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1")

    (path,) = written
    assert path.parent.parent == private and path.parent.name.startswith("nemo-gym-compose-")
    assert not path.parent.exists() and not list(private.glob("nemo-gym-compose-*"))
    assert "sk-secret" not in "".join(p.read_text() for p in private.rglob("*") if p.is_file())


async def test_a_seed_cancelled_while_the_private_document_is_written_leaves_nothing_behind(tmp_path, monkeypatch):
    """Review (CI bot, round 34): the private document was written in a worker thread under a plain await, and a
    seed cancelled during that write (an episode deadline, a shutdown) left the coroutine without the path while the
    worker went on to create the credential-bearing folder, so the file stayed for good. The write is owned past the
    cancellation: whatever the worker produces after it is removed as soon as it is done, and no group is started."""
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setenv("TMPDIR", str(private))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    server, task = sidecar_server(
        tmp_path, monkeypatch, LITELLM_API_KEY={"value": "sk-secret"}, LITELLM_BASE_URL={"value": "http://model"}
    )
    writing, release = threading.Event(), threading.Event()
    written: list[Path] = []
    write = server._write_compose_document

    def gated_write(*args, **kwargs):
        writing.set()
        assert release.wait(timeout=10)
        document, path = write(*args, **kwargs)
        written.append(path)
        return document, path

    monkeypatch.setattr(server, "_write_compose_document", gated_write)
    starts: list = []

    async def no_start(*args, **kwargs):
        starts.append(args)
        raise AssertionError("no group may start for a cancelled seed")

    monkeypatch.setattr(server, "_start_compose_group", no_start)

    # The fixture stubs `_create_compose` on the instance; this test needs the real one.
    create = asyncio.ensure_future(harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1"))
    assert await asyncio.to_thread(writing.wait, 10)
    create.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create
    assert written == []  # cancelled while the worker was still writing
    release.set()
    for _ in range(500):
        if written and not written[0].parent.exists():
            break
        await asyncio.sleep(0.01)

    (path,) = written
    assert path.parent.parent == private and path.parent.name.startswith("nemo-gym-compose-")
    assert not path.parent.exists() and not list(private.glob("nemo-gym-compose-*"))
    assert "sk-secret" not in "".join(p.read_text() for p in private.rglob("*") if p.is_file())
    assert starts == []
