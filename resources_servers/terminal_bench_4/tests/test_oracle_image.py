# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml

from resources_servers.terminal_bench_4 import environment as module
from resources_servers.terminal_bench_4.compose_config import resolve_compose
from resources_servers.terminal_bench_4.environment import Environment
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_environment import make_environment


@pytest.mark.parametrize("oracle_image", [None, "public/oracle"])
@pytest.mark.parametrize("role", ["agent", "oracle", "verifier"])
@pytest.mark.parametrize("agent_user", [None, "worker", 1000])
def test_oracle_image_changes_only_golden_image(tmp_path, monkeypatch, oracle_image, role, agent_user):
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        oracle=role == "oracle",
        verifier=role == "verifier",
        task_config={
            "oracle_docker_image": oracle_image,
            "agent": {"user": agent_user},
            "environment": {"workdir": "/work", "healthcheck": {"command": "check-agent-service"}},
            "verifier": {"user": "grader"},
        },
    )
    original = env.task.config.model_dump()
    expected = (
        "public/verifier"
        if role == "verifier"
        else oracle_image
        if role == "oracle" and oracle_image
        else "public/agent"
    )
    assert env.build_spec().image == expected
    assert env.role_user == ("grader" if role == "verifier" else agent_user)
    assert env.environment_dir == env.task.path / ("tests" if role == "verifier" else "environment")
    base = env.task.config.verifier_environment if role == "verifier" else env.task.config.environment
    assert env.settings.model_dump(exclude={"docker_image"}) == base.model_dump(exclude={"docker_image"})
    assert env.task.config.model_dump() == original
    assert env.task.config.environment.docker_image == "public/agent"
    assert env.task.config.verifier_environment.docker_image == "public/verifier"


@pytest.mark.parametrize("invalid", ["", " ", "image with spaces", 42, True, ["image"]])
def test_oracle_image_rejects_invalid_references(invalid):
    with pytest.raises(ValueError):
        TaskSettings.model_validate(
            {
                "oracle_docker_image": invalid,
                "environment": {"docker_image": "agent"},
                "verifier": {"environment_mode": "separate"},
            }
        )


def test_oracle_image_cannot_select_verifier_role(tmp_path, monkeypatch):
    env, *_ = make_environment(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="agent role"):
        Environment(env.task, env.config, "bad-role", tmp_path / "unused", oracle=True, verifier=True)


@pytest.mark.parametrize("compose", [False, True])
async def test_oracle_uses_its_startup_metadata_and_keeps_sidecars(tmp_path, monkeypatch, compose):
    env, _, sandbox_factory, compose_factory = make_environment(
        tmp_path,
        monkeypatch,
        compose=compose,
        oracle=True,
        task_config={"oracle_docker_image": "public/oracle"},
    )
    records = {
        image: {
            "image": image,
            "os": "linux",
            "architecture": "amd64",
            "config": {"Entrypoint": [entrypoint], "Cmd": ["python3"], "User": "root"},
        }
        for image, entrypoint in (("public/oracle", "/oracle-start"), ("db", "/database-start"))
    }
    catalog = tmp_path / "startup.json"
    catalog.write_text(json.dumps(records))
    env.config.compose_image_configs = catalog
    monkeypatch.setattr(module, "resolve_compose", resolve_compose)
    before = env.task.config.model_dump()
    await env.start()
    if compose:
        # The fixture explicitly names the normal image in the main overlay.
        # Oracle must replace it, not silently let the overlay win.
        document = yaml.safe_load(compose_factory.call_args.args[1].read_text())
        assert document["services"]["main"]["image"] == "public/oracle"
        assert document["services"]["main"]["entrypoint"] == ["/oracle-start"]
        assert document["services"]["main"]["command"] == ["sh", "-c", "sleep infinity"]
        assert document["services"]["db"]["image"] == "db"
        assert document["services"]["db"]["entrypoint"] == ["/database-start"]
        assert document["services"]["db"]["command"] == ["python3"]
        assert "image: public" in (env.environment_dir / "docker-compose.yaml").read_text()
    else:
        spec = sandbox_factory.call_args.args[1]
        assert spec.image == "public/oracle"
        assert spec.entrypoint == ["/oracle-start", "sh", "-c", "sleep infinity"]
    assert env.role_user == "task-user"
    assert env.task.config.model_dump() == before
    await env.stop()


def test_missing_oracle_startup_metadata_does_not_use_agent_metadata(tmp_path, monkeypatch):
    env, _, sandbox_factory, _ = make_environment(
        tmp_path, monkeypatch, oracle=True, task_config={"oracle_docker_image": "public/oracle"}
    )
    catalog = tmp_path / "startup.json"
    catalog.write_text(json.dumps({"public/agent": {"image": "public/agent", "config": {}}}))
    env.config.single_container_image_configs = catalog
    with pytest.raises(ValueError, match="public/oracle"):
        env.build_spec()
    sandbox_factory.assert_not_called()
