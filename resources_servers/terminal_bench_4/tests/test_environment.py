# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from resources_servers.terminal_bench_4 import environment as module
from resources_servers.terminal_bench_4.compose_config import resolve_compose
from resources_servers.terminal_bench_4.environment import Environment, EnvironmentConfig, HealthcheckError
from resources_servers.terminal_bench_4.task import TaskSettings


def environment_config(**overrides):
    return EnvironmentConfig.model_validate(
        {
            "cpu_enforcement_policy": "limit",
            "memory_enforcement_policy": "limit",
            "sandbox_provider": {"opensandbox": {"connection": {"domain": "example.invalid"}}},
            "sandbox_metadata": {},
            "sandbox_provider_options": {},
            "sandbox_env": {},
            "sandbox_env_by_task": {},
            "sandbox_request_gpu_type": True,
            "sandbox_split_endpoints": False,
            "compose_image_configs": None,
            "sandbox_ttl_s": 21600,
            "sandbox_ready_timeout_s": 900,
            "default_exec_timeout_s": 1800,
            "exec_shell": "bash -c",
            "image_rewrites": [],
            "workdir": None,
            "efs_logs_host_path": None,
            "efs_logs_init_image": "python:3.13-slim",
        }
        | overrides
    )


def make_environment(tmp_path, monkeypatch, *, compose=False, verifier=False, config=None, task_config=None):
    raw = {
        "environment": {
            "docker_image": "public/agent",
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 15000,
            "env": {"TASK": "value"},
        },
        "agent": {"user": "task-user"},
        "verifier": {"environment": {"docker_image": "public/verifier", "cpus": 8, "gpus": 1, "gpu_types": ["H100"]}},
    }
    for key, value in (task_config or {}).items():
        raw.setdefault(key, {}).update(value)
    task = SimpleNamespace(
        config=TaskSettings.model_validate(raw), name="terminal-bench/test", path=tmp_path / "package"
    )
    (task.path / "environment").mkdir(parents=True, exist_ok=True)
    (task.path / "tests").mkdir(exist_ok=True)
    (task.path / "environment/Dockerfile").write_text("FROM public")
    (task.path / "tests/Dockerfile").write_text("FROM public")
    cfg = environment_config(**(config or {}))
    if compose:
        (task.path / "environment/docker-compose.yaml").write_text(
            "services: {main: {image: public}, db: {image: db}}"
        )
        cfg.compose_image_configs = tmp_path / "images.json"
        cfg.compose_image_configs.write_text("{}")
        monkeypatch.setattr(
            module,
            "resolve_compose",
            lambda *a: {"services": {"main": {"image": "main", "shm_size": 64}, "db": {"image": "db"}}},
        )
    box = MagicMock()
    box._handle = SimpleNamespace(sandbox_id="owned-box")
    box.start = AsyncMock()
    box.stop = AsyncMock()

    async def execute(command, **kwargs):
        stdout = "/app\n"
        if command == "id -u":
            stdout = "0\n"
        elif "id -u && id -g" in command:
            user = kwargs.get("user")
            uid = user if isinstance(user, int) else 0 if user in (None, "root") else 1000
            stdout = f"{uid}\n{uid}\n" + (f"{uid}\n" if "id -u --" in command else "")
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    box.exec = AsyncMock(wraps=execute)
    box.serialize = AsyncMock(return_value={"sandbox_id": "owned-box", "credentials": "must not be copied"})
    create = MagicMock(return_value=box)
    monkeypatch.setattr(module, "AsyncSandbox", create)
    group = SimpleNamespace(services={"main": box, "db": box}, start=AsyncMock(), stop=AsyncMock(), project="project")
    compose_create = MagicMock(return_value=group)
    monkeypatch.setattr(module, "AsyncSandboxCompose", compose_create)
    env = Environment(task, cfg, "session", tmp_path / "result", verifier=verifier)
    return env, box, create, compose_create


@pytest.mark.parametrize("agent_gpus,verifier_gpus,pool", [(0, 0, "cpu"), (1, 0, "gpu"), (0, 1, "gpu"), (1, 1, "gpu")])
def test_specs_keep_resource_units_and_task_deployment(tmp_path, monkeypatch, agent_gpus, verifier_gpus, pool):
    # The other deployment's credentials must not be needed by any task role.
    for endpoint in ["CPU", "GPU"]:
        monkeypatch.delenv("OPENSANDBOX_DOMAIN_" + endpoint, raising=False)
        monkeypatch.delenv("OPENSANDBOX_API_KEY_" + endpoint, raising=False)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN_" + pool.upper(), pool + ".invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY_" + pool.upper(), "credential")
    args = {
        "sandbox_split_endpoints": True,
        "sandbox_request_gpu_type": False,
        "sandbox_provider_options": {"resource_requests": "limits"},
        "sandbox_env_by_task": {"test": {"OVERRIDE": "task"}},
        "sandbox_env": {"OVERRIDE": "global"},
    }
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        config=args,
        task_config={
            "environment": {"gpus": agent_gpus},
            "verifier": {"environment": {"docker_image": "public/verifier", "gpus": verifier_gpus}},
        },
    )
    verifier = Environment(env.task, env.config, "verify", tmp_path, verifier=True)
    agent_spec, verifier_spec = env.build_spec(), verifier.build_spec()
    assert env.pool == verifier.pool == pool
    assert agent_spec.resources.cpu == 2 and agent_spec.resources.memory_mib == 4096
    assert agent_spec.resources.disk_gib == 15
    assert agent_spec.resources.gpu == (agent_gpus or None)
    assert verifier_spec.resources.gpu == (verifier_gpus or None) and verifier_spec.resources.gpu_type is None
    assert agent_spec.env == {"TASK": "value", "OVERRIDE": "global"}
    assert agent_spec.provider_options == {"resource_requests": "limits"}
    assert "credential" not in str(agent_spec)
    for role in (env, verifier):
        connection = role.provider_config["opensandbox"]["connection"]
        assert connection["domain"] == pool + ".invalid"
        assert connection["api_key"] == "credential"
        assert role.build_spec().metadata["nemo-gym.nvidia.com/resource-pool"] == pool


async def test_cpu_compose_services_follow_gpu_verifier_deployment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_DOMAIN_GPU", "gpu.invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY_GPU", "gpu-credential")
    env, _, _, create = make_environment(tmp_path, monkeypatch, compose=True, config={"sandbox_split_endpoints": True})
    await env.start()
    connection = create.call_args.args[0]["opensandbox"]["connection"]
    assert connection["domain"] == "gpu.invalid"
    assert connection["api_key"] == "gpu-credential"
    for spec in create.call_args.kwargs["service_specs"].values():
        assert spec.metadata["nemo-gym.nvidia.com/resource-pool"] == "gpu"
        assert spec.resources.gpu is None


async def test_single_start_workdir_env_user_quiescence_cleanup(tmp_path, monkeypatch):
    env, box, create, _ = make_environment(tmp_path, monkeypatch)
    await env.start()
    box.start.assert_awaited_once()
    assert create.call_args.args[1].image == "public/agent"
    assert create.call_args.args[1].entrypoint is None
    assert await env.agent_workdir() == "/app"
    box.serialize.assert_not_awaited()
    assert box.exec.await_args.kwargs["user"] == "task-user"
    await env.exec("echo test", env={"TASK": "changed"}, user="another", timeout_sec=12)
    assert box.exec.await_args.kwargs["env"] == {"TASK": "changed"}
    assert box.exec.await_args.kwargs["user"] == "another"
    assert box.exec.await_args.kwargs["timeout_s"] == 12
    await env.quiesce_agent("session")
    assert "/tmp/session.pids" in box.exec.await_args.args[0]
    await env.stop()
    await env.stop()
    box.stop.assert_awaited_once()
    assert env.closed and env.resources[0]["sandbox_id"] == "owned-box"


@pytest.mark.parametrize("verifier", [False, True])
@pytest.mark.parametrize(
    "startup,expected",
    [
        (
            {"Entrypoint": ["/start", "--mode"], "Cmd": ["sh", "-c", "sleep infinity"]},
            ["/start", "--mode", "sh", "-c", "sleep infinity"],
        ),
        ({"Entrypoint": ["/start"], "Cmd": None}, ["/start", "sh", "-c", "sleep infinity"]),
        ({"Entrypoint": None, "Cmd": ["python3"]}, ["sh", "-c", "sleep infinity"]),
        ({"Entrypoint": None, "Cmd": ["/bin/bash"]}, ["sh", "-c", "sleep infinity"]),
        ({"Entrypoint": [], "Cmd": []}, ["sh", "-c", "sleep infinity"]),
        ({}, ["sh", "-c", "sleep infinity"]),
    ],
)
@pytest.mark.parametrize("catalog", ["compose_image_configs", "single_container_image_configs"])
async def test_single_container_uses_role_image_startup(tmp_path, monkeypatch, verifier, startup, expected, catalog):
    images = tmp_path / "single-images.json"
    images.write_text(
        json.dumps(
            {
                "mirror/agent": {
                    "image": "mirror/agent",
                    "os": "linux",
                    "architecture": "amd64",
                    "config": startup if not verifier else {"Entrypoint": ["wrong-role"]},
                },
                "mirror/verifier": {
                    "image": "mirror/verifier",
                    "os": "linux",
                    "architecture": "amd64",
                    "config": startup if verifier else {"Entrypoint": ["wrong-role"]},
                },
            }
        )
    )
    env, _, create, compose_create = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        config={catalog: images, "image_rewrites": [{"from": "public/", "to": "mirror/"}]},
    )
    await env.start()
    spec = create.call_args.args[1]
    assert spec.image == "mirror/" + ("verifier" if verifier else "agent")
    assert spec.entrypoint == expected
    assert not env.uses_compose
    compose_create.assert_not_called()


@pytest.mark.parametrize(
    "change,error",
    [
        ({"image": "different"}, "does not match"),
        ({"os": "windows"}, "Linux/amd64"),
        ({"architecture": "arm64"}, "Linux/amd64"),
        ({"config": None}, "requires an image config"),
        ({"config": {"Entrypoint": "/start --arg"}}, "Entrypoint.*list of strings"),
        ({"config": {"Cmd": ["sleep", 1]}}, "Cmd.*list of strings"),
    ],
)
@pytest.mark.parametrize("catalog", ["compose_image_configs", "single_container_image_configs"])
def test_single_container_rejects_incompatible_startup_metadata(tmp_path, monkeypatch, change, error, catalog):
    images = tmp_path / "single-images.json"
    record = {"image": "public/agent", "os": "linux", "architecture": "amd64", "config": {}}
    images.write_text(json.dumps({"public/agent": record | change}))
    env, *_ = make_environment(tmp_path, monkeypatch, config={catalog: images})
    with pytest.raises(ValueError, match=error):
        env.build_spec()


def test_single_container_configured_metadata_must_include_image(tmp_path, monkeypatch):
    images = tmp_path / "single-images.json"
    images.write_text("{}")
    env, *_ = make_environment(tmp_path, monkeypatch, config={"single_container_image_configs": images})
    with pytest.raises(ValueError, match="No recorded OCI startup metadata"):
        env.build_spec()


def test_compose_only_catalog_retains_legacy_standalone_keepalive(tmp_path, monkeypatch, caplog):
    images = tmp_path / "compose-images.json"
    images.write_text("{}")
    env, *_ = make_environment(tmp_path, monkeypatch, config={"compose_image_configs": images})
    assert env.build_spec().entrypoint is None
    assert "No recorded OCI startup metadata" in caplog.text


@pytest.mark.parametrize("catalog", ["compose_image_configs", "single_container_image_configs"])
@pytest.mark.parametrize("record", [None, [], "not a record"])
def test_invalid_catalog_or_present_record_does_not_fall_back(tmp_path, monkeypatch, catalog, record):
    images = tmp_path / "images.json"
    env, *_ = make_environment(tmp_path, monkeypatch, config={catalog: images})
    images.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="image mapping"):
        env.build_spec()
    images.write_text(json.dumps({"public/agent": record}))
    with pytest.raises(ValueError, match="No recorded OCI startup metadata"):
        env.build_spec()


@pytest.mark.parametrize("same_digest", [False, True])
def test_standalone_accepts_compose_catalog_digest_spelling_only_for_same_content(tmp_path, monkeypatch, same_digest):
    digest = "sha256:" + "a" * 64
    image = "registry/repo:tag@" + digest
    recorded_image = "registry/repo@" + (digest if same_digest else "sha256:" + "b" * 64)
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                image: {
                    "image": recorded_image,
                    "os": "linux",
                    "architecture": "amd64",
                    "config": {"Entrypoint": ["/start"]},
                }
            }
        )
    )
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        config={"compose_image_configs": images},
        task_config={"environment": {"docker_image": image}},
    )
    if same_digest:
        assert env.build_spec().entrypoint == ["/start", "sh", "-c", "sleep infinity"]
    else:
        with pytest.raises(ValueError, match="does not match"):
            env.build_spec()


def test_existing_standalone_catalog_override_is_still_strict(tmp_path, monkeypatch):
    images = tmp_path / "override.json"
    images.write_text("{}")
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        config={
            "single_container_image_configs": images,
            "compose_image_configs": tmp_path / "unused-catalog.json",
        },
    )
    with pytest.raises(ValueError, match="No recorded OCI startup metadata"):
        env.build_spec()


async def test_compose_specs_startup_metadata_sidecar_operations(tmp_path, monkeypatch):
    env, box, _, create = make_environment(tmp_path, monkeypatch, compose=True)
    env.config.single_container_image_configs = tmp_path / "unused-missing.json"
    await env.start()
    kwargs = create.call_args.kwargs
    assert kwargs["service_specs"]["main"].resources.cpu == 2
    assert kwargs["service_specs"]["db"].resources.cpu is None
    assert kwargs["service_specs"]["main"].entrypoint is None
    document = yaml.safe_load(create.call_args.args[1].read_text())
    assert document["services"]["main"]["labels"] == {"nemo.nvidia.com/shm": "64"}
    await env.exec("echo sidecar", service="db")
    assert box.exec.await_args.args[0].startswith("sh -c ")
    assert box.exec.await_args.kwargs["env"] is None
    await env.stop_main()
    await env.stop()
    assert env.closed and env.resources[-1]["compose_project"] == "project"


@pytest.mark.parametrize(
    "task_name,service,user,extension",
    [
        (
            "medical-claims-processing",
            "playwright-mcp",
            "pwuser",
            {"hosts": [], "resolve_environment": ["BROWSER_URL"]},
        ),
        ("payments-pipeline-fix", "kafka", "appuser", {"hosts": []}),
    ],
)
@pytest.mark.parametrize("scope", ["agent", "verifier", "other-task"])
async def test_nonroot_compose_adaptations_are_scoped_to_agent_tasks(
    tmp_path, monkeypatch, task_name, service, user, extension, scope
):
    env, _, _, create = make_environment(tmp_path, monkeypatch, compose=True)
    env.task.name = "terminal-bench/" + (task_name if scope != "other-task" else "unrelated")
    if scope == "verifier":
        env.log_role = "verifier"
        env.environment_dir = env.task.path / "tests"
    document = {
        "services": {
            "main": {"depends_on": {service: {"condition": "service_healthy"}}},
            service: {
                "image": "nonroot",
                "environment": {"BROWSER_URL": "http://workspace:18073"},
                "healthcheck": {"test": ["CMD", "true"]},
            },
            "workspace": {"image": "public/agent"},
        }
    }
    source = env.environment_dir / "docker-compose.yaml"
    source.write_text(yaml.safe_dump(document))
    original = source.read_bytes()
    images = {
        image: {
            "image": image,
            "os": "linux",
            "architecture": "amd64",
            "config": {"User": image_user, "Cmd": ["sleep", "infinity"]},
        }
        for image, image_user in [("public/agent", "root"), ("nonroot", user)]
    }
    env.config.compose_image_configs.write_text(json.dumps(images))
    monkeypatch.setattr(module, "resolve_compose", resolve_compose)
    expected = resolve_compose(deepcopy(document), "public/agent", images)
    if scope == "agent":
        expected["services"][service]["x-sandbox"] = extension
        expected["services"][service].pop("user")

    await env.start()

    generated = yaml.safe_load(create.call_args.args[1].read_text())
    assert generated == expected
    assert generated["services"][service].get("user") == (None if scope == "agent" else user)
    assert source.read_bytes() == original


async def test_environment_upload_without_build_spec(tmp_path, monkeypatch):
    env, box, _, _ = make_environment(tmp_path, monkeypatch)
    (env.environment_dir / "Dockerfile").unlink()
    (env.environment_dir / "asset.txt").write_text("asset")
    from resources_servers.terminal_bench_4 import transfers

    upload = AsyncMock()
    monkeypatch.setattr(transfers, "upload_dir", upload)
    await env.start()
    upload.assert_awaited_once_with(box, env.environment_dir, "/app")


@pytest.mark.parametrize("failure", ["logs", "workdir", "quiesce", "delete", "unavailable"])
async def test_failures_are_visible_and_preserve_cleanup_identities(tmp_path, monkeypatch, failure):
    env, box, _, _ = make_environment(tmp_path, monkeypatch)
    if failure == "unavailable":
        with pytest.raises(RuntimeError):
            env.sandbox()
        with pytest.raises(ValueError):
            env.sandbox("missing")
        return
    if failure == "logs":
        box.exec.side_effect = [
            SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
            SimpleNamespace(return_code=1, stdout="", stderr="log setup failed"),
        ]
    if failure == "logs":
        with pytest.raises(RuntimeError, match="log directories"):
            await env.start()
    else:
        await env.start()
    if failure == "workdir":
        box.exec.return_value = SimpleNamespace(return_code=1, stdout="", stderr="workdir failed")
        with pytest.raises(RuntimeError):
            await env.agent_workdir()
    if failure == "quiesce":
        box.exec.return_value = SimpleNamespace(return_code=1, stdout="", stderr="quiesce failed")
        with pytest.raises(RuntimeError):
            await env.quiesce_agent("session")
    if failure == "delete":
        box.stop.side_effect = RuntimeError("delete failed")
        with pytest.raises(RuntimeError):
            await env.stop()
        assert env.cleanup_errors[0]["resources"][0]["sandbox_id"] == "owned-box"
    else:
        await env.stop()


async def test_readiness_success_and_failure(tmp_path, monkeypatch):
    env, box, _, _ = make_environment(
        tmp_path,
        monkeypatch,
        task_config={"environment": {"healthcheck": {"command": "ready", "retries": 2, "interval_sec": 0}}},
    )
    await env.start()
    await env.healthcheck()
    box.exec.return_value = SimpleNamespace(return_code=1, stdout="", stderr="healthcheck failed")
    with pytest.raises(HealthcheckError):
        await env.healthcheck()
    env.settings.healthcheck.start_period_sec = 0.01
    env.settings.healthcheck.start_interval_sec = 0.01
    with pytest.raises(HealthcheckError):
        await env.healthcheck()


def test_offline_policy_and_unsupported_config(tmp_path, monkeypatch):
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=True,
        task_config={"verifier": {"environment": {"docker_image": "offline", "allow_internet": False}}},
    )
    assert env.build_spec().provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
    with pytest.raises(ValueError, match="Split endpoints"):
        Environment(
            env.task, environment_config(sandbox_provider={"local": {}}, sandbox_split_endpoints=True), "id", tmp_path
        )
    with pytest.raises(ValueError, match="Missing environment"):
        make_environment(tmp_path, monkeypatch, config={"sandbox_split_endpoints": True})
    with pytest.raises(ValueError, match="Offline"):
        make_environment(tmp_path, monkeypatch, compose=True, task_config={"environment": {"allow_internet": False}})
