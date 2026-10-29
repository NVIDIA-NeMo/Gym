# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from resources_servers.terminal_bench_4 import environment as module
from resources_servers.terminal_bench_4.compose_config import merge_patch, resolve_compose
from resources_servers.terminal_bench_4.environment import (
    Environment,
    EnvironmentConfig,
    HealthcheckError,
    TaskImages,
)
from resources_servers.terminal_bench_4.task import TaskSettings


REPOSITORY = Path(__file__).resolve().parents[3]


def profile_compose_overrides(benchmark):
    """The ``compose_service_overrides`` that a benchmark profile ships."""
    resources = yaml.safe_load((REPOSITORY / "benchmarks" / benchmark / "resources.yaml").read_text())
    return resources[benchmark]["resources_servers"]["terminal_bench_4"]["environment"]["compose_service_overrides"]


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
            lambda *a, **k: {"services": {"main": {"image": "main", "shm_size": 64}, "db": {"image": "db"}}},
        )
    box = MagicMock()
    box._handle = SimpleNamespace(sandbox_id="owned-box")
    box.start = AsyncMock()
    box.stop = AsyncMock()
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="/app\n", stderr=""))
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


@pytest.mark.parametrize("from_agent", [False, True])
def test_resource_less_verifiers_can_inherit_agent_resources(tmp_path, monkeypatch, from_agent):
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        config={"verifier_resources_from_agent": from_agent},
        task_config={"verifier": {"environment": {"docker_image": "public/verifier", "cpus": 8}}},
    )
    resources = Environment(env.task, env.config, "verify", tmp_path, verifier=True).build_spec().resources
    # A value the verifier declares always wins; only unset ones come from the agent.
    assert resources.cpu == 8
    assert resources.memory_mib == (4096 if from_agent else None)
    assert env.task.config.verifier.environment.memory_mb is None


def test_verifier_timeout_overrides_apply_by_short_task_name(tmp_path, monkeypatch):
    env, *_ = make_environment(tmp_path, monkeypatch, config={"verifier_timeout_overrides": {"test": 1800}})
    assert env.config.verifier_timeout_sec(env.task) == 1800
    assert environment_config().verifier_timeout_sec(env.task) == env.task.config.verifier.timeout_sec
    with pytest.raises(ValueError):
        environment_config(verifier_timeout_overrides={"test": 0})


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


async def test_compose_specs_startup_metadata_sidecar_operations(tmp_path, monkeypatch):
    env, box, _, create = make_environment(tmp_path, monkeypatch, compose=True)
    await env.start()
    kwargs = create.call_args.kwargs
    assert kwargs["service_specs"]["main"].resources.cpu == 2
    assert kwargs["service_specs"]["db"].resources.cpu is None
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
    overrides = profile_compose_overrides("terminal_bench_4")
    env, _, _, create = make_environment(
        tmp_path, monkeypatch, compose=True, config={"compose_service_overrides": overrides}
    )
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


def test_compose_service_overrides_merge_mappings_and_remove_nulls():
    service = {"user": "app", "environment": {"A": "1"}, "x-sandbox": {"hosts": ["db"]}, "expose": ["80"]}
    patch = {"user": None, "environment": {"B": "2"}, "x-sandbox": {"hosts": []}, "expose": ["90"], "absent": None}
    assert merge_patch(service, patch) == {
        "environment": {"A": "1", "B": "2"},
        "x-sandbox": {"hosts": []},
        "expose": ["90"],
    }
    assert service == {"user": "app", "environment": {"A": "1"}, "x-sandbox": {"hosts": ["db"]}, "expose": ["80"]}


async def test_compose_service_overrides_must_name_task_services(tmp_path, monkeypatch):
    overrides = {"test": {"db": {"user": None}, "cache": {"user": None}}}
    env, *_ = make_environment(tmp_path, monkeypatch, compose=True, config={"compose_service_overrides": overrides})
    with pytest.raises(ValueError, match=r"names services that terminal-bench/test lacks: \['cache'\]"):
        await env.start()


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
        box.exec.return_value.return_code = 1
    if failure == "logs":
        with pytest.raises(RuntimeError, match="log directories"):
            await env.start()
    else:
        await env.start()
    if failure == "workdir":
        box.exec.return_value.return_code = 1
        with pytest.raises(RuntimeError):
            await env.agent_workdir()
    if failure == "quiesce":
        box.exec.return_value.return_code = 1
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
    box.exec.return_value.return_code = 1
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


def dockerfile_only_task(tmp_path, *, verifier_environment=True):
    """A Terminal-Bench-Science-shaped task: Dockerfiles only, separate offline verifier."""
    verifier = {"environment_mode": "separate"}
    if verifier_environment:
        verifier["environment"] = {"network_mode": "no-network"}
    task = SimpleNamespace(
        config=TaskSettings.model_validate(
            {"environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}, "verifier": verifier}
        ),
        name="terminal-bench-science/lake-ice",
        ref="sha256:" + "ab" * 32,
        path=tmp_path / "package",
    )
    for role in ("environment", "tests"):
        (task.path / role).mkdir(parents=True, exist_ok=True)
        (task.path / role / "Dockerfile").write_text("FROM public")
    return task


def test_task_images_fill_dockerfile_only_roles(tmp_path):
    task = dockerfile_only_task(tmp_path)
    config = environment_config(
        task_images={
            "agent": "registry.invalid/tbsci/{task}:env-{digest}",
            "verifier": "registry.invalid/{org}/{task}:tests",
        },
        image_rewrites=[{"from": "registry.invalid/", "to": "mirror.invalid/"}],
    )
    agent = Environment(task, config, "session", tmp_path / "result").build_spec()
    verifier = Environment(task, config, "session__verifier", tmp_path / "result", verifier=True).build_spec()
    assert agent.image == "mirror.invalid/tbsci/lake-ice:env-" + "ab" * 32
    assert verifier.image == "mirror.invalid/terminal-bench-science/lake-ice:tests"
    assert "network_policy" not in agent.provider_options
    assert verifier.provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
    assert (agent.resources.cpu, agent.resources.memory_mib, agent.resources.disk_gib) == (2, 4096, 10)
    # As in Harbor, an explicit [verifier.environment] does not inherit the task's resources.
    assert (verifier.resources.cpu, verifier.resources.memory_mib, verifier.resources.disk_gib) == (None, None, None)


def test_task_image_precedence_and_missing_images(tmp_path):
    task = dockerfile_only_task(tmp_path)
    images = {
        "agent": "tpl/{task}",
        "verifier": "tpl/{task}-tests",
        "by_task": {"lake-ice": {"verifier": "pinned@sha256:1"}},
    }
    config = environment_config(task_images=images)
    assert Environment(task, config, "s", tmp_path).image == "tpl/lake-ice"
    assert Environment(task, config, "s", tmp_path, verifier=True).image == "pinned@sha256:1"
    task.config.environment.docker_image = "package/agent"
    assert Environment(task, config, "s", tmp_path).image == "package/agent"
    # Without [verifier.environment], separate verification reuses the task image.
    shared_image = dockerfile_only_task(tmp_path / "copy", verifier_environment=False)
    assert Environment(shared_image, config, "s", tmp_path, verifier=True).image == "tpl/lake-ice"
    with pytest.raises(ValueError, match="task_images.verifier"):
        Environment(task, environment_config(task_images={"agent": "tpl/{task}"}), "s", tmp_path, verifier=True)
    with pytest.raises(ValueError, match="task_images.agent"):
        Environment(dockerfile_only_task(tmp_path / "bare"), environment_config(), "s", tmp_path)


def test_local_docker_verifier_runs_without_network(tmp_path):
    task = dockerfile_only_task(tmp_path)
    images = {"agent": "tpl/{task}", "verifier": "tpl/{task}-tests"}
    config = environment_config(
        sandbox_provider={"docker": {}},
        sandbox_provider_options={"run_args": ["--pids-limit", "512"]},
        task_images=images,
    )
    verifier = Environment(task, config, "s", tmp_path, verifier=True).build_spec()
    agent = Environment(task, config, "s", tmp_path).build_spec()
    assert verifier.provider_options == {"run_args": ["--pids-limit", "512", "--network", "none"]}
    assert agent.provider_options == {"run_args": ["--pids-limit", "512"]}
    # The Docker provider also accepts one run argument as a string.
    config.sandbox_provider_options = {"run_args": "--read-only"}
    verifier = Environment(task, config, "s", tmp_path, verifier=True).build_spec()
    assert verifier.provider_options == {"run_args": ["--read-only", "--network", "none"]}
    with pytest.raises(ValueError, match="Offline verification"):
        Environment(
            task, environment_config(sandbox_provider={"local": {}}, task_images=images), "s", tmp_path, verifier=True
        )


@pytest.mark.parametrize(
    "template",
    [
        "registry/{name}",
        "registry/{0}",
        "registry/{}",
        "registry/{task",
        "registry/{task.upper}",
        "registry/{digest:d}",
    ],
)
def test_task_image_templates_are_validated(template):
    with pytest.raises(ValueError, match="Invalid task image template"):
        TaskImages(agent=template)


def test_task_image_templates_accept_format_specs():
    images = TaskImages(verifier="registry/{org}/{task}:tests-{digest:.12}")
    task = SimpleNamespace(name="terminal-bench-science/lake-ice", ref="sha256:" + "ab" * 32)
    assert images.resolve(task, "verifier") == "registry/terminal-bench-science/lake-ice:tests-abababababab"
    assert images.resolve(task, "agent") is None


def test_task_images_name_compose_services_built_from_dockerfiles():
    task = SimpleNamespace(name="terminal-bench-science/lake-ice", ref="sha256:" + "ab" * 32)
    images = TaskImages(
        service="registry/{task}-{service}:{digest:.12}",
        by_task={"lake-ice": {"agent": "pinned/agent", "services": {"sim": "pinned/sim@sha256:1"}}},
    )
    assert images.resolve(task, "service", "sim") == "pinned/sim@sha256:1"
    assert images.resolve(task, "service", "broker") == "registry/lake-ice-broker:abababababab"
    assert images.resolve(task, "agent") == "pinned/agent"
    assert TaskImages().resolve(task, "service", "sim") is None
    # {service} names a Compose service, so only the service template accepts it.
    with pytest.raises(ValueError, match="Invalid task image template"):
        TaskImages(agent="registry/{task}-{service}")


def dockerfile_compose_environment(tmp_path, monkeypatch, name, compose, records, **config):
    task = dockerfile_only_task(tmp_path)
    task.name = "terminal-bench-science/" + name
    (task.path / "environment/docker-compose.yaml").write_text(yaml.safe_dump(compose))
    (tmp_path / "images.json").write_text(json.dumps(records))
    box = MagicMock()
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))
    compose_create = MagicMock(
        return_value=SimpleNamespace(services={"main": box}, start=AsyncMock(), stop=AsyncMock(), project="p")
    )
    monkeypatch.setattr(module, "AsyncSandboxCompose", compose_create)
    cfg = environment_config(compose_image_configs=tmp_path / "images.json", **config)
    return Environment(task, cfg, "session", tmp_path / "result"), compose_create


def build_record(**config):
    return {"os": "linux", "architecture": "amd64", "config": config}


async def test_compose_services_built_from_dockerfiles_run_configured_images(tmp_path, monkeypatch):
    compose = {
        "services": {
            "main": {"depends_on": {"evaluator": {"condition": "service_healthy"}}},
            "evaluator": {
                "build": {"context": "${CONTEXT_DIR}/evaluator"},
                "cpus": "${CPUS:-4}",
                "mem_limit": "${MEMORY:-4096M}",
                "read_only": True,
                "networks": ["task"],
                "healthcheck": {"test": ["CMD", "true"]},
            },
        },
        "networks": {"task": {"internal": True}},
    }
    records = {
        "lake-ice/main": build_record(Cmd=["python3"], WorkingDir="/app"),
        "lake-ice/evaluator": build_record(Cmd=["python3", "/srv/server.py"], WorkingDir="/srv"),
    }
    images = {"agent": "registry/{task}:env", "by_task": {"lake-ice": {"services": {"evaluator": "registry/eval"}}}}
    env, compose_create = dockerfile_compose_environment(
        tmp_path,
        monkeypatch,
        "lake-ice",
        compose,
        records,
        task_images=images,
        compose_ignored_settings=["read_only", "networks"],
    )
    await env.start()
    document = yaml.safe_load(compose_create.call_args.args[1].read_text())
    main, evaluator = document["services"]["main"], document["services"]["evaluator"]
    assert "networks" not in document
    assert (main["image"], main["working_dir"]) == ("registry/lake-ice:env", "/app")
    assert evaluator["image"] == "registry/eval"
    assert (evaluator["command"], evaluator["working_dir"]) == (["python3", "/srv/server.py"], "/srv")
    # Harbor's CPUS and MEMORY come from the task's environment resources.
    assert (evaluator["cpus"], evaluator["mem_limit"]) == ("2", 4096 * 1024**2)
    assert "read_only" not in evaluator and "networks" not in evaluator
    with pytest.raises(ValueError, match="no configured prebuilt image"):
        await dockerfile_compose_environment(
            tmp_path / "unconfigured", monkeypatch, "lake-ice", compose, records, task_images={"agent": "a"}
        )[0].start()


@pytest.mark.parametrize("name", ["protein-active-learning", "other-task"])
async def test_protein_submission_resolves_its_assay_url_without_hosts(tmp_path, monkeypatch, name):
    compose = {
        "services": {
            "main": {"depends_on": {"submission": {"condition": "service_healthy"}}},
            "assay": {"build": {"context": "assay"}, "healthcheck": {"test": ["CMD", "true"]}},
            "submission": {
                "build": {"context": "submission"},
                "depends_on": {"assay": {"condition": "service_healthy"}},
                "healthcheck": {"test": ["CMD", "true"]},
            },
        }
    }
    records = {
        f"{name}/main": build_record(Cmd=["python3"], WorkingDir="/app"),
        f"{name}/assay": build_record(Cmd=["python3", "assay.py"], User="root"),
        f"{name}/submission": build_record(Cmd=["python3", "submission.py"], User="submission"),
    }
    images = {"agent": "registry/{task}", "service": "registry/{task}-{service}"}
    env, compose_create = dockerfile_compose_environment(
        tmp_path,
        monkeypatch,
        name,
        compose,
        records,
        task_images=images,
        compose_service_overrides=profile_compose_overrides("terminal_bench_science"),
    )
    await env.start()
    services = yaml.safe_load(compose_create.call_args.args[1].read_text())["services"]
    assert services["assay"]["user"] == "root"
    if name == "other-task":
        assert services["submission"]["user"] == "submission" and "x-sandbox" not in services["submission"]
        return
    # The non-root image cannot switch users or edit /etc/hosts.
    assert "user" not in services["submission"]
    assert services["submission"]["environment"] == {"ASSAY_STATUS": "http://assay:8080/status"}
    assert services["submission"]["x-sandbox"] == {"hosts": [], "resolve_environment": ["ASSAY_STATUS"]}
