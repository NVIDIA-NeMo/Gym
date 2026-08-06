# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the configurable Legal Agent Bench Gym-agent runner."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.legal_agent_bench_agent import app


def _runtime_archive_worker(deps_dir: str) -> tuple[str, int]:
    archive = app.ensure_runtime_archive(Path(deps_dir))
    return str(archive), archive.stat().st_size


def _config(**overrides) -> app.LegalAgentBenchAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 10000,
        "name": "lab_test_agent",
        "entrypoint": "app.py",
        "resources_server": ResourcesServerRef(name="lab", type="resources_servers"),
        "model_server": ModelServerRef(name="policy_model", type="responses_api_models"),
        "agent_server_module": "responses_api_agents.hermes_agent.app",
        "agent_server_class": "HermesAgent",
        "agent_config_class": "HermesAgentConfig",
    }
    values.update(overrides)
    return app.LegalAgentBenchAgentConfig(**values)


def _task_tree(tmp_path: Path, task_name: str = "area__task") -> tuple[Path, Path]:
    root = tmp_path / "tasks"
    task = root / task_name
    for directory in ("documents", "environment/harness", "tests"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Do the task")
    (task / "task.json").write_text(
        json.dumps({"title": "Legal task", "instructions": "Write a memo.", "criteria": [{"id": "1"}]})
    )
    (task / "task.toml").write_text(
        """
version = "1.0"
[agent]
timeout_sec = 30
[verifier]
timeout_sec = 60
[verifier.env]
LAB_JUDGE_API_KEY = "secret"  # pragma: allowlist secret
LAB_JUDGE_MODEL = "judge"
[environment]
cpus = 2
memory_mb = 4096
storage_mb = 10240
"""
    )
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /workspace/output\n")
    (task / "environment" / "harness" / "runner.py").write_text("print('ok')\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n")
    return root, task


def _skills(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    for name in app.REQUIRED_SKILLS:
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(f"# {name}\nUse {name}.")
    return root


def _successful_response(text: str = "Done") -> app.NeMoGymResponse:
    return app.NeMoGymResponse.model_validate(
        {
            "id": "resp-success",
            "created_at": 1,
            "model": "policy",
            "object": "response",
            "output": [
                {
                    "id": "msg-success",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        }
    )


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("responses_api_agents.legal_agent_bench_native_agent.app", "legal_agent_bench_native_agent"),
        ("responses_api_agents.hermes_agent.app", "hermes_agent"),
        ("responses_api_agents.claude_code_agent.app", "claude_code_agent"),
        ("responses_api_agents.codex_agent.app", "codex_agent"),
    ],
)
def test_agent_key(module, expected) -> None:
    assert app.agent_key(module) == expected


@pytest.mark.parametrize("module", ["hermes_agent", "responses_api_agents.bad-name.app", "x.y.z"])
def test_agent_key_rejects_invalid_modules(module) -> None:
    with pytest.raises(ValueError):
        app.agent_key(module)


def test_all_supported_agents_have_dependency_scripts() -> None:
    for module in (
        "responses_api_agents.legal_agent_bench_native_agent.app",
        "responses_api_agents.hermes_agent.app",
        "responses_api_agents.claude_code_agent.app",
        "responses_api_agents.codex_agent.app",
    ):
        assert app.resolve_agent_setup_script(module).is_file()


def test_dependency_runtime_cache_is_harness_and_recipe_specific(monkeypatch, tmp_path) -> None:
    package_dir = tmp_path / "legal_agent_bench_agent"
    package_dir.mkdir()
    portable = package_dir / "_portable_python.sh"
    portable.write_text("portable-v1")
    script = tmp_path / "hermes_agent_deps.sh"
    script.write_text("hermes-v1")
    agent_dir = tmp_path / "responses_api_agents" / "hermes_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "requirements.txt").write_text("nemo-gym\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'nemo-gym'\n")
    (tmp_path / "README.md").write_text("# Gym\n")
    (tmp_path / "env.yaml").write_text("judge_key: must-not-be-mounted\n")
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym" / "runtime.py").write_text("VERSION = 1\n")
    calls = []

    monkeypatch.setattr(app, "PACKAGE_DIR", package_dir)
    monkeypatch.setattr(app, "PARENT_DIR", tmp_path)
    monkeypatch.setattr(app, "PORTABLE_PYTHON_SH", portable)
    monkeypatch.setattr(app, "resolve_agent_setup_script", lambda module: script)

    def fake_run(args, **kwargs):
        if len(args) > 1 and args[1] == "run":
            mounted_bundle = next(Path(arg.split(":", 1)[0]) for arg in args if ":/nemo_gym_mount:ro" in arg)
            assert mounted_bundle != tmp_path
            assert not (mounted_bundle / "env.yaml").exists()
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="sha256:image:linux/arm64\n")

    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(app.subprocess, "run", fake_run)

    first = app.ensure_agent_runtime(
        "responses_api_agents.hermes_agent.app",
        agent_kwargs={},
        image="lab:image",
        docker_network="host",
        timeout_seconds=60,
    )
    second = app.ensure_agent_runtime(
        "responses_api_agents.hermes_agent.app",
        agent_kwargs={},
        image="lab:image",
        docker_network="host",
        timeout_seconds=60,
    )
    script.write_text("hermes-v2")
    third = app.ensure_agent_runtime(
        "responses_api_agents.hermes_agent.app",
        agent_kwargs={},
        image="lab:image",
        docker_network="host",
        timeout_seconds=60,
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'nemo-gym-updated'\n")
    fourth = app.ensure_agent_runtime(
        "responses_api_agents.hermes_agent.app",
        agent_kwargs={},
        image="lab:image",
        docker_network="host",
        timeout_seconds=60,
    )
    (tmp_path / "nemo_gym" / "runtime.py").write_text("VERSION = 2\n")
    fifth = app.ensure_agent_runtime(
        "responses_api_agents.hermes_agent.app",
        agent_kwargs={},
        image="lab:image",
        docker_network="host",
        timeout_seconds=60,
    )

    assert first == second
    assert len({first, third, fourth, fifth}) == 4
    assert all(path.parent == package_dir / ".deps" / "hermes_agent" for path in (first, third, fourth, fifth))
    assert all(path.is_dir() for path in (first, third, fourth, fifth))
    docker_runs = [call for call in calls if call[0][1] == "run"]
    assert len(docker_runs) == 4
    mounted_sources = [arg.split(":", 1)[0] for arg in docker_runs[0][0] if ":/nemo_gym_mount:ro" in arg]
    assert mounted_sources and mounted_sources[0] != str(tmp_path)
    assert "env.yaml" not in " ".join(docker_runs[0][0])
    assert "--network" in docker_runs[0][0]


@pytest.mark.parametrize(
    ("module", "kwargs", "expected"),
    [
        (
            "responses_api_agents.claude_code_agent.app",
            {"claude_code_version": "2.1.211"},
            {"CLAUDE_SPEC": "@anthropic-ai/claude-code@2.1.211"},
        ),
        (
            "responses_api_agents.codex_agent.app",
            {"codex_version": "0.144.4"},
            {"CODEX_SPEC": "@openai/codex@0.144.4"},
        ),
        ("responses_api_agents.hermes_agent.app", {}, {}),
    ],
)
def test_agent_runtime_env_uses_configured_harness_pin(module, kwargs, expected) -> None:
    assert app.agent_runtime_env(module, kwargs) == expected


@pytest.mark.parametrize(
    ("module", "field"),
    [
        ("responses_api_agents.claude_code_agent.app", "claude_code_version"),
        ("responses_api_agents.codex_agent.app", "codex_version"),
    ],
)
def test_agent_runtime_env_rejects_missing_harness_pin(module, field) -> None:
    with pytest.raises(app.LegalAgentBenchConfigurationError, match=field):
        app.agent_runtime_env(module, {})


@pytest.mark.parametrize("version", ["latest", "next", "^1.2.3", "1.2", "1.2.3 || 2.0.0"])
def test_agent_runtime_env_rejects_non_exact_harness_pin(version) -> None:
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="exact npm version"):
        app.agent_runtime_env(
            "responses_api_agents.codex_agent.app",
            {"codex_version": version},
        )


def test_dependency_provisioning_uses_pin_and_invalidates_when_it_changes(monkeypatch, tmp_path) -> None:
    package_dir = tmp_path / "legal_agent_bench_agent"
    package_dir.mkdir()
    portable = package_dir / "_portable_python.sh"
    portable.write_text("portable")
    script = tmp_path / "claude_code_agent_deps.sh"
    script.write_text("claude")
    agent_dir = tmp_path / "responses_api_agents" / "claude_code_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "requirements.txt").write_text("nemo-gym\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'nemo-gym'\n")
    (tmp_path / "README.md").write_text("# Gym\n")
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym" / "runtime.py").write_text("VERSION = 1\n")
    calls = []

    monkeypatch.setattr(app, "PACKAGE_DIR", package_dir)
    monkeypatch.setattr(app, "PARENT_DIR", tmp_path)
    monkeypatch.setattr(app, "PORTABLE_PYTHON_SH", portable)
    monkeypatch.setattr(app, "resolve_agent_setup_script", lambda module: script)

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="sha256:image:linux/arm64\n")

    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(app.subprocess, "run", fake_run)

    for version in ("2.1.211", "2.1.212"):
        app.ensure_agent_runtime(
            "responses_api_agents.claude_code_agent.app",
            agent_kwargs={"claude_code_version": version},
            image="lab:image",
            docker_network="host",
            timeout_seconds=60,
        )

    docker_runs = [call[0] for call in calls if call[0][1] == "run"]
    assert len(docker_runs) == 2
    assert "CLAUDE_SPEC=@anthropic-ai/claude-code@2.1.211" in docker_runs[0]
    assert "CLAUDE_SPEC=@anthropic-ai/claude-code@2.1.212" in docker_runs[1]


def test_runtime_archive_is_published_once_across_processes(tmp_path) -> None:
    deps = tmp_path / "hermes_agent" / "recipe"
    deps.mkdir(parents=True)
    (deps / ".installed").write_text("recipe")
    (deps / "runtime.py").write_text("VALUE = 1\n")
    context = multiprocessing.get_context("fork")

    with context.Pool(2) as pool:
        results = pool.map(_runtime_archive_worker, [str(deps), str(deps)])

    assert results[0] == results[1]
    archive = Path(results[0][0])
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as contents:
        assert "agent_deps_mount/runtime.py" in contents.getnames()
    assert app.ensure_runtime_archive(deps) == archive


def test_archive_helpers_reject_missing_unsafe_and_nonregular_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        app._create_archive(tmp_path / "missing.tar.gz", [(tmp_path / "absent", "input")])

    traversal = tarfile.TarInfo("../outside")
    with pytest.raises(app.LegalAgentBenchArtifactError, match="Unsafe"):
        app._validate_archive_member(traversal)

    device = tarfile.TarInfo("device")
    device.type = tarfile.CHRTYPE
    with pytest.raises(app.LegalAgentBenchArtifactError, match="regular files"):
        app._validate_archive_member(device)

    with pytest.raises(app.LegalAgentBenchArtifactError, match="regular downloaded file"):
        app._copy_downloaded_file(tmp_path, tmp_path / "copy")


def test_provider_and_installer_validation_rejects_ambiguous_or_missing_configuration(monkeypatch, tmp_path) -> None:
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="exactly one provider"):
        app._provider_name({"docker": {}, "ecs_fargate": {}})

    monkeypatch.setattr(app, "PARENT_DIR", tmp_path)
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="requires dependency setup script"):
        app.resolve_agent_setup_script("responses_api_agents.hermes_agent.app")


def test_recipe_hash_ignores_transient_bytecode(tmp_path) -> None:
    package = tmp_path / "package"
    (package / "__pycache__").mkdir(parents=True)
    (package / "app.py").write_text("VALUE = 1\n")
    bytecode = package / "__pycache__" / "app.pyc"
    bytecode.write_bytes(b"first")
    first = app._recipe_hash([package])

    bytecode.write_bytes(b"second")

    assert app._recipe_hash([package]) == first


def test_runtime_provisioning_requires_docker(monkeypatch) -> None:
    monkeypatch.setattr(app, "resolve_agent_setup_script", lambda _module: Path("installer.sh"))
    monkeypatch.setattr(app.shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="Docker CLI"):
        app.ensure_agent_runtime(
            "responses_api_agents.hermes_agent.app",
            agent_kwargs={},
            image="lab:image",
            docker_network="host",
            timeout_seconds=60,
        )


def test_config_defaults_are_docker_and_single_concurrency() -> None:
    config = _config()
    assert config.concurrency == 1
    assert config.sandbox_provider == {"docker": {}}
    assert config.sandbox_image is None
    assert config.docker_network == "host"
    assert config.results_dir == "results/legal_agent_bench"
    assert config.agent_timeout_seconds == 10800
    assert config.model_connect_timeout_seconds == 10
    assert config.verifier_timeout_seconds == 3600
    assert config.runtime_build_timeout_seconds == 3600


@pytest.mark.parametrize(
    ("module", "job_dir"),
    [
        ("responses_api_agents.legal_agent_bench_native_agent.app", "native_jobs"),
        ("responses_api_agents.hermes_agent.app", "hermes_jobs"),
        ("responses_api_agents.claude_code_agent.app", "claude_code_jobs"),
        ("responses_api_agents.codex_agent.app", "codex_jobs"),
    ],
)
def test_results_session_dir_is_harness_date_and_model_browsable(tmp_path, module, job_dir) -> None:
    session = app._results_session_dir(
        tmp_path,
        agent_server_module=module,
        model_name="nvidia/model name",
        timestamp=1785260796,
        session_id="0b6511a6",
    )

    timestamp = app.time.strftime("%Y%m%d-%H%M%S", app.time.localtime(1785260796))
    assert session == tmp_path / job_dir / "nvidia-model-name" / f"{timestamp}_0b6511a6"


def test_results_segment_rejects_path_syntax() -> None:
    assert app._results_segment("../../model/name", fallback="unknown") == "model-name"
    assert app._results_segment("...", fallback="unknown") == "unknown"


def test_agent_construction_does_not_create_empty_results_session(tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(results_dir=str(tmp_path)),
        server_client=SimpleNamespace(global_config_dict=OmegaConf.create({"policy_model_name": "org/model"})),
    )

    assert runner._session_results_dir.parent == tmp_path / "hermes_jobs" / "org-model"
    assert not runner._session_results_dir.exists()

    paths = runner._run_dirs("area__task")

    assert paths["root"].is_dir()
    assert runner._session_results_dir.is_dir()


def test_model_name_reads_omegaconf_global_config() -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner.server_client = SimpleNamespace(
        global_config_dict=OmegaConf.create({"policy_model_name": "nvidia/nemotron-3-ultra"})
    )

    assert runner._model_name() == "nvidia/nemotron-3-ultra"


def test_resolve_task_dir_accepts_known_task(tmp_path) -> None:
    root, task = _task_tree(tmp_path)
    assert app.resolve_task_dir(root, "legal_agent_bench::area__task") == task.resolve()


@pytest.mark.parametrize(
    "instance_id",
    [
        "wrong::area__task",
        "legal_agent_bench::../task",
        "legal_agent_bench::nested/task",
        "legal_agent_bench::",
        "area__task",
    ],
)
def test_resolve_task_dir_rejects_unsafe_or_invalid_ids(tmp_path, instance_id) -> None:
    root, _task = _task_tree(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        app.resolve_task_dir(root, instance_id)


def test_resolve_task_dir_rejects_incomplete_task(tmp_path) -> None:
    root = tmp_path / "tasks"
    (root / "broken").mkdir(parents=True)
    with pytest.raises(app.LegalAgentBenchTaskError, match="incomplete"):
        app.resolve_task_dir(root, "legal_agent_bench::broken")


def test_compose_agent_input_uses_task_and_skills_without_rubric(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    monkeypatch.setattr(app, "validate_harness_skills", lambda path: Path(path))
    params = NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=0.5)

    result = app.compose_agent_input(task, skills, params)

    assert result.temperature == 0.5
    assert len(result.input) == 2
    serialized = result.model_dump(mode="json", warnings="error")
    system = serialized["input"][0]["content"]
    assert "Write a memo." in system
    assert "Skill: docx" in system
    assert "/workspace/vdr" in system
    assert "criteria" not in system
    assert serialized["input"][1]["content"] == app.INITIAL_USER_PROMPT


def test_native_agent_input_uses_upstream_prompt_and_canonical_tools(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    monkeypatch.setattr(app, "validate_harness_skills", lambda path: Path(path))

    result = app.compose_agent_input(
        task,
        skills,
        NeMoGymResponseCreateParamsNonStreaming(input=[]),
        native=True,
    )

    serialized = result.model_dump(mode="json", warnings="error")
    system = serialized["input"][0]["content"]
    assert system.startswith(app.LAB_SYSTEM_PROMPT)
    assert "Write a memo." in system
    assert '"criteria"' not in system
    assert '"id": "1"' not in system
    assert [tool["name"] for tool in result.tools] == ["bash", "read", "write", "write_docx", "edit", "glob", "grep"]
    assert all(tool["type"] == "function" and tool["strict"] is False for tool in result.tools)
    assert result.parallel_tool_calls is False


def test_verifier_credentials_are_read_separately_from_sandbox_resources(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    assert app._verifier_env(task) == {
        "LAB_JUDGE_API_KEY": "secret",  # pragma: allowlist secret
        "LAB_JUDGE_MODEL": "judge",
    }
    resources = app._sandbox_resources(task)
    assert resources.cpu == 2
    assert resources.memory_mib == 4096
    assert resources.disk_gib == 10


def test_environment_hash_is_deterministic_and_content_sensitive(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    environment = task / "environment"
    first = app._environment_hash(environment)
    assert first == app._environment_hash(environment)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\nRUN true\n")
    assert app._environment_hash(environment) != first


def test_empty_response_is_schema_valid() -> None:
    response = app._empty_response("policy")
    assert response.model == "policy"
    assert response.output == []


def test_runner_config_preserves_dynamic_agent_configuration(tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(agent_kwargs={"max_turns": 17, "terminal_backend": "local"})
    )
    paths = {"runtime": tmp_path}
    params = NeMoGymResponseCreateParamsNonStreaming(input="Do the work")

    runner._write_runner_config(paths, params, "http://model.internal:8000")

    payload = json.loads((tmp_path / "runner.json").read_text())
    assert payload["agent_server_module"] == "responses_api_agents.hermes_agent.app"
    assert payload["agent_server_class"] == "HermesAgent"
    assert payload["agent_config_class"] == "HermesAgentConfig"
    assert payload["agent_kwargs"]["max_turns"] == 17
    assert payload["model_url"] == "http://model.internal:8000"
    assert payload["model_connect_timeout_seconds"] == 10
    assert payload["disable_endpoint_metadata_probe"] is True
    assert "inspect.signature(agent.responses)" in (tmp_path / "agent_runner.py").read_text()
    assert "SimpleNamespace(path_params={})" in (tmp_path / "agent_runner.py").read_text()
    assert "socket.create_connection" in (tmp_path / "agent_runner.py").read_text()
    assert 'os.environ.get("LAB_POLICY_MODEL_URL", runner["model_url"])' in (tmp_path / "agent_runner.py").read_text()
    assert "runner_status.json" in (tmp_path / "agent_runner.py").read_text()
    assert '"responses_api_models"' in (tmp_path / "agent_runner.py").read_text()
    assert 'os.environ.setdefault("NEMO_GYM_CONFIG_DICT", "{}")' in (tmp_path / "agent_runner.py").read_text()
    assert "hermes_model_metadata.fetch_endpoint_model_metadata" in (tmp_path / "agent_runner.py").read_text()
    assert "hermes_usage_pricing.fetch_endpoint_model_metadata" in (tmp_path / "agent_runner.py").read_text()


def test_runner_config_keeps_endpoint_metadata_for_non_hermes_harnesses(tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            agent_server_module="responses_api_agents.codex_agent.app",
            agent_server_class="CodexAgent",
            agent_config_class="CodexAgentConfig",
        )
    )
    paths = {"runtime": tmp_path}
    params = NeMoGymResponseCreateParamsNonStreaming(input="Do the work")

    runner._write_runner_config(paths, params, "http://model.internal:8000")

    payload = json.loads((tmp_path / "runner.json").read_text())
    assert payload["disable_endpoint_metadata_probe"] is False


def test_runner_config_stages_default_codex_model_catalog(tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            agent_server_module="responses_api_agents.codex_agent.app",
            agent_server_class="CodexAgent",
            agent_config_class="CodexAgentConfig",
        )
    )

    runner._write_runner_config(
        {"runtime": tmp_path},
        NeMoGymResponseCreateParamsNonStreaming(input="Do the work"),
        "http://model.internal:8000",
    )

    payload = json.loads((tmp_path / "runner.json").read_text())
    assert payload["agent_kwargs"]["extra_config"]["model_catalog_json"] == app.CODEX_MODEL_CATALOG_PATH
    catalog = json.loads((tmp_path / "codex_model_catalog.json").read_text())
    assert catalog["models"][0]["slug"] == "gym-policy-model"
    assert catalog["models"][0]["base_instructions"] == ""


def test_runner_config_preserves_explicit_codex_model_catalog(tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            agent_server_module="responses_api_agents.codex_agent.app",
            agent_server_class="CodexAgent",
            agent_config_class="CodexAgentConfig",
            agent_kwargs={"extra_config": {"model_catalog_json": "/custom/catalog.json"}},
        )
    )

    runner._write_runner_config(
        {"runtime": tmp_path},
        NeMoGymResponseCreateParamsNonStreaming(input="Do the work"),
        "http://model.internal:8000",
    )

    payload = json.loads((tmp_path / "runner.json").read_text())
    assert payload["agent_kwargs"]["extra_config"]["model_catalog_json"] == "/custom/catalog.json"
    assert not (tmp_path / "codex_model_catalog.json").exists()


def test_stage_agent_source_copies_only_selected_runtime_package(monkeypatch, tmp_path) -> None:
    repository = tmp_path / "repository"
    package = repository / "responses_api_agents" / "hermes_agent"
    (package / "tests").mkdir(parents=True)
    (package / "data").mkdir()
    (package / "runtime_helpers").mkdir()
    (package / "app.py").write_text("VALUE = 1\n")
    (package / "runtime_helpers" / "tool.py").write_text("VALUE = 2\n")
    (package / "tests" / "rubric.py").write_text("SECRET = True\n")  # pragma: allowlist secret
    (package / "data" / "example.jsonl").write_text("{}\n")
    (repository / "resources_servers" / "legal_agent_bench" / "data" / "runtime").mkdir(parents=True)
    (repository / "resources_servers" / "legal_agent_bench" / "data" / "runtime" / "rubric.json").write_text("{}")
    paths = {"agent_source": tmp_path / "staged"}
    paths["agent_source"].mkdir()
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    monkeypatch.setattr(app, "PARENT_DIR", repository)

    runner._stage_agent_source(paths)

    staged = paths["agent_source"] / "responses_api_agents" / "hermes_agent"
    assert (staged / "app.py").is_file()
    assert (staged / "runtime_helpers" / "tool.py").is_file()
    assert not (staged / "tests").exists()
    assert not (staged / "data").exists()
    assert not (paths["agent_source"] / "resources_servers").exists()


def test_model_url_uses_override_without_server_lookup() -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(sandbox_model_base_url="http://host.docker.internal:9000")
    )
    runner.server_client = SimpleNamespace(global_config_dict={})
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    assert runner._model_url(body) == "http://host.docker.internal:9000"


def test_ecs_model_url_uses_host_policy_proxy_for_reverse_tunnel(monkeypatch) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            sandbox_provider={
                "ecs_fargate": {
                    "region": "us-east-1",
                    "cluster": "test",
                }
            }
        ),
        server_client=SimpleNamespace(
            global_config_dict={},
            _build_server_base_url=lambda _config: "http://0.0.0.0:16300",
        ),
    )
    monkeypatch.setattr(app, "get_first_server_config_dict", lambda *_args: {})
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    assert runner._model_url(body) == "http://127.0.0.1:16300"


@pytest.mark.parametrize(
    ("url", "network", "platform_name", "expected"),
    [
        (
            "http://127.0.0.1:16300/rollout/run-1",
            "host",
            "darwin",
            "http://host.docker.internal:16300/rollout/run-1",
        ),
        ("http://localhost:16300", "host", "win32", "http://host.docker.internal:16300"),
        ("http://0.0.0.0:16300", None, "linux", "http://host.docker.internal:16300"),
        ("http://127.0.0.1:16300", "host", "linux", "http://127.0.0.1:16300"),
        ("http://model.internal:16300", "host", "darwin", "http://model.internal:16300"),
    ],
)
def test_sandbox_model_url_routes_loopback_for_docker(url, network, platform_name, expected) -> None:
    assert app.sandbox_model_url(url, docker_network=network, platform_name=platform_name) == expected


def test_linux_bridge_provider_registers_host_gateway(monkeypatch) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config(docker_network=None))
    monkeypatch.setattr(app.sys, "platform", "linux")

    provider = runner._provider_config()

    assert provider["docker"]["create"]["extra_run_args"] == [
        "--add-host",
        "host.docker.internal:host-gateway",
    ]


def test_hermes_synthetic_connection_error_is_an_agent_failure() -> None:
    response = app.NeMoGymResponse.model_validate(
        {
            **_successful_response().model_dump(mode="json"),
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
            "output": [
                {
                    "id": "msg-error",
                    "content": [{"annotations": [], "text": "Connection error.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "prompt_token_ids": [0],
                    "generation_token_ids": [0],
                    "generation_log_probs": [0.0],
                }
            ],
        }
    )

    failure = app.agent_response_failure(response, "responses_api_agents.hermes_agent.app")

    assert failure == "Hermes produced no model trajectory: Connection error."


def test_real_agent_activity_is_not_treated_as_an_infrastructure_failure() -> None:
    assert (
        app.agent_response_failure(
            _successful_response(),
            "responses_api_agents.hermes_agent.app",
        )
        is None
    )


def test_partial_response_with_harness_error_is_an_agent_failure() -> None:
    response = app.NeMoGymResponse.model_validate(
        {
            **_successful_response().model_dump(mode="json"),
            "status": "failed",
            "error": {"code": "server_error", "message": "adapter failed after partial output"},
        }
    )

    failure = app.agent_response_failure(response, "responses_api_agents.codex_agent.app")

    assert failure is not None
    assert "adapter failed after partial output" in failure


def test_native_timeout_failure_metadata_propagates_timeout_flag() -> None:
    response = app.NeMoGymResponse.model_validate(
        {
            **_successful_response().model_dump(mode="json"),
            "status": "failed",
            "error": {"code": "server_error", "message": "LAB model call timed out after 1800s"},
            "metadata": {app.AGENT_FAILURE_CLASS_METADATA_KEY: "agent_timed_out"},
        }
    )

    assert app.agent_response_failure_flags(response, app.NATIVE_AGENT_MODULE) == (False, True)
    assert app.agent_response_failure_flags(response, "responses_api_agents.codex_agent.app") == (False, False)


def test_native_model_connection_failure_metadata_propagates_connection_flag() -> None:
    response = app.NeMoGymResponse.model_validate(
        {
            **_successful_response().model_dump(mode="json"),
            "status": "failed",
            "error": {"code": "server_error", "message": "LAB model call failed: HTTP 500"},
            "metadata": {app.AGENT_FAILURE_CLASS_METADATA_KEY: "model_connection_failed"},
        }
    )

    assert app.agent_response_failure_flags(response, app.NATIVE_AGENT_MODULE) == (True, False)
    assert app.agent_response_failure_flags(response, "responses_api_agents.codex_agent.app") == (False, False)


def test_response_masks_harness_and_verifier_failures() -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=params,
    )

    response = runner._response(
        body=body,
        params=params,
        response=app._empty_response("policy"),
        reward_data={},
        paths=None,
        agent_failed=True,
        model_connection_failed=True,
        verifier_failed=True,
    )

    assert response.agent_failed is True
    assert response.model_connection_failed is True
    assert response.verifier_failed is True
    assert response.mask_sample is True
    assert response.model_dump()["_ng_failure_class"] == "model_connection_failed"
    assert "_ng_failure_terminal" not in response.model_dump()


@pytest.mark.parametrize(
    ("reward_data", "expected_reason"),
    [
        ({"judge_error_count": 1}, "Verifier reported 1 judge error"),
        ({"judge_error_count": 3}, "Verifier reported 3 judge errors"),
        ({"verifier_error": 1}, "Verifier reported an internal error"),
    ],
)
def test_response_synthesizes_failure_reason_for_unreliable_verifier_metrics(reward_data, expected_reason) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=params,
    )

    response = runner._response(
        body=body,
        params=params,
        response=app._empty_response("policy"),
        reward_data=reward_data,
        paths=None,
    )

    assert response.failure_reason == expected_reason
    assert response.verifier_failed is True
    assert response.mask_sample is True
    assert response.model_dump()["_ng_failure_class"] == "verifier_failed"


@pytest.mark.parametrize(
    ("failure_flag", "expected_class", "terminal"),
    [
        ("agent_failed", "agent_failed", False),
        ("model_connection_failed", "model_connection_failed", False),
        ("agent_timed_out", "agent_timed_out", False),
        ("verifier_failed", "verifier_failed", False),
        ("verifier_timed_out", "verifier_failed", False),
        ("sandbox_failed", "sandbox_failed", False),
        ("task_failed", "task_failed", True),
        ("configuration_failed", "configuration_failed", True),
    ],
)
def test_response_routes_every_failure_class(failure_flag, expected_class, terminal) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=params,
    )

    response = runner._response(
        body=body,
        params=params,
        response=app._empty_response("policy"),
        reward_data={},
        paths=None,
        **{failure_flag: True},
    )
    response_data = response.model_dump()

    assert response.mask_sample is True
    assert response_data["_ng_failure_class"] == expected_class
    assert response_data.get("_ng_failure_terminal", False) is terminal


@pytest.mark.parametrize(
    "reward_data",
    [
        {"reward": float("nan")},
        {"criteria_pass_rate": 1.1},
        {"judge_error_count": "not-a-number"},
        {"verifier_error": -1},
    ],
)
def test_response_masks_malformed_verifier_metrics_instead_of_raising(reward_data) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=params,
    )

    response = runner._response(
        body=body,
        params=params,
        response=_successful_response(),
        reward_data=reward_data,
        paths=None,
    )

    assert response.reward == 0.0
    assert response.verifier_failed is True
    assert response.mask_sample is True
    assert response.failure_reason.startswith("Invalid verifier")
    assert response.model_dump()["_ng_failure_class"] == "verifier_failed"
    assert "_ng_failure_terminal" not in response.model_dump()


@pytest.mark.asyncio
async def test_run_classifies_invalid_task_without_sandbox_failure(monkeypatch) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner._sem = app.asyncio.Semaphore(1)
    monkeypatch.setattr(runner, "_model_name", lambda: "policy")
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::../unsafe",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    response = await runner.run(None, body)

    assert response.task_failed is True
    assert response.configuration_failed is False
    assert response.model_dump()["_ng_failure_class"] == "task_failed"
    assert response.model_dump()["_ng_failure_terminal"] is True
    assert response.sandbox_failed is False
    assert response.mask_sample is True
    assert "Unsafe Legal Agent Bench task name" in response.failure_reason


@pytest.mark.asyncio
async def test_run_classifies_bad_agent_configuration_without_sandbox_failure(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner._sem = app.asyncio.Semaphore(1)
    runner._session_results_dir = tmp_path / "results"

    async def ensure_image(_task):
        return "lab:image"

    async def reject_runtime(_image):
        raise app.LegalAgentBenchConfigurationError("missing configured CLI pin")

    monkeypatch.setattr(app, "resolve_task_dir", lambda runtime, instance: task)
    monkeypatch.setattr(app, "resolve_repo_path", lambda path: skills)
    monkeypatch.setattr(app, "compose_agent_input", lambda task_dir, skills_dir, params: params)
    monkeypatch.setattr(runner, "_model_name", lambda: "policy")
    monkeypatch.setattr(runner, "_ensure_image", ensure_image)
    monkeypatch.setattr(runner, "_ensure_runtime", reject_runtime)
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=params,
    )

    response = await runner.run(None, body)

    assert response.configuration_failed is True
    assert response.task_failed is False
    assert response.sandbox_failed is False
    assert response.mask_sample is True
    assert response.model_dump()["_ng_failure_class"] == "configuration_failed"
    assert response.model_dump()["_ng_failure_terminal"] is True
    assert "missing configured CLI pin" in response.failure_reason


@pytest.mark.asyncio
async def test_concurrent_image_requests_build_once(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner._image_lock = app.asyncio.Lock()
    calls = []

    async def fake_run(args, *, cwd, timeout):
        calls.append(args)
        if args[1:3] == ["image", "inspect"]:
            inspect_count = sum(call[1:3] == ["image", "inspect"] for call in calls)
            return (1 if inspect_count == 1 else 0), "", ""
        return 0, "", ""

    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(app, "_run_process", fake_run)

    first, second = await app.asyncio.gather(runner._ensure_image(task), runner._ensure_image(task))

    assert first == second
    assert sum(call[1] == "build" for call in calls) == 1


@pytest.mark.asyncio
async def test_two_run_requests_overlap_when_server_concurrency_is_two(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    runner = app.LegalAgentBenchAgent.model_construct(config=_config(concurrency=2))
    runner._sem = app.asyncio.Semaphore(2)
    runner._session_results_dir = tmp_path / "results"
    active = 0
    maximum_active = 0
    both_started = app.asyncio.Event()

    async def ensure_image(_task):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            both_started.set()
        await app.asyncio.wait_for(both_started.wait(), timeout=1)
        active -= 1
        return "lab:image"

    async def stop_after_overlap(_image):
        raise app.LegalAgentBenchConfigurationError("stop after overlap assertion")

    monkeypatch.setattr(app, "resolve_task_dir", lambda runtime, instance: task)
    monkeypatch.setattr(app, "resolve_repo_path", lambda path: skills)
    monkeypatch.setattr(app, "compose_agent_input", lambda task_dir, skills_dir, params: params)
    monkeypatch.setattr(runner, "_model_name", lambda: "policy")
    monkeypatch.setattr(runner, "_ensure_image", ensure_image)
    monkeypatch.setattr(runner, "_ensure_runtime", stop_after_overlap)
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    first, second = await app.asyncio.gather(runner.run(None, body), runner.run(None, body))

    assert maximum_active == 2
    assert first.configuration_failed is True
    assert second.configuration_failed is True


def test_agent_sandbox_has_no_host_mounts(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "deps"
    deps.mkdir()
    paths = {name: tmp_path / name for name in ("runtime", "agent_source", "agent", "output", "workspace")}
    for path in paths.values():
        path.mkdir()
    captured = {}

    def fake_sandbox(provider, spec):
        captured["provider"] = provider
        captured["spec"] = spec
        return SimpleNamespace()

    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    monkeypatch.setattr(runner, "_provider_config", lambda: {"docker": {}})
    monkeypatch.setattr(runner, "_sandbox_metadata", lambda: {})
    monkeypatch.setattr(app, "AsyncSandbox", fake_sandbox)

    runner._agent_sandbox(
        image="lab:image",
        task_dir=task,
        skills_dir=skills,
        deps_dir=deps,
        paths=paths,
        model_url="http://model",
    )

    assert captured["provider"] == {"docker": {}}
    assert captured["spec"].provider_options == {}


def test_ecs_agent_sandbox_tunnels_derived_policy_model_url(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "deps"
    deps.mkdir()
    paths = {name: tmp_path / name for name in ("runtime", "agent_source", "agent", "output", "workspace")}
    for path in paths.values():
        path.mkdir()
    captured = {}

    def fake_sandbox(provider, spec):
        captured["provider"] = provider
        captured["spec"] = spec
        return SimpleNamespace()

    provider = {"ecs_fargate": {"region": "us-east-1", "cluster": "test"}}
    runner = app.LegalAgentBenchAgent.model_construct(config=_config(sandbox_provider=provider))
    monkeypatch.setattr(runner, "_sandbox_metadata", lambda: {})
    monkeypatch.setattr(app, "AsyncSandbox", fake_sandbox)

    runner._agent_sandbox(
        image="registry.example/lab@sha256:" + "a" * 64,
        task_dir=task,
        skills_dir=skills,
        deps_dir=deps,
        paths=paths,
        model_url="http://127.0.0.1:16300/rollout/run-1",
    )

    assert captured["provider"] == provider
    assert captured["spec"].provider_options == {
        "outside_endpoints": [
            {
                "url": "http://127.0.0.1:16300/rollout/run-1",
                "env_var": "LAB_POLICY_MODEL_URL",
            }
        ]
    }


def test_ecs_agent_sandbox_uses_explicit_reachable_model_url_without_tunnel(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    captured = {}

    def fake_sandbox(provider, spec):
        captured["spec"] = spec
        return SimpleNamespace()

    provider = {"ecs_fargate": {"region": "us-east-1", "cluster": "test"}}
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            sandbox_provider=provider,
            sandbox_model_base_url="https://model.example/v1",
        )
    )
    monkeypatch.setattr(runner, "_sandbox_metadata", lambda: {})
    monkeypatch.setattr(app, "AsyncSandbox", fake_sandbox)

    runner._agent_sandbox(
        image="registry.example/lab@sha256:" + "a" * 64,
        task_dir=task,
        skills_dir=tmp_path / "skills",
        deps_dir=tmp_path / "deps",
        paths={},
        model_url="https://model.example/v1",
    )

    assert captured["spec"].provider_options == {}


def test_verifier_sandbox_has_no_host_mounts(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    paths = {"lab_run": tmp_path / "lab-run"}
    paths["lab_run"].mkdir()
    captured = {}

    def fake_sandbox(provider, spec):
        captured["provider"] = provider
        captured["spec"] = spec
        return SimpleNamespace()

    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    monkeypatch.setattr(runner, "_provider_config", lambda: {"docker": {}})
    monkeypatch.setattr(runner, "_sandbox_metadata", lambda: {})
    monkeypatch.setattr(app, "AsyncSandbox", fake_sandbox)

    runner._verifier_sandbox(image="lab:image", task_dir=task, paths=paths)

    assert captured["spec"].workdir == "/logs/agent/artifacts/lab-run/output"
    assert captured["spec"].provider_options == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reward_blob", "expected_reward", "expected_verifier_failed"),
    [
        (b'{"reward": 1.0, "criteria_pass_rate": 1.0}', 1.0, False),
        (b'{"reward": "invalid", "criteria_pass_rate": 1.0}', 0.0, True),
    ],
)
@pytest.mark.parametrize(
    "agent_module",
    [
        "responses_api_agents.legal_agent_bench_native_agent.app",
        "responses_api_agents.hermes_agent.app",
        "responses_api_agents.claude_code_agent.app",
        "responses_api_agents.codex_agent.app",
    ],
)
async def test_incomplete_limit_outcomes_are_verified_for_every_harness(
    monkeypatch,
    tmp_path,
    capsys,
    reward_blob,
    expected_reward,
    expected_verifier_failed,
    agent_module,
) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "deps"
    deps.mkdir()
    events = []

    class PhaseSandbox:
        def __init__(self, phase):
            self.phase = phase

        async def start(self):
            events.append(f"{self.phase}:start")

        async def exec(self, *args, **kwargs):
            events.append(f"{self.phase}:exec")
            return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

        async def stop(self):
            if self.phase == "agent":
                assert not runner._session_results_dir.exists()
            events.append(f"{self.phase}:stop")

    agent_sandbox = PhaseSandbox("agent")
    verifier_sandbox = PhaseSandbox("verifier")
    runner = app.LegalAgentBenchAgent.model_construct(config=_config(agent_server_module=agent_module))
    runner._sem = app.asyncio.Semaphore(1)
    runner._session_results_dir = tmp_path / "results"

    async def ensure_image(_task):
        return "lab:image"

    async def ensure_runtime(_image):
        return deps

    async def stage_agent(*args, **kwargs):
        return None

    async def collect_agent(*args, **kwargs):
        return {}

    def write_runner(paths, params, model_url):
        incomplete = app.NeMoGymResponse.model_validate(
            _successful_response().model_dump(mode="json")
            | {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "metadata": {"nemo_gym_stop_reason": "max_turns"},
            }
        )
        (paths["runtime"] / "response.json").write_text(incomplete.model_dump_json())

    async def run_verifier(sandbox, task_dir, paths):
        assert sandbox is verifier_sandbox
        assert events == ["agent:start", "agent:exec", "agent:stop", "verifier:start"]
        assert runner._session_results_dir.is_dir()
        return {"reward.json": reward_blob}, False, None

    async def stage_agent(*args, **kwargs):
        return None

    async def collect_agent(*args, **kwargs):
        return {}

    monkeypatch.setattr(app, "resolve_task_dir", lambda runtime, instance: task)
    monkeypatch.setattr(app, "resolve_repo_path", lambda path: skills)
    monkeypatch.setattr(app, "compose_agent_input", lambda task_dir, skills_dir, params, **kwargs: params)
    monkeypatch.setattr(runner, "_ensure_image", ensure_image)
    monkeypatch.setattr(runner, "_ensure_runtime", ensure_runtime)
    monkeypatch.setattr(runner, "_stage_agent_source", lambda paths: None)
    monkeypatch.setattr(runner, "_write_runner_config", write_runner)
    monkeypatch.setattr(runner, "_model_url", lambda body: "http://model")
    monkeypatch.setattr(runner, "_model_name", lambda: "policy")
    monkeypatch.setattr(runner, "_agent_sandbox", lambda **kwargs: agent_sandbox)
    monkeypatch.setattr(runner, "_stage_agent_sandbox", stage_agent)
    monkeypatch.setattr(runner, "_collect_agent_sandbox", collect_agent)
    monkeypatch.setattr(runner, "_materialize_agent_downloads", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verifier_sandbox", lambda **kwargs: verifier_sandbox)
    monkeypatch.setattr(runner, "_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_stage_and_run_verifier", run_verifier)
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    response = await runner.run(None, body)

    assert events == ["agent:start", "agent:exec", "agent:stop", "verifier:start", "verifier:stop"]
    assert response.reward == expected_reward
    assert response.verifier_failed is expected_verifier_failed
    assert response.sandbox_failed is False
    response_data = response.model_dump()
    if expected_verifier_failed:
        assert response_data["_ng_failure_class"] == "verifier_failed"
        assert "_ng_failure_terminal" not in response_data
    else:
        assert "_ng_failure_class" not in response_data
        assert "_ng_failure_terminal" not in response_data
        assert response.mask_sample is False
    assert Path(response.artifact_dir).is_dir()
    terminal_output = capsys.readouterr().out
    assert "LAB rollout artifacts:" in terminal_output
    expected_terminal_status = "failed" if expected_verifier_failed else "complete"
    assert f"LAB rollout {expected_terminal_status}:" in terminal_output


@pytest.mark.asyncio
async def test_model_connectivity_failure_is_masked_and_skips_verifier(monkeypatch, tmp_path, capsys) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "deps"
    deps.mkdir()
    events = []

    class AgentSandbox:
        async def start(self):
            events.append("agent:start")

        async def exec(self, *args, **kwargs):
            events.append("agent:exec")
            return SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="connection failed")

        async def stop(self):
            events.append("agent:stop")

    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner._sem = app.asyncio.Semaphore(1)
    runner._session_results_dir = tmp_path / "results"
    runner._session_results_dir.mkdir()

    async def ensure_image(_task):
        return "lab:image"

    async def ensure_runtime(_image):
        return deps

    def write_runner(paths, params, model_url):
        (paths["runtime"] / "runner_status.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "phase": "model_connectivity",
                    "error": "Policy model is unreachable from the LAB sandbox: http://model",
                }
            )
        )

    async def stage_agent(*args, **kwargs):
        return None

    async def collect_agent(*args, **kwargs):
        return {}

    monkeypatch.setattr(app, "resolve_task_dir", lambda runtime, instance: task)
    monkeypatch.setattr(app, "resolve_repo_path", lambda path: skills)
    monkeypatch.setattr(app, "compose_agent_input", lambda task_dir, skills_dir, params: params)
    monkeypatch.setattr(runner, "_ensure_image", ensure_image)
    monkeypatch.setattr(runner, "_ensure_runtime", ensure_runtime)
    monkeypatch.setattr(runner, "_stage_agent_source", lambda paths: None)
    monkeypatch.setattr(runner, "_write_runner_config", write_runner)
    monkeypatch.setattr(runner, "_model_url", lambda body: "http://model")
    monkeypatch.setattr(runner, "_model_name", lambda: "policy")
    monkeypatch.setattr(runner, "_agent_sandbox", lambda **kwargs: AgentSandbox())
    monkeypatch.setattr(runner, "_stage_agent_sandbox", stage_agent)
    monkeypatch.setattr(runner, "_collect_agent_sandbox", collect_agent)
    monkeypatch.setattr(runner, "_materialize_agent_downloads", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_verifier_sandbox",
        lambda **kwargs: pytest.fail("verifier must not run after model connectivity failure"),
    )
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    response = await runner.run(None, body)

    assert events == ["agent:start", "agent:exec", "agent:stop"]
    assert response.agent_failed is True
    assert response.model_connection_failed is True
    assert response.mask_sample is True
    assert response.verifier_failed is False
    assert response.model_dump()["_ng_failure_class"] == "model_connection_failed"
    assert "_ng_failure_terminal" not in response.model_dump()
    assert "unreachable" in response.failure_reason
    summary = json.loads(Path(response.run_summary_path).read_text())
    assert summary["flags"]["model_connection_failed"] is True
    assert summary["paths"]["agent_trace"] == response.agent_trace_path
    terminal_output = capsys.readouterr().out
    assert "LAB rollout artifacts:" in terminal_output
    assert "LAB rollout failed:" in terminal_output


@pytest.mark.asyncio
async def test_native_model_timeout_is_masked_routed_and_skips_verifier(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "deps"
    deps.mkdir()
    events = []

    class AgentSandbox:
        async def start(self):
            events.append("agent:start")

        async def exec(self, *args, **kwargs):
            events.append("agent:exec")
            return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

        async def stop(self):
            events.append("agent:stop")

    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            agent_server_module=app.NATIVE_AGENT_MODULE,
            agent_server_class="LegalAgentBenchNativeAgent",
            agent_config_class="LegalAgentBenchNativeAgentConfig",
        )
    )
    runner._sem = app.asyncio.Semaphore(1)
    runner._session_results_dir = tmp_path / "results"

    async def ensure_image(_task):
        return "lab:image"

    async def ensure_runtime(_image):
        return deps

    async def stage_agent(*args, **kwargs):
        return None

    async def collect_agent(*args, **kwargs):
        return {}

    def write_runner(paths, params, model_url):
        failed = app.NeMoGymResponse.model_validate(
            {
                **_successful_response("Partial work").model_dump(mode="json"),
                "status": "failed",
                "error": {"code": "server_error", "message": "LAB model call timed out after 1800s"},
                "metadata": {app.AGENT_FAILURE_CLASS_METADATA_KEY: "agent_timed_out"},
            }
        )
        (paths["runtime"] / "response.json").write_text(failed.model_dump_json())

    monkeypatch.setattr(app, "resolve_task_dir", lambda runtime, instance: task)
    monkeypatch.setattr(app, "resolve_repo_path", lambda path: skills)
    monkeypatch.setattr(app, "compose_agent_input", lambda task_dir, skills_dir, params, **kwargs: params)
    monkeypatch.setattr(runner, "_ensure_image", ensure_image)
    monkeypatch.setattr(runner, "_ensure_runtime", ensure_runtime)
    monkeypatch.setattr(runner, "_stage_agent_source", lambda paths: None)
    monkeypatch.setattr(runner, "_write_runner_config", write_runner)
    monkeypatch.setattr(runner, "_model_url", lambda body: "http://model")
    monkeypatch.setattr(runner, "_model_name", lambda: "policy")
    monkeypatch.setattr(runner, "_agent_sandbox", lambda **kwargs: AgentSandbox())
    monkeypatch.setattr(runner, "_stage_agent_sandbox", stage_agent)
    monkeypatch.setattr(runner, "_collect_agent_sandbox", collect_agent)
    monkeypatch.setattr(runner, "_materialize_agent_downloads", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_verifier_sandbox",
        lambda **kwargs: pytest.fail("verifier must not run after a native model timeout"),
    )
    monkeypatch.setattr(runner, "_artifacts", lambda *args, **kwargs: None)
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )

    response = await runner.run(None, body)

    assert events == ["agent:start", "agent:exec", "agent:stop"]
    assert response.agent_failed is True
    assert response.agent_timed_out is True
    assert response.model_connection_failed is False
    assert response.mask_sample is True
    assert response.model_dump()["_ng_failure_class"] == "agent_timed_out"
    assert "timed out after 1800s" in response.failure_reason
    assert response.response.output


class _FakeSandbox:
    def __init__(self, reward: dict | None, *, timed_out: bool = False):
        self.reward = reward
        self.timed_out = timed_out
        self.uploaded = []
        self.exec_calls = []

    async def upload(self, source, destination):
        self.uploaded.append(destination)

    async def exec(self, command, **kwargs):
        self.exec_calls.append((command, kwargs))
        if command.startswith("test -f"):
            filename = command.rsplit("/", 1)[-1]
            exists = self.reward is not None and filename in {"reward.json", "scores.json"}
            return type(
                "Result", (), {"return_code": 0 if exists else 1, "error_type": None, "stdout": "", "stderr": ""}
            )()
        return type(
            "Result",
            (),
            {
                "return_code": 124 if self.timed_out else 0,
                "error_type": "timeout" if self.timed_out else None,
                "stdout": "",
                "stderr": "",
            },
        )()

    async def download(self, source, destination):
        if self.reward is None:
            raise FileNotFoundError(source)
        payload = self.reward if source.endswith("reward.json") else {"summary": "ok"}
        Path(destination).write_text(json.dumps(payload))


@pytest.mark.asyncio
async def test_verifier_is_staged_after_agent_and_receives_secrets_only_on_exec(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    paths = {"verifier": tmp_path / "verifier", "lab_run": tmp_path / "lab-run"}
    paths["verifier"].mkdir()
    (paths["lab_run"] / "output").mkdir(parents=True)
    sandbox = _FakeSandbox({"reward": 1.0, "criteria_pass_rate": 1.0, "judge_error_count": 0})
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())

    downloaded, timed_out, failure = await runner._stage_and_run_verifier(sandbox, task, paths)
    reward = runner._materialize_verifier_downloads(paths, downloaded)

    assert sandbox.uploaded == ["/tmp/legal-agent-bench-verifier-input.tar.gz"]
    verifier_call = next(call for call in sandbox.exec_calls if "bash /tests/test.sh" in call[0])
    assert verifier_call[1]["env"]["LAB_JUDGE_API_KEY"] == "secret"  # pragma: allowlist secret
    assert reward["reward"] == 1.0
    assert timed_out is False
    assert failure is None


@pytest.mark.asyncio
async def test_missing_verifier_reward_returns_failure(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    paths = {"verifier": tmp_path / "verifier", "lab_run": tmp_path / "lab-run"}
    paths["verifier"].mkdir()
    (paths["lab_run"] / "output").mkdir(parents=True)
    sandbox = _FakeSandbox(None, timed_out=True)
    runner = app.LegalAgentBenchAgent.model_construct(config=_config(verifier_timeout_seconds=1))

    reward, timed_out, failure = await runner._stage_and_run_verifier(sandbox, task, paths)

    assert reward == {}
    assert timed_out is True
    assert "reward.json" in failure


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_untrusted_output_archive_rejects_links_without_touching_host(tmp_path, link_type) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("safe")
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        link = tarfile.TarInfo("stdout.log")
        link.type = link_type
        link.linkname = str(victim)
        archive.addfile(link)

    with pytest.raises(app.LegalAgentBenchArtifactError, match="links"):
        app._extract_untrusted_archive(archive_path, tmp_path / "output")

    assert victim.read_text() == "safe"


def test_materialized_output_is_owned_by_invoking_user(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "memo.txt").write_text("complete")
    archive_path = tmp_path / "output.tar.gz"
    app._create_archive(archive_path, [(source, ".")])
    output = tmp_path / "materialized"

    app._extract_untrusted_archive(archive_path, output)

    assert (output / "memo.txt").stat().st_uid == os.getuid()
    assert (output / "memo.txt").stat().st_gid == os.getgid()


def test_named_remote_provider_requires_immutable_image(tmp_path) -> None:
    global_config = {
        "sandbox": {
            "default_metadata": {"sandbox-api": "remote"},
            "remote": {"endpoint": "sandbox.internal"},
        }
    }
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(sandbox_provider="sandbox", sandbox_image="registry.example/lab:latest"),
        server_client=SimpleNamespace(global_config_dict=global_config),
    )

    assert "remote" in runner._provider_config()
    assert runner._sandbox_metadata() == {"sandbox-api": "remote"}
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="sandbox_model_base_url"):
        runner._model_url(body)
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="immutable"):
        app.asyncio.run(runner._ensure_image(tmp_path))

    runner.config.sandbox_image = "registry.example/lab@sha256:" + "a" * 64
    assert app.asyncio.run(runner._ensure_image(tmp_path)) == runner.config.sandbox_image


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/lab@sha256:",
        "registry.example/lab@sha256:abc123",
        "registry.example/lab@sha512:" + "a" * 128,
        "registry.example/lab:latest@sha256:" + "g" * 64,
    ],
)
def test_named_remote_provider_rejects_incomplete_or_invalid_digest(image, tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(
            sandbox_provider={"ecs_fargate": {"region": "us-east-1"}},
            sandbox_image=image,
        )
    )

    with pytest.raises(app.LegalAgentBenchConfigurationError, match="complete immutable"):
        app.asyncio.run(runner._ensure_image(tmp_path))


@pytest.mark.asyncio
async def test_agent_staging_excludes_tests_and_judge_credentials(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "cache" / "recipe"
    deps.mkdir(parents=True)
    (deps / ".installed").write_text("recipe")
    paths = {"agent_source": tmp_path / "agent-source", "runtime": tmp_path / "runtime"}
    for path in paths.values():
        path.mkdir()
    (paths["agent_source"] / "responses_api_agents" / "hermes_agent").mkdir(parents=True)
    (paths["agent_source"] / "responses_api_agents" / "hermes_agent" / "app.py").write_text("VALUE = 1\n")
    (paths["runtime"] / "runner.json").write_text("{}")

    class CaptureSandbox:
        def __init__(self):
            self.members = {}
            self.exec_commands = []

        async def upload(self, source, destination):
            with tarfile.open(source, "r:gz") as archive:
                self.members[destination] = {member.name for member in archive.getmembers()}

        async def exec(self, command, **kwargs):
            self.exec_commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox = CaptureSandbox()
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())

    await runner._stage_agent_sandbox(
        sandbox,
        task_dir=task,
        skills_dir=skills,
        deps_dir=deps,
        paths=paths,
    )

    staged = sandbox.members["/tmp/legal-agent-bench-agent-input.tar.gz"]
    assert any(name.startswith("workspace/vdr") for name in staged)
    assert any(name.startswith("workspace/skills") for name in staged)
    assert not any("task.toml" in name or "/tests" in name or "LAB_JUDGE" in name for name in staged)
    assert "chmod -R a+rX,a-w /agent_source_mount /agent_deps_mount" in sandbox.exec_commands[0]


@pytest.mark.asyncio
async def test_run_process_returns_output_and_terminates_on_timeout(tmp_path) -> None:
    code, stdout, stderr = await app._run_process(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        cwd=tmp_path,
        timeout=5,
    )
    assert (code, stdout.strip(), stderr.strip()) == (0, "out", "err")

    with pytest.raises(TimeoutError, match="Command timed out"):
        await app._run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            timeout=0.01,
        )


@pytest.mark.asyncio
async def test_runtime_is_provisioned_once_per_agent_instance(monkeypatch, tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner._runtime_lock = app.asyncio.Lock()
    runner._deps_dir = None
    calls = []

    def provision(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "deps"

    monkeypatch.setattr(app, "ensure_agent_runtime", provision)

    first, second = await app.asyncio.gather(runner._ensure_runtime("lab:image"), runner._ensure_runtime("lab:image"))

    assert first == second == tmp_path / "deps"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_docker_image_resolution_handles_missing_cached_and_failed_builds(monkeypatch, tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    runner._image_lock = app.asyncio.Lock()

    monkeypatch.setattr(app.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="Docker CLI"):
        await runner._ensure_image(task)

    monkeypatch.setattr(app.shutil, "which", lambda _name: "/usr/bin/docker")

    async def cached(*args, **kwargs):
        return 0, "", ""

    monkeypatch.setattr(app, "_run_process", cached)
    assert await runner._ensure_image(task) == (
        f"{runner.config.image_repository}:" + app._environment_hash(task / "environment")[:16]
    )

    async def failed_build(args, **kwargs):
        return (1, "", "missing") if args[1] == "image" else (1, "", "build exploded")

    monkeypatch.setattr(app, "_run_process", failed_build)
    with pytest.raises(RuntimeError, match="build exploded"):
        await runner._ensure_image(task)


def test_model_url_reports_server_lookup_failures_and_routes_docker(monkeypatch) -> None:
    body = app.LegalAgentBenchRunRequest(
        instance_id="legal_agent_bench::area__task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(),
        server_client=SimpleNamespace(global_config_dict={}),
    )
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="Unable to resolve policy model"):
        runner._model_url(body)

    runner.server_client = SimpleNamespace(
        global_config_dict={"policy_model": {}},
        _build_server_base_url=lambda _config: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(app, "get_first_server_config_dict", lambda *_args: {})
    monkeypatch.setattr(app.LegalAgentBenchAgent, "rollout_id_from_run", lambda _self, _body: None)
    assert runner._model_url(body).startswith("http://host.docker.internal:8000")


@pytest.mark.asyncio
async def test_remote_provider_requires_an_image_and_missing_agent_source_is_configuration_error(tmp_path) -> None:
    runner = app.LegalAgentBenchAgent.model_construct(
        config=_config(sandbox_provider={"ecs_fargate": {"region": "us-east-1"}})
    )
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="require sandbox_image"):
        await runner._ensure_image(tmp_path)

    paths = runner._paths_for_root(tmp_path / "staging", create=True)
    runner.config.agent_server_module = "responses_api_agents.nonexistent_agent.app"
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="source not found"):
        runner._stage_agent_source(paths)


def test_failure_detection_covers_empty_and_zero_activity_responses() -> None:
    empty = app._empty_response("policy")
    assert app.agent_response_failure(empty, "responses_api_agents.codex_agent.app") == (
        "Agent produced an empty trajectory"
    )

    silent = _successful_response("")
    silent.usage.total_tokens = 0
    assert app.agent_response_failure(silent, "responses_api_agents.codex_agent.app") == (
        "codex_agent produced no model activity"
    )

    assert app.host_tunnel_model_url("https://model.example/v1") == "https://model.example/v1"


def test_task_prompt_and_skills_report_invalid_source_files(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])

    (task / "task.json").write_text("not json")
    with pytest.raises(app.LegalAgentBenchTaskError, match="Invalid LAB task configuration"):
        app.compose_agent_input(task, skills, params)

    (task / "task.toml").write_text("not = [valid")
    with pytest.raises(app.LegalAgentBenchTaskError, match="Invalid LAB task.toml"):
        app._task_toml(task)

    (skills / app.REQUIRED_SKILLS[0] / "SKILL.md").unlink()
    with pytest.raises(app.LegalAgentBenchConfigurationError, match="Invalid LAB skills configuration"):
        app._load_skill_prompt(skills)


@pytest.mark.asyncio
async def test_agent_staging_and_collection_surface_sandbox_failures(tmp_path) -> None:
    _root, task = _task_tree(tmp_path)
    skills = _skills(tmp_path)
    deps = tmp_path / "deps"
    deps.mkdir()
    (deps / "runtime.py").write_text("pass\n")
    paths = app.LegalAgentBenchAgent._paths_for_root(tmp_path / "staged", create=True)
    (paths["agent_source"] / "app.py").write_text("pass\n")

    class Sandbox:
        async def upload(self, source, destination):
            return None

        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=1, stdout="", stderr="sandbox command failed")

    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    with pytest.raises(RuntimeError, match="sandbox command failed"):
        await runner._stage_agent_sandbox(Sandbox(), task_dir=task, skills_dir=skills, deps_dir=deps, paths=paths)

    with pytest.raises(RuntimeError, match="sandbox command failed"):
        await runner._collect_agent_sandbox(Sandbox(), tmp_path / "downloads")


@pytest.mark.asyncio
async def test_agent_collection_and_materialization_preserve_optional_files(tmp_path) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    output_source = tmp_path / "output-source"
    output_source.mkdir()
    (output_source / "memo.txt").write_text("complete")

    class Sandbox:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def download(self, source, destination):
            destination = Path(destination)
            if source.endswith("response.json"):
                destination.write_text(_successful_response().model_dump_json())
            elif source.endswith("runner_status.json"):
                raise FileNotFoundError(source)
            else:
                app._create_archive(destination, [(output_source, ".")])

    runner = app.LegalAgentBenchAgent.model_construct(config=_config())
    downloads = await runner._collect_agent_sandbox(Sandbox(), download_dir)
    paths = runner._paths_for_root(tmp_path / "result", create=True)
    runner._materialize_agent_downloads(paths, downloads, stdout="stdout", stderr="stderr")

    assert set(downloads) == {"response.json", "output.tar.gz"}
    assert (paths["runtime"] / "response.json").is_file()
    assert (paths["output"] / "memo.txt").read_text() == "complete"
    assert (paths["agent"] / "stdout.log").read_text() == "stdout"

    with pytest.raises(app.LegalAgentBenchArtifactError, match="output archive"):
        runner._materialize_agent_downloads(paths, {}, stdout="", stderr="")


def test_verifier_materialization_rejects_bad_reward_before_writing_files(tmp_path) -> None:
    paths = {"verifier": tmp_path / "verifier"}
    paths["verifier"].mkdir()

    with pytest.raises(app.LegalAgentBenchArtifactError, match="expected an object"):
        app.LegalAgentBenchAgent._materialize_verifier_downloads(
            paths,
            {"reward.json": b"[]", "report.html": b"untrusted"},
        )
    assert list(paths["verifier"].iterdir()) == []

    with pytest.raises(app.LegalAgentBenchArtifactError, match="Invalid verifier reward.json"):
        app.LegalAgentBenchAgent._materialize_verifier_downloads(paths, {"reward.json": b"not-json"})


def test_runner_status_reports_invalid_json_and_nonobject(tmp_path) -> None:
    paths = {"runtime": tmp_path}
    assert app.LegalAgentBenchAgent._runner_status(paths) == {}

    status = tmp_path / "runner_status.json"
    status.write_text("not-json")
    assert app.LegalAgentBenchAgent._runner_status(paths)["ok"] is False

    status.write_text("[]")
    assert app.LegalAgentBenchAgent._runner_status(paths)["error"].endswith("expected an object")
