# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTaskConfig, content_hash, discover_tasks, load_task
from nemo_gym.tasks.harbor.cli import AgentSelection, PreparedTaskset, build_run, prepare_target, resolve_agent
from nemo_gym.tasks.harbor.dockerfile import base_image_only
from nemo_gym.tasks.harbor.hub import (
    HubError,
    HubRef,
    RegistryDataset,
    RegistryTask,
    fetch_dataset,
    load_registry,
    resolve_dataset,
)
from nemo_gym.tasks.harbor.materialize import materialize_task, run_config, write_rows
from nemo_gym.tasks.harbor.task import HarborTaskError


HELLO_TOML = """
schema_version = "1.4"

[task]
name = "harbor/hello-world"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
mcp_servers = []
"""


def write_task(root: Path, *, dockerfile: str = "FROM ubuntu:24.04\n\nWORKDIR /app", toml: str = HELLO_TOML) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Create a file called hello.txt.\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (root / "solution").mkdir(exist_ok=True)
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\necho hi > hello.txt\n")
    return root


class TestTaskConfig:
    def test_reads_harbor_field_names(self):
        config = HarborTaskConfig.model_validate(
            {
                "schema_version": "1.4",
                "environment": {"docker_image": "img:1", "cpus": 2, "memory_mb": 1024},
                "verifier": {"timeout_sec": 30},
            }
        )
        assert config.environment.docker_image == "img:1"
        assert config.environment.memory_mb == 1024
        assert config.verifier.timeout_sec == 30
        assert config.is_shared_verifier

    def test_deprecated_fields_are_read_but_normalized(self):
        config = HarborTaskConfig.model_validate(
            {
                "version": "1.2",
                "environment": {"image": "img:2", "memory": "2G", "storage": "512M", "allow_internet": False},
            }
        )
        assert config.schema_version == "1.2"
        assert config.environment.docker_image == "img:2"
        assert config.environment.memory_mb == 2048
        assert config.environment.storage_mb == 512
        assert config.environment.network_mode == "no-network"
        assert "image" not in config.environment.model_dump()

    def test_conflicting_deprecated_and_current_values_are_rejected(self):
        with pytest.raises(ValueError, match="Conflicting"):
            HarborTaskConfig.model_validate({"environment": {"memory": "1G", "memory_mb": 512}})

    def test_multi_step_is_rejected_for_now(self):
        with pytest.raises(ValueError, match="Multi-step"):
            HarborTaskConfig.model_validate({"steps": [{"name": "a"}]})

    def test_docker_image_is_optional(self):
        assert HarborTaskConfig.model_validate({}).environment.docker_image is None

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError):
            HarborTaskConfig.model_validate({"environment": {"dockerfile": "x"}})

    def test_mcp_server_needs_command_or_url(self):
        with pytest.raises(ValueError, match="stdio needs"):
            HarborTaskConfig.model_validate({"environment": {"mcp_servers": [{"name": "t", "transport": "stdio"}]}})
        config = HarborTaskConfig.model_validate(
            {"environment": {"mcp_servers": [{"name": "t", "transport": "http", "url": "http://x"}]}}
        )
        assert config.environment.mcp_servers[0].transport == "streamable-http"


class TestDockerfileDetection:
    def test_base_image_with_settings(self):
        base = base_image_only(
            "# SPDX header\n\nFROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS app\n"
            'WORKDIR /workspace\nENV A=1 B="two words"\nENV LEGACY value here\nUSER runner\nLABEL x=y\n'
        )
        assert base is not None
        assert base.image == "ghcr.io/astral-sh/uv:python3.13-bookworm-slim"
        assert base.workdir == "/workspace"
        assert base.env == {"A": "1", "B": "two words", "LEGACY": "value here"}
        assert base.user == "runner"

    def test_continuations_and_comments_are_ignored(self):
        base = base_image_only("FROM ubuntu:24.04 \\\n   # inside\n\n AS base\n\nWORKDIR \\\n /app\n")
        assert base is not None and base.image == "ubuntu:24.04" and base.workdir == "/app"

    @pytest.mark.parametrize(
        "text",
        [
            "FROM ubuntu\nRUN apt-get update",
            "FROM ubuntu\nCOPY . /app",
            "FROM a AS one\nFROM b",
            "ARG BASE=ubuntu\nFROM $BASE",
            "WORKDIR /app\nFROM ubuntu",
            "",
        ],
    )
    def test_anything_else_needs_a_build(self, text):
        assert base_image_only(text) is None


class TestDigest:
    def test_stable_and_sensitive_to_covered_files(self, tmp_path):
        task = write_task(tmp_path / "t")
        first = content_hash(task)
        assert first == content_hash(task)
        (task / "README.md").write_text("ignored? no, covered")
        second = content_hash(task)
        assert second != first
        (task / "tests" / "__pycache__").mkdir()
        (task / "tests" / "__pycache__" / "x.pyc").write_bytes(b"x")
        (task / "notes.txt").write_text("not covered")
        assert content_hash(task) == second


class TestLoadTask:
    def test_hello_world_shape(self, tmp_path):
        task = load_task(write_task(tmp_path / "hello-world"))
        assert task.task_id == "hello-world"
        assert task.image == "ubuntu:24.04"
        assert task.workdir == "/app"
        assert task.user is None
        assert task.needs_sandbox and task.has_solution
        assert task.instruction == "Create a file called hello.txt."
        assert task.digest == content_hash(task.path)

    def test_task_toml_wins_over_dockerfile(self, tmp_path):
        toml = HELLO_TOML + '\n[environment.env]\nA = "toml"\n'
        toml = toml.replace("gpus = 0", 'gpus = 0\nworkdir = "/work"')
        task = load_task(
            write_task(tmp_path / "t", dockerfile="FROM img\nENV A=docker B=keep\nWORKDIR /app", toml=toml)
        )
        assert task.workdir == "/work"
        assert task.env == {"A": "toml", "B": "keep"}

    def test_agent_user_wins_over_dockerfile_user(self, tmp_path):
        toml = HELLO_TOML.replace(
            "timeout_sec = 120.0\n\n[environment]", 'timeout_sec = 120.0\nuser = "agent"\n\n[environment]'
        )
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nUSER other", toml=toml))
        assert task.user == "agent"

    def test_dockerfile_needing_a_build_without_image_is_rejected(self, tmp_path):
        with pytest.raises(HarborTaskError, match="needs a build"):
            load_task(write_task(tmp_path / "t", dockerfile="FROM ubuntu\nRUN true"))

    def test_prebuilt_image_allows_a_real_dockerfile(self, tmp_path):
        toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "org/task:1"\ncpus = 1')
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM ubuntu\nRUN true", toml=toml))
        assert task.image == "org/task:1"

    def test_missing_pieces_are_reported(self, tmp_path):
        with pytest.raises(HarborTaskError, match="no task.toml"):
            load_task(tmp_path)
        task = write_task(tmp_path / "t")
        (task / "instruction.md").unlink()
        with pytest.raises(HarborTaskError, match="no instruction.md"):
            load_task(task)


class TestDiscovery:
    def test_single_task_folder(self, tmp_path):
        assert [t.task_id for t in discover_tasks(write_task(tmp_path / "solo"))] == ["solo"]

    def test_folder_of_task_folders_one_level(self, tmp_path):
        write_task(tmp_path / "ds" / "b")
        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "nested" / "deep")
        (tmp_path / "ds" / "README.md").write_text("dataset readme")
        assert [t.task_id for t in discover_tasks(tmp_path / "ds")] == ["a", "b"]

    def test_empty_folder_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(HarborTaskError, match="neither a task folder"):
            discover_tasks(tmp_path / "empty")


class TestMaterialize:
    def test_row_shape(self, tmp_path):
        task = load_task(write_task(tmp_path / "hello-world"))
        row = materialize_task(task, "hello").model_dump(mode="json", exclude_unset=True)
        assert row["task_id"] == {"taskset": "hello", "task_id": "hello-world"}
        params = row["task_input"]["responses_create_params"]
        assert params["input"][0]["role"] == "user"
        assert params["input"][0]["content"] == "Create a file called hello.txt."
        assert params["metadata"] == {"agent_timeout_sec": "120.0"}
        assert row["task_input"]["task_data"] == {DIGEST_KEY: task.digest}

    def test_write_rows_and_run_config(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a")), load_task(write_task(folder / "b"))]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path)
        assert [json.loads(line)["task_id"]["task_id"] for line in rows_path.read_text().splitlines()] == ["a", "b"]
        assert len(rows) == 2

        config = run_config(
            taskset="ds",
            folder=folder,
            tasks=tasks,
            rows_path=rows_path,
            agent_instance_name="hermes_agent",
            agent_impl="hermes_agent",
            agent_overrides={"enabled_toolsets": ["terminal"]},
        )
        assert config["environment_routing_mode"] == "taskset"
        assert config["environment_server_routes"] == {"ds": "harbor_ds_environment"}
        resources = config["harbor_ds_resources_server"]["resources_servers"]["harbor"]
        assert resources["tasksets"]["ds"]["folder"] == str(folder.resolve())
        assert resources["tasksets"]["ds"]["tasks"] == {"a": tasks[0].digest, "b": tasks[1].digest}
        assert resources["datasets"][0]["jsonl_fpath"] == str(rows_path.resolve())
        agent = config["harbor_ds_agent"]
        assert agent["_inherit_from"] == "hermes_agent"
        assert (
            agent["responses_api_agents"]["hermes_agent"]["resources_server"]["name"] == "harbor_ds_resources_server"
        )
        assert agent["responses_api_agents"]["hermes_agent"]["enabled_toolsets"] == ["terminal"]
        environment = config["harbor_ds_environment"]["environment_servers"]["single_agent_turn"]
        assert environment["agent_server"]["name"] == "harbor_ds_agent"


class TestTasksetMapping:
    def test_single_task_folder_maps_to_its_parent(self, tmp_path):
        from nemo_gym.tasks.harbor.materialize import taskset_mapping

        task = load_task(write_task(tmp_path / "solo"))
        mapping = taskset_mapping([task], tmp_path / "solo")
        assert mapping == {"folder": str(tmp_path.resolve()), "tasks": {"solo": task.digest}}
        assert Path(mapping["folder"]) / "solo" == task.path

    def test_folder_of_tasks_maps_to_itself(self, tmp_path):
        from nemo_gym.tasks.harbor.materialize import taskset_mapping

        tasks = [load_task(write_task(tmp_path / "ds" / name)) for name in ("a", "b")]
        assert taskset_mapping(tasks, tmp_path / "ds")["folder"] == str((tmp_path / "ds").resolve())


class TestHub:
    def test_ref_parsing(self):
        assert HubRef.parse("harbor:hello-world") == HubRef("hello-world", None)
        assert HubRef.parse("harbor:terminal-bench@2.0") == HubRef("terminal-bench", "2.0")
        with pytest.raises(HubError):
            HubRef.parse("hello-world")
        with pytest.raises(HubError):
            HubRef.parse("harbor:bad name")

    def test_resolve_dataset_picks_latest_version_unless_pinned(self):
        registry = [
            RegistryDataset("tb", "2.0", "", ()),
            RegistryDataset("tb", "2.10", "", ()),
            RegistryDataset("tb", "1.9", "", ()),
        ]
        assert resolve_dataset(HubRef("tb", None), registry).version == "2.10"
        assert resolve_dataset(HubRef("tb", "1.9"), registry).version == "1.9"
        with pytest.raises(HubError, match="no version"):
            resolve_dataset(HubRef("tb", "3.0"), registry)
        with pytest.raises(HubError, match="not in the Harbor registry"):
            resolve_dataset(HubRef("nope", None), registry)

    def test_load_registry_uses_cache(self, tmp_path):
        cache = tmp_path / ".harbor"
        cache.mkdir()
        (cache / "registry.json").write_text(
            json.dumps(
                [
                    {
                        "name": "hello-world",
                        "version": "1.0",
                        "description": "d",
                        "tasks": [{"name": "hello-world", "git_url": "u", "git_commit_id": "HEAD", "path": "p"}],
                    }
                ]
            )
        )
        [dataset] = load_registry(cache)
        assert dataset.tasks == (RegistryTask("hello-world", "u", "HEAD", "p"),)

    def test_fetch_dataset_pins_head_and_copies_tasks(self, tmp_path):
        repo = tmp_path / "repo"
        write_task(repo / "examples" / "tasks" / "hello-world")
        write_task(repo / "examples" / "tasks" / "other")
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "uploadpack.allowFilter", "true"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "init"],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        url = repo.as_uri()
        dataset = RegistryDataset(
            "hello-world",
            "1.0",
            "",
            (
                RegistryTask("hello-world", url, "HEAD", "examples/tasks/hello-world"),
                RegistryTask("renamed", url, head, "examples/tasks/other"),
            ),
        )

        folder = fetch_dataset(dataset, tmp_path / "datasets")

        assert folder == tmp_path / "datasets" / "hello-world"
        assert (folder / "hello-world" / "task.toml").is_file()
        assert (folder / "renamed" / "tests" / "test.sh").is_file()
        manifest = (folder / "manifest.toml").read_text()
        assert f'git_commit_id = "{head}"' in manifest
        assert "HEAD" not in manifest.replace("hello-world", "")
        assert [t.task_id for t in discover_tasks(folder)] == ["hello-world", "renamed"]

        # A rerun keeps existing folders and does not need the repository again.
        assert fetch_dataset(dataset, tmp_path / "datasets") == folder


class TestCli:
    def test_prepare_local_folder(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert prepared.taskset == "ds"
        assert prepared.rows_path == tmp_path / "out" / "ds" / "tasks.jsonl"
        assert [t.task_id for t in prepared.tasks] == ["a"]

    def test_resolve_agent_accepts_short_name(self):
        selection = resolve_agent("hermes")
        assert selection.instance_name == "hermes_agent"
        assert selection.impl_name == "hermes_agent"
        assert selection.config_path.name == "hermes_agent.yaml"
        with pytest.raises(ValueError, match="No agent config"):
            resolve_agent("no_such_agent_xyz")

    def test_build_run_writes_config_and_overrides(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a"))]
        prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
        prepared.output_dir.mkdir(parents=True)
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent")

        config_path, tokens = build_run(
            prepared, agent, sandbox="opensandbox", overrides=["+agent_name=x", "+split=train"]
        )

        written = yaml.safe_load(config_path.read_text())
        assert written["environment_server_routes"] == {"ds": "harbor_ds_environment"}
        assert written["harbor_ds_agent"]["responses_api_agents"]["hermes_agent"]["enabled_toolsets"] == ["terminal"]
        assert "+split=train" in tokens
        assert not any(token.startswith("+agent_name") for token in tokens)
        config_paths = next(token for token in tokens if token.startswith("+config_paths="))
        assert str(config_path.resolve()) in config_paths
        assert "single_agent_turn/configs/single_agent_turn.yaml" in config_paths
        assert "opensandbox/configs/opensandbox.yaml" in config_paths
        assert f"+output_jsonl_fpath={tmp_path / 'out' / 'rollouts.jsonl'}" in tokens
