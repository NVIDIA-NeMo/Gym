# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import nemo_gym.tasks.harbor.hub as hub_module
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
    validate_git_url,
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
                "environment": {"docker_image": "img:2", "memory": "2G", "storage": "512M", "allow_internet": False},
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

    def test_unknown_keys_are_ignored_and_listed(self):
        data = {
            "allowlist": ["x"],
            "version": "1.3",
            "environment": {
                "dockerfile": "x",
                "image": "img",
                "mcp_servers": [{"name": "t", "url": "http://x", "extra": 1}],
                "healthcheck": {"command": "true", "retries_sec": 1},
            },
            "verifier": {"environment": {"docker_image": "v", "nope": 1}},
        }
        config = HarborTaskConfig.model_validate(data)
        assert config.environment.docker_image is None
        assert "dockerfile" not in config.environment.model_dump()
        assert HarborTaskConfig.unknown_keys(data) == [
            "allowlist",
            "environment.dockerfile",
            "environment.image",
            "environment.mcp_servers[0].extra",
            "environment.healthcheck.retries_sec",
            "verifier.environment.nope",
        ]

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

    def test_undecodable_files_are_task_errors(self, tmp_path):
        task = write_task(tmp_path / "t")
        (task / "task.toml").write_bytes(b"\xff\xfe not utf-8")
        with pytest.raises(HarborTaskError, match="task.toml"):
            load_task(task)
        task = write_task(tmp_path / "u")
        (task / "instruction.md").write_bytes(b"\xff\xfe")
        with pytest.raises(HarborTaskError, match="instruction.md"):
            load_task(task)

    def test_unknown_task_toml_key_is_warned_once_per_key(self, tmp_path, caplog):
        toml = HELLO_TOML.replace('schema_version = "1.4"', 'schema_version = "1.4"\nallowlist = ["pypi.org"]')
        toml = toml.replace("gpus = 0", "gpus = 0\nnew_harbor_key = true")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(write_task(tmp_path / "t", toml=toml))
        assert task.config.environment.gpus == 0
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2
        assert all(str(task.path / "task.toml") in message for message in messages)
        assert any("`allowlist`" in message for message in messages)
        assert any("`environment.new_harbor_key`" in message for message in messages)


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

    def test_unknown_key_in_one_task_does_not_block_the_dataset(self, tmp_path, caplog):
        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "b", toml=HELLO_TOML + "\n[environment.allowlist]\nhosts = []\n")
        write_task(tmp_path / "ds" / "c")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            tasks = discover_tasks(tmp_path / "ds")
        assert [t.task_id for t in tasks] == ["a", "b", "c"]
        assert [r.getMessage() for r in caplog.records] == [
            f"{tmp_path.resolve() / 'ds' / 'b' / 'task.toml'}: ignoring unknown key `environment.allowlist`"
        ]

    def test_malformed_task_is_skipped_by_name(self, tmp_path):
        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "broken", toml="[task\nname = oops")
        write_task(tmp_path / "ds" / "c")
        with pytest.raises(HarborTaskError, match="broken"):
            discover_tasks(tmp_path / "ds")
        skipped: dict[str, HarborTaskError] = {}
        tasks = discover_tasks(tmp_path / "ds", skipped=skipped)
        assert [t.task_id for t in tasks] == ["a", "c"]
        assert list(skipped) == ["broken"]
        assert "broken/task.toml" in str(skipped["broken"])

    def test_all_tasks_failing_is_still_an_error(self, tmp_path):
        write_task(tmp_path / "ds" / "x", toml="not toml =")
        with pytest.raises(HarborTaskError, match="No task under"):
            discover_tasks(tmp_path / "ds", skipped={})


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
        # Verifier logs land beside the rollouts, not under the server's own folder.
        assert resources["artifacts_dir"] == str((tmp_path / "out" / "verifier").resolve())
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

    def test_mixed_numeric_and_named_versions_need_a_pin(self):
        kumo = [RegistryDataset("kumo", "1.0", "", ()), RegistryDataset("kumo", "parity", "", ())]
        with pytest.raises(HubError, match=r"1\.0, parity.*harbor:kumo@<version>"):
            resolve_dataset(HubRef("kumo", None), kumo)
        assert resolve_dataset(HubRef("kumo", "parity"), kumo).version == "parity"
        assert resolve_dataset(HubRef("kumo", "1.0"), kumo).version == "1.0"

    def test_named_versions_are_never_picked_arbitrarily(self):
        lancer = [
            RegistryDataset("swe-lancer-diamond", "diamond", "", ()),
            RegistryDataset("swe-lancer-diamond", "latest", "", ()),
        ]
        with pytest.raises(HubError, match="pin one"):
            resolve_dataset(HubRef("swe-lancer-diamond", None), lancer)
        assert resolve_dataset(HubRef("swe-lancer-diamond", "diamond"), lancer).version == "diamond"
        # One entry is unambiguous whatever its version is called.
        assert resolve_dataset(HubRef("swe-lancer-diamond", None), lancer[:1]).version == "diamond"

    def test_folder_name_includes_the_version(self):
        assert RegistryDataset("kumo", "1.0", "", ()).folder_name == "kumo-1.0"
        assert RegistryDataset("kumo", "parity", "", ()).folder_name == "kumo-parity"
        assert RegistryDataset("kumo", "a/b c", "", ()).folder_name == "kumo-a-b-c"
        assert RegistryDataset("kumo", "", "", ()).folder_name == "kumo"

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
                        "tasks": [
                            {
                                "name": "hello-world",
                                "git_url": "https://github.com/x/y",
                                "git_commit_id": "HEAD",
                                "path": "p",
                            }
                        ],
                    }
                ]
            )
        )
        [dataset] = load_registry(cache)
        assert dataset.tasks == (RegistryTask("hello-world", "https://github.com/x/y", "HEAD", "p"),)

    @pytest.mark.parametrize(
        "url",
        ["https://github.com/x/y.git", "ssh://git@github.com/x/y", "git://host/x", "git@github.com:x/y.git"],
    )
    def test_registry_git_urls_that_git_fetches_over_the_network(self, url):
        assert validate_git_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "-oProxyCommand=touch /tmp/pwned",
            "--upload-pack=touch /tmp/pwned",
            "ext::sh -c 'touch /tmp/pwned'",
            "file:///etc",
            "/etc/passwd",
            "git@-evil:x",
            "git@host:-flag",
            "https://",
        ],
    )
    def test_registry_git_urls_that_are_refused(self, tmp_path, url):
        with pytest.raises(HubError, match="Unsupported git_url"):
            validate_git_url(url)
        cache = tmp_path / ".harbor"
        cache.mkdir()
        entry = {
            "name": "d",
            "version": "1",
            "tasks": [{"name": "t", "git_url": url, "git_commit_id": "HEAD", "path": "p"}],
        }
        (cache / "registry.json").write_text(json.dumps([entry]))
        with pytest.raises(HubError, match="Unsupported git_url"):
            load_registry(cache)

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

        assert folder == tmp_path / "datasets" / "hello-world-1.0"
        assert (folder / "hello-world" / "task.toml").is_file()
        assert (folder / "renamed" / "tests" / "test.sh").is_file()
        manifest = (folder / "manifest.toml").read_text()
        assert f'git_commit_id = "{head}"' in manifest
        assert "HEAD" not in manifest.replace("hello-world", "")
        assert [t.task_id for t in discover_tasks(folder)] == ["hello-world", "renamed"]

        # A rerun keeps existing folders and does not need the repository again.
        assert fetch_dataset(dataset, tmp_path / "datasets") == folder

        # HEAD moves. Present folders keep their pin; only the new task resolves HEAD.
        write_task(repo / "examples" / "tasks" / "third")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "more"],
            check=True,
        )
        new_head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        assert new_head != head
        grown = RegistryDataset(
            dataset.name,
            dataset.version,
            "",
            (*dataset.tasks, RegistryTask("third", url, "HEAD", "examples/tasks/third")),
        )
        assert fetch_dataset(grown, tmp_path / "datasets") == folder
        pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
        assert pins["hello-world"]["git_commit_id"] == head
        assert pins["renamed"]["git_commit_id"] == head
        assert pins["third"]["git_commit_id"] == new_head
        assert (folder / "third" / "task.toml").is_file()

    def test_fetch_resolves_head_once_per_repo_and_shields_urls(self, tmp_path, monkeypatch):
        url = "https://github.com/org/tasks.git"
        sha = "a" * 40
        calls: list[list[str]] = []

        def fake_run(argv, cwd=None, **kwargs):
            calls.append(list(argv))
            if argv[1] == "ls-remote":
                return SimpleNamespace(stdout=f"{sha}\tHEAD\n", stderr="")
            if argv[1] == "clone":
                for task in dataset.tasks:
                    write_task(Path(argv[-1]) / task.path)
            return SimpleNamespace(stdout="", stderr="")

        monkeypatch.setattr(hub_module.subprocess, "run", fake_run)
        dataset = RegistryDataset(
            "big", "2.0", "", tuple(RegistryTask(f"t{i}", url, "HEAD", f"tasks/t{i}") for i in range(12))
        )

        folder = fetch_dataset(dataset, tmp_path / "datasets")

        ls_remotes = [argv for argv in calls if argv[1] == "ls-remote"]
        clones = [argv for argv in calls if argv[1] == "clone"]
        assert len(ls_remotes) == 1 and len(clones) == 1
        assert ls_remotes[0][-3:] == ["--", url, "HEAD"]
        assert clones[0][-3:-1] == ["--", url]
        assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == sorted(t.name for t in dataset.tasks)
        pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
        assert {pin["git_commit_id"] for pin in pins.values()} == {sha}


class TestCli:
    def test_prepare_local_folder(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert prepared.taskset == "ds"
        assert prepared.rows_path == tmp_path / "out" / "ds" / "tasks.jsonl"
        assert [t.task_id for t in prepared.tasks] == ["a"]
        assert prepared.skipped == {}

    def test_prepare_skips_tasks_that_do_not_load(self, tmp_path, caplog):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "bad", toml="[[steps]]\nname = 'x'")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.cli"):
            prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["a"]
        assert list(prepared.skipped) == ["bad"]
        assert "Multi-step" in prepared.skipped["bad"]
        assert any("Skipping task bad" in r.getMessage() for r in caplog.records)
        assert len(prepared.rows_path.read_text().splitlines()) == 1

    def test_resolve_agent_accepts_short_name(self):
        selection = resolve_agent("hermes")
        assert selection.instance_name == "hermes_agent"
        assert selection.impl_name == "hermes_agent"
        assert selection.config_path.name == "hermes_agent.yaml"
        with pytest.raises(ValueError, match="No agent config"):
            resolve_agent("no_such_agent_xyz")

    def test_build_run_writes_config_and_overrides(self, tmp_path):
        from hydra.core.override_parser.overrides_parser import OverridesParser

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
        parsed = OverridesParser.create().parse_overrides(tokens)
        output = next(token.value() for token in parsed if token.key_or_group == "output_jsonl_fpath")
        assert output == str(tmp_path / "out" / "rollouts.jsonl")


@pytest.mark.parametrize(
    "suffix",
    [
        "ENV PATH=/opt/bin:$PATH",
        "ENV ROOT=/app\nWORKDIR $ROOT",
        "USER $RUNNER",
        "WORKDIR /app\nWORKDIR child",
        'ENV A "x y"',
        "ENV A=ok broken",
        'ENV A="unterminated',
        'WORKDIR "/quoted"',
        "USER",
        "ENV",
        "ENV A=foo\\ bar",
        "ENV A=<<EOF\nx\nEOF",
    ],
)
def test_dockerfile_shortcut_refuses_uninterpreted_syntax(suffix):
    assert base_image_only(f"FROM ubuntu\n{suffix}\n") is None


def test_prebuilt_image_ignores_all_dockerfile_settings(tmp_path):
    toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "prebuilt:1"\ncpus = 1')
    task = load_task(write_task(tmp_path / "t", toml=toml, dockerfile="FROM other\nENV A=1\nWORKDIR /x\nUSER ghost"))
    assert (task.image, task.workdir, task.env, task.user) == ("prebuilt:1", None, {}, None)


@pytest.mark.parametrize("failure", ["encoding", "escape", "io"])
def test_discovery_keeps_file_errors_inside_task_boundary(tmp_path, monkeypatch, failure):
    write_task(tmp_path / "ds" / "good")
    bad = write_task(tmp_path / "ds" / "bad")
    if failure == "encoding":
        (bad / "environment" / "Dockerfile").write_bytes(b"\xff")
    elif failure == "escape":
        marker = tmp_path / "marker"
        marker.write_text("private marker")
        (bad / "tests" / "leak").symlink_to(marker)
    else:
        original = Path.open

        def fail_open(path, *args, **kwargs):
            if path == bad / "tests" / "test.sh":
                raise PermissionError("test read denied")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_open)
    skipped = {}
    assert [task.task_id for task in discover_tasks(tmp_path / "ds", skipped=skipped)] == ["good"]
    assert list(skipped) == ["bad"]
    assert isinstance(skipped["bad"], HarborTaskError)


@pytest.mark.parametrize("section", ["environment", "agent", "verifier", "verifier.environment"])
@pytest.mark.parametrize(
    "policy", ['network_mode = "no-network"', 'network_mode = "allowlist"', 'allowed_hosts = ["example.org"]']
)
def test_loader_rejects_unenforced_network_policies(tmp_path, section, policy):
    with pytest.raises(HarborTaskError, match="network.*not supported"):
        load_task(write_task(tmp_path / "t", toml=f"[{section}]\n{policy}\n"))


@pytest.mark.parametrize(
    "toml",
    [
        "[environment]\nallow_internet = false",
        '[environment]\nos = "Windows"',
        '[environment.tpu]\ntype = "v6e"\ntopology = "1x1"',
    ],
)
def test_loader_rejects_unsupported_execution_requirements(tmp_path, toml):
    with pytest.raises(HarborTaskError, match="not supported"):
        load_task(write_task(tmp_path / "t", toml=toml))


def test_schema_matches_harbor_legacy_and_resource_fields(tmp_path):
    config = HarborTaskConfig.model_validate(
        {"environment": {"memory": 4096, "storage": 2048, "cpus": 0, "os": "Linux", "image": "stale"}}
    )
    assert config.environment.memory_mb is None and config.environment.storage_mb is None
    assert config.environment.cpus == 0 and config.environment.os == "linux"
    assert config.environment.docker_image is None
    task = load_task(write_task(tmp_path / "t", toml='[environment]\nimage = "stale"'))
    assert task.image == "ubuntu:24.04"


@pytest.fixture
def registry_repo(tmp_path):
    repo = tmp_path / "repo"
    write_task(repo / "tasks" / "a")
    write_task(repo / "tasks" / "b")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    commit_repo(repo)
    return repo


def commit_repo(repo: Path) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "fixture"], check=True
    )
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def registry_dataset(repo: Path, *, name: str = "a", path: str = "tasks/a") -> RegistryDataset:
    return RegistryDataset("ds", "1.0", "", (RegistryTask(name, repo.as_uri(), "HEAD", path),))


@pytest.mark.parametrize("name", ["../../escaped", "/absolute", ".", "..", "a/b", "a\\b", "manifest.toml"])
def test_fetch_rejects_unsafe_destination_names(tmp_path, name):
    with pytest.raises(HubError, match="Unsafe"):
        fetch_dataset(registry_dataset(tmp_path / "unused", name=name), tmp_path / "datasets")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("path", ["../outside", "/absolute", "tasks/../../outside", "--skip-checks"])
def test_fetch_rejects_unsafe_source_paths(tmp_path, path):
    with pytest.raises(HubError, match="Unsafe"):
        fetch_dataset(registry_dataset(tmp_path / "unused", path=path), tmp_path / "datasets")


@pytest.mark.parametrize("link_kind", ["absolute", "relative_escape", "source_root", "directory"])
def test_fetch_rejects_source_symlinks_before_copying(tmp_path, registry_repo, monkeypatch, link_kind):
    marker = tmp_path / "marker"
    marker.write_text("private marker")
    task = registry_repo / "tasks" / "a"
    source_path = "tasks/a"
    if link_kind == "source_root":
        (registry_repo / "alias").symlink_to("tasks/a", target_is_directory=True)
        source_path = "alias"
    elif link_kind == "relative_escape":
        (registry_repo / "private").write_text("private marker")
        (task / "tests" / "leak").symlink_to("../../../private")
    elif link_kind == "directory":
        (tmp_path / "private-dir").mkdir()
        (tmp_path / "private-dir" / "marker").write_text("private marker")
        (task / "tests" / "leak").symlink_to(tmp_path / "private-dir", target_is_directory=True)
    else:
        (task / "tests" / "leak").symlink_to(marker)
    commit_repo(registry_repo)
    copied = []
    original = hub_module.shutil.copytree

    def record_copy(*args, **kwargs):
        copied.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(hub_module.shutil, "copytree", record_copy)
    with pytest.raises(HubError, match="symlink"):
        fetch_dataset(registry_dataset(registry_repo, path=source_path), tmp_path / "datasets")
    assert copied == []
    assert marker.read_text() == "private marker"


def test_fetch_preserves_internal_relative_symlinks(tmp_path, registry_repo):
    (registry_repo / "tasks" / "a" / "tests" / "link").symlink_to("test.sh")
    commit_repo(registry_repo)
    folder = fetch_dataset(registry_dataset(registry_repo), tmp_path / "datasets")
    assert (folder / "a" / "tests" / "link").is_symlink()
    assert (folder / "a" / "tests" / "link").read_bytes() == (folder / "a" / "tests" / "test.sh").read_bytes()


@pytest.mark.parametrize("location", ["dataset", "task", "manifest"])
def test_fetch_refuses_symlinked_destination(tmp_path, registry_repo, location):
    root = tmp_path / "datasets"
    root.mkdir()
    outside = write_task(tmp_path / "outside")
    folder = root / "ds-1.0"
    if location == "dataset":
        folder.symlink_to(outside, target_is_directory=True)
    else:
        folder.mkdir()
        if location == "task":
            (folder / "a").symlink_to(outside, target_is_directory=True)
        else:
            (folder / "manifest.toml").symlink_to(outside / "task.toml")
    before = (outside / "task.toml").read_bytes()
    with pytest.raises(HubError):
        fetch_dataset(registry_dataset(registry_repo), root)
    assert (outside / "task.toml").read_bytes() == before
    assert not (outside / "a").exists()


def test_interrupted_fetch_preserves_pins_when_head_moves(tmp_path, registry_repo, monkeypatch):
    old_head = subprocess.check_output(["git", "-C", str(registry_repo), "rev-parse", "HEAD"], text=True).strip()
    dataset = RegistryDataset(
        "ds",
        "1.0",
        "",
        tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b")),
    )
    original = hub_module.shutil.copytree

    def interrupt_second(source, *args, **kwargs):
        if source.name == "b":
            raise OSError("interrupted download")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(hub_module.shutil, "copytree", interrupt_second)
    with pytest.raises(OSError, match="interrupted"):
        fetch_dataset(dataset, tmp_path / "datasets")
    monkeypatch.setattr(hub_module.shutil, "copytree", original)
    for name in ("a", "b"):
        (registry_repo / "tasks" / name / "instruction.md").write_text("new contents")
    new_head = commit_repo(registry_repo)
    folder = fetch_dataset(dataset, tmp_path / "datasets")
    pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
    assert pins["a"]["git_commit_id"] == old_head
    assert pins["b"]["git_commit_id"] == new_head
    assert (folder / "a" / "instruction.md").read_text().startswith("Create a file")
    assert (folder / "b" / "instruction.md").read_text() == "new contents"


def test_existing_unpinned_task_is_never_relabeled(tmp_path, registry_repo):
    existing = write_task(tmp_path / "datasets" / "ds-1.0" / "a")
    (existing / "instruction.md").write_text("local contents")
    with pytest.raises(HubError, match="pin"):
        fetch_dataset(registry_dataset(registry_repo), tmp_path / "datasets")
    assert (existing / "instruction.md").read_text() == "local contents"
    assert not (existing.parent / "manifest.toml").exists()


def test_registry_versions_keep_distinct_rows_routes_and_outputs(tmp_path, monkeypatch):
    from hydra.core.override_parser.overrides_parser import OverridesParser

    import nemo_gym.tasks.harbor.cli as harbor_cli

    for version in ("1.0", "2.0"):
        write_task(tmp_path / f"ds-{version}" / "a")
    monkeypatch.setattr(harbor_cli, "fetch_ref", lambda target, **kw: tmp_path / f"ds-{target.split('@')[1]}")
    first = prepare_target("harbor:ds@1.0", output_root=tmp_path / "out")
    first_rows = first.rows_path.read_bytes()
    second = prepare_target("harbor:ds@2.0", output_root=tmp_path / "out")
    assert first.rows_path != second.rows_path and first.taskset != second.taskset
    assert first.rows_path.read_bytes() == first_rows
    agent = resolve_agent("hermes")
    configs = [build_run(prepared, agent, sandbox=None, overrides=[]) for prepared in (first, second)]
    assert configs[0][0] != configs[1][0]
    for prepared, (path, tokens) in zip((first, second), configs):
        assert json.loads(prepared.rows_path.read_text())["task_id"]["taskset"] == prepared.taskset
        assert list(yaml.safe_load(path.read_text())["environment_server_routes"]) == [prepared.taskset]
        parsed = OverridesParser.create().parse_overrides(tokens)
        output = next(token.value() for token in parsed if token.key_or_group == "output_jsonl_fpath")
        assert output == str(prepared.output_dir / "rollouts.jsonl")


def test_config_overlays_win_after_real_loading_and_inheritance(tmp_path):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    from nemo_gym.cli.main import _merge_config_paths
    from nemo_gym.global_config import GlobalConfigDictParser

    task = write_task(tmp_path / "ds" / "a")
    prepared = prepare_target(str(task.parent), output_root=tmp_path / "out")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "hermes_agent:\n  responses_api_agents:\n    hermes_agent:\n      max_turns: 71\nsingle_agent_turn:\n  environment_servers:\n    single_agent_turn:\n      default_episode_timeout_seconds: 1234\n"
    )
    _, tokens = build_run(prepared, resolve_agent("hermes"), sandbox=None, overrides=[f"+config_paths=[{overlay}]"])
    paths_token = next(t for t in _merge_config_paths(tokens) if t.startswith("+config_paths="))
    paths = OverridesParser.create().parse_override(paths_token).value()
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths(paths)
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    assert config.harbor_ds_agent.responses_api_agents.hermes_agent.max_turns == 71
    assert config.harbor_ds_environment.environment_servers.single_agent_turn.default_episode_timeout_seconds == 1234


@pytest.mark.parametrize("skip", [False, True])
def test_partial_eval_requires_explicit_opt_in_and_records_reasons(tmp_path, monkeypatch, skip):
    import sys

    import nemo_gym.cli.main as main_module

    write_task(tmp_path / "ds" / "good")
    write_task(tmp_path / "ds" / "bad", toml="bad toml")
    monkeypatch.chdir(tmp_path)
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        ["gym", "eval", "run", str(tmp_path / "ds"), "--agent", "hermes", *(["--skip-unloadable"] if skip else [])],
    )
    if skip:
        main_module.main()
        assert len(dispatched) == 1
    else:
        with pytest.raises(SystemExit) as exc:
            main_module.main()
        assert exc.value.code != 0
        assert not dispatched
    skipped = json.loads((tmp_path / "results" / "harbor" / "ds" / "skipped.json").read_text())
    assert list(skipped) == ["bad"] and "task.toml" in skipped["bad"]


def test_manifest_publication_failure_keeps_previous_pins(tmp_path, registry_repo, monkeypatch):
    folder = fetch_dataset(registry_dataset(registry_repo), tmp_path / "datasets")
    before = (folder / "manifest.toml").read_bytes()
    dataset = RegistryDataset(
        "ds",
        "1.0",
        "",
        tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b")),
    )
    original = Path.replace

    def fail_publication(path, target):
        if Path(target) == folder / "manifest.toml":
            raise OSError("interrupted manifest publication")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="interrupted manifest"):
        fetch_dataset(dataset, tmp_path / "datasets")
    assert (folder / "manifest.toml").read_bytes() == before
    assert (folder / "b" / "task.toml").is_file()
    assert not list(folder.glob(".manifest-*"))
    monkeypatch.setattr(Path, "replace", original)
    (registry_repo / "tasks" / "b" / "instruction.md").write_text("new contents")
    commit_repo(registry_repo)
    with pytest.raises(HubError, match="no trustworthy source pin"):
        fetch_dataset(dataset, tmp_path / "datasets")
    assert (folder / "manifest.toml").read_bytes() == before
    assert (folder / "b" / "instruction.md").read_text().startswith("Create a file")


def test_all_unloadable_tasks_still_record_reasons(tmp_path):
    write_task(tmp_path / "ds" / "bad", toml="bad toml")
    with pytest.raises(HarborTaskError, match="No task under"):
        prepare_target(str(tmp_path / "ds"), output_root=tmp_path / "out")
    skipped = json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text())
    assert list(skipped) == ["bad"]
    assert "task.toml" in skipped["bad"]


@pytest.mark.parametrize("taskset", ["ds", "terminal-bench-2.0", "trial=1", "中文=1"])
def test_generated_run_passes_full_config_validation(tmp_path, monkeypatch, taskset):
    import sys

    import nemo_gym.global_config as global_config
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths
    from nemo_gym.global_config import GlobalConfigDictParser, get_global_config_dict
    from nemo_gym.rollout_collection import E2ERolloutCollectionConfig
    from nemo_gym.tasks.harbor.materialize import resources_server_instance

    write_task(tmp_path / taskset / "a")
    prepared = prepare_target(str(tmp_path / taskset), output_root=tmp_path / "out")
    _, tokens = build_run(
        prepared,
        resolve_agent("hermes"),
        sandbox="opensandbox",
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'}]",
            "+policy_model_name=test-model",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    config = get_global_config_dict()
    servers = GlobalConfigDictParser().filter_for_server_instance_configs(config)
    resources = next(server for server in servers if server.name == resources_server_instance(taskset))
    assert resources.get_inner_run_server_config().domain is not None
    assert resources.datasets[0].name == taskset
    run = E2ERolloutCollectionConfig.model_validate(config)
    assert run.environment_routing_mode == "taskset"
    assert taskset in run.environment_server_routes


def test_interrupted_cache_repair_preserves_pending_pins(tmp_path, registry_repo, monkeypatch):
    import shutil

    write_task(registry_repo / "tasks" / "c")
    old_head = commit_repo(registry_repo)
    dataset = RegistryDataset(
        "ds",
        "1.0",
        "",
        tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b", "c")),
    )
    folder = fetch_dataset(dataset, tmp_path / "datasets")
    for name in ("b", "c"):
        shutil.rmtree(folder / name)
        (registry_repo / "tasks" / name / "instruction.md").write_text("new upstream instruction")
    assert commit_repo(registry_repo) != old_head
    original = hub_module.shutil.copytree

    def interrupt_last(source, *args, **kwargs):
        if source.name == "c":
            raise OSError("interrupted cache repair")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(hub_module.shutil, "copytree", interrupt_last)
    with pytest.raises(OSError, match="interrupted cache repair"):
        fetch_dataset(dataset, tmp_path / "datasets")
    pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
    assert pins["c"]["git_commit_id"] == old_head
    monkeypatch.setattr(hub_module.shutil, "copytree", original)
    fetch_dataset(dataset, tmp_path / "datasets")
    pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
    assert {pin["git_commit_id"] for pin in pins.values()} == {old_head}
    assert (folder / "c" / "instruction.md").read_text().startswith("Create a file")


@pytest.mark.parametrize("external", [False, True])
def test_symlinked_child_tasks_are_reported_without_duplicate_rows(tmp_path, external):
    folder = tmp_path / "ds"
    task = write_task(folder / "a")
    target = write_task(tmp_path / "elsewhere" / "renamed") if external else task
    (folder / "b").symlink_to(target, target_is_directory=True)
    with pytest.raises(HarborTaskError, match="symlink"):
        discover_tasks(folder)
    with pytest.raises(HarborTaskError, match="symlink"):
        load_task(folder / "b")
    prepared = prepare_target(str(folder), output_root=tmp_path / "out")
    assert [row["task_id"]["task_id"] for row in map(json.loads, prepared.rows_path.read_text().splitlines())] == ["a"]
    assert "symlink" in prepared.skipped["b"]
    assert "symlink" in json.loads((prepared.output_dir / "skipped.json").read_text())["b"]


def test_duplicate_task_ids_are_rejected_before_rows_or_mapping_are_written(tmp_path):
    from nemo_gym.tasks.harbor.materialize import taskset_mapping

    task = load_task(write_task(tmp_path / "ds" / "a"))
    rows = tmp_path / "tasks.jsonl"
    rows.write_text("previous valid rows\n")
    with pytest.raises(HarborTaskError, match="Duplicate.*a"):
        write_rows([task, task], "ds", rows)
    assert rows.read_text() == "previous valid rows\n"
    with pytest.raises(HarborTaskError, match="Duplicate.*a"):
        taskset_mapping([task, task], task.path.parent)


def test_generated_instance_names_are_override_safe_and_collision_free():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from nemo_gym.tasks.harbor.materialize import (
        agent_instance,
        environment_server_instance,
        resources_server_instance,
    )

    tasksets = ["terminal-bench-2.0", "terminal-bench-2_0", "terminal-bench-2_2e0", "ds-1.0", "ds-2.0", "中文", "a b"]
    parser = OverridesParser.create()
    for instance in (agent_instance, environment_server_instance, resources_server_instance):
        names = [instance(taskset) for taskset in tasksets]
        assert len(set(names)) == len(tasksets)
        for name in names:
            assert "." not in name
            override = parser.parse_override(f"++{name}.setting=true")
            assert override.key_or_group == f"{name}.setting"


def test_generated_agent_can_be_overridden_for_versioned_taskset(tmp_path, monkeypatch):
    import sys

    import nemo_gym.global_config as global_config
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths
    from nemo_gym.global_config import get_global_config_dict
    from nemo_gym.tasks.harbor.materialize import agent_instance

    folder = tmp_path / "ds-2.0"
    write_task(folder / "a")
    prepared = prepare_target(str(folder), output_root=tmp_path / "out")
    name = agent_instance(prepared.taskset)
    _, tokens = build_run(
        prepared,
        resolve_agent("hermes"),
        sandbox=None,
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'}]",
            "+policy_model_name=test-model",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
            f"++{name}.responses_api_agents.hermes_agent.chat_template_kwargs_enabled=true",
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    config = get_global_config_dict()
    assert config[name].responses_api_agents.hermes_agent.chat_template_kwargs_enabled is True
    assert list(config.environment_server_routes) == ["ds-2.0"]


@pytest.mark.parametrize(
    ("target", "flags", "message"),
    [
        ("missing-folder", ["--agent", "hermes"], "is not a directory"),
        ("harbor:not-present", ["--agent", "hermes"], "not in the Harbor registry"),
        ("harbor:hello-world", [], "needs `--agent"),
        ("harbor:hello-world", ["--no-serve", "--agent", "hermes"], "drop --no-serve"),
    ],
)
def test_expected_harbor_cli_errors_are_concise(tmp_path, monkeypatch, capsys, target, flags, message):
    import sys

    from nemo_gym.cli.main import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hub_module, "load_registry", lambda *args, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", target, *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert message in error and "Traceback" not in error


@pytest.mark.parametrize("flags", [["--agent", "not-an-agent"], ["--agent", "hermes", "--sandbox", "not-a-sandbox"]])
def test_harbor_selectors_fail_before_any_preparation(monkeypatch, capsys, flags):
    import sys

    import nemo_gym.tasks.harbor.cli as harbor_cli
    from nemo_gym.cli.main import main

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Invalid selectors must be rejected before preparing or fetching tasks")

    monkeypatch.setattr(harbor_cli, "prepare_target", unexpected_preparation)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "harbor:hello-world", *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "No " in capsys.readouterr().err


def test_unexpected_harbor_programming_errors_remain_visible(monkeypatch):
    import sys

    import nemo_gym.tasks.harbor.cli as harbor_cli
    from nemo_gym.cli.main import main

    def bug(*args, **kwargs):
        raise TypeError("unexpected implementation bug")

    monkeypatch.setattr(harbor_cli, "prepare_target", bug)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "harbor:hello-world", "--agent", "hermes"])
    with pytest.raises(TypeError, match="unexpected implementation bug"):
        main()


@pytest.mark.parametrize(
    "payload",
    [
        b"[{",
        b"\xff",
        b"{}",
        b"[null]",
        b"[{}]",
        b'[{"name": "ds", "tasks": [{}]}]',
        b'[{"name": "ds", "tasks": null}]',
        b'[{"name": "ds", "tasks": [null]}]',
        b'[{"name": 1}]',
        b'[{"name": "ds", "tasks": [{"name": "a", "git_url": null}]}]',
    ],
)
def test_damaged_registry_cache_reports_recovery_at_cli(tmp_path, monkeypatch, capsys, payload):
    import sys

    from nemo_gym.cli.main import main

    root = tmp_path / "datasets"
    cache = root / ".harbor" / "registry.json"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(payload)
    monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(root))
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "harbor:ds", "--agent", "hermes"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert str(cache) in message
    assert "Remove this file and retry" in message
    assert "Traceback" not in message
    assert cache.read_bytes() == payload


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("payload", [b"[{", b'[{"tasks": []}]'])
def test_invalid_registry_download_does_not_publish(tmp_path, monkeypatch, existing, payload):
    from io import BytesIO

    cache = tmp_path / "registry.json"
    previous = b'[{"name": "previous"}]'
    if existing:
        cache.write_bytes(previous)
    monkeypatch.setattr(hub_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    with pytest.raises(HubError, match="registry"):
        load_registry(tmp_path, refresh=True)
    assert cache.read_bytes() == previous if existing else not cache.exists()
    assert not list(tmp_path.glob(".registry-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_registry_publication_is_atomic_and_recovers(tmp_path, monkeypatch, existing):
    from io import BytesIO

    cache = tmp_path / "registry.json"
    previous = b'[{"name": "previous"}]'
    payload = b'[{"name": "replacement"}]'
    if existing:
        cache.write_bytes(previous)
    monkeypatch.setattr(hub_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    replace = Path.replace
    attempted = []

    def interrupted(path, target):
        if Path(target) == cache:
            assert path.parent == cache.parent
            assert path.read_bytes() == payload
            assert cache.read_bytes() == previous if existing else not cache.exists()
            attempted.append(path)
            raise OSError("interrupted registry publication")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(HubError, match="interrupted registry publication"):
        load_registry(tmp_path, refresh=True)
    assert attempted
    assert cache.read_bytes() == previous if existing else not cache.exists()
    assert not list(tmp_path.glob(".registry-*"))
    monkeypatch.setattr(Path, "replace", replace)
    assert load_registry(tmp_path, refresh=True)[0].name == "replacement"
    assert cache.read_bytes() == payload
    assert not list(tmp_path.glob(".registry-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_registry_cache_write_preserves_previous_file(tmp_path, monkeypatch, existing):
    from io import BytesIO

    cache = tmp_path / "registry.json"
    previous = b'[{"name": "previous"}]'
    payload = b'[{"name": "replacement"}]'
    if existing:
        cache.write_bytes(previous)
    monkeypatch.setattr(hub_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    create = hub_module.tempfile.NamedTemporaryFile
    attempted = []

    def interrupted_file(*args, **kwargs):
        stream = create(*args, **kwargs)
        write = stream.write

        def partial_write(data):
            write(data[:5])
            stream.flush()
            attempted.append(Path(stream.name))
            raise OSError("interrupted registry write")

        stream.write = partial_write
        return stream

    monkeypatch.setattr(hub_module.tempfile, "NamedTemporaryFile", interrupted_file)
    with pytest.raises(HubError, match="interrupted registry write"):
        load_registry(tmp_path, refresh=True)
    assert attempted
    assert cache.read_bytes() == previous if existing else not cache.exists()
    assert not list(tmp_path.glob(".registry-*"))
