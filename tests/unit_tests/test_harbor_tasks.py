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
from nemo_gym.tasks.harbor.cli import (
    AgentSelection,
    PreparedTaskset,
    build_run,
    prepare_target,
    resolve_agent,
    runs_in_sandbox,
)
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
        assert config.environment.docker_image == "img"
        assert "dockerfile" not in config.environment.model_dump()
        assert HarborTaskConfig.unknown_keys(data) == [
            "allowlist",
            "environment.dockerfile",
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
        assert config["harbor_ds_resources_server"]["_inherit_from"] == "harbor_resources_server"
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
        selection = resolve_agent("hermes_agent")
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
        assert "resources_servers/harbor/configs/harbor.yaml" in config_paths
        assert "opensandbox/configs/opensandbox.yaml" in config_paths
        assert f"+output_jsonl_fpath={tmp_path / 'out' / 'rollouts.jsonl'}" in tokens
        assert not any(token.lstrip("+").startswith("use_absolute_ip=") for token in tokens)

    def test_resolve_agent_marks_in_sandbox_harnesses(self):
        assert resolve_agent("terminus_2_sandboxed").runs_in_sandbox is True
        assert resolve_agent("hermes_agent").runs_in_sandbox is True
        assert resolve_agent("oracle").runs_in_sandbox is False

    def test_runs_in_sandbox_from_config_key_or_known_harness(self):
        assert runs_in_sandbox("anyswe_agent", {"sandbox_model_base_url": None}) is True
        assert runs_in_sandbox("anyswe_agent", {"model_server": {"name": "policy_model"}}) is False
        assert runs_in_sandbox("miniswe_sandboxed_agent", {}) is True
        assert runs_in_sandbox("oracle_agent", None) is False

    def _prepared(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a"))]
        prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
        prepared.output_dir.mkdir(parents=True)
        return prepared

    def test_build_run_advertises_node_ip_for_in_sandbox_agent(self, tmp_path, caplog):
        agent = AgentSelection(
            tmp_path / "agent.yaml", "terminus_2_sandboxed_agent", "terminus_2_sandboxed_agent", runs_in_sandbox=True
        )
        with caplog.at_level(logging.INFO, logger="nemo_gym.tasks.harbor.cli"):
            _, tokens = build_run(self._prepared(tmp_path), agent, sandbox="opensandbox", overrides=[])
        assert tokens.count("+use_absolute_ip=true") == 1
        messages = [record.getMessage() for record in caplog.records]
        assert any("use_absolute_ip=true" in m and "terminus_2_sandboxed_agent" in m for m in messages)

    def test_build_run_leaves_host_side_agent_on_loopback(self, tmp_path):
        agent = AgentSelection(tmp_path / "agent.yaml", "oracle_agent", "oracle_agent")
        _, tokens = build_run(self._prepared(tmp_path), agent, sandbox="docker", overrides=[])
        assert not any("use_absolute_ip" in token for token in tokens)

    @pytest.mark.parametrize("override", ["+use_absolute_ip=false", "++use_absolute_ip=false"])
    def test_build_run_respects_caller_use_absolute_ip(self, tmp_path, override):
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent", runs_in_sandbox=True)
        _, tokens = build_run(self._prepared(tmp_path), agent, sandbox=None, overrides=[override])
        assert [token for token in tokens if "use_absolute_ip" in token] == [override]


class TestValidationSummary:
    def test_summarizes_rollouts(self, tmp_path):
        from nemo_gym.tasks.harbor.cli import summarize_validation

        # Rows are the verify response as returned plus the collector's `_ng_task_id` stamp.
        rows = [
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "solved"},
                "reward": 1.0,
                "mask_sample": False,
                "response": {"metadata": {"oracle": "solved"}},
            },
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "wrong"},
                "reward": 0.0,
                "mask_sample": False,
                "failure_kind": "harbor:missing_reward",
                "response": {"metadata": {"oracle": "solved"}},
            },
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "skipped"},
                "reward": 0.0,
                "mask_sample": False,
                "response": {"metadata": {"oracle": "unvalidated"}},
            },
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "masked"},
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "provider_unavailable",
            },
        ]
        rollouts = tmp_path / "rollouts.jsonl"
        rollouts.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        # An episode that failed before verification lands in the failures sidecar.
        (tmp_path / "rollouts_failures.jsonl").write_text(
            json.dumps(
                {
                    "_ng_task_id": {"taskset": "ds", "task_id": "broken"},
                    "_ng_failure_class": "environment_server_failed",
                    "_ng_failure_terminal": True,
                    "_ng_failure_message": "seed exploded",
                }
            )
            + "\n"
        )

        report = summarize_validation(rollouts, ["never_ran"])

        lines = report.text.splitlines()
        assert lines[0] == "task_id\tstatus\treward"
        assert "solved\toracle solved\t1.0" in lines
        assert "wrong\toracle solved (harbor:missing_reward)\t0.0" in lines
        assert "skipped\tunvalidated (no solution/)\t-" in lines
        assert "broken\tfailed: seed exploded\t-" in lines
        assert "masked\tmasked (provider_unavailable)\t-" in lines
        assert "never_ran\tunvalidated (no solution/)\t-" in lines
        assert report.ok is False

    def test_all_good_is_ok(self, tmp_path):
        from nemo_gym.tasks.harbor.cli import summarize_validation

        rollouts = tmp_path / "rollouts.jsonl"
        rollouts.write_text(json.dumps({"_ng_task_id": {"taskset": "ds", "task_id": "a"}, "reward": 1.0}) + "\n")
        report = summarize_validation(rollouts, [])
        assert report.ok is True and "a\toracle ran\t1.0" in report.text

    def test_missing_rollouts_file(self, tmp_path):
        from nemo_gym.tasks.harbor.cli import summarize_validation

        report = summarize_validation(tmp_path / "none.jsonl", [])
        assert report.ok is False and "no rollouts written" in report.text


def test_harbor_task_data_schema_names_the_digest_key():
    """The schema module may import only pydantic, so the key is spelled out; keep it equal to DIGEST_KEY."""
    from resources_servers.harbor.task_data import TaskData

    assert TaskData.model_fields["ng_digest"].alias == DIGEST_KEY
    data = TaskData.model_validate({DIGEST_KEY: "abc", "other": 1})
    assert data.ng_digest == "abc" and data.model_dump(by_alias=True)[DIGEST_KEY] == "abc"
