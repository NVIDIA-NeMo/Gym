# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from omegaconf import OmegaConf

from environments.prime_terminal import prepare


ENV_ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_config(monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "example.invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    config = OmegaConf.to_container(
        OmegaConf.load(ENV_ROOT / "config.yaml"),
        resolve=True,
    )
    agent = config["harbor_agent"]["responses_api_agents"]["harbor_agent"]

    assert (
        agent["harbor_datasets"]["terminal_all_opensandbox"]["local_dataset_path"]
        == "environments/prime_terminal/data/tasks"
    )
    assert agent["harbor_environment_import_path"].endswith(":UploadedNemoGymSandboxEnvironment")
    assert agent["datasets"][0]["license"] == "TBD"


def test_terminal_bootstrap_restores_symlink_parent(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:24.04\nRUN ln -sfn -- ../vendor/source /app/fixture/EMPTY_PARENT/src\n",
        encoding="utf-8",
    )

    image, script = prepare.terminal_bootstrap(dockerfile)

    assert image == "ubuntu:24.04"
    assert (
        "mkdir -p /app/fixture/EMPTY_PARENT && ln -sfn -- ../vendor/source /app/fixture/EMPTY_PARENT/src"
    ) in script


def test_prepare_materializes_tasks_and_writes_manifest(tmp_path):
    source = tmp_path / "source"
    task = source / "task-one"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.13-slim-bookworm\nRUN apt-get update\n",
        encoding="utf-8",
    )
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "old-image"\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "validation.jsonl"
    output = tmp_path / "output"

    assert prepare.prepare(source, output, manifest, expected_tasks=1) == 1
    assert 'docker_image = "python:3.13-slim-bookworm"' in (output / "task-one" / "task.toml").read_text(
        encoding="utf-8"
    )
    setup = output / "task-one" / "environment" / "setup_opensandbox.sh"
    assert setup.is_file()
    assert setup.stat().st_mode & 0o111
    assert _rows(manifest) == [
        {
            "instance_id": "terminal_all_opensandbox::task-one",
            "responses_create_params": {
                "input": [],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_output_tokens": 32768,
            },
            "agent_ref": {
                "type": "responses_api_agents",
                "name": "harbor_agent",
            },
        }
    ]
