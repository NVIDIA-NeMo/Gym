# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from omegaconf import OmegaConf

from environments.prime_swe import prepare


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

    assert agent["harbor_datasets"]["swe_all_opensandbox"]["local_dataset_path"] == "environments/prime_swe/data/tasks"
    assert agent["harbor_environment_import_path"].endswith(":UploadedNemoGymSandboxEnvironment")
    assert agent["datasets"][0]["license"] == "TBD"


def test_setup_installs_build_backends():
    setup = (ENV_ROOT / "sandbox_setup" / "setup_opensandbox.sh").read_text(encoding="utf-8")
    assert "'flit_core>=3.12,<4'" in setup
    assert "'setuptools>=77'" in setup
    assert "'setuptools-scm>=8'" in setup


def test_prepare_materializes_tasks_and_writes_manifest(tmp_path):
    source = tmp_path / "source"
    task = source / "task-one"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "old-image"\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "validation.jsonl"
    output = tmp_path / "output"

    assert prepare.prepare(source, output, manifest, expected_tasks=1) == 1
    assert 'docker_image = "python:3.12-slim-bookworm"' in (output / "task-one" / "task.toml").read_text(
        encoding="utf-8"
    )
    assert (output / "task-one" / "environment" / "setup_opensandbox.sh").is_file()
    assert _rows(manifest)[0]["instance_id"] == "swe_all_opensandbox::task-one"
    assert _rows(manifest)[0]["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "harbor_agent",
    }
