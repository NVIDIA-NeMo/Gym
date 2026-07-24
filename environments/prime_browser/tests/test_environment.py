# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from omegaconf import OmegaConf

from environments.prime_browser import prepare
from responses_api_agents.harbor_agent.custom_agents.browser_terminus_2_nemo_gym import (
    BrowserTerminus2NemoGym,
)
from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import (
    Terminus2NemoGym,
)


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
        agent["harbor_datasets"]["browser_all_opensandbox"]["local_dataset_path"]
        == "environments/prime_browser/data/tasks"
    )
    assert agent["harbor_agent_import_path"].endswith(":BrowserTerminus2NemoGym")
    assert agent["datasets"][0]["license"] == "TBD"


def test_browser_agent_and_runtime_assets():
    assert issubclass(BrowserTerminus2NemoGym, Terminus2NemoGym)
    assert BrowserTerminus2NemoGym.name() == "browser-terminus-2-nemo-gym"
    assert (ENV_ROOT / "browser_tools" / "browser_open").is_file()
    assert (ENV_ROOT / "sandbox_setup" / "setup_opensandbox.sh").is_file()


def test_prepare_materializes_manifest_tasks_without_symlinks(
    tmp_path,
):
    bundle = tmp_path / "bundle"
    task = bundle / "tasks" / "task-one"
    shared_runtime = bundle / "shared-runtime"
    shared_runtime.mkdir(parents=True)
    (shared_runtime / "site.txt").write_text("site\n", encoding="utf-8")
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "runtime").symlink_to(
        shared_runtime,
        target_is_directory=True,
    )
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "old-image"\n',
        encoding="utf-8",
    )
    launcher = bundle / "scripts" / "start_local_sims.py"
    launcher.parent.mkdir()
    launcher.write_text("print('start')\n", encoding="utf-8")
    manifest = tmp_path / "validation.jsonl"
    output = tmp_path / "output"

    assert prepare.prepare(bundle, output, manifest, expected_tasks=1) == 1
    environment = output / "task-one" / "environment"
    assert not (environment / "runtime").is_symlink()
    assert (environment / "runtime" / "site.txt").read_text(encoding="utf-8") == "site\n"
    assert (environment / "setup_opensandbox.sh").is_file()
    assert (environment / "browser_open").is_file()
    assert (environment / "start_local_sims.py").is_file()
    assert _rows(manifest)[0]["instance_id"] == "browser_all_opensandbox::task-one"
    assert _rows(manifest)[0]["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "harbor_agent",
    }
