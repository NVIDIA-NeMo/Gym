# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from resources_servers.cvdp.app import CVDPResourcesServer, CVDPResourcesServerConfig
from resources_servers.cvdp.prepare import _write_context, prepare_images
from resources_servers.cvdp.testbench_runner import TestbenchRunner as CVDPTestbenchRunner


def _config() -> CVDPResourcesServerConfig:
    return CVDPResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        name="cvdp",
        entrypoint="app.py",
        oss_pnr_image="registry.example/cvdp-pnr:base",
    )


def test_default_synthesis_image_follows_configured_simulator() -> None:
    config = CVDPResourcesServerConfig(
        host="0.0.0.0", port=0, name="cvdp", entrypoint="app.py", oss_sim_image="/images/sim.sif"
    )
    assert config.oss_pnr_image == "/images/sim.sif"


def test_composed_synthesis_image_follows_simulator_override() -> None:
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/cvdp.yaml")
    config.cvdp.resources_servers.cvdp.oss_sim_image = "/images/overridden.sif"
    assert config.cvdp.resources_servers.cvdp.oss_pnr_image == "/images/overridden.sif"


def test_write_context_pins_python39_get_pip(tmp_path) -> None:
    config = _config().model_copy(update={"oss_pnr_image": "ghcr.io/hdl/impl/pnr"})
    _write_context(
        tmp_path,
        {
            "src/Dockerfile.synth": (
                "FROM __OSS_PNR_IMAGE__ AS base\nADD https://bootstrap.pypa.io/get-pip.py get-pip.py\n"
            )
        },
        config,
    )

    dockerfile = (tmp_path / "src/Dockerfile.synth").read_text()
    assert "https://bootstrap.pypa.io/pip/3.9/get-pip.py" in dockerfile


def test_prepare_builds_each_unique_recipe_once(tmp_path) -> None:
    compose = """
services:
  synth:
    build:
      dockerfile: src/Dockerfile.synth
    command: pytest /src/synth.py
"""
    dockerfile = "FROM __OSS_PNR_IMAGE__\nADD https://example.com/get-pip.py /tmp/get-pip.py\nRUN true\n"
    rows = []
    for test_name in ("first", "second"):
        rows.append(
            {
                "verifier_metadata": {
                    "harness_files": {
                        "docker-compose.yml": compose,
                        "src/Dockerfile.synth": dockerfile,
                        "src/synth.py": f"def test_{test_name}(): pass\n",
                    }
                }
            }
        )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "prepared.json"

    with patch("resources_servers.cvdp.prepare.subprocess.run") as run:
        images = prepare_images(
            [dataset],
            "registry.example/cvdp-verifier",
            manifest,
            _config(),
            push=True,
            force=False,
        )

    assert len(images) == 1
    assert run.call_count == 2
    build_command = run.call_args_list[0].args[0]
    assert build_command[:2] == ["docker", "build"]
    assert "registry.example/cvdp-verifier:" in build_command[build_command.index("-t") + 1]
    assert json.loads(manifest.read_text())["images"] == images


def test_force_rebuilds_duplicate_recipe_once(tmp_path) -> None:
    row = {
        "verifier_metadata": {
            "harness_files": {
                "docker-compose.yml": "services:\n  synth:\n    build: .\n    command: true\n",
                "Dockerfile": "FROM registry.example/base:latest\nRUN true\n",
            }
        }
    }
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with patch("resources_servers.cvdp.prepare.subprocess.run") as run:
        prepare_images(
            [dataset],
            "registry.example/cvdp-verifier",
            tmp_path / "prepared.json",
            _config(),
            push=False,
            force=True,
        )

    assert run.call_count == 1


def test_prepare_skips_dataset_without_build_services(tmp_path) -> None:
    row = {
        "verifier_metadata": {
            "harness_files": {
                "docker-compose.yml": "services:\n  direct:\n    image: ghcr.io/hdl/sim/osvb\n    command: true\n"
            }
        }
    }
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "prepared.json"

    with patch("resources_servers.cvdp.prepare.subprocess.run") as run:
        images = prepare_images(
            [dataset],
            "registry.example/cvdp-verifier",
            manifest,
            _config(),
            push=False,
            force=False,
        )

    assert images == {}
    run.assert_not_called()


def _build_dataset(tmp_path):
    row = {
        "verifier_metadata": {
            "harness_files": {
                "docker-compose.yml": "services:\n  direct:\n    build: .\n",
                "Dockerfile": "FROM registry.example/base:latest\nRUN true\n",
            }
        }
    }
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_check_prepared_images_rejects_missing_recipes_without_building(tmp_path) -> None:
    dataset = _build_dataset(tmp_path)
    with patch("resources_servers.cvdp.prepare.subprocess.run") as run:
        with pytest.raises(ValueError, match="1 missing prepared image mapping.*direct"):
            CVDPTestbenchRunner(_config()).check_prepared_images([dataset])
    run.assert_not_called()


def test_check_prepared_images_accepts_manifest_and_rejects_missing_local_image(tmp_path) -> None:
    dataset = _build_dataset(tmp_path)
    manifest = tmp_path / "images.json"
    with patch("resources_servers.cvdp.prepare.subprocess.run"):
        images = prepare_images([dataset], "registry.example/prepared", manifest, _config(), push=False, force=False)
    config = _config().model_copy(update={"prepared_image_manifest": str(manifest)})
    assert CVDPTestbenchRunner(config).check_prepared_images([dataset]) == set(images)
    config.prepared_image_manifest = ""
    config.prepared_images = dict.fromkeys(images, str(tmp_path / "missing.sif"))
    with pytest.raises(ValueError, match="verifier image does not exist"):
        CVDPTestbenchRunner(config).check_prepared_images([dataset])


def test_check_prepared_images_rejects_unconfigured_commercial_image(tmp_path) -> None:
    dataset = tmp_path / "commercial.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "verifier_metadata": {
                    "harness_files": {"docker-compose.yml": "services:\n  direct:\n    image: __VERIF_EDA_IMAGE__\n"}
                }
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="no configured image.*__VERIF_EDA_IMAGE__"):
        CVDPTestbenchRunner(_config()).check_prepared_images([dataset])
    config = _config().model_copy(update={"eda_sim_image": "registry.example/eda:test"})
    assert CVDPTestbenchRunner(config).check_prepared_images([dataset]) == set()


@pytest.mark.parametrize("selected_split", ["validation", "benchmark"])
def test_server_checks_selected_dataset_images_at_startup(tmp_path, selected_split) -> None:
    dataset = _build_dataset(tmp_path)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "split": selected_split,
        "agent": {
            "responses_api_agents": {
                "cvdp_agent": {
                    "resources_server": {"name": "cvdp"},
                    "datasets": [{"type": "validation", "jsonl_fpath": str(dataset)}],
                }
            }
        },
    }
    if selected_split == "validation":
        with pytest.raises(ValueError, match="missing prepared image mapping"):
            CVDPResourcesServer(config=_config(), server_client=client)
    else:
        CVDPResourcesServer(config=_config(), server_client=client)
