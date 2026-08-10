# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

from resources_servers.cvdp.app import CVDPResourcesServerConfig
from resources_servers.cvdp.prepare import _write_context, prepare_images


def _config() -> CVDPResourcesServerConfig:
    return CVDPResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        name="cvdp",
        entrypoint="app.py",
        oss_pnr_image="registry.example/cvdp-pnr:base",
    )


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
