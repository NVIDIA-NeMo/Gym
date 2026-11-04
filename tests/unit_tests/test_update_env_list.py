# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from scripts.update_env_list import extract_config_metadata, visit_agent_datasets, visit_resources_datasets


def _agent_config(train_dataset: dict) -> dict:
    return {
        "test_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "datasets": [train_dataset],
                }
            }
        }
    }


def test_visit_agent_datasets_reads_huggingface_source() -> None:
    metadata = visit_agent_datasets(
        _agent_config(
            {
                "name": "train",
                "type": "train",
                "license": "Creative Commons Attribution 4.0 International",
                "source": {
                    "type": "huggingface",
                    "repo_id": "nvidia/example-dataset",
                },
            }
        )
    )

    assert metadata.types == ["train"]
    assert metadata.license == "Creative Commons Attribution 4.0 International"
    assert metadata.huggingface_repo_id == "nvidia/example-dataset"


def test_visit_agent_datasets_retains_legacy_huggingface_fallback() -> None:
    metadata = visit_agent_datasets(
        _agent_config(
            {
                "name": "train",
                "type": "train",
                "license": "Apache 2.0",
                "huggingface_identifier": {"repo_id": "nvidia/legacy-dataset"},
            }
        )
    )

    assert metadata.types == ["train"]
    assert metadata.license == "Apache 2.0"
    assert metadata.huggingface_repo_id == "nvidia/legacy-dataset"


def test_resources_datasets_are_read_from_the_resources_server() -> None:
    metadata = visit_resources_datasets(
        {
            "geobenchx_resources_server": {
                "resources_servers": {
                    "geobenchx": {
                        "datasets": [
                            {"name": "benchmark", "type": "example", "license": "MIT"},
                            {"name": "train", "type": "train", "license": "MIT"},
                            {"name": "validation", "type": "validation", "license": "MIT"},
                        ]
                    }
                }
            }
        }
    )

    assert metadata.types == ["example", "train", "validation"]
    assert metadata.license == "MIT"


def test_agent_datasets_keep_priority_over_resources_datasets(tmp_path) -> None:
    path = tmp_path / "server.yaml"
    path.write_text(
        """
server:
  resources_servers:
    server:
      description: From the resources server
      datasets:
        - name: train
          type: train
          license: MIT
agent:
  responses_api_agents:
    simple_agent:
      datasets:
        - name: train
          type: train
          license: Apache 2.0
""".lstrip(),
        encoding="utf-8",
    )

    metadata = extract_config_metadata(path)

    assert metadata.types == ["train"]
    assert metadata.license == "Apache 2.0"
    assert metadata.description == "From the resources server"
