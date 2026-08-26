# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from scripts.update_env_list import visit_agent_datasets


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
                    "repo_id": "nvidia/Nemotron-RL-litmus-bench-v0.1",
                },
            }
        )
    )

    assert metadata.types == ["train"]
    assert metadata.license == "Creative Commons Attribution 4.0 International"
    assert metadata.huggingface_repo_id == "nvidia/Nemotron-RL-litmus-bench-v0.1"


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


def test_visit_agent_datasets_reads_resources_server_datasets() -> None:
    metadata = visit_agent_datasets(
        {
            "test_rs": {
                "resources_servers": {
                    "my_server": {
                        "datasets": [
                            {"name": "train", "type": "train", "license": "Apache 2.0"},
                            {"name": "validation", "type": "validation"},
                        ]
                    }
                }
            }
        }
    )

    assert metadata.types == ["train", "validation"]
    assert metadata.license == "Apache 2.0"


def test_visit_agent_datasets_ignores_harbor_fallback_when_datasets_declared() -> None:
    metadata = visit_agent_datasets(
        {
            "test_rs": {
                "resources_servers": {
                    "my_server": {
                        "datasets": [{"name": "validation", "type": "validation"}],
                    }
                }
            },
            "test_agent": {
                "responses_api_agents": {
                    "harbor_agent": {
                        "harbor_datasets": {"my_bench": {}},
                    }
                }
            },
        }
    )

    assert metadata.types == ["validation"]


def test_visit_agent_datasets_harbor_fallback_without_datasets() -> None:
    metadata = visit_agent_datasets(
        {
            "test_agent": {
                "responses_api_agents": {
                    "harbor_agent": {
                        "harbor_datasets": {"my_bench": {}},
                    }
                }
            }
        }
    )

    assert metadata.types == ["train"]
