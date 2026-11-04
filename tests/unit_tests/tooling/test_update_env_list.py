# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import scripts.update_env_list as update_env_list
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


def test_training_server_info_includes_benchmark_configs(tmp_path: Path, monkeypatch) -> None:
    resources_servers = tmp_path / "resources_servers"
    responses_api_agents = tmp_path / "responses_api_agents"
    benchmarks = tmp_path / "benchmarks"
    resources_servers.mkdir()
    responses_api_agents.mkdir()
    benchmark_dir = benchmarks / "desktop_bench"
    benchmark_dir.mkdir(parents=True)
    agent_dir = responses_api_agents / "desktop_agent" / "configs"
    agent_dir.mkdir(parents=True)
    agent_config = agent_dir / "desktop_agent.yaml"
    agent_config.write_text(
        """
desktop_agent:
  responses_api_agents:
    desktop_agent:
      domain: agent
      description: Desktop benchmark
      value: GUI evaluation
""".lstrip(),
        encoding="utf-8",
    )
    (benchmark_dir / "config.yaml").write_text(
        f"""
config_paths:
  - {agent_config}

desktop_agent:
  responses_api_agents:
    desktop_agent:
      datasets:
        - name: validation
          type: validation
          license: Apache 2.0
          huggingface_identifier:
            repo_id: example/desktop-bench
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", responses_api_agents)
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", benchmarks)

    servers = update_env_list.get_training_server_info()

    assert len(servers) == 1
    server = servers[0]
    assert server.name == "desktop_bench"
    assert server.display_name == "Desktop Agent"
    assert server.config_path == "benchmarks/desktop_bench/config.yaml"
    assert server.readme_path == "benchmarks/desktop_bench/README.md"
    assert server.config_metadata.domain == "agent"
    assert server.config_metadata.description == "Desktop benchmark"
    assert server.config_metadata.value == "GUI evaluation"
    assert server.config_metadata.types == ["validation"]
