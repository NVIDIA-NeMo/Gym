# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import scripts.update_env_list as update_env_list


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


def test_resources_owned_datasets_appear_and_helper_configs_stay_out(tmp_path: Path, monkeypatch) -> None:
    resources_servers = tmp_path / "resources_servers"
    responses_api_agents = tmp_path / "responses_api_agents"
    benchmarks = tmp_path / "benchmarks"
    configs = resources_servers / "geobenchx" / "configs"
    configs.mkdir(parents=True)
    responses_api_agents.mkdir()
    benchmarks.mkdir()
    (configs / "geobenchx.yaml").write_text(
        """
geobenchx_resources_server:
  resources_servers:
    geobenchx:
      domain: agent
      description: GeoBenchX benchmark
      datasets:
        - name: train
          type: train
          license: MIT
        - name: validation
          type: validation
          license: MIT
geobenchx_simple_agent:
  responses_api_agents:
    simple_agent:
      resources_server:
        type: resources_servers
        name: geobenchx_resources_server
""".lstrip(),
        encoding="utf-8",
    )
    (configs / "geobenchx_real_tools.yaml").write_text(
        """
geobenchx_resources_server:
  resources_servers:
    geobenchx:
      tool_backend: upstream
""".lstrip(),
        encoding="utf-8",
    )
    helper = responses_api_agents / "geobenchx_agent" / "configs"
    helper.mkdir(parents=True)
    (helper / "agent.yaml").write_text(
        """
geobenchx_simple_agent:
  responses_api_agents:
    simple_agent:
      resources_server:
        type: resources_servers
        name: geobenchx_resources_server
      datasets:
        - name: train
          type: train
          license: Apache 2.0
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", responses_api_agents)
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", benchmarks)

    servers = update_env_list.get_training_server_info()

    assert [server.config_filename for server in servers] == ["geobenchx.yaml"]
    assert servers[0].get_train_mark() == "✓"
    assert servers[0].get_validation_mark() == "✓"
    assert servers[0].get_license_or_dash() == "MIT"


def test_example_only_resources_datasets_stay_out_of_the_catalog(tmp_path: Path, monkeypatch) -> None:
    resources_servers = tmp_path / "resources_servers"
    configs = resources_servers / "chembench" / "configs"
    configs.mkdir(parents=True)
    (configs / "chembench.yaml").write_text(
        """
chembench_resources_server:
  resources_servers:
    chembench:
      domain: knowledge
      description: ChemBench
      datasets:
        - name: example
          type: example
          license: MIT
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", tmp_path / "responses_api_agents")
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", tmp_path / "benchmarks")
    (tmp_path / "responses_api_agents").mkdir()
    (tmp_path / "benchmarks").mkdir()

    assert update_env_list.get_training_server_info() == []
