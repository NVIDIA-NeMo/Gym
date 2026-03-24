# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import yaml

from nemo_gym.env_list import EnvInfo, _visit_datasets, _visit_server_metadata, get_envs


class TestVisitServerMetadata:
    def test_extracts_fields(self) -> None:
        data = {
            "my_instance": {
                "resources_servers": {
                    "my_server": {
                        "domain": "math",
                        "description": "A math env",
                        "verified": True,
                        "value": "Improve math",
                    }
                }
            }
        }
        result = _visit_server_metadata(data)
        assert result["domain"] == "math"
        assert result["description"] == "A math env"
        assert result["verified"] is True
        assert result["value"] == "Improve math"

    def test_handles_agent_server_type(self) -> None:
        data = {
            "my_agent": {
                "responses_api_agents": {
                    "simple_agent": {
                        "domain": "coding",
                        "description": "An agent env",
                    }
                }
            }
        }
        result = _visit_server_metadata(data)
        assert result["domain"] == "coding"

    def test_returns_empty_for_unknown_structure(self) -> None:
        result = _visit_server_metadata({"foo": "bar"})
        assert result == {}

    def test_returns_empty_for_empty_dict(self) -> None:
        result = _visit_server_metadata({})
        assert result == {}


class TestVisitDatasets:
    def test_extracts_dataset_types(self) -> None:
        data = {
            "my_instance": {
                "resources_servers": {
                    "my_server": {
                        "datasets": [
                            {"type": "train", "name": "train"},
                            {"type": "validation", "name": "val"},
                        ]
                    }
                }
            }
        }
        types = _visit_datasets(data)
        assert "train" in types
        assert "validation" in types

    def test_extracts_from_agent_server_type(self) -> None:
        data = {
            "my_agent": {"responses_api_agents": {"simple_agent": {"datasets": [{"type": "train", "name": "train"}]}}}
        }
        types = _visit_datasets(data)
        assert "train" in types

    def test_returns_empty_for_no_datasets(self) -> None:
        data = {"my_instance": {"resources_servers": {"my_server": {"entrypoint": "app.py"}}}}
        assert _visit_datasets(data) == []

    def test_returns_empty_for_non_dict(self) -> None:
        assert _visit_datasets({"key": "not_a_dict"}) == []


class TestGetEnvs:
    def test_returns_envs_from_resources_servers(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "resources_servers" / "my_env" / "configs"
        server_dir.mkdir(parents=True)
        config = {
            "my_instance": {
                "resources_servers": {
                    "my_env": {
                        "domain": "math",
                        "description": "Test env",
                        "datasets": [{"type": "train", "name": "train"}],
                    }
                }
            }
        }
        (server_dir / "my_env.yaml").write_text(yaml.dump(config))

        envs = get_envs(tmp_path)
        assert len(envs) == 1
        assert envs[0].name == "my_env"
        assert envs[0].domain == "math"
        assert envs[0].has_train is True
        assert envs[0].has_validation is False
        assert envs[0].is_example is False

    def test_example_env_flagged(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "resources_servers" / "example_foo" / "configs"
        server_dir.mkdir(parents=True)
        config = {
            "example_foo_instance": {
                "resources_servers": {
                    "example_foo": {
                        "domain": "agent",
                        "datasets": [{"type": "example", "name": "example"}],
                    }
                }
            }
        }
        (server_dir / "example_foo.yaml").write_text(yaml.dump(config))

        envs = get_envs(tmp_path)
        assert len(envs) == 1
        assert envs[0].is_example is True

    def test_agent_env_requires_domain(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "responses_api_agents" / "my_agent" / "configs"
        agent_dir.mkdir(parents=True)
        config = {
            "my_agent": {
                "responses_api_agents": {
                    "my_agent": {
                        "datasets": [{"type": "train", "name": "train"}],
                    }
                }
            }
        }
        (agent_dir / "my_agent.yaml").write_text(yaml.dump(config))
        assert get_envs(tmp_path) == []

    def test_agent_env_with_domain_included(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "responses_api_agents" / "my_agent" / "configs"
        agent_dir.mkdir(parents=True)
        config = {
            "my_agent": {
                "responses_api_agents": {
                    "my_agent": {
                        "domain": "coding",
                        "datasets": [{"type": "train", "name": "train"}],
                    }
                }
            }
        }
        (agent_dir / "my_agent.yaml").write_text(yaml.dump(config))

        envs = get_envs(tmp_path)
        assert len(envs) == 1
        assert envs[0].domain == "coding"

    def test_skips_empty_yaml(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "resources_servers" / "empty_env" / "configs"
        server_dir.mkdir(parents=True)
        (server_dir / "empty.yaml").write_text("")
        assert get_envs(tmp_path) == []

    def test_skips_missing_configs_dir(self, tmp_path: Path) -> None:
        (tmp_path / "resources_servers" / "no_configs").mkdir(parents=True)
        assert get_envs(tmp_path) == []

    def test_real_repo(self) -> None:
        from nemo_gym import PARENT_DIR

        envs = get_envs(PARENT_DIR)
        assert len(envs) > 0
        assert all(isinstance(e, EnvInfo) for e in envs)
        domains = {e.domain for e in envs if e.domain}
        assert "math" in domains
        assert "coding" in domains
