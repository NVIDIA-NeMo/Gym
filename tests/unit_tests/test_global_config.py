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
from contextlib import nullcontext as does_not_raise
from typing import Dict
from unittest.mock import MagicMock

from pytest import MonkeyPatch, raises

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym.global_config import (
    DEFAULT_HEAD_SERVER_PORT,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    find_open_port,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.server_utils import (
    DictConfig,
)


class TestServerUtils:
    def _mock_versions_for_testing(self, monkeypatch: MonkeyPatch) -> Dict[str, str]:
        monkeypatch.setattr(nemo_gym.global_config, "openai_version", "test openai version")
        monkeypatch.setattr(nemo_gym.global_config, "ray_version", "test ray version")

        python_version_mock = MagicMock(return_value="test python version")
        monkeypatch.setattr(nemo_gym.global_config, "python_version", python_version_mock)

        return {
            "head_server_deps": ["ray==test ray version", "openai==test openai version"],
            "python_version": "test python version",
        }

    def test_get_global_config_dict_sanity(self, monkeypatch: MonkeyPatch) -> None:
        mock_versions_for_testing = self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "disallowed_ports": [11000],
            **mock_versions_for_testing,
        } == global_config_dict

    def test_get_global_config_dict_global_exists(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", "my_dict")

        global_config_dict = get_global_config_dict()
        assert "my_dict" == global_config_dict

    def test_get_global_config_dict_global_env_var(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "a: 2")

        global_config_dict = get_global_config_dict()
        assert {"a": 2} == global_config_dict

    def test_get_global_config_dict_config_paths_sanity(self, monkeypatch: MonkeyPatch) -> None:
        mock_versions_for_testing = self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = True
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"config_paths": ["/var", "var"]})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        # Override OmegaConf.load to avoid file reads.
        omegaconf_load_mock = MagicMock()
        omegaconf_load_mock.side_effect = (
            lambda path: DictConfig({}) if "env" not in str(path) else DictConfig({"extra_dot_env_key": 2})
        )
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "config_paths": ["/var", "var"],
            "extra_dot_env_key": 2,
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "disallowed_ports": [11000],
            **mock_versions_for_testing,
        } == global_config_dict

    def test_get_global_config_dict_config_paths_recursive(self, monkeypatch: MonkeyPatch) -> None:
        mock_versions_for_testing = self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = True
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"config_paths": ["/var", "var", "recursive_config_path_parent"]})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        # Override OmegaConf.load to avoid file reads.
        omegaconf_load_mock = MagicMock()

        def omegaconf_load_mock_side_effect(path):
            if "recursive_config_path_parent" in str(path):
                return DictConfig({"config_paths": ["recursive_config_path_child"]})
            elif "recursive_config_path_child" in str(path):
                return DictConfig({"recursive_config_path_child_key": 3})
            elif "env" in str(path):
                return DictConfig({"extra_dot_env_key": 2})
            else:
                return DictConfig({})

        omegaconf_load_mock.side_effect = omegaconf_load_mock_side_effect
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "config_paths": [
                "/var",
                "var",
                "recursive_config_path_parent",
                "recursive_config_path_child",
            ],
            "extra_dot_env_key": 2,
            "recursive_config_path_child_key": 3,
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "disallowed_ports": [11000],
            **mock_versions_for_testing,
        } == global_config_dict

    def test_get_global_config_dict_server_host_port_defaults(self, monkeypatch: MonkeyPatch) -> None:
        mock_versions_for_testing = self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", find_open_port_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "a": {"responses_api_models": {"c": {"entrypoint": "app.py"}}},
                    "b": {"c": {"d": {}}},
                    "c": 2,
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "a": {"responses_api_models": {"c": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}}},
            "b": {"c": {"d": {}}},
            "c": 2,
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "disallowed_ports": [11000, 12345],
            **mock_versions_for_testing,
        } == global_config_dict

    def test_get_global_config_dict_server_refs_sanity(self, monkeypatch: MonkeyPatch) -> None:
        mock_versions_for_testing = self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.side_effect = [12345, 123456]
        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", find_open_port_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "agent_name": {
                        "responses_api_agents": {
                            "agent_type": {
                                "entrypoint": "app.py",
                                "d": {
                                    "type": "resources_servers",
                                    "name": "resources_name",
                                },
                                "e": 2,
                            }
                        }
                    },
                    "resources_name": {
                        "resources_servers": {
                            "c": {
                                "entrypoint": "app.py",
                                "domain": "other",
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "agent_name": {
                "responses_api_agents": {
                    "agent_type": {
                        "entrypoint": "app.py",
                        "d": {
                            "type": "resources_servers",
                            "name": "resources_name",
                        },
                        "e": 2,
                        "host": "127.0.0.1",
                        "port": 12345,
                    }
                }
            },
            "resources_name": {
                "resources_servers": {
                    "c": {
                        "entrypoint": "app.py",
                        "host": "127.0.0.1",
                        "port": 123456,
                        "domain": "other",
                    }
                }
            },
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "disallowed_ports": [11000, 12345, 123456],
            **mock_versions_for_testing,
        } == global_config_dict

    def test_get_global_config_dict_server_refs_errors_on_missing(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", find_open_port_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        # Test errors on missing
        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "agent_name": {
                        "responses_api_agents": {
                            "agent_type": {
                                "entrypoint": "app.py",
                                "d": {
                                    "type": "resources_servers",
                                    "name": "resources_name",
                                },
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        with raises(AssertionError):
            get_global_config_dict()

    def test_get_global_config_dict_server_refs_errors_on_wrong_type(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", find_open_port_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        # Test errors on missing
        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "agent_name": {
                        "responses_api_agents": {
                            "agent_type": {
                                "entrypoint": "app.py",
                                "d": {
                                    "type": "resources_servers",
                                    "name": "resources_name",
                                },
                            }
                        }
                    },
                    "resources_name": {
                        "responses_api_models": {
                            "c": {
                                "entrypoint": "app.py",
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        with raises(AssertionError):
            get_global_config_dict()

    def test_get_first_server_config_dict(self) -> None:
        global_config_dict = DictConfig(
            {
                "a": {
                    "b": {
                        "c": {"my_key": "my_value"},
                        "d": None,
                    },
                    "e": None,
                },
                "f": None,
            }
        )
        assert {"my_key": "my_value"} == get_first_server_config_dict(global_config_dict, "a")

    def test_find_open_port_avoids_disallowed_ports(self, monkeypatch: MonkeyPatch) -> None:
        """Test that find_open_port retries when the head server port is returned."""
        socket_mock = MagicMock()
        socket_instance = MagicMock()
        socket_mock.return_value.__enter__ = MagicMock(return_value=socket_instance)
        socket_mock.return_value.__exit__ = MagicMock(return_value=False)

        socket_instance.getsockname.side_effect = [
            ("", DEFAULT_HEAD_SERVER_PORT),  # first attempt: 11000 (conflict)
            ("", 12345),  # second attempt (safe)
        ]

        monkeypatch.setattr(nemo_gym.global_config, "socket", socket_mock)

        port = find_open_port(disallowed_ports=[DEFAULT_HEAD_SERVER_PORT])

        assert port == 12345
        assert socket_instance.getsockname.call_count == 2  # first: conflict, second: success

    def test_find_open_port_raises_after_max_retries(self, monkeypatch: MonkeyPatch) -> None:
        """Test that find_open_port raises RuntimeError after exhausting retries."""
        socket_mock = MagicMock()
        socket_instance = MagicMock()
        socket_mock.return_value.__enter__ = MagicMock(return_value=socket_instance)
        socket_mock.return_value.__exit__ = MagicMock(return_value=False)

        socket_instance.getsockname.return_value = ("", DEFAULT_HEAD_SERVER_PORT)  # force conflict

        monkeypatch.setattr(nemo_gym.global_config, "socket", socket_mock)

        with raises(RuntimeError) as exc_info:
            find_open_port(disallowed_ports=[DEFAULT_HEAD_SERVER_PORT], max_retries=5)

        assert "Unable to find an open port" in str(exc_info.value)
        assert "after 5 attempts" in str(exc_info.value)
        assert socket_instance.getsockname.call_count == 5

    def test_get_global_config_dict_prevents_port_conflict_with_head_server(self, monkeypatch: MonkeyPatch) -> None:
        """Integration test: verify that child servers never get the head server port."""
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        def mock_find_open_port(disallowed_ports=None, max_retries=50):
            return 12345  # safe port

        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", mock_find_open_port)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            """Trigger find_open_port by excluding port from the config"""
            config_dict = DictConfig(
                {"test_resource": {"resources_servers": {"test_server": {"entrypoint": "app.py", "domain": "other"}}}}
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()

        resource_port = global_config_dict["test_resource"]["resources_servers"]["test_server"]["port"]
        head_port = global_config_dict["head_server"]["port"]

        assert resource_port == 12345
        assert head_port == 11000
        assert resource_port != head_port
        assert "disallowed_ports" in global_config_dict
        assert 11000 in global_config_dict["disallowed_ports"]
        assert 12345 in global_config_dict["disallowed_ports"]

    def test_almost_servers_detection_and_warning(self, monkeypatch) -> None:
        """Test the default flag error_on_almost_servers=true raises ValueError."""
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        rich_print_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.global_config.rich, "print", rich_print_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "test_resources_server": {
                        "resources_servers": {"test_server": {"entrypoint": "app.py", "domain": "invalid_domain"}}
                    },
                    "test_agent": {
                        "responses_api_agents": {
                            "simple_agent": {
                                "entrypoint": "app.py",
                                "datasets": [
                                    {
                                        "name": "train",
                                        "type": "train",
                                        "jsonl_fpath": "data/train.jsonl",
                                        "gitlab_identifier": {
                                            "dataset_name": "test",
                                            "version": "0.0.1",
                                            "artifact_fpath": "train.jsonl",
                                        },
                                        "license": "Invalid License",
                                    }
                                ],
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        with raises(ValueError, match="almost-server.*validation errors"):
            get_global_config_dict()

    def test_almost_servers_error_flag_bypasses_value_error(self, monkeypatch: MonkeyPatch) -> None:
        """
        Test that error_on_almost_servers=false does not raise ValueError.
        Almost-servers are still detected and warnings are printed.
        """
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        rich_print_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.global_config.rich, "print", rich_print_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "error_on_almost_servers": False,
                    "test_resources_server": {
                        "resources_servers": {"test_server": {"entrypoint": "app.py", "domain": "invalid_domain"}}
                    },
                    "test_agent": {
                        "responses_api_agents": {
                            "simple_agent": {
                                "entrypoint": "app.py",
                                "datasets": [
                                    {
                                        "name": "train",
                                        "type": "train",
                                        "jsonl_fpath": "data/train.jsonl",
                                        "gitlab_identifier": {
                                            "dataset_name": "test",
                                            "version": "0.0.1",
                                            "artifact_fpath": "train.jsonl",
                                        },
                                        "license": "Invalid License",
                                    }
                                ],
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        with does_not_raise():
            global_config_dict = get_global_config_dict()

        assert global_config_dict is not None

        printed_messages = " ".join(str(call) for call in rich_print_mock.call_args_list)
        assert "Almost-Server" in printed_messages
        assert "test_resources_server" in printed_messages
        assert "test_agent" in printed_messages
        assert "Configuration Warnings" in printed_messages
        assert "license" in printed_messages
        assert "domain" in printed_messages
