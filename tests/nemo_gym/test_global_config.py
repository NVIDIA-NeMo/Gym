import nemo_gym.server_utils
import nemo_gym.global_config
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    get_global_config_dict,
    get_first_server_config_dict,
)
from nemo_gym.server_utils import (
    DictConfig,
)

from pytest import MonkeyPatch, raises
from unittest.mock import MagicMock


class TestServerUtils:
    def test_get_global_config_dict_sanity(self, monkeypatch: MonkeyPatch) -> None:
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
            "head_server": {"host": "127.0.0.1", "port": 11000}
        } == global_config_dict

    def test_get_global_config_dict_global_exists(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", "my_dict")

        global_config_dict = get_global_config_dict()
        assert "my_dict" == global_config_dict

    def test_get_global_config_dict_global_env_var(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "a: 2")

        global_config_dict = get_global_config_dict()
        assert {"a": 2} == global_config_dict

    def test_get_global_config_dict_config_paths_sanity(
        self, monkeypatch: MonkeyPatch
    ) -> None:
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
            lambda path: DictConfig({})
            if "env" not in str(path)
            else DictConfig({"extra_dot_env_key": 2})
        )
        monkeypatch.setattr(
            nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock
        )

        global_config_dict = get_global_config_dict()
        assert {
            "config_paths": ["/var", "var"],
            "extra_dot_env_key": 2,
            "head_server": {"host": "127.0.0.1", "port": 11000},
        } == global_config_dict

    def test_get_global_config_dict_config_paths_recursive(
        self, monkeypatch: MonkeyPatch
    ) -> None:
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
            config_dict = DictConfig(
                {"config_paths": ["/var", "var", "recursive_config_path_parent"]}
            )
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
        monkeypatch.setattr(
            nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock
        )

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
        } == global_config_dict

    def test_get_global_config_dict_server_host_port_defaults(
        self, monkeypatch: MonkeyPatch
    ) -> None:
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
        monkeypatch.setattr(
            nemo_gym.global_config, "find_open_port", find_open_port_mock
        )

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
            "a": {
                "responses_api_models": {
                    "c": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}
                }
            },
            "b": {"c": {"d": {}}},
            "c": 2,
            "head_server": {"host": "127.0.0.1", "port": 11000},
        } == global_config_dict

    def test_get_global_config_dict_server_refs_sanity(
        self, monkeypatch: MonkeyPatch
    ) -> None:
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
        monkeypatch.setattr(
            nemo_gym.global_config, "find_open_port", find_open_port_mock
        )

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
                    "c": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}
                }
            },
            "head_server": {"host": "127.0.0.1", "port": 11000},
        } == global_config_dict

    def test_get_global_config_dict_server_refs_errors_on_missing(
        self, monkeypatch: MonkeyPatch
    ) -> None:
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
        monkeypatch.setattr(
            nemo_gym.global_config, "find_open_port", find_open_port_mock
        )

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

    def test_get_global_config_dict_server_refs_errors_on_wrong_type(
        self, monkeypatch: MonkeyPatch
    ) -> None:
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
        monkeypatch.setattr(
            nemo_gym.global_config, "find_open_port", find_open_port_mock
        )

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
        assert {"my_key": "my_value"} == get_first_server_config_dict(
            global_config_dict, "a"
        )
