import nemo_gym.server_utils
from nemo_gym.server_utils import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    HeadServer,
    ServerClient,
    DictConfig,
    get_global_config_dict,
    get_first_server_config_dict,
    BaseServerConfig,
    BaseServer,
)

from pytest import MonkeyPatch, raises
from unittest.mock import MagicMock, AsyncMock


class TestServerUtils:
    def test_get_global_config_dict_sanity(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.server_utils.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.server_utils.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "head_server": {"host": "127.0.0.1", "port": 11000}
        } == global_config_dict

    def test_get_global_config_dict_global_exists(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", "my_dict")

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
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = True
        monkeypatch.setattr(nemo_gym.server_utils.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"config_paths": ["/var", "var"]})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.server_utils.hydra, "main", hydra_main_mock)

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

    def test_get_global_config_dict_server_host_port_defaults(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.server_utils.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(
            nemo_gym.server_utils, "find_open_port", find_open_port_mock
        )

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "a": {"responses_api_models": {"c": {}}},
                    "b": {"c": {"d": {}}},
                    "c": 2,
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.server_utils.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "a": {"responses_api_models": {"c": {"host": "127.0.0.1", "port": 12345}}},
            "b": {"c": {"d": {}}},
            "c": 2,
            "head_server": {"host": "127.0.0.1", "port": 11000},
        } == global_config_dict

    def test_get_global_config_dict_server_refs_sanity(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.server_utils.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(
            nemo_gym.server_utils, "find_open_port", find_open_port_mock
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
                                "d": {
                                    "type": "resources_servers",
                                    "name": "resources_name",
                                },
                                "e": 2,
                            }
                        }
                    },
                    "resources_name": {"resources_servers": {"c": {}}},
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.server_utils.hydra, "main", hydra_main_mock)

        global_config_dict = get_global_config_dict()
        assert {
            "agent_name": {
                "responses_api_agents": {
                    "agent_type": {
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
                "resources_servers": {"c": {"host": "127.0.0.1", "port": 12345}}
            },
            "head_server": {"host": "127.0.0.1", "port": 11000},
        } == global_config_dict

    def test_get_global_config_dict_server_refs_errors_on_missing(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.server_utils.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(
            nemo_gym.server_utils, "find_open_port", find_open_port_mock
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
                                "d": {
                                    "type": "resources_servers",
                                    "name": "resources_name",
                                }
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.server_utils.hydra, "main", hydra_main_mock)

        with raises(AssertionError):
            get_global_config_dict()

    def test_get_global_config_dict_server_refs_errors_on_wrong_type(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.server_utils.Path, "exists", exists_mock)

        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(
            nemo_gym.server_utils, "find_open_port", find_open_port_mock
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
                                "d": {
                                    "type": "resources_servers",
                                    "name": "resources_name",
                                }
                            }
                        }
                    },
                    "resources_name": {"responses_api_models": {"c": {}}},
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.server_utils.hydra, "main", hydra_main_mock)

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

    def test_ServerClient_load_head_server_config(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(
            nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock
        )
        actual_config = ServerClient.load_head_server_config()
        assert actual_config.host == ""
        assert actual_config.port == 0

    def test_ServerClient_load_from_global_config(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(
            nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock
        )

        httpx_client_mock = MagicMock()
        httpx_response_mock = MagicMock()
        httpx_client_mock.return_value = httpx_response_mock
        httpx_response_mock.content = b'"a: 2"'
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        actual_client = ServerClient.load_from_global_config()
        assert {"a": 2} == actual_client.global_config_dict

    async def test_ServerClient_get_post_sanity(self, monkeypatch: MonkeyPatch) -> None:
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="abcdef", port=12345),
            global_config_dict=DictConfig(
                {
                    "my_server": {
                        "a": {
                            "b": {
                                "host": "xyz",
                                "port": 54321,
                            }
                        }
                    }
                }
            ),
        )

        httpx_client_mock = AsyncMock()
        httpx_client_mock.return_value = "my mock response"
        monkeypatch.setattr(
            nemo_gym.server_utils.GLOBAL_HTTPX_CLIENT, "get", httpx_client_mock
        )

        actual_response = await server_client.get(
            server_name="my_server",
            url_path="blah blah",
        )
        assert "my mock response" == actual_response

        httpx_client_mock = AsyncMock()
        httpx_client_mock.return_value = "my mock response"
        monkeypatch.setattr(
            nemo_gym.server_utils.GLOBAL_HTTPX_CLIENT, "post", httpx_client_mock
        )

        actual_response = await server_client.post(
            server_name="my_server",
            url_path="blah blah",
        )
        assert "my mock response" == actual_response

    def test_BaseServer_load_config_from_global_config(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Clear any lingering env vars.
        monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "my_server")

        global_config_dict = DictConfig(
            {
                "my_server": {
                    "a": {"b": {"host": "", "port": 0, "entrypoint": "my entrypoint"}}
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(
            nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock
        )

        actual_config = BaseServer.load_config_from_global_config()
        assert "" == actual_config.host
        assert 0 == actual_config.port
        assert "my entrypoint" == actual_config.entrypoint

    def test_HeadServer_setup_webserver_sanity(self) -> None:
        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        head_server.setup_webserver()

    async def test_HeadServer_global_config_dict_yaml(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        global_config_dict = DictConfig({"a": 2})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(
            nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock
        )

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        resp = await head_server.global_config_dict_yaml()

        assert "a: 2\n" == resp
