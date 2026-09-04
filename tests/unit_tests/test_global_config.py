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
import importlib.metadata
import sys
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch, mark, raises

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym import CACHE_DIR, NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, RESULTS_DIR, WORKING_DIR
from nemo_gym._config_aliases import LEGACY_AGENT_ALIASES, LEGACY_CONFIG_PATH_ALIASES
from nemo_gym.config_types import (
    AgentCompositionError,
    AlmostServerError,
    ConfigError,
    ConfigMissingValuesError,
    ConfigPathNotFoundError,
    MalformedConfigPathsError,
    NoServerInstancesError,
    ServerRefNotFoundError,
    UnsupportedAgentOverrideError,
    UnsupportedAgentPairingError,
    UnsupportedModelPairingError,
    WANDBConfig,
)
from nemo_gym.global_config import (
    ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME,
    ALLOW_UNSUPPORTED_PAIRING_KEY_NAME,
    DEFAULT_HEAD_SERVER_PORT,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    USE_ABSOLUTE_IP,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    _openai_version_matches_nemo_gym_constraint,
    find_open_port,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.secret_utils import recursively_hide_secrets
from nemo_gym.server_utils import (
    DictConfig,
)


class TestGlobalConfig:
    def _mock_versions_for_testing(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.global_config, "openai_version", "test openai version")
        monkeypatch.setattr(nemo_gym.global_config, "ray_version", "test ray version")

        python_version_mock = MagicMock(return_value="test python version")
        monkeypatch.setattr(nemo_gym.global_config, "python_version", python_version_mock)

    @property
    def _default_global_config_dict_values(self) -> dict:
        return {
            "use_absolute_ip": False,
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "disallowed_ports": [11000],
            "port_range_low": 10_001,
            "port_range_high": 20_000,
            # From self._mock_versions_for_testing
            "head_server_deps": ["ray[default]==test ray version", "openai==test openai version"],
            "python_version": "test python version",
            "skip_venv_if_present": False,
            "dry_run": False,
            "model_endpoint_readiness_timeout_seconds": 600,
            "allow_openai_version_skew": False,
            "uv_cache_dir": str(CACHE_DIR.expanduser().resolve() / "uv"),
            "uv_venv_dir": str(WORKING_DIR),
            "results_dir": str(RESULTS_DIR.expanduser().resolve()),
            "cache_dir": str(CACHE_DIR.expanduser().resolve()),
        }

    def test_get_global_config_dict_sanity(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

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
        assert self._default_global_config_dict_values == global_config_dict

    def test_offline_resolution_uses_invalid_port_without_probing(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)
        monkeypatch.delenv("UV_CACHE_DIR", raising=False)
        probe = MagicMock(side_effect=AssertionError("offline resolution must not probe sockets"))
        hostname = MagicMock(side_effect=AssertionError("offline resolution must not resolve hostnames"))
        setup_exporters = MagicMock(side_effect=AssertionError("offline resolution must not start exporters"))
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", probe)
        monkeypatch.setattr(nemo_gym.global_config, "gethostbyname", hostname)
        monkeypatch.setattr(nemo_gym.global_config, "setup_exporters", setup_exporters)
        monkeypatch.setattr(WANDBConfig, "is_available", PropertyMock(return_value=True))

        config = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(
                    {
                        USE_ABSOLUTE_IP: True,
                        "worker": {"resources_servers": {"example": {"entrypoint": "app.py", "domain": "other"}}},
                    }
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

        assert config.worker.resources_servers.example.port == -1
        assert config.worker.resources_servers.example.host == "127.0.0.1"
        probe.assert_not_called()
        hostname.assert_not_called()
        setup_exporters.assert_not_called()
        assert "UV_CACHE_DIR" not in nemo_gym.global_config.environ

    def _mock_parse_environment(self, monkeypatch: MonkeyPatch, config_dict: "DictConfig") -> None:
        """Standard parser mocks (no env var, no .env.yaml, fixed hydra config)."""
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        hydra_main_mock = MagicMock()
        hydra_main_mock.return_value = lambda fn: (lambda: fn(config_dict))
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

    def _mock_openai_topology(self, monkeypatch: MonkeyPatch, parent_version: str) -> None:
        """Parent process on `parent_version`; nemo-gym's own constraint is openai<=2.7.2."""
        monkeypatch.setattr(nemo_gym.global_config, "openai_version", parent_version)
        monkeypatch.setattr(nemo_gym.global_config, "ray_version", "test ray version")
        monkeypatch.setattr(nemo_gym.global_config, "python_version", MagicMock(return_value="test python version"))
        monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["openai<=2.7.2"])

    def test_head_server_deps_pins_compatible_parent_openai(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_openai_topology(monkeypatch, parent_version="2.6.0")
        self._mock_parse_environment(monkeypatch, DictConfig({}))

        global_config_dict = get_global_config_dict()
        assert "openai==2.6.0" in global_config_dict["head_server_deps"]

    def test_head_server_deps_raises_on_incompatible_parent_openai(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_openai_topology(monkeypatch, parent_version="2.52.0")
        self._mock_parse_environment(monkeypatch, DictConfig({}))

        with raises(ConfigError, match="allow_openai_version_skew") as excinfo:
            get_global_config_dict()
        # The error must name both sides of the conflict.
        assert "openai==2.52.0" in str(excinfo.value)
        assert "openai<=2.7.2" in str(excinfo.value)

    def test_head_server_deps_omits_pin_with_explicit_skew_opt_in(
        self, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
    ) -> None:
        self._mock_openai_topology(monkeypatch, parent_version="2.52.0")
        self._mock_parse_environment(monkeypatch, DictConfig({"allow_openai_version_skew": True}))

        with caplog.at_level("WARNING"):
            global_config_dict = get_global_config_dict()
        assert not any(dep.startswith("openai") for dep in global_config_dict["head_server_deps"])
        assert global_config_dict["allow_openai_version_skew"] is True
        # The warning must name both sides of the skew.
        assert "openai==2.52.0" in caplog.text
        assert "openai<=2.7.2" in caplog.text

    def test_head_server_deps_rejects_non_boolean_skew_opt_in(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_openai_topology(monkeypatch, parent_version="2.52.0")
        # YAML/CLI plumbing can hand the key through as a string; "false" is
        # truthy and must not silently enable the skew.
        self._mock_parse_environment(monkeypatch, DictConfig({"allow_openai_version_skew": "false"}))

        with raises(ConfigError, match="must be a boolean"):
            get_global_config_dict()

    def test_get_global_config_dict_artifact_dir_overrides(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)
        self._mock_parse_environment(
            monkeypatch, DictConfig({"results_dir": "/shared/results", "cache_dir": "/local/cache"})
        )

        global_config_dict = get_global_config_dict()
        assert global_config_dict["results_dir"] == "/shared/results"
        assert global_config_dict["cache_dir"] == "/local/cache"
        # uv_cache_dir defaults under the (overridden) cache root.
        assert global_config_dict["uv_cache_dir"] == str(Path("/local/cache") / "uv")

    def test_get_global_config_dict_artifact_dir_normalization(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)
        self._mock_parse_environment(
            monkeypatch, DictConfig({"results_dir": "relative/results", "cache_dir": "~/gym-cache"})
        )

        global_config_dict = get_global_config_dict()
        # Children resolve relative paths against their own cwd; the parser
        # must hand them an absolute path.
        assert global_config_dict["results_dir"] == str((Path.cwd() / "relative/results").resolve())
        assert global_config_dict["cache_dir"] == str((Path.home() / "gym-cache").resolve())

    def test_get_global_config_dict_artifact_dir_rejects_non_string(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)
        self._mock_parse_environment(monkeypatch, DictConfig({"results_dir": 123}))

        with raises(ConfigError, match="results_dir must be a non-empty path string"):
            get_global_config_dict()

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
        self._mock_versions_for_testing(monkeypatch)

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
        omegaconf_load_mock.side_effect = lambda path: (
            DictConfig({}) if "env" not in str(path) else DictConfig({"extra_dot_env_key": 2})
        )
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock)

        global_config_dict = get_global_config_dict()
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": ["/var", "var"],
                "extra_dot_env_key": 2,
            }
            == global_config_dict
        )

    def test_get_global_config_dict_config_paths_ordering(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        (tmp_path / "config1.yaml").write_text(f"""\
config_paths:
- {tmp_path}/config2.yaml

a: 1
        """)
        (tmp_path / "config2.yaml").write_text("""a: 2
b: 2
        """)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = True
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        # Here we assume that the user sets sys.argv correctly (we are not trying to test Hydra) and just return some DictConfig for our test.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"config_paths": [f"{tmp_path}/config1.yaml"]})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        # Override OmegaConf.load to avoid file reads.
        omegaconf_load_mock = MagicMock()
        original_load = OmegaConf.load
        omegaconf_load_mock.side_effect = lambda path: (DictConfig({}) if "env" in str(path) else original_load(path))
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock)

        global_config_dict = get_global_config_dict()
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": [f"{tmp_path}/config1.yaml", f"{tmp_path}/config2.yaml"],
                "a": 1,
                "b": 2,
            }
            == global_config_dict
        )

    def test_duplicate_config_paths_warning_goes_to_stderr(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture
    ) -> None:
        """Diagnostics must stay off stdout, which carries the `--json` payload."""
        self._mock_versions_for_testing(monkeypatch)

        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Two roots pulling in the same child is what triggers the duplicate-path warning.
        (tmp_path / "shared.yaml").write_text("a: 1\n")
        for name in ("first.yaml", "second.yaml"):
            (tmp_path / name).write_text(f"""\
config_paths:
- {tmp_path}/shared.yaml
""")

        exists_mock = MagicMock()
        exists_mock.return_value = True
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"config_paths": [f"{tmp_path}/first.yaml", f"{tmp_path}/second.yaml"]})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        omegaconf_load_mock = MagicMock()
        original_load = OmegaConf.load
        omegaconf_load_mock.side_effect = lambda path: (DictConfig({}) if "env" in str(path) else original_load(path))
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock)

        get_global_config_dict()

        captured = capsys.readouterr()
        assert "Found configs that reference the same source config path" in captured.err
        assert f"- {tmp_path}/shared.yaml" in captured.err
        assert captured.out == ""

    def test_get_global_config_dict_config_paths_recursive(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

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
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": [
                    "/var",
                    "var",
                    "recursive_config_path_parent",
                    "recursive_config_path_child",
                ],
                "extra_dot_env_key": 2,
                "recursive_config_path_child_key": 3,
            }
            == global_config_dict
        )

    def _mock_config_paths_env(self, monkeypatch: MonkeyPatch, config_paths: list[str]) -> None:
        """Scaffolding shared by the sibling-ordering tests: stub .env.yaml, hydra and
        the loader so only the config_paths merge order is under test."""
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Explicitly handle any local .env.yaml files. Either read or don't read.
        exists_mock = MagicMock()
        exists_mock.return_value = True
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        # Override the hydra main wrapper call. At runtime, this will use sys.argv.
        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"config_paths": config_paths})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        # Override OmegaConf.load to avoid file reads.
        omegaconf_load_mock = MagicMock()
        original_load = OmegaConf.load
        omegaconf_load_mock.side_effect = lambda path: (DictConfig({}) if "env" in str(path) else original_load(path))
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "load", omegaconf_load_mock)

    def test_get_global_config_dict_config_paths_later_sibling_wins(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two configs passed by the caller, neither referencing the other: the later
        entry overrides the earlier one, which is how an override config is layered on
        top of a base config."""
        self._mock_versions_for_testing(monkeypatch)

        (tmp_path / "base.yaml").write_text("""a: base
b: base
        """)
        (tmp_path / "override.yaml").write_text("""a: override
        """)

        self._mock_config_paths_env(monkeypatch, [f"{tmp_path}/base.yaml", f"{tmp_path}/override.yaml"])

        global_config_dict = get_global_config_dict()
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": [f"{tmp_path}/base.yaml", f"{tmp_path}/override.yaml"],
                "a": "override",
                "b": "base",
            }
            == global_config_dict
        )

    def test_get_global_config_dict_config_paths_outer_beats_inner_across_siblings(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Nesting and sibling order applied together. ``override.yaml`` is listed after
        ``base.yaml`` so it wins on ``a``; ``base.yaml`` pulls in ``inner.yaml``, so it
        wins on ``b`` despite ``inner.yaml`` being merged as part of the same list."""
        self._mock_versions_for_testing(monkeypatch)

        (tmp_path / "base.yaml").write_text(f"""\
config_paths:
- {tmp_path}/inner.yaml

a: base
b: base
        """)
        (tmp_path / "inner.yaml").write_text("""b: inner
c: inner
        """)
        (tmp_path / "override.yaml").write_text("""a: override
        """)

        self._mock_config_paths_env(monkeypatch, [f"{tmp_path}/base.yaml", f"{tmp_path}/override.yaml"])

        global_config_dict = get_global_config_dict()
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": [
                    f"{tmp_path}/base.yaml",
                    f"{tmp_path}/override.yaml",
                    f"{tmp_path}/inner.yaml",
                ],
                "a": "override",
                "b": "base",
                "c": "inner",
            }
            == global_config_dict
        )

    def test_get_global_config_dict_config_paths_outer_beats_inner_two_levels_deep(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Precedence follows nesting depth rather than position, at every level:
        outer overrides middle, which overrides inner."""
        self._mock_versions_for_testing(monkeypatch)

        (tmp_path / "outer.yaml").write_text(f"""\
config_paths:
- {tmp_path}/middle.yaml

a: outer
        """)
        (tmp_path / "middle.yaml").write_text(f"""\
config_paths:
- {tmp_path}/inner.yaml

a: middle
b: middle
        """)
        (tmp_path / "inner.yaml").write_text("""a: inner
b: inner
c: inner
        """)

        self._mock_config_paths_env(monkeypatch, [f"{tmp_path}/outer.yaml"])

        global_config_dict = get_global_config_dict()
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": [
                    f"{tmp_path}/outer.yaml",
                    f"{tmp_path}/middle.yaml",
                    f"{tmp_path}/inner.yaml",
                ],
                "a": "outer",
                "b": "middle",
                "c": "inner",
            }
            == global_config_dict
        )

    def test_get_global_config_dict_config_paths_later_sibling_subtree_wins_at_uneven_depth(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sibling order covers everything a sibling pulled in, whatever depth it sits
        at. ``first.yaml`` nests one level and ``second.yaml`` two, and the key both of
        their innermost configs set resolves to the later sibling's."""
        self._mock_versions_for_testing(monkeypatch)

        (tmp_path / "first.yaml").write_text(f"""\
config_paths:
- {tmp_path}/first_inner.yaml

first: first
        """)
        (tmp_path / "first_inner.yaml").write_text("""first: first_inner
contested: first_inner
        """)
        (tmp_path / "second.yaml").write_text(f"""\
config_paths:
- {tmp_path}/second_middle.yaml

second: second
        """)
        (tmp_path / "second_middle.yaml").write_text(f"""\
config_paths:
- {tmp_path}/second_inner.yaml

second: second_middle
        """)
        (tmp_path / "second_inner.yaml").write_text("""second: second_inner
contested: second_inner
        """)

        self._mock_config_paths_env(monkeypatch, [f"{tmp_path}/first.yaml", f"{tmp_path}/second.yaml"])

        global_config_dict = get_global_config_dict()
        assert (
            self._default_global_config_dict_values
            | {
                "config_paths": [
                    f"{tmp_path}/first.yaml",
                    f"{tmp_path}/second.yaml",
                    f"{tmp_path}/first_inner.yaml",
                    f"{tmp_path}/second_middle.yaml",
                    f"{tmp_path}/second_inner.yaml",
                ],
                "first": "first",
                "second": "second",
                "contested": "second_inner",
            }
            == global_config_dict
        )

    def test_get_global_config_dict_server_host_port_defaults(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

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
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

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
        assert (
            self._default_global_config_dict_values
            | {
                "a": {"responses_api_models": {"c": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}}},
                "b": {"c": {"d": {}}},
                "c": 2,
                "disallowed_ports": [11000, 12345],
            }
            == global_config_dict
        )

    def test_get_global_config_dict_skip_verification_defaults_for_agents(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

        find_open_port_mock = MagicMock()
        find_open_port_mock.side_effect = [12345, 12346, 12347]
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        parser = GlobalConfigDictParser()
        global_config_dict = parser.parse_no_environment(
            DictConfig(
                {
                    "skip_verification": True,
                    "skip_verification_reward": -1.5,
                    "agent_name": {
                        "responses_api_agents": {
                            "agent_type": {
                                "entrypoint": "app.py",
                            }
                        }
                    },
                    "explicit_agent_name": {
                        "responses_api_agents": {
                            "agent_type": {
                                "entrypoint": "app.py",
                                "skip_verification": False,
                                "skip_verification_reward": 0.25,
                            }
                        }
                    },
                    "resources_name": {
                        "resources_servers": {
                            "resources_type": {
                                "entrypoint": "app.py",
                                "domain": "other",
                            }
                        }
                    },
                }
            )
        )

        agent_config = global_config_dict["agent_name"]["responses_api_agents"]["agent_type"]
        assert agent_config["skip_verification"] is True
        assert agent_config["skip_verification_reward"] == -1.5

        explicit_agent_config = global_config_dict["explicit_agent_name"]["responses_api_agents"]["agent_type"]
        assert explicit_agent_config["skip_verification"] is False
        assert explicit_agent_config["skip_verification_reward"] == 0.25

        resources_config = global_config_dict["resources_name"]["resources_servers"]["resources_type"]
        assert "skip_verification" not in resources_config
        assert "skip_verification_reward" not in resources_config

    def test_get_global_config_dict_server_refs_sanity(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

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
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

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
        assert (
            self._default_global_config_dict_values
            | {
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
                "disallowed_ports": [11000, 12345, 123456],
            }
            == global_config_dict
        )

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

        with raises(ServerRefNotFoundError) as exc_info:
            get_global_config_dict()

        # The error should name the offending instance, the field, and the missing ref.
        message = str(exc_info.value)
        assert "agent_name" in message
        assert "'d'" in message
        assert "resources_servers/'resources_name'" in message

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

        with raises(ServerRefNotFoundError):
            get_global_config_dict()

    def test_get_global_config_dict_server_refs_suggests_close_match(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", find_open_port_mock)

        hydra_main_mock = MagicMock()

        # The agent references "resource" but the defined resources server is "resources" — a typo.
        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "agent_name": {
                        "responses_api_agents": {
                            "agent_type": {
                                "entrypoint": "app.py",
                                "resources_server": {
                                    "type": "resources_servers",
                                    "name": "resource",
                                },
                            }
                        }
                    },
                    "resources": {
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

        with raises(ServerRefNotFoundError) as exc_info:
            get_global_config_dict()

        message = str(exc_info.value)
        # Fuzzy match should suggest the correctly-spelled resources server, scoped to the same type.
        assert "Did you mean" in message
        assert "'resources'" in message

    def test_collect_missing_value_paths(self) -> None:
        config = DictConfig(
            {
                "a": "???",
                "b": {"c": "value", "d": "???"},
                "e": ["ok", "???"],
            }
        )
        missing = GlobalConfigDictParser().collect_missing_value_paths(config)
        assert missing == ["a", "b.d", "e[1]"]

    def test_missing_value_check_ignores_deleted_branch(self) -> None:
        # A '???' inside a branch removed by _delete_key must not be reported as missing (the swap
        # runs first and deletes it), while a genuinely-unset '???' elsewhere still is. Also exercises
        # that _recursively_swap_keys no longer crashes when a live '???' is present.
        parser = GlobalConfigDictParser()
        config = DictConfig(
            {
                "policy_model": {
                    "responses_api_models": {
                        "_delete_key": "old_model",
                        "old_model": {"entrypoint": "app.py", "model": "???"},
                        "new_model": {"entrypoint": "app.py", "model": "actual-model"},
                    }
                },
                "agent": {
                    "responses_api_agents": {
                        "simple_agent": {"entrypoint": "app.py", "container_formatter": "???"},
                    }
                },
            }
        )
        parser._recursively_swap_keys(config)

        assert "old_model" not in config.policy_model.responses_api_models
        missing = parser.collect_missing_value_paths(config)
        assert missing == ["agent.responses_api_agents.simple_agent.container_formatter"]

    def test_missing_value_check_handles_copy_and_inherit(self) -> None:
        # A '???' that flows through _copy / _inherit_from into a target is still reported (the swap
        # propagates it), so missing values in copied/inherited branches are not silently lost.
        parser = GlobalConfigDictParser()
        config = DictConfig(
            {
                "base": {"a": "set", "b": "???"},
                "viacopy": "${copy:base}",  # copies base, including b='???'
                "viainherit": {"_inherit_from": "base"},  # inherits base (moves it), including b='???'
            }
        )
        parser._recursively_swap_keys(config)

        missing = parser.collect_missing_value_paths(config)
        assert "viacopy.b" in missing
        assert "viainherit.b" in missing

    def test_recursively_swap_keys_helper_new_struct(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            GlobalConfigDictParser, "parse_global_config_dict_from_cli", lambda *args, **kwargs: struct
        )

        parser = GlobalConfigDictParser()
        struct = DictConfig(
            {
                "a": {
                    "b": 1,
                },
                "a2": {
                    "_inherit_from": "a",
                    "c": 2,
                },
            },
            flags={"struct": True},
        )

        parse_config = GlobalConfigDictParserConfig(skip_load_from_cli=False, skip_load_from_dotenv=True)

        """
        Assert no error:
        omegaconf.errors.ConfigKeyError: Key 'c' is not in struct
            full_key: a.c
            object_type=dict
        """
        parser.parse(parse_config)

    def test_copy_of_missing_leaf_is_reported_not_opaque_error(self) -> None:
        # `${copy:source.model}` where source.model is unset must not raise an opaque "path does not
        # exist" error; the copy propagates MISSING so both the target and source are reported.
        parser = GlobalConfigDictParser()
        config = DictConfig({"target": "${copy:source.model}", "source": {"model": "???"}})

        parser._recursively_swap_keys(config)  # must not raise

        missing = parser.collect_missing_value_paths(config)
        assert "target" in missing
        assert "source.model" in missing

    @mark.parametrize(
        "target_value",
        [
            "${inherit_from:source.model}",  # string inherit, missing leaf
            "${inherit_from:source.model.name}",  # string inherit, missing parent
            "${copy:source.model.name}",  # string copy, missing parent
            {"_copy": "source.model"},  # property copy, missing leaf
            {"_copy": "source.model.name"},  # property copy, missing parent
            {"_inherit_from": "source.model"},  # property inherit, missing leaf
            {"_inherit_from": "source.model.name"},  # property inherit, missing parent
        ],
    )
    def test_swap_copy_inherit_of_missing_value_does_not_crash(self, target_value) -> None:
        # All four quadrants {string, property} x {missing leaf, missing parent}, for both copy and
        # inherit_from: swapping a '???' source must not raise an opaque error (AttributeError from
        # .pop() on a bare string, or OmegaConf.merge ValueError). The target inherits MISSING and is
        # reported by the aggregated scan instead.
        parser = GlobalConfigDictParser()
        config = DictConfig({"target": target_value, "source": {"model": "???"}})

        parser._recursively_swap_keys(config)  # must not raise

        assert "target" in parser.collect_missing_value_paths(config)

    def test_inherit_of_missing_surfaces_aggregated_error_not_opaque(self) -> None:
        # End-to-end (swap -> raise_on_missing_values, the core of parse()): a property inherit of an
        # unset value yields the friendly ConfigMissingValuesError, not an AttributeError/ValueError.
        parser = GlobalConfigDictParser()
        config = DictConfig({"target": {"_inherit_from": "source.model"}, "source": {"model": "???"}})

        parser._recursively_swap_keys(config)
        with raises(ConfigMissingValuesError):
            parser.raise_on_missing_values(config)

    def test_missing_value_in_list_is_reported_not_crash(self) -> None:
        # A '???' as a list element (or inside a dict nested in a list) must not crash the swap; it is
        # reported by collect_missing_value_paths with its indexed path.
        parser = GlobalConfigDictParser()
        config = DictConfig({"server": {"items": ["a", "???"], "nested": [{"k": "???"}]}})

        parser._recursively_swap_keys(config)  # must not raise

        missing = parser.collect_missing_value_paths(config)
        assert "server.items[1]" in missing
        assert "server.nested[0].k" in missing

    def test_get_global_config_dict_raises_on_missing_values(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        hydra_main_mock = MagicMock()

        # A model server that leaves a required value unset ('???') — as base configs do.
        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "policy_model": {
                        "responses_api_models": {
                            "openai_model": {
                                "entrypoint": "app.py",
                                "openai_api_key": "???",
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        with raises(ConfigMissingValuesError) as exc_info:
            get_global_config_dict()

        message = str(exc_info.value)
        # Names the full dotted path and shows how to override it.
        assert "policy_model.responses_api_models.openai_model.openai_api_key" in message
        assert "++policy_model.responses_api_models.openai_model.openai_api_key=<value>" in message

    def test_get_global_config_dict_ignores_missing_in_deleted_branch(self, monkeypatch: MonkeyPatch) -> None:
        # Regression: a '???' inside a branch removed by _delete_key must NOT raise
        # ConfigMissingValuesError. The missing-value scan runs after _recursively_swap_keys, which
        # deletes the branch first, so the only surviving config has no unset values. End-to-end
        # guard through the full get_global_config_dict() parse path.
        self._mock_versions_for_testing(monkeypatch)

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "policy_model": {
                        "responses_api_models": {
                            "_delete_key": "old_model",
                            # old_model carries the only '???' and is removed before the scan.
                            "old_model": {"entrypoint": "app.py", "openai_api_key": "???"},
                            "new_model": {"entrypoint": "app.py"},
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        # Must not raise — the only '???' lived in the deleted branch.
        config = get_global_config_dict()
        models = config["policy_model"]["responses_api_models"]
        assert "old_model" not in models
        assert "new_model" in models

    def test_get_global_config_dict_aggregates_multiple_missing(self, monkeypatch: MonkeyPatch) -> None:
        # End-to-end through get_global_config_dict() -> parse(): more than one unset '???' is
        # collected into a single ConfigMissingValuesError listing every missing path.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "policy_model": {
                        "responses_api_models": {
                            "openai_model": {
                                "entrypoint": "app.py",
                                "openai_api_key": "???",
                                "openai_model": "???",
                            }
                        }
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        with raises(ConfigMissingValuesError) as exc_info:
            get_global_config_dict()

        message = str(exc_info.value)
        assert "2 required" in message
        assert "policy_model.responses_api_models.openai_model.openai_api_key" in message
        assert "policy_model.responses_api_models.openai_model.openai_model" in message

    def test_plain_interpolation_resolves_through_parse(self, monkeypatch: MonkeyPatch) -> None:
        # Regression: a plain OmegaConf interpolation `${a.b.c}` (not a swap directive) must still
        # resolve through the full parse after _recursively_swap_keys was made missing-tolerant.
        self._mock_versions_for_testing(monkeypatch)

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({"a": {"b": {"c": 3}}, "x": "${a.b.c}"})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        config = get_global_config_dict()
        assert config["x"] == 3

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
        randint_mock = MagicMock()
        randint_mock.side_effect = [
            DEFAULT_HEAD_SERVER_PORT,  # first attempt: 11000 (conflict)
            12345,  # second attempt (safe)
        ]
        monkeypatch.setattr(nemo_gym.global_config, "randint", randint_mock)

        socket_mock = MagicMock()
        socket_instance = MagicMock()
        socket_mock.return_value.__enter__ = MagicMock(return_value=socket_instance)
        socket_mock.return_value.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(nemo_gym.global_config, "socket", socket_mock)

        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = {"port_range_low": 10_001, "port_range_high": 20_000}
        monkeypatch.setattr(nemo_gym.global_config, "get_global_config_dict", get_global_config_dict_mock)

        port = find_open_port(disallowed_ports=[DEFAULT_HEAD_SERVER_PORT])

        assert port == 12345
        assert randint_mock.call_count == 2  # first: conflict, second: success

    def test_find_open_port_raises_after_max_retries(self, monkeypatch: MonkeyPatch) -> None:
        """Test that find_open_port raises RuntimeError after exhausting retries."""
        socket_mock = MagicMock()
        socket_instance = MagicMock()
        socket_mock.return_value.__enter__ = MagicMock(return_value=socket_instance)
        socket_mock.return_value.__exit__ = MagicMock(return_value=False)
        socket_instance.bind.side_effect = OSError("")
        monkeypatch.setattr(nemo_gym.global_config, "socket", socket_mock)

        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = {"port_range_low": 10_001, "port_range_high": 20_000}
        monkeypatch.setattr(nemo_gym.global_config, "get_global_config_dict", get_global_config_dict_mock)

        with raises(RuntimeError) as exc_info:
            find_open_port(disallowed_ports=[], max_retries=5)

        assert "Unable to find an open port" in str(exc_info.value)
        assert "after 5 attempts" in str(exc_info.value)
        assert socket_instance.bind.call_count == 5

    def test_get_global_config_dict_prevents_port_conflict_with_head_server(self, monkeypatch: MonkeyPatch) -> None:
        """Integration test: verify that child servers never get the head server port."""
        # Clear any lingering env vars.
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        mock_find_open_port_mock = MagicMock(return_value=12345)
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", mock_find_open_port_mock)

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

        # AlmostServerError is a ConfigError, so the CLI reports it cleanly (no traceback).
        with raises(AlmostServerError, match="almost-server.*validation errors") as exc_info:
            get_global_config_dict()
        assert isinstance(exc_info.value, ConfigError)
        # Diagnostics must stay off stdout, which carries the `--json` payload.
        assert all(call.kwargs.get("file") is sys.stderr for call in rich_print_mock.call_args_list)

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
        # Diagnostics must stay off stdout, which carries the `--json` payload.
        assert all(call.kwargs.get("file") is sys.stderr for call in rich_print_mock.call_args_list)

    def test_use_absolute_ip(self, monkeypatch: MonkeyPatch) -> None:
        """Test that use_absolute_ip=True uses machine's hostname ip for default_host."""
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "use_absolute_ip": True,
                    "test_resource": {"responses_api_models": {"test_model": {"entrypoint": "app.py"}}},
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        expected_ip = "abcd ip"
        gethostbyname_mock = MagicMock(return_value=expected_ip)
        monkeypatch.setattr(nemo_gym.global_config, "gethostbyname", gethostbyname_mock)

        global_config_dict = get_global_config_dict()

        assert global_config_dict["head_server"]["host"] == expected_ip
        assert global_config_dict["test_resource"]["responses_api_models"]["test_model"]["host"] == expected_ip

    def test_recursively_hide_secrets(self) -> None:
        dict_config = DictConfig(
            {
                "dict": {
                    "key": "key",
                    "not": "not",
                },
                "list": [
                    {"key": "key", "not": "not"},
                ],
                "key": "key",
                "not": "not",
            }
        )
        recursively_hide_secrets(dict_config)
        assert OmegaConf.to_container(dict_config) == {
            "dict": {"key": "****", "not": "not"},
            "list": [{"key": "****", "not": "not"}],
            "key": "****",
            "not": "not",
        }

    def test_recursively_replace_keys(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "policy_model": "${inherit_from:test_resource}",
                    "test_resource": {"responses_api_models": {"test_model": {"entrypoint": "app.py"}}},
                    "policy_model_2": {
                        "_inherit_from": "test_resource_2",
                        "responses_api_models": {"test_model": {"entrypoint": "app2.py"}},
                    },
                    "test_resource_2": {"responses_api_models": {"test_model": {"entrypoint": "app.py"}}},
                    "a": {"b": {"c": 3}},
                    "a_prime": {"b_prime": "${inherit_from:a.b.c}"},
                    "test_resource_3_copy1": "${copy:test_resource_3}",
                    "test_resource_3": {"responses_api_models": {"test_model": {"entrypoint": "app.py"}}},
                    "test_resource_3_copy2": {
                        "_copy": "test_resource_3",
                        "responses_api_models": {"test_model": {"entrypoint": "app2.py"}},
                    },
                    "test_resource_3_copy3_delete": {
                        "_copy": "test_resource_3",
                        "_delete_key": "responses_api_models",
                        "responses_api_models_2": {"test_model": {"entrypoint": "app.py"}},
                    },
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        actual_global_config_dict = OmegaConf.to_container(get_global_config_dict())
        expected_global_config_dict = self._default_global_config_dict_values | {
            "policy_model": {
                "responses_api_models": {"test_model": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}}
            },
            "policy_model_2": {
                "responses_api_models": {"test_model": {"entrypoint": "app2.py", "host": "127.0.0.1", "port": 12345}}
            },
            "test_resource_3_copy1": {
                "responses_api_models": {"test_model": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}}
            },
            "test_resource_3": {
                "responses_api_models": {"test_model": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}}
            },
            "test_resource_3_copy2": {
                "responses_api_models": {"test_model": {"entrypoint": "app2.py", "host": "127.0.0.1", "port": 12345}},
            },
            "disallowed_ports": [11000, 12345, 12345, 12345, 12345, 12345],
            "a": {"b": {}},
            "a_prime": {"b_prime": 3},
            "test_resource_3_copy3_delete": {
                "responses_api_models_2": {"test_model": {"entrypoint": "app.py"}},
            },
        }

        assert expected_global_config_dict == actual_global_config_dict

    def test_recursively_replace_keys_multiple_ref_one(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "policy_model": "${inherit_from:test_resource}",
                    "test_resource": {"responses_api_models": {"test_model": {"entrypoint": "app.py"}}},
                    "policy_model_2": {
                        "_inherit_from": "test_resource",
                        "responses_api_models": {"test_model": {"entrypoint": "app2.py"}},
                    },
                    "a": {"b": {"c": 3}},
                    "a_prime": {"b_prime": "${inherit_from:a.b.c}"},
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        actual_global_config_dict = OmegaConf.to_container(get_global_config_dict())
        expected_global_config_dict = self._default_global_config_dict_values | {
            "policy_model": {
                "responses_api_models": {"test_model": {"entrypoint": "app.py", "host": "127.0.0.1", "port": 12345}}
            },
            "policy_model_2": {
                "responses_api_models": {"test_model": {"entrypoint": "app2.py", "host": "127.0.0.1", "port": 12345}}
            },
            "disallowed_ports": [11000, 12345, 12345],
            "a": {"b": {}},
            "a_prime": {"b_prime": 3},
        }

        assert expected_global_config_dict == actual_global_config_dict

    def test_dummy_model_sanity(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig({})
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        actual_global_config_dict = OmegaConf.to_container(
            get_global_config_dict(
                global_config_dict_parser_config=GlobalConfigDictParserConfig(
                    initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                )
            )
        )
        expected_global_config_dict = self._default_global_config_dict_values | {
            "disallowed_ports": [11000, 12345],
            "policy_model": {
                "responses_api_models": {
                    "dummy_model": {
                        "entrypoint": "app.py",
                        "host": "127.0.0.1",
                        "port": 12345,
                    }
                }
            },
            "policy_base_url": "",
            "policy_api_key": "",
            "policy_model_name": "",
        }

        assert expected_global_config_dict == actual_global_config_dict

    def test_dummy_model_override(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_versions_for_testing(monkeypatch)

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        exists_mock = MagicMock()
        exists_mock.return_value = False
        monkeypatch.setattr(nemo_gym.global_config.Path, "exists", exists_mock)

        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", find_open_port_mock)

        hydra_main_mock = MagicMock()

        def hydra_main_wrapper(fn):
            config_dict = DictConfig(
                {
                    "policy_model": {"responses_api_models": {"test_model": {"entrypoint": "app.py"}}},
                }
            )
            return lambda: fn(config_dict)

        hydra_main_mock.return_value = hydra_main_wrapper
        monkeypatch.setattr(nemo_gym.global_config.hydra, "main", hydra_main_mock)

        actual_global_config_dict = OmegaConf.to_container(
            get_global_config_dict(
                global_config_dict_parser_config=GlobalConfigDictParserConfig(
                    initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                )
            )
        )
        expected_global_config_dict = self._default_global_config_dict_values | {
            "disallowed_ports": [11000, 12345],
            "policy_model": {
                "responses_api_models": {
                    "test_model": {
                        "entrypoint": "app.py",
                        "host": "127.0.0.1",
                        "port": 12345,
                    }
                }
            },
            "policy_base_url": "",
            "policy_api_key": "",
            "policy_model_name": "",
        }

        assert expected_global_config_dict == actual_global_config_dict

    def test_load_extra_config_paths_prefers_cwd(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        parser = GlobalConfigDictParser()

        (tmp_path / "my_config.yaml").write_text("my_key: from_cwd\n")
        monkeypatch.chdir(tmp_path)

        config_paths, extra_configs = parser.load_extra_config_paths(["my_config.yaml"])
        assert extra_configs[0]["my_key"] == "from_cwd"

    def test_load_extra_config_paths_falls_back_to_parent_dir(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        parser = GlobalConfigDictParser()

        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        (parent_dir / "my_config.yaml").write_text("my_key: from_parent\n")

        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr("nemo_gym.PARENT_DIR", parent_dir)
        monkeypatch.setattr("nemo_gym.WORKING_DIR", parent_dir)

        config_paths, extra_configs = parser.load_extra_config_paths(["my_config.yaml"])
        assert extra_configs[0]["my_key"] == "from_parent"

    def test_env_yaml_loaded_from_cwd(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        self._mock_versions_for_testing(monkeypatch)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        (tmp_path / "env.yaml").write_text("custom_env_key: from_cwd\n")
        monkeypatch.chdir(tmp_path)
        empty_parent = tmp_path / "empty_parent"
        empty_parent.mkdir()
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr("nemo_gym.PARENT_DIR", empty_parent)
        monkeypatch.setattr("nemo_gym.WORKING_DIR", empty_parent)

        parser = GlobalConfigDictParser()
        global_config_dict = parser.parse(GlobalConfigDictParserConfig(skip_load_from_cli=True))
        assert global_config_dict["custom_env_key"] == "from_cwd"

    def test_env_yaml_falls_back_to_parent_dir(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        self._mock_versions_for_testing(monkeypatch)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        (parent_dir / "env.yaml").write_text("custom_env_key: from_parent\n")

        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr("nemo_gym.PARENT_DIR", parent_dir)
        monkeypatch.setattr("nemo_gym.WORKING_DIR", parent_dir)

        parser = GlobalConfigDictParser()
        global_config_dict = parser.parse(GlobalConfigDictParserConfig(skip_load_from_cli=True))
        assert global_config_dict["custom_env_key"] == "from_parent"

    def test_help(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["++abc=2", "--help"])

        # Without the help override, this will SystemExit.
        GlobalConfigDictParser.parse_global_config_dict_from_cli(None)


class TestConfigLoadErrors:
    """Actionable, fail-fast errors for bad/malformed/empty config_paths (no raw traceback)."""

    def test_load_extra_config_paths_missing_relative_lists_both_locations(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, parent = tmp_path / "cwd", tmp_path / "parent"
        cwd.mkdir()
        parent.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr("nemo_gym.PARENT_DIR", parent)
        monkeypatch.setattr("nemo_gym.WORKING_DIR", parent)

        parser = GlobalConfigDictParser()
        with raises(ConfigPathNotFoundError) as exc_info:
            parser.load_extra_config_paths(["missing/nope.yaml"])

        message = str(exc_info.value)
        assert "missing/nope.yaml" in message
        assert str(cwd / "missing/nope.yaml") in message
        assert str(parent / "missing/nope.yaml") in message
        assert "spelled correctly" in message

    def test_load_extra_config_paths_missing_dedups_when_cwd_is_install_root(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr("nemo_gym.PARENT_DIR", tmp_path)
        monkeypatch.setattr("nemo_gym.WORKING_DIR", tmp_path)

        parser = GlobalConfigDictParser()
        with raises(ConfigPathNotFoundError) as exc_info:
            parser.load_extra_config_paths(["missing/nope.yaml"])

        assert str(exc_info.value).count("  - ") == 1

    def test_load_extra_config_paths_missing_absolute_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.yaml"
        parser = GlobalConfigDictParser()
        with raises(ConfigPathNotFoundError) as exc_info:
            parser.load_extra_config_paths([str(missing)])

        message = str(exc_info.value)
        assert str(missing) in message
        assert message.count("  - ") == 1

    @mark.parametrize(("legacy", "canonical"), LEGACY_CONFIG_PATH_ALIASES.items())
    def test_load_extra_config_paths_resolves_legacy_alias(
        self, caplog: LogCaptureFixture, legacy: str, canonical: str
    ) -> None:
        parser = GlobalConfigDictParser()

        with caplog.at_level("WARNING"):
            config_paths, configs = parser.load_extra_config_paths([legacy])

        assert config_paths == [canonical]
        assert len(configs) == 1
        assert f"Config path `{legacy}` is deprecated; use `{canonical}`." in caplog.text

    def test_existing_legacy_config_path_takes_precedence(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, caplog: LogCaptureFixture
    ) -> None:
        legacy = next(iter(LEGACY_CONFIG_PATH_ALIASES))
        local_config = tmp_path / legacy
        local_config.parent.mkdir(parents=True)
        local_config.write_text("local: true\n")
        monkeypatch.chdir(tmp_path)

        parser = GlobalConfigDictParser()
        with caplog.at_level("WARNING"):
            config_paths, configs = parser.load_extra_config_paths([legacy])

        assert config_paths == [legacy]
        assert configs[0].local is True
        assert "deprecated" not in caplog.text

    @mark.parametrize(("legacy", "canonical"), LEGACY_AGENT_ALIASES.items())
    def test_legacy_agent_names_route_to_canonical_instance(
        self, caplog: LogCaptureFixture, legacy: str, canonical: str
    ) -> None:
        config = OmegaConf.create(
            {
                canonical: {},
                "agent_name": legacy,
                "agent_map": {"source": legacy},
                "fan_out": {"source": [legacy]},
            }
        )

        with caplog.at_level("WARNING"):
            GlobalConfigDictParser.apply_legacy_agent_aliases(config)

        assert config.agent_name == canonical
        assert config.agent_map.source == canonical
        assert config.agent_map[legacy] == canonical
        assert config.fan_out.source == [canonical]
        assert f"`{legacy}` -> `{canonical}`" in caplog.text

    def test_legacy_agent_alias_follows_composed_agent_route(self) -> None:
        legacy, canonical = next(iter(LEGACY_AGENT_ALIASES.items()))
        composed = "reasoning_gym_custom_agent"
        config = OmegaConf.create({composed: {}, "agent_map": {canonical: composed}})

        GlobalConfigDictParser.apply_legacy_agent_aliases(config)

        assert config.agent_map[legacy] == composed

    def test_load_extra_config_paths_malformed_yaml_raises_config_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: [1, 2\nbar: : :\n")  # invalid YAML syntax
        parser = GlobalConfigDictParser()
        with raises(ConfigError) as exc_info:
            parser.load_extra_config_paths([str(bad)])

        message = str(exc_info.value)
        assert "Malformed YAML" in message
        assert str(bad) in message
        assert "line" in message and "column" in message

    def test_malformed_dotenv_yaml_raises_config_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "env.yaml"
        bad.write_text("policy_api_key: [oops\n")  # invalid YAML syntax
        parser = GlobalConfigDictParser()
        parse_config = GlobalConfigDictParserConfig(dotenv_path=bad, skip_load_from_cli=True)
        with raises(ConfigError) as exc_info:
            parser.parse(parse_config)

        message = str(exc_info.value)
        assert "Malformed YAML" in message
        assert str(bad) in message

    def test_parse_malformed_config_paths_raises_actionable_error(self) -> None:
        parser = GlobalConfigDictParser()
        parse_config = GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig({"config_paths": "not_a_list.yaml"}),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
        with raises(MalformedConfigPathsError) as exc_info:
            parser.parse(parse_config)

        message = str(exc_info.value)
        assert "config_paths" in message
        assert "list" in message

    def test_raise_on_no_server_instances_raises_when_empty(self) -> None:
        parser = GlobalConfigDictParser()
        config = DictConfig({"config_paths": [], "head_server": {"port": 11000}})
        with raises(NoServerInstancesError) as exc_info:
            parser.raise_on_no_server_instances(config)
        assert "gym env start" in str(exc_info.value)

    def test_raise_on_no_server_instances_passes_with_a_server(self) -> None:
        parser = GlobalConfigDictParser()
        config = DictConfig({"my_server": {"resources_servers": {"x": {"entrypoint": "app.py", "domain": "other"}}}})
        parser.raise_on_no_server_instances(config)

    def test_all_repo_configs_load_without_duplicate_keys(self) -> None:
        # OmegaConf.load (the loader `gym env start` actually uses) rejects duplicate YAML keys,
        # but a plain PyYAML parse silently allows them (last-writer-wins). A repeated key like a
        # second `datasets:` block can land unnoticed and only surface at runtime. Sweep every real
        # config in the repo so this fails fast in CI instead.
        repo_root = Path(__file__).resolve().parents[2]
        config_dirs = ("responses_api_agents", "resources_servers", "nemo_gym/sandbox/providers")
        config_paths = [p for d in config_dirs for p in (repo_root / d).rglob("*.yaml")]
        assert config_paths, "no config files discovered -- sweep dirs may have moved"

        failures = []
        for path in config_paths:
            try:
                OmegaConf.load(path)
            except Exception as e:
                failures.append(f"{path.relative_to(repo_root)}: {e}")

        assert not failures, "malformed config(s) found:\n" + "\n".join(failures)


class TestOpenAIVersionMatchesNemoGymConstraint:
    @mark.parametrize(
        "requirements, version, expected",
        [
            # Parent openai satisfies nemo-gym's constraint -> keep pinning it.
            (["openai<=2.7.2"], "2.7.2", True),
            (["openai<=2.7.2"], "2.6.0", True),
            # Parent openai violates the constraint -> the pin would be unsatisfiable.
            (["openai<=2.7.2"], "2.52.0", False),
            (["openai>=2.0,<3"], "2.52.0", True),
            # No openai requirement at all -> preserve the pin-the-parent behavior.
            (["ray[default]==2.49.0"], "2.52.0", True),
            # Package names are case-insensitive (PEP 503): still recognized.
            (["OpenAI<=2.7.2"], "2.52.0", False),
            # Marker'd requirements are skipped (conservative fallback).
            (['openai<=2.7.2; extra == "adapters"'], "2.52.0", True),
            # importlib.metadata.requires may return None.
            (None, "2.52.0", True),
        ],
    )
    def test_constraint_matching(
        self, monkeypatch: MonkeyPatch, requirements: list, version: str, expected: bool
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "requires", lambda name: requirements)
        assert _openai_version_matches_nemo_gym_constraint(version) is expected

    def test_falls_back_to_pinning_when_constraint_lookup_fails(self, monkeypatch: MonkeyPatch) -> None:
        def raise_not_found(name: str) -> None:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "requires", raise_not_found)
        assert _openai_version_matches_nemo_gym_constraint("2.52.0") is True


class TestModelServerPairing:
    _ABSENT = object()

    def _config(self, model_type: str = "openai_model", declared=_ABSENT) -> DictConfig:
        resources_config = {"entrypoint": "app.py", "domain": "other"}
        if declared is not self._ABSENT:
            resources_config["allowed_model_types"] = declared
        return DictConfig(
            {
                "audio_resources_server": {"resources_servers": {"audio": resources_config}},
                "audio_simple_agent": {
                    "responses_api_agents": {
                        "simple_agent": {
                            "entrypoint": "app.py",
                            "resources_server": {"type": "resources_servers", "name": "audio_resources_server"},
                            "model_server": {"type": "responses_api_models", "name": "policy_model"},
                        }
                    }
                },
                "policy_model": {"responses_api_models": {model_type: {"entrypoint": "app.py"}}},
            }
        )

    def test_rejects_an_unsupported_model_adapter_with_an_actionable_error(self) -> None:
        config = self._config(declared=["vllm_model", "local_vllm_model"])

        with raises(UnsupportedModelPairingError) as error:
            GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

        message = str(error.value)
        assert "audio_simple_agent" in message
        assert "policy_model" in message
        assert "openai_model" in message
        assert "audio_resources_server" in message
        assert "--model-type vllm_model" in message
        assert "Supported model types: vllm_model, local_vllm_model" in message
        assert "--allow-unsupported-pairing" in message

    def test_allows_a_declared_model_adapter(self) -> None:
        config = self._config(model_type="vllm_model", declared=["vllm_model"])

        GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

    @mark.parametrize("declared", [_ABSENT, None, []], ids=["key absent", "null", "empty list"])
    def test_resources_server_without_a_restriction_remains_compatible(self, declared) -> None:
        config = self._config(declared=declared)

        GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

    def test_accepts_a_bare_model_type_string(self) -> None:
        config = self._config(model_type="vllm_model", declared="vllm_model")

        GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

    @mark.parametrize("declared", [5, {"vllm_model": True}], ids=["scalar", "mapping"])
    def test_rejects_a_malformed_model_type_declaration(self, declared) -> None:
        config = self._config(declared=declared)

        with raises(ConfigError) as error:
            GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

        message = str(error.value)
        assert "audio_resources_server.resources_servers.audio.allowed_model_types" in message
        assert "must be a list of model types" in message

    def test_pairing_override_also_waives_model_compatibility(self) -> None:
        config = self._config(declared=["vllm_model"])
        config[ALLOW_UNSUPPORTED_PAIRING_KEY_NAME] = True

        GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

    def test_dummy_model_does_not_break_model_free_parser_clients(self) -> None:
        config = self._config(model_type="dummy_model", declared=["vllm_model"])

        GlobalConfigDictParser().raise_on_unsupported_model_pairings(config)

    @staticmethod
    def _parse_librispeech(model_config_path: str) -> DictConfig:
        initial = DictConfig(
            {
                "config_paths": [
                    "benchmarks/librispeech_pc/config.yaml",
                    model_config_path,
                ],
                "policy_base_url": "https://example.test/v1",
                "policy_api_key": "test-key",
                "policy_model_name": "test-model",
            }
        )
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=initial,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    def test_librispeech_config_fails_during_parse_with_openai_model(self) -> None:
        model_config_path = "responses_api_models/openai_model/configs/openai_model.yaml"

        with raises(UnsupportedModelPairingError) as error:
            self._parse_librispeech(model_config_path)

        message = str(error.value)
        assert "librispeech_pc_asr_with_pc_simple_agent" in message
        assert "type 'openai_model'" in message
        assert "--model-type vllm_model" in message

    def test_librispeech_config_resolves_with_vllm_model(self) -> None:
        model_config_path = "responses_api_models/vllm_model/configs/vllm_model.yaml"

        resolved = self._parse_librispeech(model_config_path)

        assert "vllm_model" in resolved["policy_model"]["responses_api_models"]


class TestComposeUnboundAgent:
    """Composition of a standalone agent config onto the environment's agent instances."""

    HARNESS_INSTANCE = "hermes_agent"

    def _harness(self, **overrides) -> dict:
        """The agent as its standalone config ships it, with `resources_server.name` left unset."""
        block = {
            "entrypoint": "app.py",
            "resources_server": {"type": "resources_servers", "name": "???"},
            # Deliberately not the environment's `policy_model`: assertions must be able to tell which
            # side a carried-over binding came from.
            "model_server": {"type": "responses_api_models", "name": "harness_model"},
            "max_turns": 30,
            "terminal_backend": "local",
        }
        block.update(overrides)
        return {"responses_api_agents": {"hermes_agent": block}}

    def _environment_agent(self, resources_server: str, **overrides) -> dict:
        block = {
            "entrypoint": "app.py",
            "resources_server": {"type": "resources_servers", "name": resources_server},
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
            "max_steps": 6,
            "datasets": [
                {
                    "name": "gpqa",
                    "type": "benchmark",
                    "jsonl_fpath": "benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl",
                    "prompt_config": "benchmarks/prompts/eval/aai/mcq-4choices.yaml",
                    "prepare_script": "benchmarks/gpqa/prepare.py",
                    "num_repeats": 8,
                    "license": "CC-BY-4.0",
                }
            ],
        }
        block.update(overrides)
        return {"responses_api_agents": {"simple_agent": block}}

    def _config(self, **extra) -> DictConfig:
        config = {
            self.HARNESS_INSTANCE: self._harness(),
            "gpqa_mcqa_simple_agent": self._environment_agent("gpqa_mcqa_resources_server"),
            "gpqa_mcqa_resources_server": {
                "resources_servers": {"mcqa": {"entrypoint": "app.py", "domain": "knowledge"}}
            },
        }
        config.update(extra)
        return DictConfig(config)

    def _composed_name(self, original: str) -> str:
        """The name an instance takes once rehosted: its environment prefix plus the new agent type."""
        stem = original.removesuffix("_simple_agent").removesuffix("simple_agent").rstrip("_")
        if stem == original:
            stem = original.removesuffix("_agent").rstrip("_")
        return f"{stem}_hermes_agent" if stem else "hermes_agent"

    def _composed_block(self, config: DictConfig, instance: str) -> DictConfig:
        renamed = self._composed_name(instance)
        assert instance == renamed or instance not in config, f"{instance} was not renamed after rehosting"
        agents = config[renamed]["responses_api_agents"]
        assert list(agents) == ["hermes_agent"], f"{renamed} was not rehosted on the harness"
        return agents["hermes_agent"]

    def _guarded_config(self, *required_agents: str) -> DictConfig:
        """The standard config with the environment's resources server declaring `allowed_agents`."""
        config = self._config()
        config["gpqa_mcqa_resources_server"]["resources_servers"]["mcqa"]["allowed_agents"] = list(required_agents)
        return config

    def _parse(self, config: DictConfig) -> DictConfig:
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, config
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    def _parse_config_paths(self, *paths: str) -> DictConfig:
        return self._parse(DictConfig({"config_paths": list(paths)}))

    def test_noop_when_no_instance_is_unbound(self) -> None:
        config = self._config()
        config.pop(self.HARNESS_INSTANCE)
        before = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert OmegaConf.to_container(config, resolve=False, throw_on_missing=False) == before

    def test_absent_resources_server_is_not_a_source(self) -> None:
        # Self-contained agents omit `resources_server` entirely, which is not the same as leaving it unset.
        self_contained = {
            "responses_api_agents": {"swe_agents": {"entrypoint": "app.py", "agent_framework": "openhands"}}
        }
        config = self._config()
        config.pop(self.HARNESS_INSTANCE)
        config["swe_agents"] = self_contained

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config["swe_agents"]["responses_api_agents"]) == ["swe_agents"]
        assert list(config["gpqa_mcqa_simple_agent"]["responses_api_agents"]) == ["simple_agent"]

    @mark.parametrize("declared", [None, []], ids=["key absent", "empty list"])
    def test_swap_allowed_when_the_verifier_declares_nothing(self, declared) -> None:
        config = self._config() if declared is None else self._guarded_config(*declared)

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]) == ["hermes_agent"]

    def test_swap_allowed_when_the_selected_agent_is_declared(self) -> None:
        config = self._guarded_config("simple_agent", "hermes_agent")

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]) == ["hermes_agent"]

    def _set_allowed_agents(self, declared) -> DictConfig:
        """The standard config with `allowed_agents` set to an arbitrary value, not just a list."""
        config = self._config()
        config["gpqa_mcqa_resources_server"]["resources_servers"]["mcqa"]["allowed_agents"] = declared
        return config

    def test_a_single_agent_may_be_declared_as_a_bare_string(self) -> None:
        config = self._set_allowed_agents("hermes_agent")

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert "gpqa_mcqa_hermes_agent" in config
        assert "gpqa_mcqa_simple_agent" not in config
        assert list(config["gpqa_mcqa_hermes_agent"]["responses_api_agents"]) == ["hermes_agent"]

    def test_a_bare_string_still_rejects_a_different_agent(self) -> None:
        config = self._set_allowed_agents("scicode_agent")

        with raises(UnsupportedAgentPairingError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        assert "accepts scicode_agent" in str(error.value)

    @mark.parametrize("declared", [5, {"a": 1}], ids=["scalar", "mapping"])
    def test_a_value_that_is_neither_string_nor_list_is_reported(self, declared) -> None:
        config = self._set_allowed_agents(declared)

        with raises(ConfigError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        message = str(error.value)
        assert "gpqa_mcqa_resources_server.resources_servers.mcqa.allowed_agents" in message
        assert "must be a list of agent types" in message

    def test_swap_rejected_when_the_selected_agent_is_not_declared(self) -> None:
        config = self._guarded_config("scicode_agent")

        with raises(UnsupportedAgentPairingError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        message = str(error.value)
        assert "gpqa_mcqa_simple_agent" in message
        assert "scicode_agent" in message and "hermes_agent" in message
        assert "--allow-unsupported-pairing" in message
        # The rejected swap must not have been applied.
        assert list(config["gpqa_mcqa_simple_agent"]["responses_api_agents"]) == ["simple_agent"]

    def test_override_config_key_allows_the_rejected_swap(self) -> None:
        config = self._guarded_config("scicode_agent")
        config[ALLOW_UNSUPPORTED_PAIRING_KEY_NAME] = True

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]) == ["hermes_agent"]

    @mark.parametrize("value, allowed", [("1", True), ("true", True), ("0", False), ("", False)])
    def test_override_env_var(self, monkeypatch: MonkeyPatch, value: str, allowed: bool) -> None:
        monkeypatch.setenv(ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, value)
        config = self._guarded_config("scicode_agent")

        if allowed:
            GlobalConfigDictParser().compose_unbound_agent(config)
            assert list(config[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]) == [
                "hermes_agent"
            ]
        else:
            with raises(UnsupportedAgentPairingError):
                GlobalConfigDictParser().compose_unbound_agent(config)

    def test_swap_allowed_when_the_requirement_cannot_be_determined(self) -> None:
        """The guard permits what it cannot read, so the permissive paths are asserted, not assumed.

        A server instance pins exactly one implementation, so a block with two is already invalid and is
        reported by almost-server detection; the guard must not turn it into a pairing error.
        """
        config = self._config()
        config["gpqa_mcqa_resources_server"]["resources_servers"]["extra"] = DictConfig(
            {"entrypoint": "app.py", "domain": "knowledge", "allowed_agents": ["scicode_agent"]}
        )

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]) == ["hermes_agent"]

    def test_every_target_is_checked_not_just_the_first(self) -> None:
        """A permissive verifier must not vouch for a restrictive one in the same config."""
        config = self._guarded_config("hermes_agent")
        config["strict_resources_server"] = DictConfig(
            {"resources_servers": {"scicode": {"entrypoint": "app.py", "allowed_agents": ["scicode_agent"]}}}
        )
        config["strict_agent"] = DictConfig(self._environment_agent("strict_resources_server"))

        with raises(UnsupportedAgentPairingError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        assert "strict_agent" in str(error.value)

    def test_reports_every_rejected_instance_at_once(self) -> None:
        """Fixing one instance per re-resolve is the slow path; name them all in one error."""
        config = self._guarded_config("simple_agent")
        for name, required in (("scicode", ["scicode_agent", "simple_agent"]), ("critpt", ["critpt_agent"])):
            config[f"{name}_resources_server"] = DictConfig(
                {"resources_servers": {name: {"entrypoint": "app.py", "allowed_agents": required}}}
            )
            config[f"{name}_agent"] = DictConfig(self._environment_agent(f"{name}_resources_server"))

        with raises(UnsupportedAgentPairingError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        message = str(error.value)
        # All three rejected instances, each with what it accepts, not just the first one hit.
        for instance in ("gpqa_mcqa_simple_agent", "scicode_agent", "critpt_agent"):
            assert instance in message, f"{instance} missing from:\n{message}"
        assert "3 of the agent instance(s)" in message
        # critpt accepts no agent the others do, so there is nothing to recommend.
        assert "No single agent satisfies all of them." in message

    def test_recommendation_accounts_for_targets_that_accept_the_selection(self) -> None:
        """An instance that accepts the selected agent still constrains what could replace it."""
        config = self._guarded_config("hermes_agent")
        config["strict_resources_server"] = DictConfig(
            {"resources_servers": {"scicode": {"entrypoint": "app.py", "allowed_agents": ["scicode_agent"]}}}
        )
        config["strict_agent"] = DictConfig(self._environment_agent("strict_resources_server"))

        with raises(UnsupportedAgentPairingError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        message = str(error.value)
        assert "No single agent satisfies all of them." in message
        assert "Select one of" not in message, "scicode_agent would only move the failure to gpqa"

    def test_recommends_the_agent_every_rejected_instance_accepts(self) -> None:
        config = self._guarded_config("scicode_agent", "simple_agent")
        config["critpt_resources_server"] = DictConfig(
            {
                "resources_servers": {
                    "critpt": {"entrypoint": "app.py", "allowed_agents": ["critpt_agent", "simple_agent"]}
                }
            }
        )
        config["critpt_agent"] = DictConfig(self._environment_agent("critpt_resources_server"))

        with raises(UnsupportedAgentPairingError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        assert "Select one of: simple_agent." in str(error.value)

    def test_instance_is_renamed_after_the_agent_that_runs_it(self) -> None:
        """#2657: the instance advertises the harness actually running it, not the one it was written for."""
        config = self._config()

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert "gpqa_mcqa_simple_agent" not in config
        assert list(config["gpqa_mcqa_hermes_agent"]["responses_api_agents"]) == ["hermes_agent"]

    @mark.parametrize(
        "original, expected",
        [
            ("gpqa_mcqa_simple_agent", "gpqa_mcqa_hermes_agent"),
            # Names not ending in their agent type keep their whole stem rather than losing part of it.
            ("osworld_nano_omni", "osworld_nano_omni_hermes_agent"),
            # Falls back to the generic `_agent` suffix rather than keeping the whole old name.
            ("scicode_benchmark_agent", "scicode_benchmark_hermes_agent"),
            ("simple_agent", "hermes_agent"),
        ],
    )
    def test_rename_substitutes_the_trailing_agent_type(self, original: str, expected: str) -> None:
        config = self._config()
        config[original] = DictConfig(self._environment_agent("gpqa_mcqa_resources_server"))
        if original != "gpqa_mcqa_simple_agent":
            config.pop("gpqa_mcqa_simple_agent")

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert expected in config, f"expected {expected}, got {sorted(config)}"

    def test_raises_when_the_rename_would_collide(self) -> None:
        config = self._config()
        config["gpqa_mcqa_hermes_agent"] = DictConfig(
            {"resources_servers": {"mcqa": {"entrypoint": "app.py", "domain": "knowledge"}}}
        )

        with raises(AgentCompositionError) as error:
            GlobalConfigDictParser().compose_unbound_agent(config)

        assert "same name" in str(error.value)

    def test_rehosts_environment_instance_on_the_standalone_agent(self) -> None:
        config = self._config()

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert self.HARNESS_INSTANCE not in config
        # _composed_block asserts the instance kept its name and now hosts the harness.
        block = self._composed_block(config, "gpqa_mcqa_simple_agent")

        # The harness's own configuration comes across.
        assert block["terminal_backend"] == "local"
        assert block["max_turns"] == 30

        # The environment's bindings and data are carried over, its agent parameters are not.
        assert block["resources_server"] == {"type": "resources_servers", "name": "gpqa_mcqa_resources_server"}
        assert block["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
        assert "max_steps" not in block
        assert OmegaConf.to_container(block["datasets"]) == [
            {
                "name": "gpqa",
                "type": "benchmark",
                "jsonl_fpath": "benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl",
                "prompt_config": "benchmarks/prompts/eval/aai/mcq-4choices.yaml",
                "prepare_script": "benchmarks/gpqa/prepare.py",
                "num_repeats": 8,
                "license": "CC-BY-4.0",
            }
        ]

    def test_preserves_model_server_per_instance(self) -> None:
        # Two instances differing only by model server, as genrm_compare does.
        config = self._config(
            genrm_reasoning_on=self._environment_agent(
                "genrm_resources_server",
                model_server={"type": "responses_api_models", "name": "policy_model"},
            ),
            genrm_reasoning_off=self._environment_agent(
                "genrm_resources_server",
                model_server={"type": "responses_api_models", "name": "policy_model_reasoning_off"},
            ),
        )

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert self._composed_block(config, "genrm_reasoning_on")["model_server"]["name"] == "policy_model"
        assert (
            self._composed_block(config, "genrm_reasoning_off")["model_server"]["name"] == "policy_model_reasoning_off"
        )

    def test_replaces_every_agent_instance(self) -> None:
        # jailbreak_detection declares five agent instances, one per category.
        categories = [f"jailbreak_{index}" for index in range(5)]
        config = self._config(**{name: self._environment_agent("jailbreak_resources_server") for name in categories})
        config.pop("gpqa_mcqa_simple_agent")

        GlobalConfigDictParser().compose_unbound_agent(config)

        for name in categories:
            assert self._composed_block(config, name)["resources_server"]["name"] == "jailbreak_resources_server"

    def test_falls_back_to_harness_model_when_environment_has_none(self) -> None:
        environment_agent = self._environment_agent("gpqa_mcqa_resources_server")
        environment_agent["responses_api_agents"]["simple_agent"].pop("model_server")
        config = self._config(no_model_server_agent=environment_agent)

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert self._composed_block(config, "no_model_server_agent")["model_server"]["name"] == "harness_model"

    def test_skips_an_instance_whose_agent_block_is_unset(self) -> None:
        # Resolving the block would raise MissingMandatoryValue, which is not a ConfigError, so the CLI
        # would print a traceback instead of the usual missing-value report.
        config = self._config(broken={"responses_api_agents": {"simple_agent": "???"}})

        names = {instance.name for instance in GlobalConfigDictParser()._agent_instances(config)}

        assert "broken" not in names

    def test_parse_reports_an_unset_agent_block(self) -> None:
        config = self._config(broken={"responses_api_agents": {"simple_agent": "???"}})

        with raises(ConfigMissingValuesError) as exc_info:
            self._parse(config)

        assert "broken.responses_api_agents.simple_agent" in str(exc_info.value)

    def test_carries_over_a_binding_the_environment_leaves_unset(self) -> None:
        # OmegaConf reports a '???' value as absent, so skipping on `in` alone let the incoming agent's own
        # model server stand in for one the environment deliberately left for the user to supply.
        config = self._config(unset_model=self._environment_agent("gpqa_mcqa_resources_server", model_server="???"))

        GlobalConfigDictParser().compose_unbound_agent(config)

        block = self._composed_block(config, "unset_model")
        assert OmegaConf.is_missing(block, "model_server"), "an unset binding must stay unset, not silently resolve"

    def test_carry_over_leaves_an_absent_binding_to_the_incoming_agent(self) -> None:
        # Absent is not unset: the environment simply has no opinion, so the agent's own value stands.
        environment = self._environment_agent("gpqa_mcqa_resources_server")
        del environment["responses_api_agents"]["simple_agent"]["model_server"]
        config = self._config(no_model=environment)

        GlobalConfigDictParser().compose_unbound_agent(config)

        harness_model = self._harness()["responses_api_agents"]["hermes_agent"]["model_server"]["name"]
        assert self._composed_block(config, "no_model")["model_server"]["name"] == harness_model

    def test_leaves_self_contained_agents_alone(self) -> None:
        # A self-contained agent declares no resources server, so there is no task to hand the new agent.
        self_contained = self._environment_agent("unused")
        agents = self_contained["responses_api_agents"]
        agents["tau2"] = agents.pop("simple_agent")
        del agents["tau2"]["resources_server"]
        config = self._config(tau2_benchmark_agent=self_contained)

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config["tau2_benchmark_agent"]["responses_api_agents"]) == ["tau2"]
        composed = self._composed_block(config, "gpqa_mcqa_simple_agent")
        assert composed["resources_server"]["name"] == "gpqa_mcqa_resources_server"

    def test_reads_a_self_contained_agent_from_a_struct_config(self) -> None:
        # Hydra hands the command line over in struct mode, where reading the absent `resources_server`
        # raises instead of reporting it missing.
        self_contained = self._environment_agent("unused")
        agents = self_contained["responses_api_agents"]
        agents["tau2"] = agents.pop("simple_agent")
        del agents["tau2"]["resources_server"]
        config = self._config(tau2_benchmark_agent=self_contained)
        OmegaConf.set_struct(config, True)

        GlobalConfigDictParser().compose_unbound_agent(config)

        assert list(config["tau2_benchmark_agent"]["responses_api_agents"]) == ["tau2"]

    def test_raises_when_two_instances_are_unbound(self) -> None:
        config = self._config(second_harness=self._harness())

        with raises(AgentCompositionError) as exc_info:
            GlobalConfigDictParser().compose_unbound_agent(config)

        message = str(exc_info.value)
        assert "2 agent instances leave their 'resources_server' unset" in message
        assert "'hermes_agent'" in message
        assert "'second_harness'" in message

    def test_raises_when_there_is_nothing_to_rehost(self) -> None:
        config = DictConfig({self.HARNESS_INSTANCE: self._harness()})

        with raises(AgentCompositionError) as exc_info:
            GlobalConfigDictParser().compose_unbound_agent(config)

        assert "no other agent instance to rehost it on" in str(exc_info.value)

    def test_parse_fills_unbound_binding_before_missing_value_check(self) -> None:
        resolved = self._parse(self._config())

        block = resolved[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]["hermes_agent"]
        assert block["resources_server"]["name"] == "gpqa_mcqa_resources_server"

    def test_raises_when_every_other_agent_is_self_contained(self) -> None:
        # Self-contained agents are left alone, so there is nothing here for the selected agent to run.
        environment_agent = self._environment_agent("gpqa_mcqa_resources_server")
        environment_agent["responses_api_agents"]["simple_agent"].pop("resources_server")
        config = self._config()
        config.pop("gpqa_mcqa_simple_agent")
        config["self_contained_agent"] = environment_agent

        with raises(AgentCompositionError) as exc_info:
            self._parse(config)

        assert "no other agent instance to rehost it on" in str(exc_info.value)

    def test_real_benchmark_composes_onto_real_harness(self) -> None:
        resolved = self._parse_config_paths(
            "benchmarks/gpqa/config.yaml",
            "responses_api_agents/hermes_agent/configs/hermes_agent.yaml",
        )

        block = resolved[self._composed_name("gpqa_mcqa_simple_agent")]["responses_api_agents"]["hermes_agent"]
        assert block["resources_server"]["name"] == "gpqa_mcqa_resources_server"
        assert block["max_turns"] == 30
        assert "max_steps" not in block
        assert OmegaConf.to_container(block["datasets"])[0]["num_repeats"] == 8
        assert "hermes_agent" not in resolved

    def _cli_dict(self, config: dict) -> DictConfig:
        """A command line dict as Hydra hands it over: struct, so an undeclared key is an error."""
        dict_config = DictConfig(config)
        OmegaConf.set_struct(dict_config, True)
        return dict_config

    def _cli_override(self, instance: str, **block) -> DictConfig:
        """An override addressed to an instance's agent, as `++<instance>...=<value>` produces."""
        return self._cli_dict({instance: {"responses_api_agents": {"hermes_agent": block}}})

    def _parse_with_cli(self, config: DictConfig, cli: DictConfig, monkeypatch: MonkeyPatch) -> DictConfig:
        monkeypatch.setattr(GlobalConfigDictParser, "parse_global_config_dict_from_cli", lambda self: cli)
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, config
                ),
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    def test_predicts_the_names_composition_will_produce(self) -> None:
        names = GlobalConfigDictParser()._composed_instance_names(self._config())

        assert names == {self._composed_name("gpqa_mcqa_simple_agent")}

    def test_predicts_no_names_when_nothing_is_unbound(self) -> None:
        config = self._config()
        config.pop(self.HARNESS_INSTANCE)

        assert GlobalConfigDictParser()._composed_instance_names(config) == set()

    def test_applies_command_line_override_to_the_composed_agent(self, monkeypatch: MonkeyPatch) -> None:
        renamed = self._composed_name("gpqa_mcqa_simple_agent")

        resolved = self._parse_with_cli(self._config(), self._cli_override(renamed, max_turns=99), monkeypatch)

        assert resolved[renamed]["responses_api_agents"]["hermes_agent"]["max_turns"] == 99

    def test_command_line_override_outranks_the_carried_over_bindings(self, monkeypatch: MonkeyPatch) -> None:
        # The override must carry only the fields the user set, leaving the environment's bindings intact.
        config = self._config(
            **{
                self.HARNESS_INSTANCE: self._harness(
                    model_server={"type": "responses_api_models", "name": "policy_model"}
                )
            }
        )
        renamed = self._composed_name("gpqa_mcqa_simple_agent")

        resolved = self._parse_with_cli(config, self._cli_override(renamed, max_turns=99), monkeypatch)

        block = resolved[renamed]["responses_api_agents"]["hermes_agent"]
        assert block["resources_server"]["name"] == "gpqa_mcqa_resources_server"
        assert block["max_turns"] == 99

    def test_holds_back_only_the_names_composition_produces(self) -> None:
        renamed = self._composed_name("gpqa_mcqa_simple_agent")
        cli = self._cli_dict(
            {
                renamed: {"responses_api_agents": {"hermes_agent": {"max_turns": 99}}},
                "some_other_server": {"responses_api_models": {"a_model": {"entrypoint": "app.py"}}},
            }
        )

        held = GlobalConfigDictParser()._hold_back_composed_agent_overrides(cli, self._config())

        assert list(held) == [renamed], "only the composed instance should be held back"
        assert list(cli) == ["some_other_server"], "everything else must still reach the merge"

    def test_rejects_an_override_naming_an_instance_that_never_appears(self, monkeypatch: MonkeyPatch) -> None:
        # The names are predicted before inheritance runs, which can drop an instance a base defined. An
        # override held for one of those would otherwise vanish without trace.
        config = self._config(base_simple_agent=self._environment_agent("gpqa_mcqa_resources_server"))
        config["gpqa_mcqa_simple_agent"]["_inherit_from"] = "base_simple_agent"
        cli = self._cli_override(self._composed_name("base_simple_agent"), max_turns=99)

        with raises(UnsupportedAgentOverrideError) as exc_info:
            self._parse_with_cli(config, cli, monkeypatch)

        assert self._composed_name("base_simple_agent") in str(exc_info.value)

    def test_rejects_a_field_the_incoming_agent_does_not_declare(self, monkeypatch: MonkeyPatch) -> None:
        renamed = self._composed_name("gpqa_mcqa_simple_agent")

        with raises(ConfigKeyError):
            self._parse_with_cli(self._config(), self._cli_override(renamed, no_such_field=1), monkeypatch)
