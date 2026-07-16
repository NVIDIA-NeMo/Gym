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

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.registry import _discover_environments_in_dir, discover_environments


def _make_env(environments_dir: Path, name: str, config_body: str) -> Path:
    env_dir = environments_dir / name
    env_dir.mkdir(parents=True)
    config_path = env_dir / "config.yaml"
    config_path.write_text(config_body)
    return config_path


_ENV_CONFIG = """{name}:
  resources_servers:
    {name}:
      entrypoint: app.py
      domain: agent
      description: {name} test environment
{name}_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: {name}
      model_server:
        type: responses_api_models
        name: policy_model
"""


class TestDiscoverEnvironments:
    def test_discovers_by_directory_name_with_metadata(self, tmp_path: Path) -> None:
        envs_dir = tmp_path / "environments"
        _make_env(envs_dir, "alpha", _ENV_CONFIG.format(name="alpha"))
        _make_env(envs_dir, "beta", _ENV_CONFIG.format(name="beta"))

        environments = _discover_environments_in_dir(envs_dir)

        assert set(environments) == {"alpha", "beta"}
        alpha = environments["alpha"]
        assert alpha.name == "alpha"
        assert alpha.config_path == envs_dir / "alpha" / "config.yaml"
        assert alpha.path == envs_dir / "alpha"
        assert alpha.description == "alpha test environment"
        assert alpha.domain == "agent"

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert _discover_environments_in_dir(tmp_path / "does_not_exist") == {}

    def test_ignores_dirs_without_config_and_loose_files(self, tmp_path: Path) -> None:
        envs_dir = tmp_path / "environments"
        _make_env(envs_dir, "real", _ENV_CONFIG.format(name="real"))
        (envs_dir / "not_an_env").mkdir()  # dir without a config.yaml
        (envs_dir / "__init__.py").write_text("")  # loose file

        assert set(_discover_environments_in_dir(envs_dir)) == {"real"}

    def test_unparseable_or_metadataless_configs_still_discovered(self, tmp_path: Path) -> None:
        # Configs without a parseable resources_servers block (or malformed YAML) must still be
        # discovered by name, just with no description/domain — never crash discovery.
        envs_dir = tmp_path / "environments"
        _make_env(envs_dir, "no_rs", "agent_only:\n  responses_api_agents:\n    a: {}\n")  # no resources_servers
        _make_env(envs_dir, "top_list", "- x\n- y\n")  # top-level not a mapping
        _make_env(envs_dir, "scalar_top", "top: just_a_string\n")  # top-level value not a dict
        _make_env(envs_dir, "rs_not_dict", "top:\n  resources_servers: not_a_mapping\n")  # rs not a dict
        _make_env(envs_dir, "broken", "key: [unclosed\n")  # malformed YAML -> load raises

        environments = _discover_environments_in_dir(envs_dir)

        assert set(environments) == {"no_rs", "top_list", "scalar_top", "rs_not_dict", "broken"}
        for entry in environments.values():
            assert entry.description is None
            assert entry.domain is None

    def test_metadata_tolerates_unset_interpolations(self, tmp_path: Path) -> None:
        # A config referencing an unset interpolation must still be discoverable: the shared metadata
        # reader reads the inline `domain` from the raw config and tolerates the unresolved value.
        envs_dir = tmp_path / "environments"
        _make_env(
            envs_dir,
            "needs_key",
            "needs_key:\n"
            "  resources_servers:\n"
            "    needs_key:\n"
            "      entrypoint: app.py\n"
            "      domain: other\n"
            "      api_key: ${some_unset_key}\n",
        )

        environments = _discover_environments_in_dir(envs_dir)
        assert "needs_key" in environments
        assert environments["needs_key"].domain == "other"


class TestRealEnvironments:
    def test_workplace_assistant_is_discoverable(self) -> None:
        # The repo ships environments/workplace_assistant/ — the registry must find it by name.
        environments = discover_environments()
        assert "workplace_assistant" in environments
        assert environments["workplace_assistant"].config_path.name == "config.yaml"


class TestDiscoverEnvironmentsAcrossRoots:
    def test_extra_root_surfaces_user_environments_alongside_builtins(self, tmp_path: Path, monkeypatch) -> None:
        _make_env(tmp_path / "environments", "custom_env", _ENV_CONFIG.format(name="custom_env"))
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))

        environments = discover_environments()

        assert "custom_env" in environments  # a user-supplied environment is discovered
        assert "workplace_assistant" in environments  # ...alongside the built-ins

    def test_cwd_is_scanned_by_default(self, tmp_path: Path, monkeypatch) -> None:
        _make_env(tmp_path / "environments", "cwd_env", _ENV_CONFIG.format(name="cwd_env"))
        monkeypatch.chdir(tmp_path)

        assert "cwd_env" in discover_environments()
