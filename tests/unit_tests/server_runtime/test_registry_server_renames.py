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
"""What the head server and its dependency keys were called before their renames keeps working, with a warning."""

import importlib
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

import nemo_gym.client.server_client as server_client_module
import nemo_gym.rollouts.rollout_collection as rollout_collection_module
from nemo_gym._moved_paths import _RENAMED_REGISTRY_SERVER_NAMES, moved_names_getattr
from nemo_gym.client.server_client import ServerClient, setup_server_client
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.config.renamed_keys import ConflictingRenamedKeyError
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper
from nemo_gym.rollouts.rollout_recovery import _configuration_identity


REGISTRY = BaseServerConfig(host="127.0.0.1", port=12345)

RENAMED = sorted(
    (module, name, moved.new)
    for module, names in _RENAMED_REGISTRY_SERVER_NAMES.items()
    for name, moved in names.items()
)


def _new_object(target: str) -> object:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(("old_module", "old_name", "target"), RENAMED)
def test_an_old_name_resolves_to_its_renamed_object_with_a_warning(
    old_module: str, old_name: str, target: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module(old_module)
        # Through the module itself, so a module that lost its `__getattr__` fails here.
        assert getattr(module, old_name) is _new_object(target)
    # A fresh getter keeps its own warn-once record, so an earlier read of the old name cannot swallow the warning.
    with pytest.warns(DeprecationWarning, match=rf"{old_name}.*{target.partition(':')[2]}"):
        assert moved_names_getattr(module.__name__)(old_name) is _new_object(target)


def _parse(*, initial: dict | None = None, cli: dict | None = None, config_paths: list[str] | None = None) -> dict:
    # Stands in for Hydra, which reads sys.argv; the parser merges what it returns as the command line.
    with patch.object(GlobalConfigDictParser, "parse_global_config_dict_from_cli", lambda self: DictConfig(cli or {})):
        parsed = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig({**(initial or {}), "config_paths": config_paths or []}),
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
    return OmegaConf.to_container(parsed)


@pytest.fixture
def expected() -> dict:
    parsed = _parse(initial={"registry_server": {"port": 12345}})
    assert parsed["registry_server"] == {"host": "127.0.0.1", "port": 12345}
    return parsed


def test_the_old_key_in_the_initial_config_parses_like_the_new_key(expected: dict) -> None:
    # NeMo RL passes the registry server's address in the initial config.
    with pytest.warns(DeprecationWarning, match="'head_server' was renamed 'registry_server'"):
        parsed = _parse(initial={"head_server": {"port": 12345}})
    assert parsed == expected


def test_the_old_key_in_a_yaml_config_parses_like_the_new_key(tmp_path: Path, expected: dict) -> None:
    config_path = tmp_path / "old_key.yaml"
    config_path.write_text(OmegaConf.to_yaml({"head_server": {"port": 12345}}))
    with pytest.warns(DeprecationWarning, match="'registry_server'"):
        parsed = _parse(config_paths=[str(config_path)])
    parsed.pop("config_paths")
    expected.pop("config_paths")
    assert parsed == expected


def test_the_old_key_on_the_command_line_parses_like_the_new_key(expected: dict) -> None:
    with pytest.warns(DeprecationWarning, match="'registry_server'"):
        parsed = _parse(cli={"head_server": {"port": 12345}})
    assert parsed == expected


def test_the_old_and_the_new_key_set_to_different_values_is_an_error() -> None:
    with pytest.raises(ConflictingRenamedKeyError, match="remove 'head_server' and keep only 'registry_server'"):
        _parse(cli={"head_server": {"port": 12345}, "registry_server": {"port": 12346}})


def test_the_old_and_the_new_key_set_to_the_same_value_parse_like_the_new_key(expected: dict) -> None:
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial={"head_server": {"port": 12345}, "registry_server": {"port": 12345}})
    assert parsed == expected


@pytest.mark.parametrize("old_key", ["head_server_deps", "head_server_constraints"])
def test_an_old_dependency_key_is_dropped(old_key: str, expected: dict) -> None:
    # The parser computes the dependency keys on every parse, so a value a config sets was never used, before or after.
    parsed = _parse(initial={"registry_server": {"port": 12345}, old_key: ["ignored==1.0"]})
    assert old_key not in parsed
    assert parsed == expected


def test_a_run_saved_before_the_rename_keeps_its_identity() -> None:
    # Resuming compares this identity with the one saved with the run, so the key's new name must not change it.
    servers = {"my_agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", "port": 1}}}}
    before = _configuration_identity({}, {**servers, "head_server": {"host": "127.0.0.1", "port": 11000}}, [])
    after = _configuration_identity({}, {**servers, "registry_server": {"host": "127.0.0.1", "port": 12345}}, [])
    assert before == after


def test_a_server_client_built_with_the_old_field_name_warns() -> None:
    with pytest.warns(DeprecationWarning, match="'head_server_config' was renamed 'registry_server_config'"):
        client = ServerClient(head_server_config=REGISTRY, global_config_dict=DictConfig({}))
    assert client.registry_server_config == REGISTRY
    with pytest.warns(DeprecationWarning, match="ServerClient.head_server_config was renamed registry_server_config"):
        assert client.head_server_config == REGISTRY


def test_a_server_client_built_with_both_field_names_set_differently_is_an_error() -> None:
    other = BaseServerConfig(host="127.0.0.1", port=12346)
    with pytest.raises(ValueError, match="remove 'head_server_config' and keep only 'registry_server_config'"):
        ServerClient(head_server_config=REGISTRY, registry_server_config=other, global_config_dict=DictConfig({}))


def test_the_old_config_loader_returns_the_registry_server_config(monkeypatch: pytest.MonkeyPatch) -> None:
    global_config = DictConfig({"registry_server": {"host": "127.0.0.1", "port": 12345}})
    monkeypatch.setattr(server_client_module, "get_global_config_dict", lambda: global_config)
    with pytest.warns(DeprecationWarning, match="load_head_server_config was renamed load_registry_server_config"):
        assert ServerClient.load_head_server_config() == REGISTRY


def test_building_a_client_accepts_the_old_argument_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_client_module, "_has_injected_global_config_env", lambda: True)
    monkeypatch.setattr(server_client_module, "get_global_config_dict", lambda: DictConfig({}))
    with pytest.warns(DeprecationWarning, match="argument 'head_server_config' was renamed 'registry_server_config'"):
        client = ServerClient.load_from_global_config(head_server_config=REGISTRY)
    assert client.registry_server_config == REGISTRY


def test_passing_both_argument_names_is_an_error() -> None:
    with pytest.raises(TypeError, match="got both 'head_server_config' and 'registry_server_config'"):
        ServerClient.load_from_global_config(head_server_config=REGISTRY, registry_server_config=REGISTRY)


def test_setup_server_client_accepts_the_old_argument_name(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []

    def load_from_global_config(registry_server_config: BaseServerConfig) -> Any:
        received.append(registry_server_config)
        raise RuntimeError("stop after the config is passed on")

    monkeypatch.setattr(ServerClient, "load_from_global_config", load_from_global_config)
    with pytest.warns(DeprecationWarning, match="'registry_server_config'"), pytest.raises(RuntimeError):
        setup_server_client(head_server_config=REGISTRY)
    assert received == [REGISTRY]


def test_the_helpers_client_setup_accepts_the_old_argument_name(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr(rollout_collection_module, "setup_server_client_utils", received.append)
    with pytest.warns(DeprecationWarning, match="'registry_server_config'"):
        RolloutCollectionHelper().setup_server_client(head_server_config=REGISTRY)
    assert received == [REGISTRY]


@pytest.mark.parametrize("method", ["run_examples", "run_outcomes"])
def test_running_examples_accepts_the_old_argument_name(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # NeMo RL passes the registry server's config to run_examples by keyword.
    received = []

    def run_examples_with_metadata(self: Any, examples: list, **kwargs: Any) -> Any:
        received.append(kwargs["registry_server_config"])
        return iter(())

    monkeypatch.setattr(RolloutCollectionHelper, "_run_examples_with_metadata", run_examples_with_metadata)
    with pytest.warns(DeprecationWarning, match="'registry_server_config'"):
        getattr(RolloutCollectionHelper(), method)([], head_server_config=REGISTRY)
    assert received == [REGISTRY]


def test_the_wait_script_reads_the_old_wait_variable(tmp_path: Path) -> None:
    script = Path(__file__).parents[3] / "scripts" / "wait_for_servers.sh"
    # This process stands in for `gym env start`; nothing listens on the port, so the wait runs out.
    env = {**os.environ, "HEAD_MAX_WAIT": "2"}
    env.pop("REGISTRY_MAX_WAIT", None)
    result = subprocess.run(
        ["bash", str(script), str(os.getpid()), "1"], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 1
    assert "Registry server did not respond within 2s" in result.stdout
