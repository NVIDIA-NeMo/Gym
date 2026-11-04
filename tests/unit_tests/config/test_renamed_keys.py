# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A config, model or call that uses a renamed key under its old name behaves exactly as one that uses the new name."""

import asyncio
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, model_validator

from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.config.renamed_keys import (
    RENAMED_SERVER_INSTANCE_KEYS,
    RENAMED_TOP_LEVEL_KEYS,
    ConflictingRenamedKeyError,
    rename_deprecated_keys,
    renamed_attribute,
    renamed_keyword,
    with_renamed_keys,
)


BLOCK = {"enabled": True, "dir": "/tmp/example"}


def _servers(agent_key: str) -> dict[str, Any]:
    return {"my_agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", agent_key: True}}}}


OLD = {"old_block": BLOCK, **_servers("old_flag")}
NEW = {"new_block": BLOCK, **_servers("new_flag")}


@pytest.fixture(autouse=True)
def renames(monkeypatch: pytest.MonkeyPatch) -> None:
    # Example renames, one top-level key and one agent instance key, added to the registered ones for each test.
    monkeypatch.setitem(RENAMED_TOP_LEVEL_KEYS, "old_block", "new_block")
    monkeypatch.setitem(RENAMED_SERVER_INSTANCE_KEYS, "responses_api_agents", {"old_flag": "new_flag"})


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
    parsed = _parse(initial=NEW)
    assert parsed["new_block"] == BLOCK and "old_block" not in parsed
    agent = parsed["my_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["new_flag"] is True and "old_flag" not in agent
    return parsed


def test_old_keys_in_the_initial_config_parse_like_new_keys(expected: dict) -> None:
    with pytest.warns(DeprecationWarning) as record:
        parsed = _parse(initial=OLD)
    assert parsed == expected
    messages = "\n".join(str(warning.message) for warning in record)
    assert "'old_block' was renamed 'new_block'" in messages
    assert "'my_agent.responses_api_agents.simple_agent.old_flag' was renamed" in messages


def test_old_keys_in_a_yaml_config_parse_like_new_keys(tmp_path: Path, expected: dict) -> None:
    config_path = tmp_path / "old_keys.yaml"
    config_path.write_text(OmegaConf.to_yaml(OLD))
    with pytest.warns(DeprecationWarning, match="'new_block'"):
        parsed = _parse(config_paths=[str(config_path)])
    parsed.pop("config_paths")
    expected.pop("config_paths")
    assert parsed == expected


def test_an_old_key_on_the_command_line_parses_like_the_new_key(expected: dict) -> None:
    with pytest.warns(DeprecationWarning, match="'new_block'"):
        parsed = _parse(initial=_servers("new_flag"), cli={"old_block": BLOCK})
    assert parsed == expected


def test_an_old_key_on_the_command_line_overrides_an_old_key_in_a_config(tmp_path: Path, expected: dict) -> None:
    config_path = tmp_path / "old_keys.yaml"
    config_path.write_text(OmegaConf.to_yaml({**OLD, "old_block": {**BLOCK, "enabled": False}}))
    with pytest.warns(DeprecationWarning):
        parsed = _parse(config_paths=[str(config_path)], cli={"old_block": {"enabled": True}})
    assert parsed["new_block"] == BLOCK


@pytest.mark.parametrize("source", ["cli", "initial"])
def test_an_old_key_overrides_the_new_key_from_a_lower_precedence_config(
    tmp_path: Path, expected: dict, source: str
) -> None:
    # A shipped config already uses the new keys; an override that still uses the old ones must win, not conflict.
    config_path = tmp_path / "new_keys.yaml"
    shipped = {
        "new_block": {"enabled": False},
        "my_agent": {"responses_api_agents": {"simple_agent": {"new_flag": False}}},
    }
    config_path.write_text(OmegaConf.to_yaml(OmegaConf.merge(NEW, shipped)))
    with pytest.warns(DeprecationWarning):
        parsed = _parse(config_paths=[str(config_path)], **{source: OLD})
    parsed.pop("config_paths")
    expected.pop("config_paths")
    assert parsed == expected


def test_an_old_key_overrides_the_new_key_it_inherits() -> None:
    shipped = {"shipped": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", "new_flag": False}}}}
    mine = {"_inherit_from": "shipped.responses_api_agents.simple_agent", "old_flag": True}
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial={**NEW, **shipped, "mine": {"responses_api_agents": {"simple_agent": mine}}})
    agent = parsed["mine"]["responses_api_agents"]["simple_agent"]
    assert agent["new_flag"] is True and "old_flag" not in agent


def test_an_old_key_fills_a_new_key_a_config_leaves_unset(tmp_path: Path, expected: dict) -> None:
    config_path = tmp_path / "required.yaml"
    config_path.write_text(OmegaConf.to_yaml({"new_block": "???", **_servers("new_flag")}))
    with pytest.warns(DeprecationWarning):
        parsed = _parse(config_paths=[str(config_path)], cli={"old_block": BLOCK})
    parsed.pop("config_paths")
    expected.pop("config_paths")
    assert parsed == expected


def test_a_config_that_refers_to_another_config_is_renamed_without_resolving(tmp_path: Path) -> None:
    # On its own, this config cannot resolve its reference; only the merged config can.
    config_path = tmp_path / "refers.yaml"
    config_path.write_text(OmegaConf.to_yaml({**OLD, "block_dir": "${defined_elsewhere}"}))
    with pytest.warns(DeprecationWarning):
        parsed = _parse(config_paths=[str(config_path)], initial={"defined_elsewhere": "/tmp/elsewhere"})
    assert parsed["new_block"] == BLOCK and parsed["block_dir"] == "${defined_elsewhere}"


@pytest.mark.parametrize(
    "conflict",
    [
        {"new_block": {**BLOCK, "enabled": False}},
        {"my_agent": {"responses_api_agents": {"simple_agent": {"new_flag": False}}}},
    ],
)
@pytest.mark.parametrize("source", ["cli", "initial"])
def test_an_old_and_a_new_key_set_to_different_values_in_one_config_is_an_error(conflict: dict, source: str) -> None:
    with pytest.raises(ConflictingRenamedKeyError, match="keep only"):
        _parse(**{source: OmegaConf.to_container(OmegaConf.merge(OLD, conflict))})


def test_a_conflict_on_the_command_line_is_reported_without_a_traceback(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exit_info:
        exit_cleanly_on_config_error(_parse)(cli={**OLD, "new_block": {**BLOCK, "enabled": False}})
    assert exit_info.value.code == 1
    assert capsys.readouterr().out.startswith("Error: Config sets both 'old_block' and 'new_block'")


def test_a_new_key_on_the_command_line_overrides_an_old_key_in_the_initial_config(expected: dict) -> None:
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial={"old_block": {**BLOCK, "enabled": False}, **_servers("old_flag")}, cli=NEW)
    assert parsed == expected


def test_an_old_and_a_new_key_set_to_the_same_value_parse_like_the_new_key(expected: dict) -> None:
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial={**OLD, "new_block": BLOCK})
    assert parsed == expected


def test_an_instance_key_of_another_server_type_is_left_alone() -> None:
    model = {"policy_model": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py", "old_flag": True}}}}
    parsed = _parse(initial={**NEW, **model})
    assert parsed["policy_model"]["responses_api_models"]["vllm_model"]["old_flag"] is True


def test_an_old_key_set_to_an_interpolation_keeps_it_unresolved() -> None:
    # A child server resolves the config itself, so a reference such as ${oc.env:...} must reach it as written.
    servers = {"my_agent": {"responses_api_agents": {"simple_agent": {"old_flag": "${flag_source}"}}}}
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial={"new_block": BLOCK, **servers, "flag_source": True})
    agent = parsed["my_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["new_flag"] == "${flag_source}" and "old_flag" not in agent


def test_an_old_and_a_new_key_set_to_equal_values_keep_the_new_value() -> None:
    config = {"old": 1, "new": True}
    with pytest.warns(DeprecationWarning):
        rename_deprecated_keys(config, {"old": "new"})
    assert config == {"new": True} and config["new"] is True


def test_renaming_moves_each_old_key_and_leaves_other_keys_alone() -> None:
    config = {"old": 1, "unrelated": 2}
    with pytest.warns(DeprecationWarning, match="'server.old' was renamed 'server.new'"):
        rename_deprecated_keys(config, {"old": "new", "absent": "other"}, where="server")
    assert config == {"new": 1, "unrelated": 2}


class _Settings(BaseModel):
    new_name: int

    _accept_renamed_keys = model_validator(mode="before")(
        classmethod(lambda cls, data: with_renamed_keys(data, {"old_name": "new_name"}))
    )
    old_name = renamed_attribute("old_name", "new_name", owner="_Settings")


def test_a_model_built_with_an_old_field_name_reads_under_both_names() -> None:
    data = {"old_name": 3}
    with pytest.warns(DeprecationWarning, match="'old_name' was renamed 'new_name'"):
        settings = _Settings.model_validate(data)
    assert settings.new_name == 3
    assert data == {"old_name": 3}
    with pytest.warns(DeprecationWarning, match=r"_Settings\.old_name was renamed new_name"):
        assert settings.old_name == 3


@renamed_keyword("old_arg", "new_arg")
def _call(*, new_arg: int = 0) -> int:
    return new_arg


@renamed_keyword("old_arg", "new_arg")
async def _call_async(*, new_arg: int = 0) -> int:
    return new_arg


def test_a_call_with_an_old_argument_name_passes_it_under_the_new_name() -> None:
    with pytest.warns(DeprecationWarning, match="argument 'old_arg' was renamed 'new_arg'") as record:
        assert _call(old_arg=5) == 5
    # The warning points at the caller, not at the decorator.
    assert record[0].filename == __file__


def test_an_async_call_with_an_old_argument_name_stays_async() -> None:
    assert inspect.iscoroutinefunction(_call_async)
    with pytest.warns(DeprecationWarning, match="'old_arg' was renamed 'new_arg'"):
        assert asyncio.run(_call_async(old_arg=7)) == 7


def test_passing_both_argument_names_is_an_error() -> None:
    with pytest.raises(TypeError, match="got both 'old_arg' and 'new_arg'"):
        _call(old_arg=1, new_arg=2)
