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
"""A config that sets a token capture key under its name from before the rename parses exactly as one that sets the
current name."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.config.renamed_keys import ConflictingRenamedKeyError


BLOCK = {"enabled": True, "dir": "/tmp/capture", "delta_records": True}
DECLARED = [{"method": "GET", "path": "/custom/metadata"}]


def _servers(agent_key: str, model_key: str) -> dict[str, Any]:
    return {
        "policy_model": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py", model_key: DECLARED}}},
        "my_agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", agent_key: True}}},
    }


OLD = {"token_id_capture": BLOCK, **_servers("token_id_capture", "token_id_capture_non_generating_requests")}
NEW = {"token_capture": BLOCK, **_servers("token_capture", "token_capture_non_generating_requests")}


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


def _capture_settings(parsed: dict) -> dict:
    agent = parsed["my_agent"]["responses_api_agents"]["simple_agent"]
    model = parsed["policy_model"]["responses_api_models"]["vllm_model"]
    return {
        "block": parsed.get("token_capture"),
        "agent": agent.get("token_capture"),
        "model": model.get("token_capture_non_generating_requests"),
        "old keys left": "token_id_capture" in parsed
        or "token_id_capture" in agent
        or "token_id_capture_non_generating_requests" in model,
    }


@pytest.fixture
def expected() -> dict:
    parsed = _parse(initial=NEW)
    assert _capture_settings(parsed) == {"block": BLOCK, "agent": True, "model": DECLARED, "old keys left": False}
    return parsed


def test_old_keys_in_the_initial_config_parse_like_new_keys(expected: dict) -> None:
    with pytest.warns(DeprecationWarning) as record:
        parsed = _parse(initial=OLD)
    assert parsed == expected
    messages = "\n".join(str(warning.message) for warning in record)
    assert "'token_id_capture' was renamed 'token_capture'" in messages
    assert "'my_agent.responses_api_agents.simple_agent.token_id_capture' was renamed" in messages
    assert (
        "'policy_model.responses_api_models.vllm_model.token_id_capture_non_generating_requests' was renamed"
        in messages
    )


def test_old_keys_in_a_yaml_config_parse_like_new_keys(tmp_path: Path, expected: dict) -> None:
    config_path = tmp_path / "old_keys.yaml"
    config_path.write_text(OmegaConf.to_yaml(OLD))
    with pytest.warns(DeprecationWarning, match="'token_capture'"):
        parsed = _parse(config_paths=[str(config_path)])
    parsed.pop("config_paths")
    expected.pop("config_paths")
    assert parsed == expected


def test_an_old_key_on_the_command_line_parses_like_the_new_key(expected: dict) -> None:
    servers = _servers("token_capture", "token_capture_non_generating_requests")
    with pytest.warns(DeprecationWarning, match="'token_capture'"):
        parsed = _parse(initial=servers, cli={"token_id_capture": BLOCK})
    assert parsed == expected


def test_an_old_key_on_the_command_line_overrides_an_old_key_in_a_config(tmp_path: Path, expected: dict) -> None:
    config_path = tmp_path / "old_keys.yaml"
    config_path.write_text(OmegaConf.to_yaml({**OLD, "token_id_capture": {**BLOCK, "enabled": False}}))
    with pytest.warns(DeprecationWarning):
        parsed = _parse(config_paths=[str(config_path)], cli={"token_id_capture": {"enabled": True}})
    assert parsed["token_capture"] == BLOCK


@pytest.mark.parametrize(
    "conflict",
    [
        {"token_capture": {**BLOCK, "enabled": False}},
        {"my_agent": {"responses_api_agents": {"simple_agent": {"token_capture": False}}}},
        {"policy_model": {"responses_api_models": {"vllm_model": {"token_capture_non_generating_requests": []}}}},
    ],
)
def test_an_old_and_a_new_key_set_to_different_values_in_one_source_is_an_error(conflict: dict) -> None:
    with pytest.raises(ConflictingRenamedKeyError, match="keep only"):
        _parse(cli=OmegaConf.to_container(OmegaConf.merge(OLD, conflict)))


def test_a_new_key_in_a_higher_precedence_source_overrides_an_old_key() -> None:
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial=OLD, cli={"token_capture": {**BLOCK, "enabled": False}})
    assert parsed["token_capture"] == {**BLOCK, "enabled": False}


def test_an_old_and_a_new_key_set_to_the_same_value_parse_like_the_new_key(expected: dict) -> None:
    with pytest.warns(DeprecationWarning):
        parsed = _parse(initial={**OLD, "token_capture": BLOCK})
    assert parsed == expected
