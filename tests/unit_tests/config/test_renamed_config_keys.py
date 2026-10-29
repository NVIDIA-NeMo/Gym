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
"""The old token_id_capture config keys and names keep working, under their new token_capture names."""

import importlib
import logging
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

import nemo_gym.config.global_config
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import (
    TOKEN_CAPTURE_BLOCK,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    _load_config_yaml,
    rename_deprecated_config_keys,
)
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig
from nemo_gym.models.token_capture.config import TokenCaptureConfig


OLD_STYLE = {
    "token_id_capture": {"enabled": True, "dir": "/tmp/capture"},
    "policy": {
        "responses_api_models": {
            "vllm_model": {"token_id_capture_non_generating_requests": [{"method": "POST", "path": "/tokenize"}]}
        }
    },
    "agent": {"responses_api_agents": {"simple_agent": {"token_id_capture": True}}},
}


def test_old_keys_become_new_keys_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    config = OmegaConf.create(OLD_STYLE)
    with caplog.at_level(logging.WARNING, logger=nemo_gym.config.global_config.__name__):
        rename_deprecated_config_keys(config, source="recipe.yaml")

    assert OmegaConf.to_container(config) == {
        "token_capture": {"enabled": True, "dir": "/tmp/capture"},
        "policy": {
            "responses_api_models": {
                "vllm_model": {"token_capture_non_generating_requests": [{"method": "POST", "path": "/tokenize"}]}
            }
        },
        "agent": {"responses_api_agents": {"simple_agent": {"token_capture": True}}},
    }
    [record] = caplog.records
    assert "recipe.yaml" in record.getMessage()
    assert "agent.responses_api_agents.simple_agent.token_id_capture" in record.getMessage()


def test_new_keys_are_left_alone_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    config = OmegaConf.create({"token_capture": {"enabled": False}})
    with caplog.at_level(logging.WARNING):
        rename_deprecated_config_keys(config, source="recipe.yaml")
    assert OmegaConf.to_container(config) == {"token_capture": {"enabled": False}}
    assert not caplog.records


def test_setting_both_spellings_is_refused() -> None:
    config = OmegaConf.create({"token_id_capture": {"enabled": True}, "token_capture": {"enabled": False}})
    with pytest.raises(ConfigError, match="'token_id_capture' and 'token_capture' are both set in recipe.yaml"):
        rename_deprecated_config_keys(config, source="recipe.yaml")


def test_a_config_file_with_old_keys_loads_with_new_keys(tmp_path: Path) -> None:
    path = tmp_path / "recipe.yaml"
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(OLD_STYLE)))
    loaded = _load_config_yaml(path)
    assert loaded.token_capture.enabled is True
    assert "token_id_capture" not in loaded


def test_an_old_key_from_one_source_and_a_new_key_from_another_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each source is rewritten before the merge, so mixing spellings across sources is not a conflict."""
    monkeypatch.setattr(nemo_gym.config.global_config, "setup_exporters", lambda *args, **kwargs: None)
    parsed = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(
                {
                    **GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    "token_id_capture": {"enabled": True, "dir": "/tmp/capture"},
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert parsed[TOKEN_CAPTURE_BLOCK].enabled is True
    assert "token_id_capture" not in parsed


def test_models_built_without_the_loader_accept_the_old_keys() -> None:
    assert TokenCaptureConfig.model_validate({"token_id_capture": {"enabled": True, "dir": "/tmp/c"}}).enabled
    agent = BaseResponsesAPIAgentConfig.model_validate(
        {"host": "127.0.0.1", "port": 1, "name": "agent", "entrypoint": "app.py", "token_id_capture": True}
    )
    assert agent.token_capture is True
    model = BaseResponsesAPIModelConfig.model_validate(
        {
            "host": "127.0.0.1",
            "port": 1,
            "name": "policy",
            "entrypoint": "app.py",
            "token_id_capture_non_generating_requests": [{"method": "POST", "path": "/tokenize"}],
        }
    )
    assert [(r.method, r.path) for r in model.token_capture_non_generating_requests] == [("POST", "/tokenize")]


def test_an_old_name_resolves_to_its_renamed_object_with_a_warning() -> None:
    old = importlib.import_module("nemo_gym.models.token_capture.config")
    with pytest.warns(DeprecationWarning, match="TokenIdCaptureConfig.*TokenCaptureConfig"):
        assert old.TokenIdCaptureConfig is TokenCaptureConfig
