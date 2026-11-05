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
"""Names renamed with the token capture package keep working under their old names, with a warning."""

import importlib
import warnings

import pytest

from nemo_gym._moved_paths import _RENAMED_TOKEN_CAPTURE_NAMES, moved_names_getattr
from nemo_gym._public_api import lazy_exports
from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config.global_config import TOKEN_CAPTURE_BLOCK
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig
from nemo_gym.models.token_capture.config import TokenCaptureConfig, token_capture_enabled_for_agent


RENAMED = sorted(
    (module, name, moved.new)
    for module, names in _RENAMED_TOKEN_CAPTURE_NAMES.items()
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
    # A fresh getter keeps its own warn-once record, so an earlier read of the old name cannot swallow the warning.
    with pytest.warns(DeprecationWarning, match=rf"{old_name}.*{target.partition(':')[2]}"):
        assert moved_names_getattr(module.__name__)(old_name) is _new_object(target)


def test_an_old_name_resolves_through_a_package_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh public getter makes the warning assertion independent of earlier legacy-name reads."""
    package = importlib.import_module("nemo_gym.config")
    getter, _ = lazy_exports(package.__name__, package._PUBLIC_API)
    monkeypatch.setattr(package, "__getattr__", getter)
    with pytest.warns(DeprecationWarning, match="TOKEN_ID_CAPTURE_BLOCK"):
        # Read through the package object: this test is about the package's public API itself.
        TOKEN_ID_CAPTURE_BLOCK = importlib.import_module("nemo_gym.config").TOKEN_ID_CAPTURE_BLOCK
    assert TOKEN_ID_CAPTURE_BLOCK is TOKEN_CAPTURE_BLOCK


def test_the_run_wide_block_is_named_token_capture() -> None:
    assert TOKEN_CAPTURE_BLOCK == "token_capture"


def test_a_block_under_its_old_key_configures_capture_with_a_warning() -> None:
    block = {"enabled": True, "dir": "/tmp/capture", "delta_records": True}
    with pytest.warns(DeprecationWarning, match="'token_id_capture'.*'token_capture'"):
        old = TokenCaptureConfig.model_validate({"token_id_capture": block})
    assert old == TokenCaptureConfig.model_validate({"token_capture": block})
    assert old.enabled and old.token_capture.delta_records


def test_a_block_under_both_keys_with_different_values_is_an_error() -> None:
    with pytest.raises(ValueError, match="remove 'token_id_capture' and keep only 'token_capture'"):
        TokenCaptureConfig.model_validate({"token_id_capture": {"enabled": True}, "token_capture": {"enabled": False}})


def test_the_old_block_attribute_returns_the_block_with_a_warning() -> None:
    config = TokenCaptureConfig.model_validate({"token_capture": {"enabled": True, "dir": "/tmp/capture"}})
    with pytest.warns(DeprecationWarning, match="TokenCaptureConfig.token_capture"):
        assert config.token_id_capture is config.token_capture


def test_an_agent_flag_under_its_old_key_opts_the_agent_in() -> None:
    with pytest.warns(DeprecationWarning, match="'token_id_capture'.*'token_capture'"):
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="app.py", name="", token_id_capture=True)
    assert config.token_capture is True
    with pytest.warns(DeprecationWarning, match="BaseResponsesAPIAgentConfig.token_capture"):
        assert config.token_id_capture is True


def test_a_raw_config_with_old_keys_opts_its_agent_in() -> None:
    # Frameworks may hand a config dict straight to this function without the parser renaming its keys.
    config = {
        "token_id_capture": {"enabled": True, "dir": "/tmp/capture"},
        "captured": {"responses_api_agents": {"implementation": {"token_id_capture": True}}},
    }
    with pytest.warns(DeprecationWarning):
        assert token_capture_enabled_for_agent(config, "captured") is True


def test_model_non_generating_requests_under_their_old_key_are_declared() -> None:
    declared = [{"method": "get", "path": "/custom/metadata"}]
    with pytest.warns(DeprecationWarning, match="'token_capture_non_generating_requests'"):
        config = BaseResponsesAPIModelConfig(
            host="", port=0, entrypoint="app.py", name="", token_id_capture_non_generating_requests=declared
        )
    assert [(r.method, r.path) for r in config.token_capture_non_generating_requests] == [("GET", "/custom/metadata")]
    with pytest.warns(DeprecationWarning, match="BaseResponsesAPIModelConfig.token_capture_non_generating_requests"):
        assert config.token_id_capture_non_generating_requests is config.token_capture_non_generating_requests
