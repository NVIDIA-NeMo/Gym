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
"""Python names renamed with the token capture package keep resolving at their old paths; config keys do not change."""

import importlib
import warnings

import pytest

from nemo_gym._moved_paths import _RENAMED_TOKEN_CAPTURE_NAMES, moved_names_getattr
from nemo_gym._public_api import lazy_exports
from nemo_gym.config.global_config import TOKEN_CAPTURE_BLOCK
from nemo_gym.models.token_capture.config import TokenCaptureConfig


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


def test_config_keys_keep_their_token_id_capture_spelling() -> None:
    assert TOKEN_CAPTURE_BLOCK == "token_id_capture"
    config = TokenCaptureConfig.model_validate({"token_id_capture": {"enabled": True, "dir": "/tmp/capture"}})
    assert config.token_id_capture.enabled and config.enabled
