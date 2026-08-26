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
from omegaconf import OmegaConf

from nemo_gym.secret_utils import recursively_hide_secrets


class TestRecursivelyHideSecrets:
    def test_scalar_token_and_key_leaves_are_masked(self) -> None:
        cfg = OmegaConf.create({"api_key": "sk-real-value", "auth_token": "tok-real-value", "other": "kept"})

        recursively_hide_secrets(cfg)

        assert cfg.api_key == "****"
        assert cfg.auth_token == "****"
        assert cfg.other == "kept"

    def test_nested_dict_is_recursed_into(self) -> None:
        cfg = OmegaConf.create({"server": {"nested": {"api_key": "sk-nested"}}})

        recursively_hide_secrets(cfg)

        assert cfg.server.nested.api_key == "****"

    def test_list_of_tokens_under_a_key_named_field_is_masked_entrywise(self) -> None:
        # A list value under a "*key*"/"*token*" name (e.g. multiple API keys) is masked element-by-element,
        # not replaced wholesale, so the list's length still tells you how many secrets were configured.
        cfg = OmegaConf.create({"api_keys": ["sk-one", "sk-two", "sk-three"]})

        recursively_hide_secrets(cfg)

        assert list(cfg.api_keys) == ["****", "****", "****"]

    def test_list_of_nested_dicts_under_a_non_secret_key_is_recursed_into(self) -> None:
        # A list that isn't itself a "*key*"/"*token*" field (e.g. a list of server configs) must still
        # have its dict elements walked, so a secret nested inside one of them is masked too.
        cfg = OmegaConf.create(
            {
                "servers": [
                    {"name": "a", "api_key": "sk-a"},
                    {"name": "b", "api_key": "sk-b"},
                ]
            }
        )

        recursively_hide_secrets(cfg)

        assert cfg.servers[0].api_key == "****"
        assert cfg.servers[1].api_key == "****"
        assert cfg.servers[0].name == "a"
