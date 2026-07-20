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
import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from nemo_gym.cli.models import list_models
from nemo_gym.model_registry import ModelEntry


def _mock_global_config(config: dict = None):
    return OmegaConf.create(config or {})


def _entry(name: str, group: str) -> ModelEntry:
    config_path = Path("responses_api_models") / group / "configs" / f"{name.split('/')[-1]}.yaml"
    return ModelEntry(name=name, model_group=group, config_path=config_path)


_MODELS = {
    "my_model": _entry("my_model", "my_model"),
    "my_model/some_other_flavor": _entry("my_model/some_other_flavor", "my_model"),
    "another_model": _entry("another_model", "another_model"),
}


class TestListModels:
    def test_lists_per_variant_rows(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.models.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.models.discover_models", return_value=_MODELS),
        ):
            list_models()
        out = capsys.readouterr().out

        variants = list(_MODELS)
        assert len(variants) == 3
        # one data row per variant (data rows use the light "│"; the header uses the heavy "┃")
        assert sum(1 for line in out.splitlines() if "│" in line) == 3
        for variant in variants:
            assert variant in out

    def test_no_models(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.models.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.models.discover_models", return_value={}),
        ):
            list_models()
        assert "No models found" in capsys.readouterr().out

    def test_query_filters_rows(self, capsys) -> None:
        # `gym search models <query>` reuses this command via the `query` config key (token + model group).
        with (
            patch(
                "nemo_gym.cli.models.get_global_config_dict",
                return_value=_mock_global_config({"query": "some_other_flavor"}),
            ),
            patch("nemo_gym.cli.models.discover_models", return_value=_MODELS),
        ):
            list_models()
        out = capsys.readouterr().out
        assert "my_model/some_other_flavor" in out and "Models matching" in out
        assert "another_model" not in out

    def test_json_output_is_per_variant_rows(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.models.get_global_config_dict", return_value=_mock_global_config({"json": True})),
            patch("nemo_gym.cli.models.discover_models", return_value=_MODELS),
        ):
            list_models()
        payload = json.loads(capsys.readouterr().out)
        expected = [
            {"model": "my_model", "model_group": "my_model"},
            {"model": "my_model/some_other_flavor", "model_group": "my_model"},
            {"model": "another_model", "model_group": "another_model"},
        ]
        assert len(payload) == len(expected)
        for row in expected:
            assert row in payload
