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
from pathlib import Path

from nemo_gym.model_registry import _discover_models_in_dir


def _make_model(models_dir: Path, name: str, *, app: bool = True, flavors=()) -> Path:
    model_dir = models_dir / name
    model_dir.mkdir(parents=True)
    if app:
        (model_dir / "app.py").write_text("# app\n")
    if flavors:
        configs_dir = model_dir / "configs"
        configs_dir.mkdir()
        for flavor in flavors:
            (configs_dir / f"{flavor}.yaml").write_text("{}\n")
    return model_dir


class TestDiscoverModels:
    def test_discovers_by_directory_name(self, tmp_path: Path) -> None:
        _make_model(tmp_path, "my_model", flavors=("my_model", "some_other_flavor"))
        _make_model(tmp_path, "another_model", flavors=("another_model",))

        models = _discover_models_in_dir(tmp_path)

        assert set(models) == {"my_model", "another_model"}
        assert list(models["my_model"].variants) == ["my_model", "some_other_flavor"]

    def test_model_types_are_the_model_type_tokens(self, tmp_path: Path) -> None:
        # The flavor named after the model is the default (bare `<name>`); others are `<name>/<flavor>`.
        _make_model(tmp_path, "my_model", flavors=("my_model", "some_other_flavor"))

        assert _discover_models_in_dir(tmp_path)["my_model"].model_types == [
            "my_model",
            "my_model/some_other_flavor",
        ]

    def test_dirs_without_a_config_are_skipped(self, tmp_path: Path) -> None:
        # Only a dir that ships a config (something to pass to --model-type) is a model: a stray .egg-info,
        # or a dir with just an app.py and no configs, has nothing selectable and is not listed.
        (tmp_path / "my_model.egg-info").mkdir()
        _make_model(tmp_path, "app_only_model", app=True, flavors=())
        _make_model(tmp_path, "another_model", flavors=("another_model",))

        assert set(_discover_models_in_dir(tmp_path)) == {"another_model"}

    def test_missing_directory_yields_no_models(self, tmp_path: Path) -> None:
        assert _discover_models_in_dir(tmp_path / "nope") == {}
