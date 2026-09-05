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

from nemo_gym.discovery import merge_by_name
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
    def test_keys_by_model_name(self, tmp_path: Path) -> None:
        # One entry per --model-type name: `<dir>` for the flavor named after the model, `<dir>/<flavor>`
        # for the rest.
        _make_model(tmp_path, "my_model", flavors=("my_model", "some_other_flavor"))
        _make_model(tmp_path, "another_model", flavors=("another_model",))

        models = _discover_models_in_dir(tmp_path)

        assert set(models) == {"my_model", "my_model/some_other_flavor", "another_model"}
        assert models["my_model/some_other_flavor"].model_group == "my_model"
        assert models["my_model"].config_path == tmp_path / "my_model" / "configs" / "my_model.yaml"

    def test_user_flavor_shadows_only_its_name_not_the_whole_group(self, tmp_path: Path) -> None:
        # A user's `vllm_model` flavor shadows only the `vllm_model` name; the built-in
        # `vllm_model/vllm_model_for_training` flavor stays discoverable.
        user = tmp_path / "user"
        builtin = tmp_path / "builtin"
        _make_model(user, "vllm_model", flavors=("vllm_model",))
        _make_model(builtin, "vllm_model", flavors=("vllm_model", "vllm_model_for_training"))

        merged = merge_by_name([_discover_models_in_dir(user), _discover_models_in_dir(builtin)])

        assert set(merged) == {"vllm_model", "vllm_model/vllm_model_for_training"}
        assert merged["vllm_model"].config_path == user / "vllm_model" / "configs" / "vllm_model.yaml"  # user wins
        survivor = merged["vllm_model/vllm_model_for_training"]  # built-in flavor survives
        assert survivor.config_path == builtin / "vllm_model" / "configs" / "vllm_model_for_training.yaml"

    def test_dirs_without_a_config_are_skipped(self, tmp_path: Path) -> None:
        # Only a dir that ships a config (something to pass to --model-type) is a model: a stray .egg-info,
        # or a dir with just an app.py and no configs, has nothing selectable and is not listed.
        (tmp_path / "my_model.egg-info").mkdir()
        _make_model(tmp_path, "app_only_model", app=True, flavors=())
        _make_model(tmp_path, "another_model", flavors=("another_model",))

        assert set(_discover_models_in_dir(tmp_path)) == {"another_model"}

    def test_missing_directory_yields_no_models(self, tmp_path: Path) -> None:
        assert _discover_models_in_dir(tmp_path / "nope") == {}
