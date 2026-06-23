# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_gym.model_server_rename import rename_file, rename_references


OLD, NEW = "local_vllm_model", "vllm_server"


class TestRenameReferences:
    def test_rewrites_config_paths_directory(self) -> None:
        text = "config_paths:\n- responses_api_models/local_vllm_model/configs/Qwen/Qwen3.5-27B.yaml\n"
        out, count = rename_references(text, OLD, NEW)
        assert count == 1
        assert "responses_api_models/vllm_server/configs/Qwen/Qwen3.5-27B.yaml" in out
        assert "local_vllm_model" not in out

    def test_rewrites_dotted_override_path(self) -> None:
        text = "++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.x=1\n"
        out, count = rename_references(text, OLD, NEW)
        # The path token responses_api_models/<old> isn't present (dotted form); the bare key under a
        # parent isn't either, so a dotted-attr reference is left untouched (handled at dir-move time).
        assert count == 0
        assert out == text

    def test_rewrites_config_key_under_responses_api_models(self) -> None:
        text = "policy_model:\n  responses_api_models:\n    local_vllm_model:\n      entrypoint: app.py\n"
        out, count = rename_references(text, OLD, NEW)
        assert count == 1
        assert "    vllm_server:" in out
        assert "local_vllm_model:" not in out

    def test_does_not_rewrite_same_name_key_elsewhere(self) -> None:
        # A key named like the server but NOT under responses_api_models: must be left alone.
        text = "some_block:\n  local_vllm_model:\n    foo: bar\n"
        out, count = rename_references(text, OLD, NEW)
        assert count == 0
        assert out == text

    def test_rewrites_both_path_and_key_together(self) -> None:
        text = (
            "config_paths:\n- responses_api_models/local_vllm_model/configs/x.yaml\n"
            "policy_model:\n  responses_api_models:\n    local_vllm_model:\n      entrypoint: app.py\n"
        )
        out, count = rename_references(text, OLD, NEW)
        assert count == 2
        assert "responses_api_models/vllm_server/configs/x.yaml" in out
        assert "    vllm_server:" in out

    def test_does_not_touch_python_import_path(self) -> None:
        # Imports use dotted module paths, not slash paths; left to the dir-move step.
        text = "from responses_api_models.local_vllm_model.app import LocalVLLMModel\n"
        out, count = rename_references(text, OLD, NEW)
        assert count == 0
        assert out == text

    def test_no_references_is_unchanged(self) -> None:
        text = "config_paths:\n- responses_api_models/openai_model/configs/openai_model.yaml\n"
        out, count = rename_references(text, OLD, NEW)
        assert count == 0
        assert out == text

    def test_key_block_scope_resets_after_dedent(self) -> None:
        # After leaving the responses_api_models block, a matching key must not be rewritten.
        text = "a:\n  responses_api_models:\n    local_vllm_model:\n      x: 1\nb:\n  local_vllm_model:\n    y: 2\n"
        out, count = rename_references(text, OLD, NEW)
        assert count == 1  # only the one under responses_api_models:
        assert "b:\n  local_vllm_model:" in out


class TestRenameFile:
    def test_rewrites_in_place(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        f.write_text("config_paths:\n- responses_api_models/local_vllm_model/configs/x.yaml\n")
        assert rename_file(f, OLD, NEW) == 1
        assert "vllm_server" in f.read_text()

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        original = "config_paths:\n- responses_api_models/local_vllm_model/configs/x.yaml\n"
        f.write_text(original)
        assert rename_file(f, OLD, NEW, dry_run=True) == 1
        assert f.read_text() == original
