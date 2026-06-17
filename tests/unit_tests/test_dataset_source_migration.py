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
import warnings
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.config_types import DatasetConfig
from nemo_gym.dataset_source_migration import migrate_config_text, migrate_file


_GITLAB_BLOCK = """\
datasets:
- name: train
  type: train
  jsonl_fpath: data/train.jsonl
  num_repeats: 1
  gitlab_identifier:
    dataset_name: my_dataset
    version: 0.0.1
    artifact_fpath: train.jsonl
  license: Apache 2.0
"""

_HF_BLOCK = """\
datasets:
- name: example
  type: example
  jsonl_fpath: data/example.jsonl
  num_repeats: 1
  huggingface_identifier:
    repo_id: org/dataset
"""


class TestMigrateConfigText:
    def test_gitlab_identifier_becomes_source(self) -> None:
        out, count = migrate_config_text(_GITLAB_BLOCK)

        assert count == 1
        assert "gitlab_identifier:" not in out
        assert "  source:\n    type: gitlab\n" in out
        # The nested fields and the trailing license line are preserved verbatim.
        assert "    dataset_name: my_dataset" in out
        assert "    version: 0.0.1" in out
        assert "  license: Apache 2.0" in out

    def test_huggingface_identifier_becomes_source(self) -> None:
        out, count = migrate_config_text(_HF_BLOCK)

        assert count == 1
        assert "huggingface_identifier:" not in out
        assert "  source:\n    type: huggingface\n" in out
        assert "    repo_id: org/dataset" in out

    def test_multiple_blocks_in_one_file(self) -> None:
        text = _GITLAB_BLOCK + _HF_BLOCK
        out, count = migrate_config_text(text)

        assert count == 2
        assert "  type: gitlab" in out
        assert "  type: huggingface" in out

    def test_comments_and_indentation_are_preserved(self) -> None:
        text = "  gitlab_identifier:  # fetched from registry\n"
        out, count = migrate_config_text(text)

        # A trailing comment on the key line means it is not a bare key; leave it untouched
        # rather than risk dropping the comment.
        assert count == 0
        assert out == text

    def test_flow_style_is_left_untouched(self) -> None:
        text = "  gitlab_identifier: {dataset_name: x, version: 0.0.1, artifact_fpath: t.jsonl}\n"
        out, count = migrate_config_text(text)

        assert count == 0
        assert "gitlab_identifier:" in out

    def test_config_without_legacy_identifier_is_unchanged(self) -> None:
        text = "datasets:\n- name: example\n  type: example\n  jsonl_fpath: data/example.jsonl\n"
        out, count = migrate_config_text(text)

        assert count == 0
        assert out == text

    def test_migrated_text_validates_without_deprecation_warning(self) -> None:
        out, _ = migrate_config_text(_GITLAB_BLOCK)
        dataset = OmegaConf.to_container(OmegaConf.create(out)["datasets"][0], resolve=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            cfg = DatasetConfig.model_validate(dataset)

        assert cfg.source is not None
        assert cfg.source.type == "gitlab"


class TestMigrateFile:
    def test_migrate_file_rewrites_in_place(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(_GITLAB_BLOCK)

        count = migrate_file(config)

        assert count == 1
        assert "gitlab_identifier:" not in config.read_text()
        assert "type: gitlab" in config.read_text()

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(_GITLAB_BLOCK)

        count = migrate_file(config, dry_run=True)

        assert count == 1
        assert config.read_text() == _GITLAB_BLOCK

    def test_file_without_legacy_identifier_is_not_rewritten(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        original = "datasets:\n- name: example\n  type: example\n  jsonl_fpath: data/example.jsonl\n"
        config.write_text(original)

        count = migrate_file(config)

        assert count == 0
        assert config.read_text() == original
