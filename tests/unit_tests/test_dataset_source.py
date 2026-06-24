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
from pydantic import ValidationError
from pytest import raises, warns

from nemo_gym.config_types import (
    DatasetConfig,
    GitlabDatasetSource,
    HuggingFaceDatasetSource,
)


def _dataset(**extra) -> dict:
    return {"name": "ds", "type": "example", "jsonl_fpath": "data.jsonl", **extra}


class TestDatasetSource:
    def test_source_gitlab_backfills_legacy_identifier(self) -> None:
        cfg = DatasetConfig.model_validate(
            _dataset(
                source={
                    "type": "gitlab",
                    "dataset_name": "my_dataset",
                    "version": "0.0.1",
                    "artifact_fpath": "train.jsonl",
                }
            )
        )

        assert isinstance(cfg.source, GitlabDatasetSource)
        # Existing consumers read the legacy field; it must be back-filled from `source`.
        assert cfg.gitlab_identifier is not None
        assert cfg.gitlab_identifier.dataset_name == "my_dataset"
        assert cfg.gitlab_identifier.version == "0.0.1"
        assert cfg.gitlab_identifier.artifact_fpath == "train.jsonl"
        assert cfg.huggingface_identifier is None

    def test_source_huggingface_backfills_legacy_identifier(self) -> None:
        cfg = DatasetConfig.model_validate(
            _dataset(source={"type": "huggingface", "repo_id": "org/dataset", "artifact_fpath": "train.jsonl"})
        )

        assert isinstance(cfg.source, HuggingFaceDatasetSource)
        assert cfg.huggingface_identifier is not None
        assert cfg.huggingface_identifier.repo_id == "org/dataset"
        assert cfg.huggingface_identifier.artifact_fpath == "train.jsonl"
        assert cfg.gitlab_identifier is None

    def test_legacy_gitlab_identifier_mirrors_into_source_with_warning(self) -> None:
        with warns(DeprecationWarning, match="gitlab_identifier"):
            cfg = DatasetConfig.model_validate(
                _dataset(
                    gitlab_identifier={
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    }
                )
            )

        assert isinstance(cfg.source, GitlabDatasetSource)
        assert cfg.source.dataset_name == "my_dataset"
        assert cfg.source.version == "0.0.1"
        assert cfg.source.artifact_fpath == "train.jsonl"
        # Legacy field stays populated so nothing that already reads it breaks.
        assert cfg.gitlab_identifier is not None

    def test_legacy_huggingface_identifier_mirrors_into_source_with_warning(self) -> None:
        with warns(DeprecationWarning, match="huggingface_identifier"):
            cfg = DatasetConfig.model_validate(_dataset(huggingface_identifier={"repo_id": "org/dataset"}))

        assert isinstance(cfg.source, HuggingFaceDatasetSource)
        assert cfg.source.repo_id == "org/dataset"
        assert cfg.source.artifact_fpath is None
        assert cfg.huggingface_identifier is not None

    def test_specifying_source_and_legacy_identifier_is_rejected(self) -> None:
        with raises(ValidationError, match="set only one"):
            DatasetConfig.model_validate(
                _dataset(
                    source={
                        "type": "gitlab",
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    },
                    gitlab_identifier={
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    },
                )
            )

    def test_both_legacy_identifiers_together_is_allowed(self) -> None:
        # A gitlab-primary / huggingface-fallback pair (backend chosen at download time) must stay
        # valid; the single discriminated `source:` can't represent both, so it is left unset.
        with warns(DeprecationWarning, match="gitlab_identifier"):
            cfg = DatasetConfig.model_validate(
                _dataset(
                    gitlab_identifier={
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    },
                    huggingface_identifier={"repo_id": "org/dataset"},
                )
            )

        assert cfg.source is None
        assert cfg.gitlab_identifier is not None
        assert cfg.huggingface_identifier is not None

    def test_no_source_is_allowed(self) -> None:
        cfg = DatasetConfig.model_validate(_dataset())

        assert cfg.source is None
        assert cfg.gitlab_identifier is None
        assert cfg.huggingface_identifier is None

    def test_source_discriminator_selects_backend(self) -> None:
        with raises(ValidationError):
            # Missing repo_id for the huggingface branch.
            DatasetConfig.model_validate(_dataset(source={"type": "huggingface", "artifact_fpath": "train.jsonl"}))
