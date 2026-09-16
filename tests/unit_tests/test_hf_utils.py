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
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from huggingface_hub import DatasetCard
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

from nemo_gym.config_types import UploadJsonlDatasetHuggingFaceConfig
from nemo_gym.hf_utils import HfApi, upload_jsonl_dataset


@pytest.fixture
def upload_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resource_config = tmp_path / "environments/example/config.yaml"
    resource_config.parent.mkdir(parents=True)
    resource_config.write_text(yaml.safe_dump({"example": {"resources_servers": {"example": {"domain": "math"}}}}))
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"responses_create_params": {"input": []}}\n')
    config = UploadJsonlDatasetHuggingFaceConfig(
        hf_token="test-token",
        hf_organization="test-org",
        hf_collection_name="Environments",
        hf_collection_slug="123",
        input_jsonl_fpath=str(data_file),
        resource_config_path="environments/example/config.yaml",
    )
    client = Mock(spec=HfApi)
    client.repo_info.return_value = SimpleNamespace(sha="a" * 40)
    client.get_collection.return_value = SimpleNamespace(
        items=[SimpleNamespace(item_id="test-org/Nemotron-RL-math-example", item_type="dataset")]
    )
    monkeypatch.setattr("nemo_gym.hf_utils.create_huggingface_client", Mock(return_value=client))
    return config, client


class TestHFUtils:
    def test_sanity(self) -> None:
        HfApi()

    @pytest.mark.parametrize(
        "card_text",
        [
            None,
            "# Existing card\n",
            "---\nlicense: mit\nconfigs:\n- config_name: default\n  data_files: train.jsonl\ntags: [math]\n---\n# Existing card\n",
            "---\ntags: [nemo-gym]\n---\n# Existing card\n",
            "---\ntags: [rl-environment, nemo-gym, math]\n---\n# Existing card\n",
        ],
    )
    @pytest.mark.parametrize("create_pr, revision, split", [(False, None, "train"), (True, "release", "validation")])
    def test_upload_environment_tags(self, upload_setup, tmp_path, monkeypatch, card_text, create_pr, revision, split):
        config, client = upload_setup
        config.create_pr, config.revision, config.split = create_pr, revision, split
        config.commit_message, config.commit_description = "Upload tasks", "Existing description"
        download = Mock()
        if card_text is None:
            download.side_effect = RemoteEntryNotFoundError(
                "README.md does not exist",
                response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/README.md")),
            )
        else:
            card_path = tmp_path / "README.md"
            card_path.write_text(card_text)
            download.return_value = str(card_path)
        monkeypatch.setattr("nemo_gym.hf_utils.hf_hub_download", download)

        upload_jsonl_dataset(config)

        client.repo_info.assert_called_once_with(
            "test-org/Nemotron-RL-math-example", repo_type="dataset", revision=revision
        )
        download.assert_called_once_with(
            repo_id="test-org/Nemotron-RL-math-example",
            filename="README.md",
            repo_type="dataset",
            revision="a" * 40,
            token="test-token",
        )
        client.create_commit.assert_called_once()
        kwargs = client.create_commit.call_args.kwargs
        operations = kwargs.pop("operations")
        assert kwargs == dict(
            repo_id="test-org/Nemotron-RL-math-example",
            token="test-token",
            repo_type="dataset",
            create_pr=create_pr,
            revision=revision,
            parent_commit="a" * 40,
            commit_message="Upload tasks",
            commit_description="Existing description",
        )
        assert operations[0].path_in_repo == "train.jsonl"
        assert operations[0].path_or_fileobj == config.input_jsonl_fpath
        original_card = DatasetCard(card_text or "")
        if set(original_card.data.get("tags") or []) >= {"rl-environment", "nemo-gym"}:
            assert len(operations) == 1
        else:
            assert len(operations) == 2
            assert operations[1].path_in_repo == "README.md"
            card = DatasetCard(operations[1].path_or_fileobj.decode("utf-8"))
            metadata = card.data.to_dict()
            expected_tags = list(
                dict.fromkeys([*(original_card.data.get("tags") or []), "rl-environment", "nemo-gym"])
            )
            assert metadata.pop("tags") == expected_tags
            original_metadata = original_card.data.to_dict()
            original_metadata.pop("tags", None)
            assert metadata == original_metadata
            assert card.text == original_card.text
        client.upload_file.assert_not_called()

    def test_upload_default_commit_message(self, upload_setup, monkeypatch):
        config, client = upload_setup
        monkeypatch.setattr(
            "nemo_gym.hf_utils.hf_hub_download",
            Mock(
                side_effect=RemoteEntryNotFoundError(
                    "README.md does not exist",
                    response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/README.md")),
                )
            ),
        )

        upload_jsonl_dataset(config)

        assert client.create_commit.call_args.kwargs["commit_message"] == "Upload train.jsonl"

    @pytest.mark.parametrize(
        "error",
        [
            HfHubHTTPError(
                "Access denied",
                response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/README.md")),
            ),
            LocalEntryNotFoundError("Offline"),
        ],
    )
    def test_upload_does_not_replace_unreadable_card(self, upload_setup, monkeypatch, error):
        config, client = upload_setup
        monkeypatch.setattr("nemo_gym.hf_utils.hf_hub_download", Mock(side_effect=error))

        with pytest.raises(type(error)):
            upload_jsonl_dataset(config)

        client.create_commit.assert_not_called()
        client.upload_file.assert_not_called()

    def test_upload_rejects_invalid_training_data(self, upload_setup):
        config, client = upload_setup
        with open(config.input_jsonl_fpath, "w") as f:
            f.write('{"rollout": []}\n')

        upload_jsonl_dataset(config)

        client.create_repo.assert_not_called()
        client.create_commit.assert_not_called()
