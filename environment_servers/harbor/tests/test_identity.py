# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The environment server accepts the identity the loader records (dataset digest and the per-row digest keys).

Lives with the server tests so the core unit run never imports the server package, whose coverage the server
suite measures.
"""

from environment_servers.harbor.app import HarborTasksetConfig
from environment_servers.harbor.task_data import TaskData
from nemo_gym.tasks.harbor.digest import DATASET_DIGEST_KEY, DIGEST_KEY


def test_the_environment_server_accepts_the_recorded_identity(tmp_path):
    config = HarborTasksetConfig.model_validate(
        {
            "folder": str(tmp_path),
            "tasks": {"a": "1" * 64},
            "dataset_digest": "2" * 64,
            "dataset_toml_digest": "3" * 64,
        }
    )
    assert config.dataset_digest == "2" * 64
    assert TaskData.model_fields["ng_dataset_digest"].alias == DATASET_DIGEST_KEY
    data = TaskData.model_validate({DIGEST_KEY: "abc", DATASET_DIGEST_KEY: "def"})
    assert data.ng_dataset_digest == "def" and TaskData.model_validate({DIGEST_KEY: "abc"}).ng_dataset_digest is None
