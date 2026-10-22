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

import yaml
from omegaconf import OmegaConf

from nemo_gym.path_utils import resolved_config_path_for
from nemo_gym.resolved_config import write_resolved_config
from nemo_gym.secret_utils import MASKED_VALUE, hide_secrets_in_container


def test_resolved_config_path_for() -> None:
    assert resolved_config_path_for(Path("out/weather_rollouts.jsonl")) == Path(
        "out/weather_rollouts_resolved_config.yaml"
    )


def test_hide_secrets_in_container_masks_leaves_and_keeps_input() -> None:
    original = {
        "policy_api_key": "sk-1",
        "model": "m",
        "nested": [{"password": "p", "name": "n"}],
        "tokens": ["a", "b"],
    }
    masked = hide_secrets_in_container(original)
    assert masked == {
        "policy_api_key": MASKED_VALUE,
        "model": "m",
        "nested": [{"password": MASKED_VALUE, "name": "n"}],
        "tokens": [MASKED_VALUE, MASKED_VALUE],
    }
    assert original["policy_api_key"] == "sk-1"


def test_write_resolved_config_masks_secrets_and_keeps_interpolations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("UNSET_ENV_FOR_TEST", raising=False)
    config = OmegaConf.create(
        {
            "policy_api_key": "${oc.env:UNSET_ENV_FOR_TEST}",
            "base": "x",
            "url": "http://${base}/v1",
            "agent": {"auth_token": "literal"},
        }
    )
    rollouts = tmp_path / "run" / "rollouts.jsonl"

    path = write_resolved_config(config, rollouts)

    assert path == tmp_path / "run" / "rollouts_resolved_config.yaml"
    assert yaml.safe_load(path.read_text()) == {
        "policy_api_key": MASKED_VALUE,
        "base": "x",
        "url": "http://${base}/v1",
        "agent": {"auth_token": MASKED_VALUE},
    }
    assert "literal" not in path.read_text()


def test_write_resolved_config_overwrites_and_leaves_no_temp_files(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    write_resolved_config(OmegaConf.create({"attempt": 1}), rollouts)
    path = write_resolved_config(OmegaConf.create({"attempt": 2}), rollouts)

    assert yaml.safe_load(path.read_text()) == {"attempt": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["rollouts_resolved_config.yaml"]
