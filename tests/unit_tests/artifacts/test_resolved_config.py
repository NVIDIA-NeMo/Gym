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

from nemo_gym.artifacts.resolved_config import write_resolved_config
from nemo_gym.config.secret_utils import MASKED_VALUE, hide_secrets_in_container
from nemo_gym.utils.path_utils import resolved_config_path_for


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


def test_write_resolved_config_accepts_a_plain_dict(tmp_path: Path) -> None:
    config = {"policy_api_key": "sk-canary", "url": "http://${base}/v1", "base": "x"}  # pragma: allowlist secret

    path = write_resolved_config(config, tmp_path / "rollouts.jsonl")

    assert yaml.safe_load(path.read_text()) == {
        "policy_api_key": MASKED_VALUE,
        "url": "http://${base}/v1",
        "base": "x",
    }
    assert "sk-canary" not in path.read_text()


def test_write_resolved_config_overwrites_and_leaves_no_temp_files(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    write_resolved_config(OmegaConf.create({"attempt": 1}), rollouts)
    path = write_resolved_config(OmegaConf.create({"attempt": 2}), rollouts)

    assert yaml.safe_load(path.read_text()) == {"attempt": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["rollouts_resolved_config.yaml"]


def test_hide_secrets_in_container_masks_every_header_value_and_keeps_header_names() -> None:
    original = {
        "openai_default_headers": {"X-Auth": "credential-canary", "Authorization": "Bearer another-canary"},
        "token_id_capture": {"enabled": True, "sink": "pkg:Sink"},
    }

    masked = hide_secrets_in_container(original)

    assert masked["openai_default_headers"] == {"X-Auth": MASKED_VALUE, "Authorization": MASKED_VALUE}
    assert masked["token_id_capture"] == {"enabled": True, "sink": "pkg:Sink"}
    assert original["openai_default_headers"]["X-Auth"] == "credential-canary"


def test_write_resolved_config_never_writes_header_credentials(tmp_path: Path) -> None:
    config = OmegaConf.create({"openai_default_headers": {"X-Auth": "credential-canary", "Api-Id": "another-canary"}})

    path = write_resolved_config(config, tmp_path / "rollouts.jsonl")

    text = path.read_text()
    assert "credential-canary" not in text and "another-canary" not in text
    assert yaml.safe_load(text) == {"openai_default_headers": {"X-Auth": MASKED_VALUE, "Api-Id": MASKED_VALUE}}


def test_no_serve_collection_saves_the_config_before_collecting(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.eval as cli_eval
    import nemo_gym.rollouts.rollout_collection as rollout_collection
    import nemo_gym.telemetry.setup as telemetry_setup

    monkeypatch.setenv("NO_SERVE_BASE_FOR_TEST", "base")
    rollouts = tmp_path / "run.jsonl"
    attempt = {"n": 1}

    def global_config():
        return OmegaConf.create(
            {
                "input_jsonl_fpath": str(tmp_path / "tasks.jsonl"),
                "output_jsonl_fpath": str(rollouts),
                "disable_health_check": True,
                "policy_api_key": "sk-canary",
                "url": "${oc.env:NO_SERVE_BASE_FOR_TEST}/v1",
                "attempt": attempt["n"],
            }
        )

    seen: dict = {}

    class FakeHelper:
        async def run_from_config(self, config) -> None:
            seen["yaml"] = yaml.safe_load((tmp_path / "run_resolved_config.yaml").read_text())

    monkeypatch.setattr(cli_eval, "get_global_config_dict", global_config)
    monkeypatch.setattr(rollout_collection, "RolloutCollectionHelper", FakeHelper)
    monkeypatch.setattr(telemetry_setup, "init_telemetry", lambda **kwargs: None)
    monkeypatch.setattr(telemetry_setup, "shutdown_telemetry", lambda *args, **kwargs: None)

    cli_eval.collect_rollouts()
    assert seen["yaml"]["policy_api_key"] == MASKED_VALUE
    assert seen["yaml"]["url"] == "${oc.env:NO_SERVE_BASE_FOR_TEST}/v1"
    assert seen["yaml"]["attempt"] == 1

    attempt["n"] = 2
    cli_eval.collect_rollouts()
    assert seen["yaml"]["attempt"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run_resolved_config.yaml"]
