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
import json

import pytest

from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, failures_path_for
from nemo_gym.rollout_reverification import InputRolloutPair, _build_verify_payload


@pytest.mark.parametrize(
    "failure_class,patch,expected_pending",
    [
        ("judge_failed", "saved patch", 0),
        ("judge_failed", "", 0),
        ("judge_failed", None, 0),
        ("agent_run_error", None, 1),
    ],
)
def test_saved_patch_is_not_regenerated_by_collection(tmp_path, monkeypatch, failure_class, patch, expected_pending):
    monkeypatch.setenv("NG_MAX_ROLLOUT_ATTEMPTS", "3")
    output = tmp_path / "rollouts.jsonl"
    cfg = RolloutCollectionConfig(input_jsonl_fpath=str(tmp_path / "inputs.jsonl"), output_jsonl_fpath=str(output))
    identity = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    cfg.materialized_jsonl_fpath.write_text(json.dumps(identity | {"agent_ref": {"name": "agent"}}) + "\n")
    output.write_text("")
    failure = identity | {"_ng_failure_class": failure_class, "response": {"output": []}, "model_patch": patch}
    sidecar = failures_path_for(output)
    sidecar.write_text(json.dumps(failure) + "\n")
    original = sidecar.read_bytes()
    pending, _, _, _ = RolloutCollectionHelper()._load_from_cache(cfg)
    assert len(pending) == expected_pending
    assert sidecar.read_bytes() == original
    assert "_ng_failure_terminal" not in json.loads(original)
    if pending:
        assert pending[0]["_ng_attempt_index"] == 1


@pytest.mark.parametrize("patch", ["saved patch", ""])
def test_reverify_refuses_to_drop_the_saved_patch(patch):
    pair = InputRolloutPair(
        input={"patch": "reference patch"},
        rollout={
            "_ng_failure_class": "judge_failed",
            "model_patch": patch,
            "response": {"output": []},
        },
    )
    with pytest.raises(ConfigError, match="saved model_patch replay is unsupported"):
        _build_verify_payload(pair)
    assert pair.rollout["model_patch"] == patch
    assert pair.input["patch"] == "reference patch"


def test_text_judge_failure_keeps_existing_reverify_payload():
    pair = InputRolloutPair(
        input={"question": "q"},
        rollout={
            "_ng_failure_class": "judge_failed",
            "response": {"output": []},
        },
    )
    assert _build_verify_payload(pair) == {"question": "q", "response": {"output": []}}


@pytest.mark.parametrize("patch", ["saved patch", ""])
def test_typed_ungraded_patch_cannot_fall_back_to_reference_patch(patch):
    pair = InputRolloutPair(
        input={"patch": "reference patch"},
        rollout={
            "failure_kind": "verifier_error",
            "evaluation_completed": False,
            "mask_sample": True,
            "model_patch": patch,
            "response": {"output": []},
        },
    )
    with pytest.raises(ConfigError, match="saved model_patch replay is unsupported"):
        _build_verify_payload(pair)
    assert pair.rollout["model_patch"] == patch
    assert pair.input["patch"] == "reference patch"
