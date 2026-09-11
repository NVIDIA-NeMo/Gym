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

import subprocess
import sys
from copy import deepcopy

import pytest

from nemo_gym.failure_routing import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    build_failure_result,
    minimal_failure_response,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY as ROLLOUT_FAILURE_CLASS_KEY,
)
from nemo_gym.rollout_collection import (
    NG_NO_PERSIST_KEY as ROLLOUT_NO_PERSIST_KEY,
)
from nemo_gym.rollout_collection import (
    NG_TERMINAL_KEY as ROLLOUT_TERMINAL_KEY,
)


def test_failure_routing_import_stays_independent_of_rollout_collection() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import nemo_gym.failure_routing; "
                "assert 'nemo_gym.rollout_collection' not in sys.modules; "
                "assert 'wandb' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_rollout_collection_reexports_failure_routing_keys() -> None:
    assert ROLLOUT_FAILURE_CLASS_KEY == NG_FAILURE_CLASS_KEY
    assert ROLLOUT_NO_PERSIST_KEY == NG_NO_PERSIST_KEY
    assert ROLLOUT_TERMINAL_KEY == NG_TERMINAL_KEY


def test_minimal_failure_response_is_valid_and_configurable() -> None:
    response = minimal_failure_response(
        response_id="failed-task",
        model="policy-model",
        message_id="failed-message",
        text="Timed out: deadline exceeded",
        created_at=42.0,
        tool_choice="none",
    )

    validated = NeMoGymResponse.model_validate(response)
    assert validated.id == "failed-task"
    assert validated.model == "policy-model"
    assert validated.created_at == 42.0
    assert validated.tool_choice == "none"
    assert validated.output[0].id == "failed-message"
    assert validated.output_text == "Timed out: deadline exceeded"


def test_build_failure_result_replaces_stale_reserved_keys_without_mutating_record() -> None:
    record = {
        "task_id": "task-1",
        "reward": 1.0,
        "response": {"stale": True},
        "error": "old error",
        NG_FAILURE_CLASS_KEY: "old_failure",
        NG_NO_PERSIST_KEY: True,
        NG_TERMINAL_KEY: True,
    }
    original = deepcopy(record)

    result = build_failure_result(record, failure_class="retryable", error="fresh error")

    assert record == original
    assert result["task_id"] == "task-1"
    assert result["reward"] == 0.0
    assert result["error"] == "fresh error"
    assert result[NG_FAILURE_CLASS_KEY] == "retryable"
    assert NG_NO_PERSIST_KEY not in result
    assert NG_TERMINAL_KEY not in result
    assert NeMoGymResponse.model_validate(result["response"]).id == "failure"


@pytest.mark.parametrize(
    ("terminal", "no_persist"),
    [(True, False), (False, True), (True, True)],
)
def test_build_failure_result_sets_requested_routing_flags(terminal: bool, no_persist: bool) -> None:
    result = build_failure_result(
        {},
        failure_class="classified",
        error="failed",
        terminal=terminal,
        no_persist=no_persist,
    )

    assert result.get(NG_TERMINAL_KEY, False) is terminal
    assert result.get(NG_NO_PERSIST_KEY, False) is no_persist


def test_build_failure_result_supports_custom_error_key_extra_metadata_and_response() -> None:
    response = minimal_failure_response(response_id="custom", model="agent", text="no trajectory")
    result = build_failure_result(
        {
            "grading_notes": "stale notes",
            "error": "stale standard error",
            NG_NO_PERSIST_KEY: True,
        },
        failure_class="timeout_exceeded",
        error="run failed: TimeoutError",
        response=response,
        terminal=True,
        error_key="grading_notes",
        extra={
            "status": "error",
            "grading_type": "unknown",
            "reward": 99.0,
            NG_FAILURE_CLASS_KEY: "extra must not win",
            NG_NO_PERSIST_KEY: True,
        },
    )

    assert result["grading_notes"] == "run failed: TimeoutError"
    assert "error" not in result
    assert result["status"] == "error"
    assert result["grading_type"] == "unknown"
    assert result["reward"] == 0.0
    assert result[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert result[NG_TERMINAL_KEY] is True
    assert NG_NO_PERSIST_KEY not in result
    assert result["response"]["id"] == "custom"


def test_build_failure_result_accepts_a_serialized_response() -> None:
    response = minimal_failure_response(response_id="serialized")

    result = build_failure_result({}, failure_class="failed", error="error", response=response)

    assert result["response"] == response


def test_build_failure_result_rejects_a_reserved_error_key() -> None:
    with pytest.raises(ValueError, match="reserved failure-result key"):
        build_failure_result({}, failure_class="failed", error="error", error_key=NG_TERMINAL_KEY)
