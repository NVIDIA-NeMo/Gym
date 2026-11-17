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
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.core.episode_context import (
    current_rollout_skips_token_capture,
    skip_token_capture_context,
    skips_token_capture,
)
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({}, False),
        ({"_ng_skip_token_capture": False}, False),
        # A table that sets the column on some rows writes null on the others.
        ({"_ng_skip_token_capture": None}, False),
        ({"_ng_skip_token_capture": True}, True),
    ],
)
def test_a_row_skips_token_capture_only_with_true(row: dict[str, Any], expected: bool) -> None:
    assert skips_token_capture({"responses_create_params": {"input": "task"}, **row}) is expected


@pytest.mark.parametrize("value", ["true", 1])
def test_a_row_with_a_value_that_is_not_a_boolean_is_refused(value: Any) -> None:
    with pytest.raises(ValueError, match="_ng_skip_token_capture must be true or false"):
        skips_token_capture({"_ng_skip_token_capture": value})


def test_a_refused_value_names_its_row() -> None:
    with pytest.raises(ValueError, match=r"\(row \{'_ng_task_index': 3, '_ng_rollout_index': 1\}\); got 'yes'"):
        skips_token_capture({"_ng_task_index": 3, "_ng_rollout_index": 1, "_ng_skip_token_capture": "yes"})


def test_a_run_request_reads_the_key_and_keeps_it_out_of_its_serialized_form() -> None:
    body = BaseRunRequest.model_validate(
        {"responses_create_params": {"input": "task"}, "_ng_skip_token_capture": True}
    )

    assert skips_token_capture(body) is True
    assert "_ng_skip_token_capture" not in body.model_dump(by_alias=True)
    unset = BaseRunRequest.model_validate(
        {"responses_create_params": {"input": "task"}, "_ng_skip_token_capture": None}
    )
    assert skips_token_capture(unset) is False
    with pytest.raises(ValidationError):
        BaseRunRequest.model_validate({"responses_create_params": {"input": "task"}, "_ng_skip_token_capture": "true"})


def test_a_run_request_model_without_the_field_reads_its_extra_key() -> None:
    class AgentRunRequest(BaseModel):
        model_config = ConfigDict(extra="allow")

    assert skips_token_capture(AgentRunRequest.model_validate({"_ng_skip_token_capture": True})) is True
    assert skips_token_capture(AgentRunRequest.model_validate({})) is False


def test_no_body_leaves_capture_to_the_run() -> None:
    assert skips_token_capture(None) is False


def test_the_skip_context_skips_capture_inside_the_block_only() -> None:
    assert current_rollout_skips_token_capture() is False
    with skip_token_capture_context(True):
        assert current_rollout_skips_token_capture() is True
        with skip_token_capture_context(False):
            assert current_rollout_skips_token_capture() is False
        assert current_rollout_skips_token_capture() is True
    assert current_rollout_skips_token_capture() is False
