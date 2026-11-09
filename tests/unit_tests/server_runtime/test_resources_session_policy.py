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

from typing import Annotated

import pytest
from pydantic import BeforeValidator, TypeAdapter, ValidationError

from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseSeedSessionRequest,
    ResourcesSeedSessionRequest,
    reject_partial_typed_seed,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="tasks", task_id="task")
IDENTITY = (EPISODE, TASK)
SEED = ResourcesSeedSessionRequest(resources_session_id="session", episode_id=EPISODE, task_id=TASK, task_data={})
SEED_BODY = TypeAdapter(
    ResourcesSeedSessionRequest | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)]
)


@pytest.mark.parametrize("missing", ["resources_session_id", "episode_id", "task_id", "task_data"])
def test_partial_typed_seed_cannot_fall_back_to_cookie_seed(missing: str) -> None:
    body = SEED.model_dump(mode="json")
    del body[missing]
    with pytest.raises(ValidationError):
        SEED_BODY.validate_python(body)


@pytest.mark.parametrize("field", ["episode_id", "task_id", "task_data"])
def test_cookie_seed_keeps_dataset_fields_that_share_a_typed_seed_name(field: str) -> None:
    # A /run seed posts the whole dataset row, whose own columns may be named like the typed seed's fields.
    body = {"verifier_metadata": {"city": "Paris"}, field: "row-1"}
    assert isinstance(SEED_BODY.validate_python(body), BaseSeedSessionRequest)


def test_seed_union_accepts_both_complete_wire_formats() -> None:
    assert SEED_BODY.validate_python(SEED.model_dump(mode="json")) == SEED
    assert isinstance(SEED_BODY.validate_python({"verifier_metadata": {"city": "Paris"}}), BaseSeedSessionRequest)
