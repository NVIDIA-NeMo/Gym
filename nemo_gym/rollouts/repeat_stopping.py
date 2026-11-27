# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The independent-repeat sampling policy recorded on materialized evaluation inputs."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY
from nemo_gym.config.global_config import TASK_INDEX_KEY_NAME, rollout_run_key
from nemo_gym.models.token_capture.delivery import MASK_SAMPLE_KEY


STOP_ON_SUCCESS_THRESHOLD_KEY = "_ng_stop_on_success_threshold"
STOP_ON_SUCCESS_REASON = "Earlier repeat reached the verifier reward threshold"


def repeat_group(row: Mapping[str, Any]) -> tuple[int, str | None]:
    return row[TASK_INDEX_KEY_NAME], rollout_run_key(row)


def verified_reward(result: Mapping[str, Any]) -> float | None:
    reward = result.get("reward")
    if (
        result.get(NG_FAILURE_CLASS_KEY) is not None
        or result.get(NG_NO_PERSIST_KEY)
        or result.get(MASK_SAMPLE_KEY)
        or type(reward) not in (int, float)
        or not isfinite(reward)
    ):
        return None
    return reward


def is_verified_success(result: Mapping[str, Any], threshold: float) -> bool:
    reward = verified_reward(result)
    return reward is not None and reward >= threshold
