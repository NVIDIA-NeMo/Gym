# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection and interpretation of saved rollout evidence."""

from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.rollouts.model_calls import (
        ModelCallRecord,
        clear_model_call_captures_for_rollouts,
        merge_model_call_capture_into_record,
        model_call_capture_dirs_from_config,
        observability_enabled_from_config,
    )
    from nemo_gym.rollouts.rollout_collection import (
        ENVIRONMENT_SERVER_FAILURE_CLASS,
        NG_ENVIRONMENT_SERVER_KEY,
        NG_RESULT_TYPE_KEY,
        NG_TERMINAL_KEY,
        NG_TRAJECTORY_KEY,
        RolloutCollectionHelper,
        get_max_rollout_attempts,
        is_terminal_failure,
        load_resubmit_offset,
        observed_elapsed,
        record_resubmit_offset,
    )
    from nemo_gym.rollouts.rollout_perf import DispatchLatencyTracker


# The public API: what code outside the library, such as servers, imports from this package. Each name's
# submodule loads on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "DispatchLatencyTracker": "rollout_perf",
    "ENVIRONMENT_SERVER_FAILURE_CLASS": "rollout_collection",
    "ModelCallRecord": "model_calls",
    "NG_ENVIRONMENT_SERVER_KEY": "rollout_collection",
    "NG_RESULT_TYPE_KEY": "rollout_collection",
    "NG_TERMINAL_KEY": "rollout_collection",
    "NG_TRAJECTORY_KEY": "rollout_collection",
    "RolloutCollectionHelper": "rollout_collection",
    "clear_model_call_captures_for_rollouts": "model_calls",
    "get_max_rollout_attempts": "rollout_collection",
    "is_terminal_failure": "rollout_collection",
    "load_resubmit_offset": "rollout_collection",
    "merge_model_call_capture_into_record": "model_calls",
    "model_call_capture_dirs_from_config": "model_calls",
    "observability_enabled_from_config": "model_calls",
    "observed_elapsed": "rollout_collection",
    "record_resubmit_offset": "rollout_collection",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
