# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection and interpretation of saved rollout evidence."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.rollouts.harness_capabilities.behavior import inspect_behavior
    from nemo_gym.rollouts.harness_capabilities.checker import NAMES, EvidenceScope, inspect_record
    from nemo_gym.rollouts.harness_capabilities.cli import digest_file, inspect_bundle, json_rows, main
    from nemo_gym.rollouts.harness_capabilities.health import inspect_health
    from nemo_gym.rollouts.harness_capabilities.results import gate_passes, render_matrices
    from nemo_gym.rollouts.health.types import Verdict
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
        DispatchLatencyTracker,
        RolloutCollectionConfig,
        RolloutCollectionHelper,
        get_max_rollout_attempts,
        is_terminal_failure,
        load_resubmit_offset,
        observed_elapsed,
        record_resubmit_offset,
    )
    from nemo_gym.rollouts.rollout_health import run_health_checks


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "DispatchLatencyTracker": "rollout_collection",
    "ENVIRONMENT_SERVER_FAILURE_CLASS": "rollout_collection",
    "EvidenceScope": "harness_capabilities.checker",
    "ModelCallRecord": "model_calls",
    "NAMES": "harness_capabilities.checker",
    "NG_ENVIRONMENT_SERVER_KEY": "rollout_collection",
    "NG_RESULT_TYPE_KEY": "rollout_collection",
    "NG_TERMINAL_KEY": "rollout_collection",
    "NG_TRAJECTORY_KEY": "rollout_collection",
    "RolloutCollectionConfig": "rollout_collection",
    "RolloutCollectionHelper": "rollout_collection",
    "Verdict": "health.types",
    "clear_model_call_captures_for_rollouts": "model_calls",
    "digest_file": "harness_capabilities.cli",
    "gate_passes": "harness_capabilities.results",
    "get_max_rollout_attempts": "rollout_collection",
    "inspect_behavior": "harness_capabilities.behavior",
    "inspect_bundle": "harness_capabilities.cli",
    "inspect_health": "harness_capabilities.health",
    "inspect_record": "harness_capabilities.checker",
    "is_terminal_failure": "rollout_collection",
    "json_rows": "harness_capabilities.cli",
    "load_resubmit_offset": "rollout_collection",
    "main": "harness_capabilities.cli",
    "merge_model_call_capture_into_record": "model_calls",
    "model_call_capture_dirs_from_config": "model_calls",
    "observability_enabled_from_config": "model_calls",
    "observed_elapsed": "rollout_collection",
    "record_resubmit_offset": "rollout_collection",
    "render_matrices": "harness_capabilities.results",
    "run_health_checks": "rollout_health",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
