# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection and interpretation of saved rollout evidence."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.rollouts.health.timeouts import ELAPSED_S_KEY, read_rollout_timing, summarize_timeouts
    from nemo_gym.rollouts.health.types import Verdict
    from nemo_gym.rollouts.model_calls import (
        aggregate_model_call_metrics,
        await_model_call_capture_quiescence,
        clear_model_call_captures_for_rollouts,
        merge_model_call_capture_into_record,
        model_call_capture_dirs_from_config,
        observability_enabled_from_config,
        read_available_model_call_records,
        read_model_call_records,
    )
    from nemo_gym.rollouts.rollout_collection import (
        AGENT_REQUEST_FAILED_FAILURE_CLASS,
        ENVIRONMENT_SERVER_FAILURE_CLASS,
        NG_ENVIRONMENT_SERVER_KEY,
        NG_RESULT_TYPE_KEY,
        NG_TERMINAL_KEY,
        RolloutCollectionConfig,
        RolloutCollectionHelper,
        SharedRolloutCollectionConfig,
        enrich_run_result,
        get_max_rollout_attempts,
    )
    from nemo_gym.rollouts.rollout_health import run_health_checks
    from nemo_gym.rollouts.rollout_outcomes import RolloutFailureRecord
    from nemo_gym.rollouts.rollout_perf import DispatchLatencyTracker
    from nemo_gym.rollouts.rollout_records import (
        coverage_path_for,
        journal_path_for,
        logical_rollout_id,
        materialized_path_for,
    )
    from nemo_gym.rollouts.rollout_recovery import (
        IncompleteEvaluationError,
        RunManifest,
        is_terminal_failure,
        load_resubmit_offset,
        manifest_path_for,
        observed_elapsed,
        record_resubmit_offset,
    )
    from nemo_gym.rollouts.rollout_store import RolloutStore
    from nemo_gym.rollouts.segment_rewards import resolve_segment_rewards


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "enrich_run_result": "rollout_collection",
    "resolve_segment_rewards": "segment_rewards",
    "AGENT_REQUEST_FAILED_FAILURE_CLASS": "rollout_collection",
    "DispatchLatencyTracker": "rollout_perf",
    "ELAPSED_S_KEY": "health.timeouts",
    "ENVIRONMENT_SERVER_FAILURE_CLASS": "rollout_collection",
    "IncompleteEvaluationError": "rollout_recovery",
    "NG_ENVIRONMENT_SERVER_KEY": "rollout_collection",
    "NG_RESULT_TYPE_KEY": "rollout_collection",
    "NG_TERMINAL_KEY": "rollout_collection",
    "RolloutCollectionConfig": "rollout_collection",
    "RolloutCollectionHelper": "rollout_collection",
    "RolloutFailureRecord": "rollout_outcomes",
    "RolloutStore": "rollout_store",
    "RunManifest": "rollout_recovery",
    "SharedRolloutCollectionConfig": "rollout_collection",
    "Verdict": "health.types",
    "aggregate_model_call_metrics": "model_calls",
    "await_model_call_capture_quiescence": "model_calls",
    "clear_model_call_captures_for_rollouts": "model_calls",
    "coverage_path_for": "rollout_records",
    "get_max_rollout_attempts": "rollout_collection",
    "is_terminal_failure": "rollout_recovery",
    "journal_path_for": "rollout_records",
    "load_resubmit_offset": "rollout_recovery",
    "logical_rollout_id": "rollout_records",
    "manifest_path_for": "rollout_recovery",
    "materialized_path_for": "rollout_records",
    "merge_model_call_capture_into_record": "model_calls",
    "model_call_capture_dirs_from_config": "model_calls",
    "observability_enabled_from_config": "model_calls",
    "observed_elapsed": "rollout_recovery",
    "read_available_model_call_records": "model_calls",
    "read_model_call_records": "model_calls",
    "read_rollout_timing": "health.timeouts",
    "record_resubmit_offset": "rollout_recovery",
    "run_health_checks": "rollout_health",
    "summarize_timeouts": "health.timeouts",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
