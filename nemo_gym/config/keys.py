# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration keys, independent of configuration resolution."""

from typing import Tuple


ENTRYPOINT_KEY_NAME = "entrypoint"
AGENT_SERVER_TYPE_KEY_NAME = "responses_api_agents"
ENVIRONMENT_SERVER_TYPE_KEY_NAME = "environment_servers"
MODEL_SERVER_TYPE_KEY_NAME = "responses_api_models"
RESOURCES_SERVER_TYPE_KEY_NAME = "resources_servers"
# Public config-group keys, in discovery order, for Gym and downstream integrations.
SERVER_TYPE_KEYS: Tuple[str, ...] = (
    RESOURCES_SERVER_TYPE_KEY_NAME,
    AGENT_SERVER_TYPE_KEY_NAME,
    MODEL_SERVER_TYPE_KEY_NAME,
    ENVIRONMENT_SERVER_TYPE_KEY_NAME,
)
# Data keys
TASK_INDEX_KEY_NAME = "_ng_task_index"
ROLLOUT_INDEX_KEY_NAME = "_ng_rollout_index"
# Resume re-dispatch attempt counter (0 on the first attempt); distinguishes retries of the same
# (task, rollout) so their captured model calls stay separable.
ATTEMPT_INDEX_KEY_NAME = "_ng_attempt_index"
# An explicit capture id replaces the task and rollout derivation.
# Set it when dispatches reuse task and rollout indices.
# Otherwise two dispatches would share one capture key.
# The suffix "-a<N>" is reserved for attempt-qualified ids, so a dataset row's id must not end with it.
ROLLOUT_ID_KEY_NAME = "_ng_rollout_id"
AGENT_REF_KEY_NAME = "agent_ref"
# Stamped by rollout collection on every record: the Environment Server that ran the rollout.
ENVIRONMENT_SERVER_STAMP_KEY_NAME = "_ng_environment_server"
