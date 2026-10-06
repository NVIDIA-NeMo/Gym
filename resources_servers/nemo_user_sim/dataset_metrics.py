# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared-input metrics for NeMo UserSim materialized tasks."""

from collections.abc import Mapping
from typing import get_args

from resources_servers.nemo_user_sim.task_data import UserSimAgentRole


AGENT_ROLES = get_args(UserSimAgentRole)


def compute_task_metrics(task_input: Mapping[str, object]) -> dict[str, bool | str | None]:
    """Return deterministic, dependency-free metrics for one UserSim task input."""
    resolved_row = task_input.get("resolved_row")
    if not isinstance(resolved_row, Mapping):
        raise ValueError("UserSim task_input.resolved_row must be a mapping")
    usersim_config = resolved_row.get("usersim_config")
    if not isinstance(usersim_config, Mapping):
        raise ValueError("UserSim task_input.resolved_row.usersim_config must be a mapping")
    seed = usersim_config.get("random_seed")
    role_request_params = task_input.get("role_request_params", {})
    if not isinstance(role_request_params, Mapping):
        raise ValueError("UserSim role_request_params must be a mapping")

    locale = resolved_row.get("locale")
    probe_type = resolved_row.get("probe_type")
    return {
        "UserSim locales": locale if isinstance(locale, str) else None,
        "UserSim probe types": probe_type if isinstance(probe_type, str) else None,
        "UserSim sampling seeds": str(seed) if isinstance(seed, int) else None,
        **{f"UserSim {role} Responses override coverage": role in role_request_params for role in AGENT_ROLES},
    }
