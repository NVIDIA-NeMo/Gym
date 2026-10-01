# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared-input metrics for native NeMo UserSim tasks."""

from collections.abc import Mapping


AGENT_ROLES = ("user", "assistant", "judge", "summary")


def compute_task_metrics(task_input: Mapping[str, object]) -> dict[str, bool | str | None]:
    """Return deterministic, dependency-free metrics for one UserSim task input."""
    resolved_row = task_input.get("resolved_row")
    if not isinstance(resolved_row, Mapping):
        raise ValueError("UserSim task_input.resolved_row must be a mapping")
    config = resolved_row.get("usersim_config")
    if not isinstance(config, Mapping):
        raise ValueError("UserSim resolved row usersim_config must be a mapping")

    responses_create_params = task_input.get("responses_create_params", {})
    if not isinstance(responses_create_params, Mapping):
        raise ValueError("UserSim task_input.responses_create_params must be a mapping")

    locale = resolved_row.get("locale")
    probe_type = resolved_row.get("probe_type")
    seed = config.get("random_seed")
    return {
        "UserSim locales": locale if isinstance(locale, str) else None,
        "UserSim probe types": probe_type if isinstance(probe_type, str) else None,
        "UserSim sampling seeds": str(seed) if isinstance(seed, int) else None,
        **{f"UserSim {role} Responses override coverage": role in responses_create_params for role in AGENT_ROLES},
    }
