# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared-input metrics for native NeMo UserSim tasks."""

from collections.abc import Mapping


AGENT_ROLES = ("user", "assistant", "judge", "summary")


def compute_task_metrics(task_input: Mapping[str, object]) -> dict[str, bool | str | None]:
    """Return deterministic, dependency-free metrics for one UserSim task input."""
    scenario = task_input.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError("UserSim task_input.scenario must be a mapping")
    context = task_input.get("usersim_context")
    if not isinstance(context, Mapping):
        raise ValueError("UserSim task_input.usersim_context must be a mapping")

    responses_create_params = task_input.get("responses_create_params", {})
    if not isinstance(responses_create_params, Mapping):
        raise ValueError("UserSim task_input.responses_create_params must be a mapping")

    locale = scenario.get("locale")
    probe_type = scenario.get("probe_type")
    seed = context.get("seed")
    return {
        "UserSim locales": locale if isinstance(locale, str) else None,
        "UserSim probe types": probe_type if isinstance(probe_type, str) else None,
        "UserSim sampling seeds": str(seed) if isinstance(seed, int) else None,
        **{f"UserSim {role} Responses override coverage": role in responses_create_params for role in AGENT_ROLES},
    }
