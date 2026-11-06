# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure rollout identity and reporting labels."""

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional

from nemo_gym.config.keys import AGENT_REF_KEY_NAME, ENVIRONMENT_SERVER_STAMP_KEY_NAME


def rollout_run_key(row: Mapping[str, Any]) -> Optional[str]:
    """Identify what ran a rollout, for grouping: its Environment Server.

    Records written before rollout collection stamped the Environment Server fall back to their agent.
    """
    server = row.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
    if server is not None:
        return server
    return (row.get(AGENT_REF_KEY_NAME) or {}).get("name")


def label_runs(agent_by_key: Mapping[str, Optional[str]]) -> Dict[str, str]:
    """Label each run key by its agent's name when that name identifies exactly one run, else by the key.

    A run with one Environment Server per agent keeps its agent's name, so existing labels do not change.
    Every run of an agent that several Environment Servers front is labelled by its own Environment Server.
    A label that would still repeat, because one server's name equals another run's agent name, also falls back
    to the key. The result depends only on the mapping, not on its order, and every label is unique.
    """
    keys_by_agent: Dict[str, set] = defaultdict(set)
    for key, agent_name in agent_by_key.items():
        if agent_name is not None:
            keys_by_agent[agent_name].add(key)
    labels = {
        key: agent_name if agent_name is not None and len(keys_by_agent[agent_name]) == 1 else key
        for key, agent_name in agent_by_key.items()
    }
    while True:
        counts = Counter(labels.values())
        clashing = [key for key, label in labels.items() if counts[label] > 1 and label != key]
        if not clashing:
            return labels
        for key in clashing:
            labels[key] = key


def rollout_run_labels(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Label each ``rollout_run_key`` for reports, the same way rollout collection labels aggregate metrics.

    See ``label_runs``. A row without an ``agent_ref`` is labelled by its Environment Server.
    """
    agent_by_key: Dict[str, Optional[str]] = {}
    for row in rows:
        key = rollout_run_key(row)
        if key is not None and key not in agent_by_key:
            agent_by_key[key] = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
    return label_runs(agent_by_key)


def rollout_agent_label(row: Mapping[str, Any]) -> Optional[str]:
    """Name the agent that acted in one rollout, for per-rollout output such as trajectories and debug lines.

    Rows without an ``agent_ref``, such as episode rows, use their Environment Server.
    """
    agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
    if agent_name is not None:
        return agent_name
    return row.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
