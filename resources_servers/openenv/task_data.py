# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the openenv server.

Generic wrapper around Meta's OpenEnv environments. The task lives entirely OUT of the row:
environment selection is config-driven (``OpenEnvResourcesServerConfig.env_class`` /
``action_class`` / ``reset_kwargs`` / ``is_mcp``, ``app.py:53``), the per-session env is created
at /seed_session with only the configured ``reset_kwargs``, and verify() (``app.py:234``) takes a
bare ``BaseVerifyRequest`` and reads zero row fields — reward is the per-session
``accumulated_reward`` populated by /step and the MCP tool endpoints during the rollout. Rows
carry no task-specific fields at all: per-env variation (coding / echo / maze prompts and tools)
lives inside ``responses_create_params``, so the schema is empty. The empty shape matches the
gymnasium base and blackjack but is redefined here (no fields to share; openenv is not even a
gymnasium-family server).
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    """No task-owned fields: the environment and its reset arguments come from server config."""

    model_config = ConfigDict(extra="allow")
