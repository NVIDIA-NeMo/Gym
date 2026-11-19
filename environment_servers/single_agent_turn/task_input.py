# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task input single_agent_turn's episode request validates.

Collation and ``gym env validate`` load this module without the server's dependencies (see ``nemo_gym/data/task_input.py``).
"""

from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnTaskInput as TaskInput


__all__ = ["TaskInput"]
