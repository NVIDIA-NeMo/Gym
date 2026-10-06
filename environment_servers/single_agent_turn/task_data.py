# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields that ``single_agent_turn`` reads itself.

It sends ``responses_create_params`` to its agent and passes every other field to its resources
server, whose ``TaskData`` declares them. ``SingleAgentTurnTaskInput`` is the runtime model.
"""

from nemo_gym.task_data import SingleAgentTaskData as TaskData


__all__ = ["TaskData"]
