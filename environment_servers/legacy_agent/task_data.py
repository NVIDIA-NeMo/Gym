# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields of a ``legacy_agent`` relay: a single-agent run request for the agent's ``/run``.

The agent's resources server, or the agent itself when it is self-contained, declares the other fields.
"""

from nemo_gym.task_data import SingleAgentTaskData as TaskData


__all__ = ["TaskData"]
