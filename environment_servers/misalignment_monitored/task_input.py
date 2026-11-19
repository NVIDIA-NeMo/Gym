# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""misalignment_monitored is a relay: it has no episode task input.

Its ``/run`` passes a flat row to the agent's ``/run``, which validates a ``BaseRunRequest``.
It takes no task rows (see ``nemo_gym/data/task_input.py``).
"""

TaskInput = None
