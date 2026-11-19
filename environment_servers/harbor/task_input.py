# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task input Harbor's episode request validates.

Data tooling loads this module without the server's dependencies (see ``nemo_gym/data/task_input.py``).
"""

from nemo_gym.data.tasks.harbor import HarborTaskInput as TaskInput


__all__ = ["TaskInput"]
