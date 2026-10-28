# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Harbor task folders and hub references into NeMo Gym materialized tasks.

Harbor is the input format; Gym owns execution. A task is a folder holding
``task.toml``, ``instruction.md``, ``environment/`` and ``tests/``. A dataset is a
folder of task folders. ``harbor:<dataset>[@<version>]`` names a dataset in the
Harbor registry, which this package fetches into ``environments/harbor/`` in the Gym checkout, or
under the working directory when Gym is installed as a package (``gym dataset pull``;
:func:`nemo_gym.tasks.harbor.hub.datasets_dir`); a committed entry there declares its dataset as
``[gym] source`` and is pulled into its own folder (:mod:`nemo_gym.tasks.harbor.entries`).
"""

from nemo_gym.tasks.harbor.digest import DIGEST_KEY, content_hash
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from nemo_gym.tasks.harbor.task import HarborTask, discover_tasks, load_task


__all__ = ["DIGEST_KEY", "HarborTask", "HarborTaskConfig", "content_hash", "discover_tasks", "load_task"]
