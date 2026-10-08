# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run rollout collection for the checkpoint e2e suite: ``python collect.py '<RolloutCollectionConfig JSON>'``.

The Gym deployment's config comes from ``NEMO_GYM_CONFIG_DICT``.
Exits 75 when a preemption signal checkpointed the run and stopped it.
"""

import asyncio
import json
import sys

from nemo_gym._checkpoint.collection import CollectionStopped
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


STOPPED = 75


def main() -> int:
    config = RolloutCollectionConfig.model_validate(json.loads(sys.argv[1]))
    try:
        asyncio.run(RolloutCollectionHelper().run_from_config(config))
    except CollectionStopped as stopped:
        print(stopped, flush=True)
        return STOPPED
    return 0


if __name__ == "__main__":
    sys.exit(main())
