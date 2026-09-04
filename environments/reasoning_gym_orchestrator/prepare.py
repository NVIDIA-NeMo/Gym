# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated entry point for ``langgraph_orchestrator_reasoning_gym.prepare``."""

import logging

from environments.langgraph_orchestrator_reasoning_gym.prepare import main


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.warning(
        "`environments/reasoning_gym_orchestrator/prepare.py` is deprecated; "
        "use `environments/langgraph_orchestrator_reasoning_gym/prepare.py`."
    )
    main()
