# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated entry point for ``langgraph_parallel_thinking_reasoning_gym.prepare``."""

import logging

from environments.langgraph_parallel_thinking_reasoning_gym.prepare import main


if __name__ == "__main__":
    logging.getLogger(__name__).warning(
        "`environments/reasoning_gym_parallel_thinking/prepare.py` is deprecated; "
        "use `environments/langgraph_parallel_thinking_reasoning_gym/prepare.py`."
    )
    main()
