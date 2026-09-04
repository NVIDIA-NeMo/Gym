# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable server entrypoint for single-agent-turn orchestration."""

from nemo_gym.processors.single_agent_turn import SingleAgentTurnProcessor


if __name__ == "__main__":
    SingleAgentTurnProcessor.run_webserver()
