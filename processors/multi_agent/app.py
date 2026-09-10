# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable entrypoint for the core round-robin multi-agent processor."""

from nemo_gym.processors.multi_agent import MultiAgentProcessor


if __name__ == "__main__":
    MultiAgentProcessor.run_webserver()
