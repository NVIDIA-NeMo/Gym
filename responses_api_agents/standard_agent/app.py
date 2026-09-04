# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable entrypoint for the core policy-only Responses API agent."""

from nemo_gym.agents.responses_api_agent import StandardResponsesAPIAgent


if __name__ == "__main__":
    StandardResponsesAPIAgent.run_webserver()
