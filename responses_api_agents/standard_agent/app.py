# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable entrypoint for the core policy-only Responses API agent."""

from nemo_gym.agents.responses_api_agent import ResponsesAPIAgent


if __name__ == "__main__":
    ResponsesAPIAgent.run_webserver()
