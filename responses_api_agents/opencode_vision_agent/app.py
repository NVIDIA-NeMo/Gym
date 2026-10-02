# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode in a venv with an image-analysis Python.

The OpenCode CLI runs its bash tool under this server's environment, so `python3` in the
model's shell is this venv: pillow, numpy, scipy and opencv (see requirements.txt). The agent
itself is responses_api_agents/opencode_agent; set attach_images/vision for image tasks.
"""

from responses_api_agents.opencode_agent.app import OpenCodeAgent


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
