# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row metadata keys that carry a Harbor task's agent settings to the Harbor harness agent.

The agent never reads the task, so rows written for harbor_tasks carry the task.toml [agent] timeout and user in
responses_create_params.metadata. This module imports nothing, so code without Harbor installed can write those rows.
"""

HARBOR_AGENT_TIMEOUT_METADATA_KEY = "harbor_agent_timeout_sec"
HARBOR_AGENT_USER_METADATA_KEY = "harbor_agent_user"
