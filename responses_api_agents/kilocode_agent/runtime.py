# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled runtime files staged into Kilo sessions."""

from pathlib import Path

# Kilo shares OpenCode's correlation hook, its model retry limit plugin and the classifier for that plugin's error;
# keep this cross-agent dependency explicit.
from responses_api_agents.opencode_agent.runtime import (
    ASSISTANT_MESSAGE_PLUGIN_SOURCE as ASSISTANT_MESSAGE_PLUGIN_SOURCE,
)
from responses_api_agents.opencode_agent.runtime import MODEL_RETRY_PLUGIN_FILE as MODEL_RETRY_PLUGIN_FILE
from responses_api_agents.opencode_agent.runtime import MODEL_RETRY_PLUGIN_SOURCE as MODEL_RETRY_PLUGIN_SOURCE
from responses_api_agents.opencode_agent.runtime import retry_limit_outcome as retry_limit_outcome


RUNTIME_INSTALLER_SOURCE = Path(__file__).with_name("install_kilo_runtime.sh")
RUNTIME_INSTALLER_FILE = RUNTIME_INSTALLER_SOURCE.name
TASK_ENVIRONMENT_PLUGIN_SOURCE = Path(__file__).with_name("task_environment.mjs")
TASK_ENVIRONMENT_PLUGIN_FILE = TASK_ENVIRONMENT_PLUGIN_SOURCE.name
# Kilo's task-environment module imports the shared hook as an ES module.
ASSISTANT_MESSAGE_PLUGIN_FILE = ASSISTANT_MESSAGE_PLUGIN_SOURCE.with_suffix(".mjs").name
# Kilo's own cap on resends of one model step. The plugin above sets the same rule in sandbox sessions; the local
# run loads no plugins (--pure), so there this count is the only bound.
MODEL_RETRY_LIMIT_ENV = {"KILO_SESSION_RETRY_LIMIT": "2"}
