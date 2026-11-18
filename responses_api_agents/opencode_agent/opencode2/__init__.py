# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What differs when ``opencode_agent`` runs OpenCode 2 (``opencode_version: 2.x``).

OpenCode 2 changed the interfaces the adapter uses: its configuration schema, its plugin API, its session store and
its ``session export``/``session import`` commands. ``config`` writes the run's configuration and command line,
``plugin/`` is the Gym plugin that replaces the OpenCode 1 plugin files, and ``store`` turns the OpenCode 2 session
store into the OpenCode 1 tables, so the session lifecycle and every reader of the store are shared.
"""

from pathlib import Path


PLUGIN_SOURCE = Path(__file__).with_name("plugin") / "index.js"
# Where the plugin directory goes in a session's private storage; OpenCode 2 loads a plugin from a directory.
PLUGIN_DIRECTORY = "nemo-gym-plugin"
# The plugin records here which OpenCode message each labeled model call became.
LABEL_FILE = "model-call-labels.jsonl"
