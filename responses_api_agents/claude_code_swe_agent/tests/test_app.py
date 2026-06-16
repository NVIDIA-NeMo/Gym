# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Claude Code sandbox-bound custom agent's pure logic."""

import json

from responses_api_agents.claude_code_swe_agent.app import claude_command, claude_settings_json


def test_claude_command_shape():
    cmd = claude_command(
        prompt="fix it",
        model="claude-sonnet-4-5",
        max_turns=20,
        system_prompt="be terse",
        allowed_tools=None,
        disallowed_tools=None,
    )
    assert "claude -p --output-format stream-json" in cmd
    assert "--max-turns 20" in cmd
    assert "--model claude-sonnet-4-5" in cmd
    assert "--append-system-prompt 'be terse'" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert cmd.strip().endswith("'fix it'")


def test_claude_settings_telemetry_off():
    settings = json.loads(claude_settings_json())
    assert settings["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
    assert settings["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
