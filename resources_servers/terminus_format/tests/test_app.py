# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any
from unittest.mock import MagicMock

from pytest import fixture

from nemo_gym.server_utils import ServerClient
from resources_servers.terminus_format.app import (
    TerminusFormatResourcesServer,
    TerminusFormatResourcesServerConfig,
)


class TestApp:
    @fixture
    def config(self) -> TerminusFormatResourcesServerConfig:
        return TerminusFormatResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    @fixture
    def server(self, config: TerminusFormatResourcesServerConfig) -> TerminusFormatResourcesServer:
        server_client = MagicMock(spec=ServerClient)
        return TerminusFormatResourcesServer(config=config, server_client=server_client)

    def _make_valid_payload(self, overrides: dict[str, Any] | None = None) -> str:
        """Create a JSON string that matches COMMAND_BATCH_RESPONSE_SCHEMA, with optional overrides."""
        base: dict[str, Any] = {
            "state_analysis": "Current directory contains project files.",
            "explanation": "List files, then change into the project directory.",
            "commands": [
                {
                    "keystrokes": "ls -la\n",
                    "is_blocking": True,
                    "timeout_sec": 5.0,
                },
                {
                    "keystrokes": "cd project\n",
                    "is_blocking": False,
                    "timeout_sec": 2.0,
                },
            ],
            "is_task_complete": False,
        }
        if overrides:
            # shallow update is enough for tests; nested modifications can pass dicts directly
            base.update(overrides)
        return json.dumps(base)

    def test_sanity_server_can_be_constructed(self, config: TerminusFormatResourcesServerConfig) -> None:
        TerminusFormatResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_evaluate_valid_json_returns_full_reward(self, server: TerminusFormatResourcesServer) -> None:
        response_text = self._make_valid_payload()
        reward = server.evaluate_terminus_format_response_json(response_text)
        assert reward == 1.0

    def test_evaluate_invalid_json_returns_zero(self, server: TerminusFormatResourcesServer) -> None:
        # Broken JSON (missing closing brace)
        response_text = '{"state_analysis": "oops"'
        reward = server.evaluate_terminus_format_response_json(response_text)
        assert reward == 0.0

    def test_missing_required_field_returns_zero(self, server: TerminusFormatResourcesServer) -> None:
        # Drop a required top-level field: "commands"
        payload = json.loads(self._make_valid_payload())
        payload.pop("commands")
        response_text = json.dumps(payload)

        reward = server.evaluate_terminus_format_response_json(response_text)
        assert reward == 0.0

    def test_wrong_type_in_command_returns_zero(self, server: TerminusFormatResourcesServer) -> None:
        # "timeout_sec" must be a number; here we make it a string
        payload = json.loads(self._make_valid_payload())
        payload["commands"][0]["timeout_sec"] = "not-a-number"
        response_text = json.dumps(payload)

        reward = server.evaluate_terminus_format_response_json(response_text)
        assert reward == 0.0

    def test_extra_top_level_field_returns_zero(self, server: TerminusFormatResourcesServer) -> None:
        # additionalProperties=False at the top level -> extra field should fail
        payload = json.loads(self._make_valid_payload())
        payload["extra_field"] = "not allowed"
        response_text = json.dumps(payload)

        reward = server.evaluate_terminus_format_response_json(response_text)
        assert reward == 0.0

    def test_extra_field_in_command_object_returns_zero(self, server: TerminusFormatResourcesServer) -> None:
        # Command definition also has additionalProperties=False
        payload = json.loads(self._make_valid_payload())
        payload["commands"][0]["extra_field"] = "also not allowed"
        response_text = json.dumps(payload)

        reward = server.evaluate_terminus_format_response_json(response_text)
        assert reward == 0.0
