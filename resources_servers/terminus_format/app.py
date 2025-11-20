# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json

from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class TerminusFormatResourcesServerConfig(BaseResourcesServerConfig):
    pass


COMMAND_BATCH_RESPONSE_SCHEMA = {
    "title": "CommandBatchResponse",
    "type": "object",
    "additionalProperties": False,
    "definitions": {
        "Command": {
            "title": "Command",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keystrokes": {
                    "title": "Keystrokes",
                    "description": (
                        "Keystrokes to execute in the terminal. Use tmux-style escape "
                        "sequences for modifier keys (e.g. C-c for ctrl-c). Modifier keys "
                        "must be sent as their own commands otherwise the characters will "
                        "be interpreted literally."
                    ),
                    "type": "string",
                },
                "is_blocking": {
                    "title": "Is Blocking",
                    "description": (
                        "Whether to wait for and return the terminal output after executing "
                        "these keystrokes. This will append '; tmux wait -S done' to your "
                        "command. DO NOT block on modifier keys or inside interactive "
                        "programs (e.g. vim or less). Only block when the command is "
                        "executed in the command line, is not interactive, and you expect "
                        "the output to be returned with no intervention. When in doubt, "
                        "wait instead of blocking."
                    ),
                    "type": "boolean",
                },
                "timeout_sec": {
                    "title": "Timeout Sec",
                    "description": "The number of expected seconds to wait for the command to complete.",
                    "type": "number",
                },
            },
            "required": ["keystrokes", "is_blocking", "timeout_sec"],
        }
    },
    "properties": {
        "state_analysis": {
            "title": "State Analysis",
            "description": "Description of the current state of the terminal",
            "type": "string",
        },
        "explanation": {
            "title": "Explanation",
            "description": "Brief explanation of what these commands will do",
            "type": "string",
        },
        "commands": {
            "title": "Commands",
            "description": "List of shell interactions to execute in the Docker container",
            "type": "array",
            "items": {
                "$ref": "#/definitions/Command",
            },
        },
        "is_task_complete": {
            "title": "Is Task Complete",
            "description": (
                "Whether the task is complete following the execution of these commands. "
                "Make sure to check that the command you last executed worked before "
                "saying you're done."
            ),
            "type": "boolean",
        },
    },
    "required": ["state_analysis", "explanation", "commands", "is_task_complete"],
}


class TerminusFormatResourcesServer(SimpleResourcesServer):
    config: TerminusFormatResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        response_text = "".join(assistant_responses)

        reward = self.evaluate_terminus_format_response_json(response_text)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    # ----- JSON Helpers ----- #
    def evaluate_terminus_format_response_json(self, response_text: str) -> float:
        """Validate the model response against the fixed terminus format schema."""
        try:
            response_obj = json.loads(response_text)
        except Exception:
            # Not valid JSON
            return 0.0

        try:
            validate_against_schema_openapi(response_obj, COMMAND_BATCH_RESPONSE_SCHEMA)
        except Exception:
            # JSON but does not match schema
            return 0.0

        # Valid JSON and matches schema
        return 1.0


if __name__ == "__main__":
    TerminusFormatResourcesServer.run_webserver()
