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
from typing import Any, Dict

from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class TerminusFormatServerResourcesServerConfig(BaseResourcesServerConfig):
    pass


# Fixed JSON schema for the terminal agent response.
TERMINUS_FORMAT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "terminal_agent_response",
    "properties": {
        "analysis": {"type": "string"},
        "plan": {"type": "string"},
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keystrokes": {"type": "string"},
                    "duration": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0,
                    },
                },
                "required": ["keystrokes"],
                "additionalProperties": False,
            },
        },
        "task_complete": {
            "type": "boolean",
            "default": False,
        },
    },
    "required": ["analysis", "plan", "commands"],
    "additionalProperties": False,
    # commands must be EITHER:
    #   - empty array: []
    #   - OR array with ≥1 item (and keystrokes required per item)
    "anyOf": [
        {
            "properties": {
                "commands": {
                    "type": "array",
                    "maxItems": 0,
                }
            }
        },
        {
            "properties": {
                "commands": {
                    "type": "array",
                    "minItems": 1,
                }
            }
        },
    ],
}


class TerminusFormatServerResourcesServer(SimpleResourcesServer):
    config: TerminusFormatServerResourcesServerConfig

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
            validate_against_schema_openapi(response_obj, TERMINUS_FORMAT_SCHEMA)
        except Exception:
            # JSON but does not match schema
            return 0.0

        # Valid JSON and matches schema
        return 1.0


if __name__ == "__main__":
    TerminusFormatServerResourcesServer.run_webserver()
