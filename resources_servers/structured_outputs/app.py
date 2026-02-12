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
from enum import StrEnum
from typing import Any, Dict

from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class StructuredOutputsResourcesServerConfig(BaseResourcesServerConfig):
    pass


class SchemaType(StrEnum):
    JSON = "json"


class StructuredOutputsVerifyRequest(BaseVerifyRequest):
    # string representation of schema. For JSON, it is a json dictionary.
    schema_str: str
    schema_type: SchemaType


class StructuredOutputsResourcesServer(SimpleResourcesServer):
    config: StructuredOutputsResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: StructuredOutputsVerifyRequest) -> BaseVerifyResponse:
        schema_type = body.schema_type
        schema_str = body.schema_str

        # get model generation.
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)
        response_text = "".join(assistant_responses)

        # verify based on schema type
        match schema_type:
            case SchemaType.JSON:
                reward = self.evaluate_structured_output_response_json(schema_str, response_text)
            case _:
                raise NotImplementedError(f"SchemaType must be one of {list(SchemaType)}, got {schema_type} !")

        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    # ----- JSON Helpers ----- #
    def strictify_schema_json(self, schema: Dict[str, Any]):
        """Make a schema strict as per OpenAPI guidelines"""
        if isinstance(schema, Dict):
            if "properties" in schema:
                schema["required"] = list(schema["properties"])
                schema["additionalProperties"] = False
            for k, v in schema.items():
                self.strictify_schema_json(v)

    def evaluate_structured_output_response_json(self, schema_str: str, response_text: str) -> bool:
        try:
            schema = json.loads(schema_str)
            self.strictify_schema_json(schema)
            response_obj = json.loads(response_text)
            validate_against_schema_openapi(response_obj, schema)
            return 1.0
        except Exception:
            return 0.0


if __name__ == "__main__":
    StructuredOutputsResourcesServer.run_webserver()
