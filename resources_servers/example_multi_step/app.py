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
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ExampleMultiStepResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetSynonymValueRequest(BaseModel):
    synonym: str


class GetSynonymValueResponse(BaseModel):
    synonym_value: int


class ExtractSynonymValuesRequest(BaseModel):
    synonym_values: List[int]


class ExtractSynonymValuesResponse(BaseModel):
    success: bool


class ExampleMultiStepRunRequest(BaseRunRequest):
    id: int
    expected_synonym_values: List[int]
    expected_synonyms: List[str]
    minefield_label: str
    minefield_label_value: int


class ExampleMultiStepVerifyRequest(ExampleMultiStepRunRequest, BaseVerifyRequest):
    pass


class ExampleMultiStepVerifyResponse(BaseVerifyResponse):
    parsed_synonym_values: List[int]
    accuracy: bool
    set_overlap: float
    original_term_minefield_hit: bool
    order_instruction_following_failure: bool


class ExampleMultiStepResourcesServer(SimpleResourcesServer):
    config: ExampleMultiStepResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_synonym_value")(self.get_synonym_value)
        app.post("/extract_synonym_values")(self.extract_synonym_values)

        return app

    async def get_synonym_value(self, body: GetSynonymValueRequest) -> GetSynonymValueResponse:
        return GetSynonymValueResponse(synonym_value=sum(map(ord, body.synonym)))

    async def extract_synonym_values(self, body: ExtractSynonymValuesRequest) -> ExtractSynonymValuesResponse:
        return ExtractSynonymValuesResponse(success=True)

    async def verify(self, body: ExampleMultiStepVerifyRequest) -> ExampleMultiStepVerifyResponse:
        expected = body.expected_synonym_values

        actual = []
        for output in reversed(body.response.output):
            if output.type == "function_call" and output.name == "extract_synonym_values":
                actual = json.loads(output.arguments)["synonym_values"]
                break

        accuracy = expected == actual
        set_overlap = len(set(actual) & set(expected)) / len(expected)
        return ExampleMultiStepVerifyResponse(
            **body.model_dump(),
            reward=float(accuracy),
            parsed_synonym_values=actual,
            accuracy=accuracy,
            set_overlap=set_overlap,
            original_term_minefield_hit=body.minefield_label in actual or body.minefield_label_value in actual,
            order_instruction_following_failure=not accuracy and set_overlap == 1.0,
        )


if __name__ == "__main__":
    ExampleMultiStepResourcesServer.run_webserver()
