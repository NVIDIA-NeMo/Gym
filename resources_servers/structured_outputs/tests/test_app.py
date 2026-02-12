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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.structured_outputs.app import (
    SchemaType,
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    StructuredOutputsVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> StructuredOutputsResourcesServerConfig:
        return StructuredOutputsResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    def _create_response(self, id: str, output_item: NeMoGymResponseOutputItem) -> dict[str, Any]:
        return NeMoGymResponse(
            id=id,
            created_at=1234.5,
            model="response_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    def _create_response_output_message(self, message_text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id=f"ID for {message_text}",
            content=[NeMoGymResponseOutputText(annotations=[], text=message_text, type="output_text")],
            role="assistant",
            status="in_progress",
            type="message",
        )

    async def test_verify_json(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        test_schema = {
            "type": "object",
            "properties": {
                "studentId": {"type": "string"},
                "examSubject": {"type": "string"},
                "plannedStudyHours": {"type": "integer"},
                "isFullTimeStudent": {"type": "boolean"},
                "studyMaterials": {
                    "type": "object",
                    "properties": {
                        "textbooks": {"type": "array", "items": {"type": "string"}},
                        "onlineResources": {"type": "array", "items": {"type": "string"}},
                        "practiceExams": {
                            "type": "object",
                            "properties": {
                                "completedCount": {"type": "integer"},
                                "averageScore": {"type": "number"},
                                "mostRecentDate": {"type": "string", "format": "date"},
                            },
                            "required": ["completedCount", "averageScore", "mostRecentDate"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["textbooks", "onlineResources", "practiceExams"],
                    "additionalProperties": False,
                },
                "studySchedule": {
                    "type": "object",
                    "properties": {
                        "weeklyHours": {"type": "integer"},
                        "sessionsPerWeek": {"type": "integer"},
                        "preferredTimeOfDay": {"type": "string", "enum": ["morning", "afternoon", "evening"]},
                        "studyDays": {"type": "array", "items": {"type": "string"}},
                        "breakSchedule": {
                            "type": "object",
                            "properties": {
                                "shortBreakMinutes": {"type": "integer"},
                                "longBreakMinutes": {"type": "integer"},
                                "breakFrequencyMinutes": {"type": "integer"},
                            },
                            "required": ["shortBreakMinutes", "longBreakMinutes", "breakFrequencyMinutes"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["weeklyHours", "sessionsPerWeek", "preferredTimeOfDay", "studyDays", "breakSchedule"],
                    "additionalProperties": False,
                },
                "preparationStatus": {
                    "type": "string",
                    "enum": ["not_started", "in_progress", "review_only", "ready"],
                },
            },
        }
        test_completion = '{"studentId":"STU12345","examSubject":"Calculus II","plannedStudyHours":120,"isFullTimeStudent":true,"studyMaterials":{"textbooks":["Calculus: Early Transcendentals","Schaum\u2019s Outline of Calculus","The Humongous Book of Calculus Problems"],"onlineResources":["Khan Academy","Paul\u2019s Online Math Notes","Coursera Calculus Course"],"practiceExams":{"completedCount":8,"averageScore":87.5,"mostRecentDate":"2024-05-10"}},"studySchedule":{"weeklyHours":15,"sessionsPerWeek":5,"preferredTimeOfDay":"evening","studyDays":["Monday","Wednesday","Friday","Saturday","Sunday"],"breakSchedule":{"shortBreakMinutes":10,"longBreakMinutes":25,"breakFrequencyMinutes":50}},"preparationStatus":"in_progress"}'

        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid JSON ---
        valid_output_item = self._create_response_output_message(test_completion)
        valid_response = NeMoGymResponse(
            id="valid_response_id",
            created_at=1234.5,
            model="test_model",
            object="response",
            output=[valid_output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        valid_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=valid_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )

        valid_verify_response = await resources_server.verify(valid_request)
        assert valid_verify_response.reward == 1.0
        assert valid_verify_response.response == valid_response

        # --- Test 2: Invalid JSON (Not parsable) ---
        invalid_json_completion = '{"studentId":"STU12345", '  # Broken JSON
        invalid_json_output_item = self._create_response_output_message(invalid_json_completion)
        invalid_json_response = valid_response.model_copy(
            deep=True, update={"id": "invalid_json_id", "output": [invalid_json_output_item]}
        )

        invalid_json_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=invalid_json_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )

        invalid_json_verify_response = await resources_server.verify(invalid_json_request)
        assert invalid_json_verify_response.reward == 0.0

        # --- Test 3: Schema Mismatch (Missing field) ---
        # `strictify_schema_json` makes all fields required.
        parsed_completion = json.loads(test_completion)
        del parsed_completion["studentId"]
        missing_field_completion = json.dumps(parsed_completion)

        missing_field_output_item = self._create_response_output_message(missing_field_completion)
        missing_field_response = valid_response.model_copy(
            deep=True, update={"id": "missing_field_id", "output": [missing_field_output_item]}
        )

        missing_field_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_field_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )

        missing_field_verify_response = await resources_server.verify(missing_field_request)
        assert missing_field_verify_response.reward == 0.0

        # --- Test 4: Schema Mismatch (Extra field) ---
        # `strictify_schema_json` sets additionalProperties=False.
        parsed_completion = json.loads(test_completion)
        parsed_completion["extraField"] = "some value"
        extra_field_completion = json.dumps(parsed_completion)

        extra_field_output_item = self._create_response_output_message(extra_field_completion)
        extra_field_response = valid_response.model_copy(
            deep=True, update={"id": "extra_field_id", "output": [extra_field_output_item]}
        )

        extra_field_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_field_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )

        extra_field_verify_response = await resources_server.verify(extra_field_request)
        assert extra_field_verify_response.reward == 0.0

        # --- Test 5: Schema Mismatch (Wrong type) ---
        parsed_completion = json.loads(test_completion)
        parsed_completion["plannedStudyHours"] = "one hundred"  # Should be integer
        wrong_type_completion = json.dumps(parsed_completion)

        wrong_type_output_item = self._create_response_output_message(wrong_type_completion)
        wrong_type_response = valid_response.model_copy(
            deep=True, update={"id": "wrong_type_id", "output": [wrong_type_output_item]}
        )

        wrong_type_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=wrong_type_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )

        wrong_type_verify_response = await resources_server.verify(wrong_type_request)
        assert wrong_type_verify_response.reward == 0.0

        # --- Test 6: Schema Mismatch (Nested extra field) ---
        # Test that `strictify_schema_json` recurses correctly
        parsed_completion = json.loads(test_completion)
        parsed_completion["studyMaterials"]["practiceExams"]["extraNestedField"] = "bad value"
        nested_extra_field_completion = json.dumps(parsed_completion)

        nested_extra_field_output_item = self._create_response_output_message(nested_extra_field_completion)
        nested_extra_field_response = valid_response.model_copy(
            deep=True, update={"id": "nested_extra_id", "output": [nested_extra_field_output_item]}
        )

        nested_extra_field_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=nested_extra_field_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )

        nested_extra_field_verify_response = await resources_server.verify(nested_extra_field_request)
        assert nested_extra_field_verify_response.reward == 0.0
