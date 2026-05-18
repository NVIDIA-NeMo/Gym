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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import xmltodict
import yaml
from pytest import fixture

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
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

    def _create_response_function_call(
        self, *, name: str = "extract_record", arguments: str = "{}"
    ) -> NeMoGymResponseFunctionToolCall:
        return NeMoGymResponseFunctionToolCall(
            arguments=arguments,
            call_id="call_1",
            name=name,
            type="function_call",
        )

    async def test_verify_tool_call_modes(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        schema_str = json.dumps(schema)
        valid_payload = {"name": "Alice", "age": 30}
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        def make_response(output_item: NeMoGymResponseOutputItem | list[NeMoGymResponseOutputItem]) -> NeMoGymResponse:
            output = output_item if isinstance(output_item, list) else [output_item]
            return NeMoGymResponse(
                id="tool_call_response_id",
                created_at=1234.5,
                model="test_model",
                object="response",
                output=output,
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
            )

        direct_response = make_response(self._create_response_function_call(arguments=json.dumps(valid_payload)))
        direct_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=direct_response,
            schema_str=schema_str,
            response_mode="tool_call",
            tool_name="extract_record",
            tool_schema_mode="direct",
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        direct_result = await resources_server.verify(direct_request)
        assert direct_result.reward == 1.0
        assert direct_result.error_type is None

        extraction_wrapper_response = make_response(
            self._create_response_function_call(arguments=json.dumps({"extraction": valid_payload}))
        )
        extraction_wrapper_request = direct_request.model_copy(
            deep=True,
            update={
                "response": extraction_wrapper_response,
                "tool_schema_mode": "extraction_wrapper",
                "tool_payload_key": "extraction",
            },
        )
        extraction_wrapper_result = await resources_server.verify(extraction_wrapper_request)
        assert extraction_wrapper_result.reward == 1.0
        assert extraction_wrapper_result.error_type is None

        random_wrapper_response = make_response(
            self._create_response_function_call(arguments=json.dumps({"summary": valid_payload}))
        )
        random_wrapper_request = direct_request.model_copy(
            deep=True,
            update={
                "response": random_wrapper_response,
                "tool_schema_mode": "random_wrapper",
                "tool_payload_key": "summary",
            },
        )
        random_wrapper_result = await resources_server.verify(random_wrapper_request)
        assert random_wrapper_result.reward == 1.0
        assert random_wrapper_result.error_type is None

        wrong_tool_response = make_response(
            self._create_response_function_call(name="distractor_tool", arguments=json.dumps(valid_payload))
        )
        wrong_tool_request = direct_request.model_copy(deep=True, update={"response": wrong_tool_response})
        wrong_tool_result = await resources_server.verify(wrong_tool_request)
        assert wrong_tool_result.reward == 0.0
        assert wrong_tool_result.error_type == "wrong_tool_name"

        missing_tool_response = make_response(self._create_response_output_message("I cannot extract that."))
        missing_tool_request = direct_request.model_copy(deep=True, update={"response": missing_tool_response})
        missing_tool_result = await resources_server.verify(missing_tool_request)
        assert missing_tool_result.reward == 0.0
        assert missing_tool_result.error_type == "missing_tool_call"

        multiple_tool_response = make_response(
            [
                self._create_response_function_call(arguments=json.dumps(valid_payload)),
                self._create_response_function_call(name="distractor_tool", arguments=json.dumps(valid_payload)),
            ]
        )
        multiple_tool_request = direct_request.model_copy(deep=True, update={"response": multiple_tool_response})
        multiple_tool_result = await resources_server.verify(multiple_tool_request)
        assert multiple_tool_result.reward == 0.0
        assert multiple_tool_result.error_type == "multiple_tool_calls"

        invalid_json_response = make_response(self._create_response_function_call(arguments='{"name":'))
        invalid_json_request = direct_request.model_copy(deep=True, update={"response": invalid_json_response})
        invalid_json_result = await resources_server.verify(invalid_json_request)
        assert invalid_json_result.reward == 0.0
        assert invalid_json_result.error_type == "tool_arguments_parse_error"

        missing_wrapper_response = make_response(
            self._create_response_function_call(arguments=json.dumps({"result": valid_payload}))
        )
        missing_wrapper_request = direct_request.model_copy(
            deep=True,
            update={
                "response": missing_wrapper_response,
                "tool_schema_mode": "extraction_wrapper",
                "tool_payload_key": "extraction",
            },
        )
        missing_wrapper_result = await resources_server.verify(missing_wrapper_request)
        assert missing_wrapper_result.reward == 0.0
        assert missing_wrapper_result.error_type == "missing_tool_payload_key"

        bad_payload_response = make_response(
            self._create_response_function_call(arguments=json.dumps({"name": "Alice", "age": "thirty"}))
        )
        bad_payload_request = direct_request.model_copy(deep=True, update={"response": bad_payload_response})
        bad_payload_result = await resources_server.verify(bad_payload_request)
        assert bad_payload_result.reward == 0.0
        assert bad_payload_result.error_type == "validation_error"

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

    async def test_verify_yaml(self, config: StructuredOutputsResourcesServerConfig) -> None:
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
        test_completion_obj = {
            "studentId": "STU12345",
            "examSubject": "Calculus II",
            "plannedStudyHours": 120,
            "isFullTimeStudent": True,
            "studyMaterials": {
                "textbooks": ["Calculus: Early Transcendentals", "Schaum\u2019s Outline of Calculus"],
                "onlineResources": ["Khan Academy", "Coursera Calculus Course"],
                "practiceExams": {"completedCount": 8, "averageScore": 87.5, "mostRecentDate": "2024-05-10"},
            },
            "studySchedule": {
                "weeklyHours": 15,
                "sessionsPerWeek": 5,
                "preferredTimeOfDay": "evening",
                "studyDays": ["Monday", "Wednesday", "Friday"],
                "breakSchedule": {"shortBreakMinutes": 10, "longBreakMinutes": 25, "breakFrequencyMinutes": 50},
            },
            "preparationStatus": "in_progress",
        }
        test_completion_yaml = yaml.dump(test_completion_obj, default_flow_style=False)

        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid YAML ---
        valid_output_item = self._create_response_output_message(test_completion_yaml)
        valid_response = NeMoGymResponse(
            id="valid_yaml_response_id",
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
            schema_type=SchemaType.YAML,
        )

        valid_verify_response = await resources_server.verify(valid_request)
        assert valid_verify_response.reward == 1.0
        assert valid_verify_response.response == valid_response

        # --- Test 2: Invalid YAML (Not parsable) ---
        invalid_yaml_completion = "key: value\n  bad_indent: oops\n notvalid"
        invalid_yaml_output_item = self._create_response_output_message(invalid_yaml_completion)
        invalid_yaml_response = valid_response.model_copy(
            deep=True, update={"id": "invalid_yaml_id", "output": [invalid_yaml_output_item]}
        )

        invalid_yaml_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=invalid_yaml_response,
            schema_str=schema_str,
            schema_type=SchemaType.YAML,
        )

        invalid_yaml_verify_response = await resources_server.verify(invalid_yaml_request)
        assert invalid_yaml_verify_response.reward == 0.0

        # --- Test 3: Schema Mismatch (Missing field) ---
        missing_field_obj = {k: v for k, v in test_completion_obj.items() if k != "studentId"}
        missing_field_completion = yaml.dump(missing_field_obj, default_flow_style=False)

        missing_field_output_item = self._create_response_output_message(missing_field_completion)
        missing_field_response = valid_response.model_copy(
            deep=True, update={"id": "missing_field_yaml_id", "output": [missing_field_output_item]}
        )

        missing_field_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_field_response,
            schema_str=schema_str,
            schema_type=SchemaType.YAML,
        )

        missing_field_verify_response = await resources_server.verify(missing_field_request)
        assert missing_field_verify_response.reward == 0.0

        # --- Test 4: Schema Mismatch (Extra field) ---
        extra_field_obj = {**test_completion_obj, "extraField": "some value"}
        extra_field_completion = yaml.dump(extra_field_obj, default_flow_style=False)

        extra_field_output_item = self._create_response_output_message(extra_field_completion)
        extra_field_response = valid_response.model_copy(
            deep=True, update={"id": "extra_field_yaml_id", "output": [extra_field_output_item]}
        )

        extra_field_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_field_response,
            schema_str=schema_str,
            schema_type=SchemaType.YAML,
        )

        extra_field_verify_response = await resources_server.verify(extra_field_request)
        assert extra_field_verify_response.reward == 0.0

        # --- Test 5: Schema Mismatch (Wrong type) ---
        wrong_type_obj = {**test_completion_obj, "plannedStudyHours": "one hundred"}
        wrong_type_completion = yaml.dump(wrong_type_obj, default_flow_style=False)

        wrong_type_output_item = self._create_response_output_message(wrong_type_completion)
        wrong_type_response = valid_response.model_copy(
            deep=True, update={"id": "wrong_type_yaml_id", "output": [wrong_type_output_item]}
        )

        wrong_type_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=wrong_type_response,
            schema_str=schema_str,
            schema_type=SchemaType.YAML,
        )

        wrong_type_verify_response = await resources_server.verify(wrong_type_request)
        assert wrong_type_verify_response.reward == 0.0

        # --- Test 6: Schema Mismatch (Nested extra field) ---
        nested_extra_obj = json.loads(json.dumps(test_completion_obj))
        nested_extra_obj["studyMaterials"]["practiceExams"]["extraNestedField"] = "bad value"
        nested_extra_field_completion = yaml.dump(nested_extra_obj, default_flow_style=False)

        nested_extra_field_output_item = self._create_response_output_message(nested_extra_field_completion)
        nested_extra_field_response = valid_response.model_copy(
            deep=True, update={"id": "nested_extra_yaml_id", "output": [nested_extra_field_output_item]}
        )

        nested_extra_field_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=nested_extra_field_response,
            schema_str=schema_str,
            schema_type=SchemaType.YAML,
        )

        nested_extra_field_verify_response = await resources_server.verify(nested_extra_field_request)
        assert nested_extra_field_verify_response.reward == 0.0

    async def test_verify_xml(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        test_schema = {
            "type": "object",
            "properties": {
                "root": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "score": {"type": "number"},
                        "active": {"type": "boolean"},
                        "tag": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }
        valid_obj = {"root": {"name": "Alice", "age": 25, "score": 95.5, "active": True, "tag": ["python", "ml"]}}
        valid_xml = xmltodict.unparse(valid_obj)

        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid XML (with coercion enabled by default) ---
        valid_output_item = self._create_response_output_message(valid_xml)
        valid_response = NeMoGymResponse(
            id="valid_xml_response_id",
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
            schema_type=SchemaType.XML,
        )

        valid_verify_response = await resources_server.verify(valid_request)
        assert valid_verify_response.reward == 1.0
        assert valid_verify_response.response == valid_response

        # --- Test 2: Malformed XML ---
        malformed_xml = "<root><name>Alice</name><age>25"
        malformed_output_item = self._create_response_output_message(malformed_xml)
        malformed_response = valid_response.model_copy(
            deep=True, update={"id": "malformed_xml_id", "output": [malformed_output_item]}
        )

        malformed_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=malformed_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )

        malformed_verify_response = await resources_server.verify(malformed_request)
        assert malformed_verify_response.reward == 0.0

        # --- Test 3: Schema Mismatch (Missing field) ---
        missing_obj = {"root": {"name": "Alice", "score": 95.5, "active": True, "tag": ["python", "ml"]}}
        missing_xml = xmltodict.unparse(missing_obj)

        missing_output_item = self._create_response_output_message(missing_xml)
        missing_response = valid_response.model_copy(
            deep=True, update={"id": "missing_field_xml_id", "output": [missing_output_item]}
        )

        missing_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )

        missing_verify_response = await resources_server.verify(missing_request)
        assert missing_verify_response.reward == 0.0

        # --- Test 4: Schema Mismatch (Extra field) ---
        extra_obj = {**valid_obj["root"], "extraField": "bad"}
        extra_xml = xmltodict.unparse({"root": extra_obj})

        extra_output_item = self._create_response_output_message(extra_xml)
        extra_response = valid_response.model_copy(
            deep=True, update={"id": "extra_field_xml_id", "output": [extra_output_item]}
        )

        extra_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )

        extra_verify_response = await resources_server.verify(extra_request)
        assert extra_verify_response.reward == 0.0

        # --- Test 5: Coercion disabled -- non-string types fail validation ---
        no_coerce_config = StructuredOutputsResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            xml_coerce_types=False,
        )
        no_coerce_server = StructuredOutputsResourcesServer(config=no_coerce_config, server_client=server_mock)

        no_coerce_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=valid_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )

        no_coerce_verify_response = await no_coerce_server.verify(no_coerce_request)
        assert no_coerce_verify_response.reward == 0.0

    async def test_verify_toml(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "buildNumber": {"type": "integer"},
                "isStable": {"type": "boolean"},
                "score": {"type": "number"},
                "settings": {
                    "type": "object",
                    "properties": {
                        "debug": {"type": "boolean"},
                        "timeout": {"type": "integer"},
                        "logLevel": {"type": "string"},
                    },
                    "required": ["debug", "timeout", "logLevel"],
                    "additionalProperties": False,
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        valid_toml = (
            'name = "my-project"\n'
            'version = "1.2.3"\n'
            "buildNumber = 42\n"
            "isStable = true\n"
            "score = 98.5\n"
            'tags = ["python", "ml"]\n'
            "\n"
            "[settings]\n"
            "debug = false\n"
            "timeout = 30\n"
            'logLevel = "info"\n'
        )

        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid TOML ---
        valid_output_item = self._create_response_output_message(valid_toml)
        valid_response = NeMoGymResponse(
            id="valid_toml_response_id",
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
            schema_type=SchemaType.TOML,
        )

        valid_verify_response = await resources_server.verify(valid_request)
        assert valid_verify_response.reward == 1.0
        assert valid_verify_response.response == valid_response

        # --- Test 2: Unparseable TOML ---
        broken_toml = 'name = "hello\n[invalid'
        broken_output_item = self._create_response_output_message(broken_toml)
        broken_response = valid_response.model_copy(
            deep=True, update={"id": "broken_toml_id", "output": [broken_output_item]}
        )

        broken_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=broken_response,
            schema_str=schema_str,
            schema_type=SchemaType.TOML,
        )

        broken_verify_response = await resources_server.verify(broken_request)
        assert broken_verify_response.reward == 0.0
        assert broken_verify_response.error_type == "parse_error"

        # --- Test 3: Missing field ---
        missing_field_toml = (
            'version = "1.2.3"\n'
            "buildNumber = 42\n"
            "isStable = true\n"
            "score = 98.5\n"
            'tags = ["python", "ml"]\n'
            "\n"
            "[settings]\n"
            "debug = false\n"
            "timeout = 30\n"
            'logLevel = "info"\n'
        )
        missing_output_item = self._create_response_output_message(missing_field_toml)
        missing_response = valid_response.model_copy(
            deep=True, update={"id": "missing_field_toml_id", "output": [missing_output_item]}
        )

        missing_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_response,
            schema_str=schema_str,
            schema_type=SchemaType.TOML,
        )

        missing_verify_response = await resources_server.verify(missing_request)
        assert missing_verify_response.reward == 0.0
        assert missing_verify_response.error_type == "validation_error"

        # --- Test 4: Extra field ---
        extra_field_toml = (
            'name = "my-project"\n'
            'version = "1.2.3"\n'
            "buildNumber = 42\n"
            "isStable = true\n"
            "score = 98.5\n"
            'tags = ["python", "ml"]\n'
            'extraField = "some value"\n'
            "\n"
            "[settings]\n"
            "debug = false\n"
            "timeout = 30\n"
            'logLevel = "info"\n'
        )
        extra_output_item = self._create_response_output_message(extra_field_toml)
        extra_response = valid_response.model_copy(
            deep=True, update={"id": "extra_field_toml_id", "output": [extra_output_item]}
        )

        extra_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_response,
            schema_str=schema_str,
            schema_type=SchemaType.TOML,
        )

        extra_verify_response = await resources_server.verify(extra_request)
        assert extra_verify_response.reward == 0.0
        assert extra_verify_response.error_type == "validation_error"

        # --- Test 5: Wrong type (buildNumber as string) ---
        wrong_type_toml = (
            'name = "my-project"\n'
            'version = "1.2.3"\n'
            'buildNumber = "forty-two"\n'
            "isStable = true\n"
            "score = 98.5\n"
            'tags = ["python", "ml"]\n'
            "\n"
            "[settings]\n"
            "debug = false\n"
            "timeout = 30\n"
            'logLevel = "info"\n'
        )
        wrong_type_output_item = self._create_response_output_message(wrong_type_toml)
        wrong_type_response = valid_response.model_copy(
            deep=True, update={"id": "wrong_type_toml_id", "output": [wrong_type_output_item]}
        )

        wrong_type_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=wrong_type_response,
            schema_str=schema_str,
            schema_type=SchemaType.TOML,
        )

        wrong_type_verify_response = await resources_server.verify(wrong_type_request)
        assert wrong_type_verify_response.reward == 0.0
        assert wrong_type_verify_response.error_type == "validation_error"

        # --- Test 6: Nested extra field ---
        nested_extra_toml = (
            'name = "my-project"\n'
            'version = "1.2.3"\n'
            "buildNumber = 42\n"
            "isStable = true\n"
            "score = 98.5\n"
            'tags = ["python", "ml"]\n'
            "\n"
            "[settings]\n"
            "debug = false\n"
            "timeout = 30\n"
            'logLevel = "info"\n'
            'extraNestedField = "bad value"\n'
        )
        nested_extra_output_item = self._create_response_output_message(nested_extra_toml)
        nested_extra_response = valid_response.model_copy(
            deep=True, update={"id": "nested_extra_toml_id", "output": [nested_extra_output_item]}
        )

        nested_extra_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=nested_extra_response,
            schema_str=schema_str,
            schema_type=SchemaType.TOML,
        )

        nested_extra_verify_response = await resources_server.verify(nested_extra_request)
        assert nested_extra_verify_response.reward == 0.0
        assert nested_extra_verify_response.error_type == "validation_error"

    async def test_verify_csv(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        test_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "score": {"type": "number"},
                    "active": {"type": "boolean"},
                },
            },
        }
        valid_csv = "name,age,score,active\nAlice,30,95.5,true\nBob,25,88.0,false\n"

        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid CSV (type coercion handles str -> int/float/bool) ---
        valid_output_item = self._create_response_output_message(valid_csv)
        valid_response = NeMoGymResponse(
            id="valid_csv_response_id",
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
            schema_type=SchemaType.CSV,
        )

        valid_verify_response = await resources_server.verify(valid_request)
        assert valid_verify_response.reward == 1.0
        assert valid_verify_response.response == valid_response

        # --- Test 2: Empty response ---
        empty_output_item = self._create_response_output_message("   ")
        empty_response = valid_response.model_copy(
            deep=True, update={"id": "empty_csv_id", "output": [empty_output_item]}
        )

        empty_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=empty_response,
            schema_str=schema_str,
            schema_type=SchemaType.CSV,
        )

        empty_verify_response = await resources_server.verify(empty_request)
        assert empty_verify_response.reward == 0.0
        assert empty_verify_response.error_type == "empty_response"

        # --- Test 3: Missing column ---
        missing_col_csv = "age,score,active\n30,95.5,true\n25,88.0,false\n"
        missing_col_output_item = self._create_response_output_message(missing_col_csv)
        missing_col_response = valid_response.model_copy(
            deep=True, update={"id": "missing_col_csv_id", "output": [missing_col_output_item]}
        )

        missing_col_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_col_response,
            schema_str=schema_str,
            schema_type=SchemaType.CSV,
        )

        missing_col_verify_response = await resources_server.verify(missing_col_request)
        assert missing_col_verify_response.reward == 0.0
        assert missing_col_verify_response.error_type == "validation_error"

        # --- Test 4: Extra column ---
        extra_col_csv = "name,age,score,active,bonus\nAlice,30,95.5,true,extra\nBob,25,88.0,false,extra\n"
        extra_col_output_item = self._create_response_output_message(extra_col_csv)
        extra_col_response = valid_response.model_copy(
            deep=True, update={"id": "extra_col_csv_id", "output": [extra_col_output_item]}
        )

        extra_col_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_col_response,
            schema_str=schema_str,
            schema_type=SchemaType.CSV,
        )

        extra_col_verify_response = await resources_server.verify(extra_col_request)
        assert extra_col_verify_response.reward == 0.0
        assert extra_col_verify_response.error_type == "validation_error"

        # --- Test 5: Wrong type (age is non-numeric, coercion fails) ---
        wrong_type_csv = "name,age,score,active\nAlice,not_a_number,95.5,true\n"
        wrong_type_output_item = self._create_response_output_message(wrong_type_csv)
        wrong_type_response = valid_response.model_copy(
            deep=True, update={"id": "wrong_type_csv_id", "output": [wrong_type_output_item]}
        )

        wrong_type_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=wrong_type_response,
            schema_str=schema_str,
            schema_type=SchemaType.CSV,
        )

        wrong_type_verify_response = await resources_server.verify(wrong_type_request)
        assert wrong_type_verify_response.reward == 0.0
        assert wrong_type_verify_response.error_type == "validation_error"

        # --- Test 6: Nullable type coercion (empty value -> None) ---
        nullable_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": ["integer", "null"]},
                },
            },
        }
        nullable_csv = "name,age\nAlice,\nBob,25\n"
        nullable_schema_str = json.dumps(nullable_schema)

        nullable_output_item = self._create_response_output_message(nullable_csv)
        nullable_response = valid_response.model_copy(
            deep=True, update={"id": "nullable_csv_id", "output": [nullable_output_item]}
        )

        nullable_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=nullable_response,
            schema_str=nullable_schema_str,
            schema_type=SchemaType.CSV,
        )

        nullable_verify_response = await resources_server.verify(nullable_request)
        assert nullable_verify_response.reward == 1.0

    async def test_verify_json_with_oneof_strictifies_branches(
        self, config: StructuredOutputsResourcesServerConfig
    ) -> None:
        """strictify_schema must recurse into oneOf branch lists so that extra
        or missing fields inside the selected branch are caught by validation.

        On ``main`` strictify does not descend into list values, so oneOf
        branches keep their loose contract and the ``extra_field`` /
        ``missing_required`` cases below FAIL on main (incorrectly accepted)
        and PASS on this PR. anyOf and allOf are covered by the dedicated
        tests below.
        """
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        # Tool-call-style schema: `arguments` is a oneOf over two object branches.
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["run_migration", "list_migrations"]},
                "arguments": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "migration_name": {"type": "string"},
                                "target_database": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer"},
                            },
                        },
                    ],
                },
            },
        }
        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid output matching the first oneOf branch exactly ---
        valid_completion = json.dumps(
            {
                "name": "run_migration",
                "arguments": {"migration_name": "add_users_table", "target_database": "staging"},
            }
        )
        valid_output_item = self._create_response_output_message(valid_completion)
        valid_response = NeMoGymResponse(
            id="valid_oneof_id",
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

        # --- Test 2: Extra field inside a oneOf branch must be rejected ---
        # Before the fix, strictify_schema did not recurse into the oneOf list,
        # so `additionalProperties: False` was never applied to the branch and
        # this output would have been incorrectly accepted.
        extra_in_branch_completion = json.dumps(
            {
                "name": "run_migration",
                "arguments": {
                    "migration_name": "add_users_table",
                    "target_database": "staging",
                    "unexpected_field": "should_be_rejected",
                },
            }
        )
        extra_output_item = self._create_response_output_message(extra_in_branch_completion)
        extra_response = valid_response.model_copy(
            deep=True, update={"id": "extra_in_branch_id", "output": [extra_output_item]}
        )
        extra_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        extra_verify_response = await resources_server.verify(extra_request)
        assert extra_verify_response.reward == 0.0

        # --- Test 3: Missing required field inside a oneOf branch must be rejected ---
        missing_in_branch_completion = json.dumps(
            {"name": "run_migration", "arguments": {"migration_name": "add_users_table"}}
        )
        missing_output_item = self._create_response_output_message(missing_in_branch_completion)
        missing_response = valid_response.model_copy(
            deep=True, update={"id": "missing_in_branch_id", "output": [missing_output_item]}
        )
        missing_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        missing_verify_response = await resources_server.verify(missing_request)
        assert missing_verify_response.reward == 0.0

        # --- Test 4: Output matching the SECOND oneOf branch also validates ---
        second_branch_completion = json.dumps({"name": "list_migrations", "arguments": {"limit": 10}})
        second_branch_output_item = self._create_response_output_message(second_branch_completion)
        second_branch_response = valid_response.model_copy(
            deep=True, update={"id": "second_branch_oneof_id", "output": [second_branch_output_item]}
        )
        second_branch_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=second_branch_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        second_branch_verify_response = await resources_server.verify(second_branch_request)
        assert second_branch_verify_response.reward == 1.0

    async def test_verify_xml_with_defs_refs(self, config: StructuredOutputsResourcesServerConfig) -> None:
        """coerce_xml_types must resolve $ref pointers against the top-level
        schema's $defs so that $ref-based fields with non-string types are
        coerced correctly.

        On ``main`` $ref is not followed during coercion, so the xmltodict
        leaf "42" stays a string and fails validation against
        ``{"type": "integer"}`` reached via $ref. The valid case below FAILS
        on main and PASSES on this PR; the invalid case is a regression
        guard.
        """
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        test_schema = {
            "type": "object",
            "properties": {
                "root": {
                    "type": "object",
                    "properties": {
                        "id": {"$ref": "#/$defs/identifier"},
                        "score": {"$ref": "#/$defs/score"},
                    },
                },
            },
            "$defs": {
                "identifier": {"type": "integer"},
                "score": {"type": "number"},
            },
        }
        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid XML where $ref-backed fields coerce correctly ---
        # Before the fix, coerce_xml_types did not resolve $ref, so "42" stayed
        # a string and validation failed against {"type": "integer"}.
        valid_xml = xmltodict.unparse({"root": {"id": 42, "score": 87.5}})
        valid_output_item = self._create_response_output_message(valid_xml)
        valid_response = NeMoGymResponse(
            id="valid_defs_xml_id",
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
            schema_type=SchemaType.XML,
        )
        valid_verify_response = await resources_server.verify(valid_request)
        assert valid_verify_response.reward == 1.0

        # --- Test 2: Non-numeric content under a $ref integer field fails validation ---
        invalid_xml = xmltodict.unparse({"root": {"id": "not-a-number", "score": 87.5}})
        invalid_output_item = self._create_response_output_message(invalid_xml)
        invalid_response = valid_response.model_copy(
            deep=True, update={"id": "invalid_defs_xml_id", "output": [invalid_output_item]}
        )
        invalid_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=invalid_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )
        invalid_verify_response = await resources_server.verify(invalid_request)
        assert invalid_verify_response.reward == 0.0

    async def test_verify_json_with_anyof_strictifies_branches(
        self, config: StructuredOutputsResourcesServerConfig
    ) -> None:
        """strictify_schema must recurse into anyOf branch lists.

        Without the recursion fix in this PR, anyOf object branches keep
        their loose default (no ``required``, no ``additionalProperties``):
        any object validates against an empty branch contract, so the
        ``missing_required`` and ``extra_field`` cases below would FAIL on
        ``main`` (incorrectly accepted) and PASS on this PR.
        """
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        test_schema = {
            "type": "object",
            "properties": {
                "payload": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "email": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "service_account": {"type": "string"},
                                "scope": {"type": "string"},
                            },
                        },
                    ]
                }
            },
        }
        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Valid output matches the first anyOf branch exactly ---
        valid_completion = json.dumps({"payload": {"user_id": "u_42", "email": "a@b.com"}})
        valid_output_item = self._create_response_output_message(valid_completion)
        valid_response = NeMoGymResponse(
            id="valid_anyof_id",
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

        # --- Test 2: Missing required field in branch is rejected ---
        missing_completion = json.dumps({"payload": {"user_id": "u_42"}})
        missing_output_item = self._create_response_output_message(missing_completion)
        missing_response = valid_response.model_copy(
            deep=True, update={"id": "missing_anyof_id", "output": [missing_output_item]}
        )
        missing_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        missing_verify_response = await resources_server.verify(missing_request)
        assert missing_verify_response.reward == 0.0

        # --- Test 3: Field not declared by any branch is rejected ---
        extra_completion = json.dumps({"payload": {"user_id": "u_42", "email": "a@b.com", "unexpected": "x"}})
        extra_output_item = self._create_response_output_message(extra_completion)
        extra_response = valid_response.model_copy(
            deep=True, update={"id": "extra_anyof_id", "output": [extra_output_item]}
        )
        extra_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=extra_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        extra_verify_response = await resources_server.verify(extra_request)
        assert extra_verify_response.reward == 0.0

        # --- Test 4: Output matching the SECOND branch also validates ---
        second_branch_completion = json.dumps({"payload": {"service_account": "ci-bot", "scope": "deploy"}})
        second_branch_output_item = self._create_response_output_message(second_branch_completion)
        second_branch_response = valid_response.model_copy(
            deep=True, update={"id": "second_branch_anyof_id", "output": [second_branch_output_item]}
        )
        second_branch_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=second_branch_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        second_branch_verify_response = await resources_server.verify(second_branch_request)
        assert second_branch_verify_response.reward == 1.0

        # --- Test 5: Output that doesn't match any branch is rejected ---
        no_match_completion = json.dumps({"payload": {"unrelated": "value"}})
        no_match_output_item = self._create_response_output_message(no_match_completion)
        no_match_response = valid_response.model_copy(
            deep=True, update={"id": "no_match_anyof_id", "output": [no_match_output_item]}
        )
        no_match_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=no_match_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        no_match_verify_response = await resources_server.verify(no_match_request)
        assert no_match_verify_response.reward == 0.0

    async def test_verify_json_with_allof_does_not_overstrictify_branches(
        self, config: StructuredOutputsResourcesServerConfig
    ) -> None:
        """Regression guard: strictify_schema must NOT descend into ``allOf``
        branches.

        ``allOf`` is a conjunction: if each branch were independently given
        ``additionalProperties: False``, a branch that declares only a subset
        of the merged object's properties would reject fields contributed by
        sibling branches. The valid composition below would PASS on ``main``
        (no list recursion) and PASS on this PR (allOf skipped). It would FAIL
        if recursion were applied to ``allOf`` branches without an exemption.
        """
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        # Merged shape: {id: str, name: str, version: str, owner: str}.
        # Branch A declares id+name; branch B declares version+owner. If we
        # blanket-strictify each branch, branch A would reject `version` /
        # `owner` as additional properties (and vice versa), so a valid output
        # carrying all four fields would score 0.0.
        test_schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["id", "name"],
                },
                {
                    "type": "object",
                    "properties": {"version": {"type": "string"}, "owner": {"type": "string"}},
                    "required": ["version", "owner"],
                },
            ]
        }
        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Output satisfies both branches ---
        valid_completion = json.dumps({"id": "svc-1", "name": "auth", "version": "2.3.1", "owner": "platform"})
        valid_output_item = self._create_response_output_message(valid_completion)
        valid_response = NeMoGymResponse(
            id="valid_allof_id",
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

        # --- Test 2: Missing a field required by one of the branches still fails ---
        missing_completion = json.dumps({"id": "svc-1", "name": "auth", "version": "2.3.1"})
        missing_output_item = self._create_response_output_message(missing_completion)
        missing_response = valid_response.model_copy(
            deep=True, update={"id": "missing_allof_id", "output": [missing_output_item]}
        )
        missing_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=missing_response,
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        missing_verify_response = await resources_server.verify(missing_request)
        assert missing_verify_response.reward == 0.0

    async def test_verify_xml_with_nullable_union_types(self, config: StructuredOutputsResourcesServerConfig) -> None:
        """coerce_xml_types must handle union type lists like ["integer", "null"]
        so nullable numeric/boolean XML fields don't false-negative.

        On ``main`` the function reads ``schema_type = schema["type"]`` as a
        single string, so for a list it skips all coercion branches: an
        xmltodict-parsed leaf "7" stays a string and fails validation against
        ``["integer", "null"]``. The ``populated_values`` case below FAILS on
        ``main`` and PASSES on this PR. The ``empty_tags`` and
        ``invalid_content`` cases are regression guards (PASS / FAIL on both
        sides). Mirrors the CSV path's _coerce_csv_scalar.
        """
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        test_schema = {
            "type": "object",
            "properties": {
                "root": {
                    "type": "object",
                    "properties": {
                        "count": {"type": ["integer", "null"]},
                        "ratio": {"type": ["number", "null"]},
                        "active": {"type": ["boolean", "null"]},
                    },
                },
            },
        }
        schema_str = json.dumps(test_schema)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        # --- Test 1: Populated values coerce to their non-null member ---
        populated_xml = xmltodict.unparse({"root": {"count": 7, "ratio": 0.42, "active": "true"}})
        populated_output_item = self._create_response_output_message(populated_xml)
        populated_response = NeMoGymResponse(
            id="populated_nullable_id",
            created_at=1234.5,
            model="test_model",
            object="response",
            output=[populated_output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        populated_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=populated_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )
        populated_verify_response = await resources_server.verify(populated_request)
        assert populated_verify_response.reward == 1.0

        # --- Test 2: Empty tags map to null and validate against the union ---
        # xmltodict parses <count/> as None, which previously stayed None and
        # failed validation against {"type": "integer"}. With union support it
        # validates against the "null" member.
        nullable_xml = "<root><count/><ratio/><active/></root>"
        nullable_output_item = self._create_response_output_message(nullable_xml)
        nullable_response = populated_response.model_copy(
            deep=True, update={"id": "empty_nullable_id", "output": [nullable_output_item]}
        )
        nullable_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=nullable_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )
        nullable_verify_response = await resources_server.verify(nullable_request)
        assert nullable_verify_response.reward == 1.0

        # --- Test 3: Non-numeric content under a nullable integer still fails ---
        invalid_xml = xmltodict.unparse({"root": {"count": "abc", "ratio": 0.42, "active": "true"}})
        invalid_output_item = self._create_response_output_message(invalid_xml)
        invalid_response = populated_response.model_copy(
            deep=True, update={"id": "invalid_nullable_id", "output": [invalid_output_item]}
        )
        invalid_request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=invalid_response,
            schema_str=schema_str,
            schema_type=SchemaType.XML,
        )
        invalid_verify_response = await resources_server.verify(invalid_request)
        assert invalid_verify_response.reward == 0.0

    async def test_example_complex_schema_jsonl_round_trips(
        self, config: StructuredOutputsResourcesServerConfig
    ) -> None:
        """End-to-end smoke test for ``data/example_complex_schema.jsonl``.

        Loads each toy row in the bundled example file, pairs it with a
        hand-crafted gold output, and runs the full verify path. Each row
        must score reward=1.0 with the gold output. This is the test
        anchor JK asked for ("test against a small example_complex_schema.jsonl").
        """
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        data_path = Path(__file__).resolve().parent.parent / "data" / "example_complex_schema.jsonl"
        rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        # One gold output per row, in the same order: oneOf, anyOf, allOf,
        # $defs/XML, nullable XML.
        gold_outputs = [
            json.dumps(
                {
                    "name": "run_migration",
                    "arguments": {"migration_name": "add_users_table", "target_database": "staging"},
                }
            ),
            json.dumps({"payload": {"user_id": "u_42", "email": "a@b.com"}}),
            json.dumps({"id": "svc-1", "name": "auth", "version": "2.3.1", "owner": "platform"}),
            xmltodict.unparse({"root": {"id": 42, "score": 87.5}}),
            xmltodict.unparse({"root": {"count": 7, "ratio": 0.42, "active": "true"}}),
        ]
        assert len(rows) == len(gold_outputs), f"row/gold mismatch: {len(rows)} rows, {len(gold_outputs)} golds"

        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        for idx, (row, gold) in enumerate(zip(rows, gold_outputs)):
            output_item = self._create_response_output_message(gold)
            response = NeMoGymResponse(
                id=f"complex_schema_row_{idx}",
                created_at=1234.5,
                model="test_model",
                object="response",
                output=[output_item],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )
            request = StructuredOutputsVerifyRequest(
                responses_create_params=dummy_create_params,
                response=response,
                schema_str=row["schema_str"],
                schema_type=SchemaType(row["schema_type"]),
            )
            verify_response = await resources_server.verify(request)
            assert verify_response.reward == 1.0, (
                f"row {idx} ({row['schema_type']}) failed: "
                f"error_type={verify_response.error_type}, "
                f"error_message={verify_response.error_message}"
            )
