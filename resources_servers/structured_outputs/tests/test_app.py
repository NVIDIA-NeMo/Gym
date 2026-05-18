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

import orjson
import xmltodict
import yaml
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
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

    async def test_verify_xml_composition(self, config: StructuredOutputsResourcesServerConfig) -> None:
        """Test XML coercion handles oneOf, anyOf, allOf, and $ref composition."""
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        def make_response(xml_text: str, resp_id: str = "xml_comp") -> NeMoGymResponse:
            return NeMoGymResponse(
                id=resp_id,
                created_at=1234.5,
                model="test_model",
                object="response",
                output=[self._create_response_output_message(xml_text)],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )

        # --- Test 1: oneOf with type coercion ---
        oneof_schema = {
            "type": "object",
            "properties": {
                "record": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "oneOf": [
                                {"type": "integer"},
                                {"type": "string"},
                            ]
                        }
                    },
                }
            },
        }
        xml_int = "<record><value>42</value></record>"
        r1 = await resources_server.verify(StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=make_response(xml_int, "oneof_int"),
            schema_str=json.dumps(oneof_schema),
            schema_type=SchemaType.XML,
        ))
        assert r1.reward == 1.0

        # --- Test 2: anyOf with boolean coercion ---
        anyof_schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "flag": {
                            "anyOf": [
                                {"type": "boolean"},
                                {"type": "string"},
                            ]
                        }
                    },
                }
            },
        }
        xml_bool = "<data><flag>true</flag></data>"
        r3 = await resources_server.verify(StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=make_response(xml_bool, "anyof_bool"),
            schema_str=json.dumps(anyof_schema),
            schema_type=SchemaType.XML,
        ))
        assert r3.reward == 1.0

    def test_compute_metrics_semantic_reward(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        tasks = [
            [
                {"reward": 1.0, "schema_type": "json", "semantic_reward": 0.9, "problem_type": "generic_extraction"},
                {"reward": 1.0, "schema_type": "json", "semantic_reward": 1.0, "problem_type": "generic_extraction"},
                {"reward": 0.0, "schema_type": "xml", "semantic_reward": 0.8},
            ],
            [
                {"reward": 1.0, "schema_type": "json"},
                {"reward": 1.0, "schema_type": "yaml", "semantic_reward": 0.6},
            ],
        ]

        metrics = resources_server.compute_metrics(tasks)

        assert metrics["mean/semantic_reward"] == approx(0.825)
        assert metrics["mean/semantic_reward_json"] == approx(0.95)
        assert metrics["mean/semantic_reward_xml"] == approx(0.8)
        assert metrics["mean/semantic_reward_yaml"] == approx(0.6)
        assert metrics["mean/semantic_reward_generic_extraction"] == approx(0.95)
        assert "mean/reward_json" in metrics
        assert "mean/reward_xml" in metrics
        assert "mean/reward_yaml" in metrics

    def test_compute_metrics_no_semantic(self, config: StructuredOutputsResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        tasks = [[{"reward": 1.0, "schema_type": "json"}, {"reward": 0.0, "schema_type": "xml"}]]
        metrics = resources_server.compute_metrics(tasks)

        assert "mean/semantic_reward" not in metrics
        assert "mean/reward_json" in metrics

    async def test_verify_combined_reward_no_semantic(self, config: StructuredOutputsResourcesServerConfig) -> None:
        """When no semantic config, semantic_reward defaults to 1.0 and combined reward = syntax * 1.0."""
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)
        assert config.reward_mode == "combined"

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        valid_json = '{"name": "Alice"}'
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        output_item = self._create_response_output_message(valid_json)
        response = NeMoGymResponse(
            id="combined_test", created_at=1234.5, model="test",
            object="response", output=[output_item],
            parallel_tool_calls=False, tool_choice="none", tools=[],
        )
        request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=response,
            schema_str=json.dumps(schema),
            schema_type=SchemaType.JSON,
        )
        result = await resources_server.verify(request)
        assert result.reward == 1.0
        assert result.semantic_reward == 1.0
        assert result.semantic_results is None

    async def test_verify_combined_reward_syntax_fail(self, config: StructuredOutputsResourcesServerConfig) -> None:
        """Syntax fail (0) * semantic (1.0) = 0 in combined mode."""
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=config, server_client=server_mock)

        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        invalid_json = '{"name": "Alice", "extra": true}'
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        output_item = self._create_response_output_message(invalid_json)
        response = NeMoGymResponse(
            id="combined_fail_test", created_at=1234.5, model="test",
            object="response", output=[output_item],
            parallel_tool_calls=False, tool_choice="none", tools=[],
        )
        request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=response,
            schema_str=json.dumps(schema),
            schema_type=SchemaType.JSON,
        )
        result = await resources_server.verify(request)
        assert result.reward == 0.0
        assert result.semantic_reward == 1.0

    async def test_verify_independent_reward_mode(self, config: StructuredOutputsResourcesServerConfig) -> None:
        """In independent mode, reward = syntax only, semantic is separate."""
        independent_config = config.model_copy(update={"reward_mode": "independent"})
        server_mock = MagicMock(spec=ServerClient)
        resources_server = StructuredOutputsResourcesServer(config=independent_config, server_client=server_mock)

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        valid_json = '{"name": "Alice"}'
        dummy_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])

        output_item = self._create_response_output_message(valid_json)
        response = NeMoGymResponse(
            id="independent_test", created_at=1234.5, model="test",
            object="response", output=[output_item],
            parallel_tool_calls=False, tool_choice="none", tools=[],
        )
        request = StructuredOutputsVerifyRequest(
            responses_create_params=dummy_create_params,
            response=response,
            schema_str=json.dumps(schema),
            schema_type=SchemaType.JSON,
        )
        result = await resources_server.verify(request)
        assert result.reward == 1.0
        assert result.semantic_reward == 1.0

    # ---- Thorough reward mode tests with mocked judge ---- #

    def _make_judge_config(self, reward_mode: str = "combined") -> StructuredOutputsResourcesServerConfig:
        return StructuredOutputsResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="",
            reward_mode=reward_mode,
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def _mock_judge_response(self, verdict_text: str) -> MagicMock:
        """Build a mock for server_client.post that returns a NeMoGymResponse with the given judge text."""
        gym_resp = NeMoGymResponse(
            id="judge-resp",
            created_at=0.0,
            model="mock-judge",
            object="response",
            output=[NeMoGymResponseOutputMessage(
                id="msg-judge",
                content=[NeMoGymResponseOutputText(
                    text=verdict_text, type="output_text", annotations=[],
                )],
                role="assistant",
                status="completed",
                type="message",
            )],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(gym_resp.model_dump()))
        return mock_response

    def _make_request(self, json_text: str, semantic_config: dict | None = None) -> StructuredOutputsVerifyRequest:
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "integer"}}}
        response = NeMoGymResponse(
            id="test", created_at=0.0, model="test", object="response",
            output=[self._create_response_output_message(json_text)],
            parallel_tool_calls=False, tool_choice="none", tools=[],
        )
        return StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=response,
            schema_str=json.dumps(schema),
            schema_type=SchemaType.JSON,
            semantic_verifier_config=semantic_config,
        )

    async def test_combined_syntax_pass_semantic_all_pass(self) -> None:
        """Combined: syntax=1.0 * semantic=1.0 -> reward=1.0"""
        cfg = self._make_judge_config("combined")
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(return_value=self._mock_judge_response("Good output. [[PASS]]"))
        server = StructuredOutputsResourcesServer(config=cfg, server_client=server_mock)

        svc = {"criteria": [
            {"name": "c1", "type": "llmaaj", "weight": "major", "rubric": "check something"},
            {"name": "c2", "type": "llmaaj", "weight": "minor", "rubric": "check something else"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.reward == 1.0
        assert result.semantic_reward == 1.0
        assert len(result.semantic_results) == 2
        assert all(r.passed for r in result.semantic_results)

    async def test_combined_syntax_pass_semantic_partial(self) -> None:
        """Combined: syntax=1.0 * semantic=0.333 -> reward=0.333 (1 major fail, 1 minor pass)"""
        cfg = self._make_judge_config("combined")
        call_count = 0

        async def alternating_judge(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._mock_judge_response("Bad output. [[FAIL]]")
            return self._mock_judge_response("Good output. [[PASS]]")

        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = alternating_judge
        cfg_serial = cfg.model_copy(update={"parallel_evaluation": False})
        server = StructuredOutputsResourcesServer(config=cfg_serial, server_client=server_mock)

        svc = {"criteria": [
            {"name": "major_criterion", "type": "llmaaj", "weight": "major", "rubric": "important check"},
            {"name": "minor_criterion", "type": "llmaaj", "weight": "minor", "rubric": "minor check"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.semantic_results[0].name == "major_criterion"
        assert result.semantic_results[0].passed is False
        assert result.semantic_results[1].name == "minor_criterion"
        assert result.semantic_results[1].passed is True
        assert result.semantic_reward == approx(1.0 / 3.0)
        assert result.reward == approx(1.0 * (1.0 / 3.0))

    async def test_combined_syntax_pass_semantic_all_fail(self) -> None:
        """Combined: syntax=1.0 * semantic=0.0 -> reward=0.0"""
        cfg = self._make_judge_config("combined")
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(return_value=self._mock_judge_response("Terrible. [[FAIL]]"))
        server = StructuredOutputsResourcesServer(config=cfg, server_client=server_mock)

        svc = {"criteria": [
            {"name": "c1", "type": "llmaaj", "weight": "major", "rubric": "check1"},
            {"name": "c2", "type": "llmaaj", "weight": "major", "rubric": "check2"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.reward == 0.0
        assert result.semantic_reward == 0.0
        assert all(not r.passed for r in result.semantic_results)

    async def test_combined_syntax_fail_semantic_all_pass(self) -> None:
        """Combined: syntax=0.0 * semantic=1.0 -> reward=0.0"""
        cfg = self._make_judge_config("combined")
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(return_value=self._mock_judge_response("Looks good. [[PASS]]"))
        server = StructuredOutputsResourcesServer(config=cfg, server_client=server_mock)

        svc = {"criteria": [
            {"name": "c1", "type": "llmaaj", "weight": "major", "rubric": "check1"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice"}', svc))

        assert result.reward == 0.0
        assert result.semantic_reward == 1.0

    async def test_independent_syntax_pass_semantic_partial(self) -> None:
        """Independent: reward = syntax only (1.0), semantic tracked separately (0.333)"""
        cfg = self._make_judge_config("independent")
        call_count = 0

        async def alternating_judge(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._mock_judge_response("Bad. [[FAIL]]")
            return self._mock_judge_response("Good. [[PASS]]")

        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = alternating_judge
        cfg_serial = cfg.model_copy(update={"parallel_evaluation": False})
        server = StructuredOutputsResourcesServer(config=cfg_serial, server_client=server_mock)

        svc = {"criteria": [
            {"name": "major_c", "type": "llmaaj", "weight": "major", "rubric": "check"},
            {"name": "minor_c", "type": "llmaaj", "weight": "minor", "rubric": "check"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.reward == 1.0
        assert result.semantic_reward == approx(1.0 / 3.0)

    async def test_combined_weight_calculation(self) -> None:
        """Verify weight math: 2 major PASS + 1 minor FAIL = (2*1 + 2*1 + 1*0) / (2+2+1) = 0.8"""
        cfg = self._make_judge_config("combined")
        call_count = 0

        async def judge_by_order(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return self._mock_judge_response("[[PASS]]")
            return self._mock_judge_response("[[FAIL]]")

        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = judge_by_order
        cfg_serial = cfg.model_copy(update={"parallel_evaluation": False})
        server = StructuredOutputsResourcesServer(config=cfg_serial, server_client=server_mock)

        svc = {"criteria": [
            {"name": "major1", "type": "llmaaj", "weight": "major", "rubric": "r1"},
            {"name": "major2", "type": "llmaaj", "weight": "major", "rubric": "r2"},
            {"name": "minor1", "type": "llmaaj", "weight": "minor", "rubric": "r3"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.semantic_reward == approx(4.0 / 5.0)
        assert result.reward == approx(4.0 / 5.0)
        assert result.semantic_results[0].passed is True
        assert result.semantic_results[1].passed is True
        assert result.semantic_results[2].passed is False

    async def test_semantic_skipped_without_judge_config(self) -> None:
        """No judge_model_server configured -> semantic is None -> defaults to 1.0."""
        cfg = StructuredOutputsResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="",
        )
        assert cfg.judge_model_server is None
        server_mock = MagicMock(spec=ServerClient)
        server = StructuredOutputsResourcesServer(config=cfg, server_client=server_mock)

        svc = {"criteria": [
            {"name": "c1", "type": "llmaaj", "weight": "major", "rubric": "r1"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.reward == 1.0
        assert result.semantic_reward == 1.0
        assert result.semantic_results is None

    async def test_semantic_skipped_with_only_deterministic_criteria(self) -> None:
        """Only deterministic criteria (no llmaaj) -> semantic is None -> defaults to 1.0."""
        cfg = self._make_judge_config("combined")
        server_mock = MagicMock(spec=ServerClient)
        server = StructuredOutputsResourcesServer(config=cfg, server_client=server_mock)

        svc = {"criteria": [
            {"name": "exact_match", "type": "deterministic", "weight": "major"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.reward == 1.0
        assert result.semantic_reward == 1.0
        assert result.semantic_results is None

    async def test_judge_error_counts_as_fail(self) -> None:
        """If the judge call raises an exception, the criterion counts as FAIL."""
        cfg = self._make_judge_config("combined")
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock(side_effect=Exception("connection refused"))
        server = StructuredOutputsResourcesServer(config=cfg, server_client=server_mock)

        svc = {"criteria": [
            {"name": "c1", "type": "llmaaj", "weight": "major", "rubric": "r1"},
        ]}
        result = await server.verify(self._make_request('{"name": "Alice", "value": 42}', svc))

        assert result.semantic_reward == 0.0
        assert result.reward == 0.0
        assert result.semantic_results[0].passed is False
