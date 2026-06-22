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
import csv
import io
import json
import tomllib
from collections import defaultdict
from enum import StrEnum
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import xmltodict
import yaml
from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class StructuredOutputsResourcesServerConfig(BaseResourcesServerConfig):
    xml_coerce_types: bool = True


class SchemaType(StrEnum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    CSV = "csv"


class StructuredOutputsVerifyRequest(BaseVerifyRequest):
    schema_str: str
    schema_type: SchemaType = SchemaType.JSON
    problem_type: Optional[str] = None
    schema_repr: Optional[str] = None
    source_format: Optional[str] = None
    num_turns: Optional[int] = None
    source_record_id: Optional[str] = None
    response_mode: Optional[str] = "text"
    tool_name: Optional[str] = None
    tool_schema_mode: Optional[str] = None
    tool_payload_key: Optional[str] = None
    tool_choice: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    source_schema_type: Optional[str] = None
    num_tools: Optional[int] = None
    num_distractors: Optional[int] = None
    has_distractors: Optional[bool] = None
    instruction_layout: Optional[str] = None
    instruction_detail_level: Optional[str] = None
    system_instruction_style: Optional[str] = None
    tool_name_style: Optional[str] = None
    distractor_style: Optional[str] = None
    tool_union_mode: Optional[str] = None


class StructuredOutputsVerifyResponse(BaseVerifyResponse):
    schema_str: str
    schema_type: SchemaType
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    problem_type: Optional[str] = None
    schema_repr: Optional[str] = None
    source_format: Optional[str] = None
    num_turns: Optional[int] = None
    source_record_id: Optional[str] = None
    response_mode: Optional[str] = "text"
    tool_name: Optional[str] = None
    tool_schema_mode: Optional[str] = None
    tool_payload_key: Optional[str] = None
    tool_choice: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    source_schema_type: Optional[str] = None
    num_tools: Optional[int] = None
    num_distractors: Optional[int] = None
    has_distractors: Optional[bool] = None
    instruction_layout: Optional[str] = None
    instruction_detail_level: Optional[str] = None
    system_instruction_style: Optional[str] = None
    tool_name_style: Optional[str] = None
    distractor_style: Optional[str] = None
    tool_union_mode: Optional[str] = None


class StructuredOutputsResourcesServer(SimpleResourcesServer):
    config: StructuredOutputsResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        by_fmt: Dict[str, List[float]] = defaultdict(list)
        by_problem: Dict[str, List[float]] = defaultdict(list)
        by_repr: Dict[str, List[float]] = defaultdict(list)
        by_response_mode: Dict[str, List[float]] = defaultdict(list)
        by_tool_schema_mode: Dict[str, List[float]] = defaultdict(list)
        by_tool_choice: Dict[str, List[float]] = defaultdict(list)
        by_parallel_tool_calls: Dict[str, List[float]] = defaultdict(list)
        by_num_tools: Dict[str, List[float]] = defaultdict(list)
        by_has_distractors: Dict[str, List[float]] = defaultdict(list)
        by_instruction_layout: Dict[str, List[float]] = defaultdict(list)
        by_instruction_detail_level: Dict[str, List[float]] = defaultdict(list)
        by_system_instruction_style: Dict[str, List[float]] = defaultdict(list)
        by_tool_name_style: Dict[str, List[float]] = defaultdict(list)
        by_distractor_style: Dict[str, List[float]] = defaultdict(list)
        by_tool_union_mode: Dict[str, List[float]] = defaultdict(list)

        for rollouts in tasks:
            for r in rollouts:
                reward = r.get("reward", 0.0)
                by_fmt[r.get("schema_type", "unknown")].append(reward)
                pt = r.get("problem_type")
                if pt:
                    by_problem[pt].append(reward)
                sr = r.get("schema_repr")
                if sr:
                    by_repr[sr].append(reward)
                response_mode = r.get("response_mode")
                if response_mode:
                    by_response_mode[response_mode].append(reward)
                tool_schema_mode = r.get("tool_schema_mode")
                if tool_schema_mode:
                    by_tool_schema_mode[tool_schema_mode].append(reward)
                tool_choice = r.get("tool_choice")
                if tool_choice:
                    by_tool_choice[tool_choice].append(reward)
                parallel_tool_calls = r.get("parallel_tool_calls")
                if parallel_tool_calls is not None:
                    by_parallel_tool_calls[str(bool(parallel_tool_calls)).lower()].append(reward)
                num_tools = r.get("num_tools")
                if num_tools is not None:
                    by_num_tools[str(num_tools)].append(reward)
                has_distractors = r.get("has_distractors")
                if has_distractors is not None:
                    by_has_distractors[str(bool(has_distractors)).lower()].append(reward)
                instruction_layout = r.get("instruction_layout")
                if instruction_layout:
                    by_instruction_layout[instruction_layout].append(reward)
                instruction_detail_level = r.get("instruction_detail_level")
                if instruction_detail_level:
                    by_instruction_detail_level[instruction_detail_level].append(reward)
                system_instruction_style = r.get("system_instruction_style")
                if system_instruction_style:
                    by_system_instruction_style[system_instruction_style].append(reward)
                tool_name_style = r.get("tool_name_style")
                if tool_name_style:
                    by_tool_name_style[tool_name_style].append(reward)
                distractor_style = r.get("distractor_style")
                if distractor_style:
                    by_distractor_style[distractor_style].append(reward)
                tool_union_mode = r.get("tool_union_mode")
                if tool_union_mode:
                    by_tool_union_mode[tool_union_mode].append(reward)

        metrics = {f"mean/reward_{k}": mean(v) for k, v in by_fmt.items() if v}
        metrics.update({f"mean/reward_{k}": mean(v) for k, v in by_problem.items() if v})
        metrics.update({f"mean/reward_repr_{k}": mean(v) for k, v in by_repr.items() if v})
        metrics.update({f"mean/reward_response_mode_{k}": mean(v) for k, v in by_response_mode.items() if v})
        metrics.update({f"mean/reward_tool_schema_mode_{k}": mean(v) for k, v in by_tool_schema_mode.items() if v})
        metrics.update({f"mean/reward_tool_choice_{k}": mean(v) for k, v in by_tool_choice.items() if v})
        metrics.update(
            {f"mean/reward_parallel_tool_calls_{k}": mean(v) for k, v in by_parallel_tool_calls.items() if v}
        )
        metrics.update({f"mean/reward_num_tools_{k}": mean(v) for k, v in by_num_tools.items() if v})
        metrics.update({f"mean/reward_has_distractors_{k}": mean(v) for k, v in by_has_distractors.items() if v})
        metrics.update({f"mean/reward_instruction_layout_{k}": mean(v) for k, v in by_instruction_layout.items() if v})
        metrics.update(
            {f"mean/reward_instruction_detail_level_{k}": mean(v) for k, v in by_instruction_detail_level.items() if v}
        )
        metrics.update(
            {f"mean/reward_system_instruction_style_{k}": mean(v) for k, v in by_system_instruction_style.items() if v}
        )
        metrics.update({f"mean/reward_tool_name_style_{k}": mean(v) for k, v in by_tool_name_style.items() if v})
        metrics.update({f"mean/reward_distractor_style_{k}": mean(v) for k, v in by_distractor_style.items() if v})
        metrics.update({f"mean/reward_tool_union_mode_{k}": mean(v) for k, v in by_tool_union_mode.items() if v})
        return metrics

    async def verify(self, body: StructuredOutputsVerifyRequest) -> StructuredOutputsVerifyResponse:
        schema_type = body.schema_type
        schema_str = body.schema_str

        if schema_type not in list(SchemaType):
            raise NotImplementedError(f"SchemaType must be one of {list(SchemaType)}, got {schema_type} !")

        if body.response_mode == "tool_call":
            response_obj, error_type, error_message = self.extract_tool_call_payload(body)
            if error_type is not None:
                return StructuredOutputsVerifyResponse(
                    **body.model_dump(), reward=0.0, error_type=error_type, error_message=error_message
                )

            reward, error_type, error_message = self.evaluate_structured_object_response(schema_str, response_obj)
            return StructuredOutputsVerifyResponse(
                **body.model_dump(), reward=reward, error_type=error_type, error_message=error_message
            )

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

        reward, error_type, error_message = self.evaluate_structured_output_response(
            schema_type, schema_str, response_text
        )
        return StructuredOutputsVerifyResponse(
            **body.model_dump(), reward=reward, error_type=error_type, error_message=error_message
        )

    # ----- Helpers ----- #
    def extract_tool_call_payload(
        self, body: StructuredOutputsVerifyRequest
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        if body.response.error is not None:
            return None, "api_error", str(body.response.error)[:200]

        function_calls = [output_item for output_item in body.response.output if output_item.type == "function_call"]
        if not function_calls:
            return None, "missing_tool_call", "No function_call item found in assistant response"
        if len(function_calls) > 1:
            called_names = ", ".join(function_call.name for function_call in function_calls)
            return None, "multiple_tool_calls", f"Expected exactly one function_call, got: {called_names}"

        selected_call = None
        if body.tool_name:
            for function_call in function_calls:
                if function_call.name == body.tool_name:
                    selected_call = function_call
                    break
            if selected_call is None:
                called_names = ", ".join(function_call.name for function_call in function_calls)
                return None, "wrong_tool_name", f"Expected tool '{body.tool_name}', got: {called_names}"
        else:
            selected_call = function_calls[0]

        arguments = selected_call.arguments
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return None, "tool_arguments_parse_error", f"{type(e).__name__}: {str(e)[:200]}"
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            return None, "tool_arguments_parse_error", f"Unsupported arguments type: {type(arguments).__name__}"

        if body.tool_payload_key:
            if not isinstance(parsed_arguments, dict) or body.tool_payload_key not in parsed_arguments:
                return None, "missing_tool_payload_key", f"Missing tool payload key: {body.tool_payload_key}"
            return parsed_arguments[body.tool_payload_key], None, None

        return parsed_arguments, None, None

    def parse_content(self, schema_type: SchemaType, content: str):
        match schema_type.lower():
            case SchemaType.JSON:
                parsed = json.loads(content)
            case SchemaType.YAML:
                parsed = yaml.safe_load(content)
            case SchemaType.XML:
                parsed = xmltodict.parse(content)
            case SchemaType.TOML:
                parsed = tomllib.loads(content)
            case SchemaType.CSV:
                parsed = list(csv.DictReader(io.StringIO(content)))
            case _:
                parsed = None
        return parsed

    def strictify_schema(self, schema: Dict[str, Any]):
        """Make a schema strict as per OpenAPI guidelines"""
        if isinstance(schema, Dict):
            if "properties" in schema:
                schema["required"] = list(schema["properties"])
                schema["additionalProperties"] = False
            for k, v in schema.items():
                self.strictify_schema(v)

    def coerce_xml_types(self, data: Any, schema: Dict[str, Any]) -> Any:
        """Recursively coerce xmltodict string values to match the JSON schema types.

        xmltodict.parse() returns all leaf values as strings. This method walks the
        parsed data alongside the schema and converts values where possible.
        On conversion failure the original value is returned so that schema
        validation can report the error.
        """
        if not isinstance(schema, dict) or "type" not in schema:
            return data

        schema_type = schema["type"]

        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            coerced = {}
            for key, value in data.items():
                if key in properties:
                    coerced[key] = self.coerce_xml_types(value, properties[key])
                else:
                    coerced[key] = value
            return coerced

        if schema_type == "array":
            items_schema = schema.get("items", {})
            # xmltodict represents repeated child elements as {"tagName": [values]},
            # e.g. <skills><string>a</string><string>b</string></skills> becomes
            # {"string": ["a", "b"]}. For single elements, xmltodict gives
            # {"string": "python"} instead of a list. In both cases, unwrap the
            # single-key dict since we're at an array schema position -- a dict here
            # is always the xmltodict wrapping artifact, not a meaningful structure.
            if isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            if not isinstance(data, list):
                data = [data] if data is not None else []
            return [self.coerce_xml_types(item, items_schema) for item in data]

        # xmltodict returns None for empty tags like <field/> or <field></field>.
        # Coerce to "" only for string types (parity with JSON/YAML where "" is valid).
        # Non-string types (integer, boolean, etc.) intentionally left as None so
        # they fail validation -- 0 and False are meaningful values, not "empty".
        if data is None and schema_type == "string":
            return ""

        if isinstance(data, str):
            try:
                if schema_type == "integer":
                    return int(data)
                if schema_type == "number":
                    return float(data)
                if schema_type == "boolean":
                    lower = data.lower()
                    if lower in ("true", "1"):
                        return True
                    if lower in ("false", "0"):
                        return False
            except (ValueError, AttributeError):
                pass

        return data

    def coerce_csv_types(self, rows: List[Dict[str, str]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coerce CSV string values to match JSON Schema types.

        csv.DictReader returns all values as strings.  Walk rows alongside the
        items schema and convert values where possible.  On conversion failure
        the original string is kept so that schema validation reports the error.
        """
        items_schema = schema.get("items", schema)
        properties = items_schema.get("properties", {})

        coerced_rows = []
        for row in rows:
            coerced: Dict[str, Any] = {}
            for key, value in row.items():
                prop_schema = properties.get(key, {})
                prop_type = prop_schema.get("type", "string")
                coerced[key] = self._coerce_csv_scalar(value, prop_type)
            coerced_rows.append(coerced)
        return coerced_rows

    def _coerce_csv_scalar(self, value: str, target_type) -> Any:
        """Coerce a single CSV string to the target JSON Schema type."""
        if isinstance(target_type, list):
            if (value is None or value == "") and "null" in target_type:
                return None
            for t in target_type:
                if t == "null":
                    continue
                result = self._coerce_csv_scalar(value, t)
                if not isinstance(result, str) or t == "string":
                    return result
            return value

        if value is None or value == "":
            return value

        try:
            if target_type == "integer":
                return int(value)
            if target_type == "number":
                return float(value)
            if target_type == "boolean":
                lower = value.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
        except (ValueError, AttributeError):
            pass
        return value

    def evaluate_structured_object_response(
        self, schema_str: str, response_obj: Any
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, error_type, error_message) for already-parsed tool-call arguments."""
        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0, "schema_error", str(e)[:200]

        self.strictify_schema(schema)

        try:
            validate_against_schema_openapi(response_obj, schema)
            return 1.0, None, None
        except Exception as e:
            return 0.0, "validation_error", f"{type(e).__name__}: {str(e)[:200]}"

    def evaluate_structured_output_response(
        self, schema_type: SchemaType, schema_str: str, response_text: str
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, error_type, error_message)."""
        if not response_text or not response_text.strip():
            return 0.0, "empty_response", "No assistant response text"

        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0, "schema_error", str(e)[:200]

        self.strictify_schema(schema)

        try:
            response_obj = self.parse_content(schema_type, response_text)
        except Exception as e:
            return 0.0, "parse_error", f"{type(e).__name__}: {str(e)[:200]}"

        try:
            if schema_type == SchemaType.XML and self.config.xml_coerce_types:
                response_obj = self.coerce_xml_types(response_obj, schema)
            if schema_type == SchemaType.CSV:
                response_obj = self.coerce_csv_types(response_obj, schema)
            validate_against_schema_openapi(response_obj, schema)
            return 1.0, None, None
        except Exception as e:
            return 0.0, "validation_error", f"{type(e).__name__}: {str(e)[:200]}"


if __name__ == "__main__":
    StructuredOutputsResourcesServer.run_webserver()
