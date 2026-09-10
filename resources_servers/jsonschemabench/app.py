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

"""JSONSchemaBench verifier that preserves the benchmark's JSON Schema contract."""

import json
from typing import Any, Optional

import jsonschema
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


FORMAT_CHECKER = jsonschema.FormatChecker()


class JSONSchemaBenchResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the deterministic JSONSchemaBench verifier."""

    name: str = "jsonschemabench"


class JSONSchemaBenchVerifyRequest(BaseVerifyRequest):
    """One JSONSchemaBench response and its untouched source schema."""

    model_config = ConfigDict(extra="allow")

    schema_str: str
    schema_type: str = "json"
    problem_type: Optional[str] = None
    source_record_id: Optional[str] = None
    shaper_canonical_eval_row_id: Optional[int] = None


class JSONSchemaBenchVerifyResponse(BaseVerifyResponse):
    """Binary conformance result with stable failure diagnostics."""

    model_config = ConfigDict(extra="allow")

    schema_str: str
    schema_type: str
    problem_type: Optional[str] = None
    source_record_id: Optional[str] = None
    shaper_canonical_eval_row_id: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def strip_code_fences(text: str) -> str:
    """Remove the optional JSON fence accepted by the benchmark evaluator."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


def assistant_text(body: JSONSchemaBenchVerifyRequest) -> str:
    """Concatenate text emitted in assistant message output blocks."""

    return "".join(
        item.text
        for output in body.response.output
        if output.type == "message"
        for item in output.content
        if item.type == "output_text"
    )


def evaluate_response(schema_str: str, response_text: str) -> tuple[float, Optional[str], Optional[str]]:
    """Validate one response against the source Draft 2020-12 schema without rewriting it."""

    if not response_text.strip():
        return 0.0, "empty_response", "No assistant response text"

    try:
        schema: Any = json.loads(schema_str)
        jsonschema.Draft202012Validator.check_schema(schema)
    except (json.JSONDecodeError, jsonschema.SchemaError) as error:
        return 0.0, "schema_error", f"{type(error).__name__}: {str(error)[:200]}"

    try:
        instance = json.loads(strip_code_fences(response_text))
    except json.JSONDecodeError as error:
        return 0.0, "parse_error", f"{type(error).__name__}: {str(error)[:200]}"

    try:
        jsonschema.Draft202012Validator(schema, format_checker=FORMAT_CHECKER).validate(instance)
        return 1.0, None, None
    except jsonschema.ValidationError as error:
        return 0.0, "validation_error", f"{type(error).__name__}: {str(error)[:200]}"
    except Exception as error:  # noqa: BLE001 - remote references can fail broadly.
        return 0.0, "validation_runtime_error", f"{type(error).__name__}: {str(error)[:200]}"


class JSONSchemaBenchResourcesServer(SimpleResourcesServer):
    """Score model text with the benchmark's unmodified Draft 2020-12 schemas."""

    config: JSONSchemaBenchResourcesServerConfig

    async def verify(self, body: JSONSchemaBenchVerifyRequest) -> JSONSchemaBenchVerifyResponse:
        if body.schema_type != "json":
            return JSONSchemaBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                error_type="schema_type_error",
                error_message=f"JSONSchemaBench requires schema_type='json', got {body.schema_type!r}",
            )

        reward, error_type, error_message = evaluate_response(body.schema_str, assistant_text(body))
        return JSONSchemaBenchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            error_type=error_type,
            error_message=error_message,
        )


if __name__ == "__main__":
    JSONSchemaBenchResourcesServer.run_webserver()
