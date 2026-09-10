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

import json
from unittest.mock import MagicMock

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.jsonschemabench.app import (
    JSONSchemaBenchResourcesServer,
    JSONSchemaBenchResourcesServerConfig,
    JSONSchemaBenchVerifyRequest,
    evaluate_response,
)


def make_response(text: str) -> NeMoGymResponse:
    """Create one text-only model response."""

    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def make_server() -> JSONSchemaBenchResourcesServer:
    """Create a verifier without starting an HTTP server."""

    config = JSONSchemaBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="jsonschemabench",
    )
    return JSONSchemaBenchResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def make_request(schema: object, response_text: str) -> JSONSchemaBenchVerifyRequest:
    """Create one verifier request with stable benchmark metadata."""

    return JSONSchemaBenchVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response(response_text),
        schema_str=json.dumps(schema),
        schema_type="json",
        problem_type="easy",
        source_record_id="row-1",
        shaper_canonical_eval_row_id=7,
    )


def test_optional_property_remains_optional() -> None:
    schema = {"type": "object", "properties": {"author": {"type": "string"}}}
    reward, error_type, _ = evaluate_response(json.dumps(schema), "{}")
    assert reward == 1.0
    assert error_type is None


def test_additional_property_allowed_when_schema_does_not_forbid_it() -> None:
    schema = {"type": "object", "properties": {"author": {"type": "string"}}}
    reward, error_type, _ = evaluate_response(json.dumps(schema), '{"extra": true}')
    assert reward == 1.0
    assert error_type is None


def test_explicit_required_property_is_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {"author": {"type": "string"}},
        "required": ["author"],
    }
    reward, error_type, _ = evaluate_response(json.dumps(schema), "{}")
    assert reward == 0.0
    assert error_type == "validation_error"


def test_explicit_additional_properties_false_is_enforced() -> None:
    schema = {"type": "object", "additionalProperties": False}
    reward, error_type, _ = evaluate_response(json.dumps(schema), '{"extra": true}')
    assert reward == 0.0
    assert error_type == "validation_error"


def test_draft_2020_12_keywords_are_enforced() -> None:
    schema = {"type": "array", "prefixItems": [{"const": 1}], "items": False}
    assert evaluate_response(json.dumps(schema), "[1]")[0] == 1.0
    assert evaluate_response(json.dumps(schema), "[1, 2]")[0] == 0.0


def test_format_checker_is_enabled() -> None:
    schema = {"type": "string", "format": "email"}
    reward, error_type, _ = evaluate_response(json.dumps(schema), '"not-an-email"')
    assert reward == 0.0
    assert error_type == "validation_error"


def test_optional_json_code_fence_is_accepted() -> None:
    schema = {"type": "object"}
    reward, error_type, _ = evaluate_response(json.dumps(schema), "```json\n{}\n```")
    assert reward == 1.0
    assert error_type is None


def test_invalid_json_is_a_parse_failure() -> None:
    reward, error_type, _ = evaluate_response('{"type":"object"}', "not json")
    assert reward == 0.0
    assert error_type == "parse_error"


def test_invalid_schema_is_a_schema_failure() -> None:
    reward, error_type, _ = evaluate_response('{"type":"not-a-type"}', "{}")
    assert reward == 0.0
    assert error_type == "schema_error"


async def test_verify_preserves_benchmark_metadata() -> None:
    result = await make_server().verify(
        make_request(
            {"type": "object", "properties": {"author": {"type": "string"}}},
            "{}",
        )
    )
    assert result.reward == 1.0
    assert result.error_type is None
    assert result.problem_type == "easy"
    assert result.source_record_id == "row-1"
    assert result.shaper_canonical_eval_row_id == 7


async def test_verify_rejects_non_json_schema_type() -> None:
    request = make_request({"type": "object"}, "{}")
    request.schema_type = "yaml"
    result = await make_server().verify(request)
    assert result.reward == 0.0
    assert result.error_type == "schema_type_error"
