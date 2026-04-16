# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


class PalantirFdeEvalConfig(BaseResourcesServerConfig):
    # Judge configuration (JUDGE-03)
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_prompt_template_fpath: str = "prompt_templates/tool_call_judge.txt"
    judge_equal_label: str = "[[PASS]]"
    judge_not_equal_label: str = "[[FAIL]]"
    judge_endpoint_max_concurrency: Optional[int] = 64

    # Oracle configuration (CFG-02, CFG-03) -- wired but not used in verify()
    oracle_model_server: Optional[ModelServerRef] = None

    # Schema-aware validation (DATA-05)
    tool_schemas_fpath: Optional[str] = None


class PalantirFdeEvalVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Dict[str, Any] = Field(default_factory=dict)


class PalantirFdeEvalVerifyResponse(BaseVerifyResponse):
    num_expected: int = 0
    num_predicted: int = 0
    tool_name_match: bool = False
    structure_valid: bool = True
    judge_score: Optional[bool] = None
    predicted_calls: List[Dict[str, Any]] = Field(default_factory=list)


class PalantirFdeEvalResourcesServer(SimpleResourcesServer):
    config: PalantirFdeEvalConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{tool_name}")(self.handle_tool_call)
        return app

    async def handle_tool_call(self, tool_name: str, request: Request) -> PlainTextResponse:
        """Catch-all for tool call endpoints. This is an eval-only server with no
        access to a live Palantir AIP environment, so tool execution is not supported."""
        return PlainTextResponse(
            json.dumps({"error": "Tool execution not implemented. This is an evaluation-only server."})
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_semaphore = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_semaphore = nullcontext()

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

        # Load tool schemas for schema-aware validation (DATA-05)
        if self.config.tool_schemas_fpath:
            with open(self.config.tool_schemas_fpath, "r") as f:
                tool_list = json.load(f)
            self._tool_schemas: Dict[str, dict] = {tool["name"]: tool["parameters"] for tool in tool_list}
        else:
            self._tool_schemas: Dict[str, dict] = {}

    def _extract_function_calls(self, body: BaseVerifyRequest) -> List[Dict[str, Any]]:
        """Extract function_call items from the first model turn only.

        Stops at the first function_call_output (which indicates a multi-turn
        agent loop). This ensures single-turn evaluation regardless of agent config.
        """
        function_calls = []
        for output_item in body.response.output:
            if output_item.type == "function_call_output":
                break
            if output_item.type == "function_call":
                arguments = output_item.arguments
                try:
                    parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    parsed_args = {}
                function_calls.append(
                    {
                        "name": output_item.name,
                        "arguments": parsed_args,
                    }
                )
        return function_calls

    def _has_stringified_params(self, arguments: Dict[str, Any]) -> bool:
        """Detect if any parameter value is a stringified JSON object/array.

        Returns True if a stringified parameter is found (structure is INVALID).
        Only flags dict/list, NOT primitive JSON values (bool, int, float, null).
        Recurses into nested dicts and list items.
        """
        for key, value in arguments.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, (dict, list)):
                        return True
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(value, dict):
                if self._has_stringified_params(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if self._has_stringified_params(item):
                            return True
                    elif isinstance(item, str):
                        try:
                            parsed = json.loads(item)
                            if isinstance(parsed, (dict, list)):
                                return True
                        except (json.JSONDecodeError, ValueError):
                            pass
        return False

    def _value_matches_variant(self, value: Any, variant: dict) -> bool:
        """Check if value structurally matches a single anyOf variant type."""
        vtype = variant.get("type")
        if vtype == "null":
            return value is None
        if vtype == "boolean":
            return isinstance(value, bool)
        if vtype == "integer":
            # Use type() to exclude bool (isinstance(True, int) is True in Python)
            return type(value) is int
        if vtype == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if vtype == "string":
            return isinstance(value, str)
        if vtype == "array":
            return isinstance(value, list)
        if vtype == "object":
            return isinstance(value, dict)
        # Unknown type -> don't block
        return True

    def _check_anyof_violation(self, value: Any, prop_schema: dict) -> bool:
        """Return True if value violates an anyOf constraint (matches NONE of the variants)."""
        anyof_variants = prop_schema.get("anyOf")
        if not anyof_variants:
            return False
        return not any(self._value_matches_variant(value, variant) for variant in anyof_variants)

    def _has_schema_violations(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if arguments violate anyOf schema constraints for the given tool.

        Returns True if any top-level parameter violates its anyOf constraint.
        Returns False for unknown tools (graceful degradation).
        Only checks top-level parameter anyOf -- does NOT recurse into nested schemas.
        """
        schema = self._tool_schemas.get(tool_name)
        if not schema:
            return False
        properties = schema.get("properties", {})
        for param_name, param_value in arguments.items():
            prop_schema = properties.get(param_name)
            if prop_schema and self._check_anyof_violation(param_value, prop_schema):
                return True
        return False

    def _tool_names_match(
        self,
        predicted_calls: List[Dict[str, Any]],
        expected_calls: List[Dict[str, Any]],
    ) -> bool:
        """Check if all predicted tool names are a subset of expected tool names (subset matching).

        Aligns with colleague's evaluate.py methodology: predicted tools must be
        a subset of expected tools. This allows the model to call fewer tools than
        expected while still passing, which is the correct semantic for evaluating
        whether the model picked correct tools.
        """
        predicted_names = set(call["name"] for call in predicted_calls)
        expected_names = set(call["name"] for call in expected_calls)
        if not predicted_names:
            return False
        return predicted_names.issubset(expected_names)

    async def _judge_parameters(
        self,
        predicted_calls: List[Dict[str, Any]],
        expected_calls: List[Dict[str, Any]],
    ) -> bool:
        """Judge parameter quality by comparing predicted vs expected via LLM judge.

        Only pairs predicted tools against their matching expected tools by name.
        Expected tools not present in predicted calls are skipped (subset semantics).
        For each pair, formats prompt with json.dumps and calls judge model server.
        Returns True only if ALL parameter pairs pass.
        """
        cfg = self.config
        predicted_names = {c["name"] for c in predicted_calls}
        matched_expected = [c for c in expected_calls if c["name"] in predicted_names]
        sorted_predicted = sorted(predicted_calls, key=lambda c: c["name"])
        sorted_expected = sorted(matched_expected, key=lambda c: c["name"])

        for predicted, expected in zip(sorted_predicted, sorted_expected):
            user_prompt = self._judge_prompt_template.format(
                predicted_params=json.dumps(predicted["arguments"], indent=2),
                expected_params=json.dumps(expected["arguments"], indent=2),
            )

            msgs = [NeMoGymEasyInputMessage(role="user", content=user_prompt)]
            responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
            responses_create_params.input = msgs

            async with self._judge_semaphore:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))

            # Parse verdict from last output message
            try:
                last_output = judge_response.output[-1]
                if getattr(last_output, "type", None) != "message":
                    return False
                last_content = last_output.content[-1]
                text = getattr(last_content, "text", "")
            except Exception:
                return False

            # Position-based verdict: whichever label appears first wins
            eq_pos = text.find(cfg.judge_equal_label)
            neq_pos = text.find(cfg.judge_not_equal_label)

            if eq_pos < 0 and neq_pos < 0:
                return False
            if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
                continue  # This pair passes
            return False  # This pair fails

        return True

    async def verify(self, body: PalantirFdeEvalVerifyRequest) -> PalantirFdeEvalVerifyResponse:
        """Verify tool call correctness with composite three-layer reward.

        Reward logic (short-circuit evaluation):
        - No tool call expected and none predicted -> 1.0 (bypass judge)
        - No tool call expected but model predicted one -> 0.0 (bypass judge)
        - Tool call expected but model predicted none -> 0.0 (bypass judge)
        - Structure invalid (stringified params) -> 0.0 (bypass judge)
        - Tool name mismatch -> 0.0 (bypass judge)
        - Structure valid AND tool names match -> call judge
          - Judge pass -> 1.0
          - Judge fail -> 0.0
        """
        # 1. Extract predicted function calls (EVAL-01)
        predicted_calls = self._extract_function_calls(body)

        # 2. Get expected tool calls from verifier_metadata
        expected_calls = body.verifier_metadata.get("expected_tool_calls", [])
        has_tool_call = body.verifier_metadata.get("has_tool_call", True)

        # 3. Handle no-tool-call case (bypass judge entirely)
        if not has_tool_call and not predicted_calls:
            return PalantirFdeEvalVerifyResponse(
                **body.model_dump(),
                reward=1.0,
                num_expected=0,
                num_predicted=0,
                tool_name_match=True,
                structure_valid=True,
                predicted_calls=[],
            )

        if not has_tool_call and predicted_calls:
            return PalantirFdeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                num_expected=0,
                num_predicted=len(predicted_calls),
                tool_name_match=False,
                structure_valid=True,
                predicted_calls=predicted_calls,
            )

        # 4. Check structure validity (EVAL-02) -- short-circuit before judge
        structure_valid = True
        for call in predicted_calls:
            if self._has_stringified_params(call["arguments"]):
                structure_valid = False
                break

        # 4b. Check schema-aware structure validity (EVAL-03)
        if structure_valid and self._tool_schemas:
            for call in predicted_calls:
                if self._has_schema_violations(call["name"], call["arguments"]):
                    structure_valid = False
                    break

        if not structure_valid:
            return PalantirFdeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                num_expected=len(expected_calls),
                num_predicted=len(predicted_calls),
                tool_name_match=False,
                structure_valid=False,
                judge_score=None,
                predicted_calls=predicted_calls,
            )

        # 5. Check tool name match (EVAL-04) -- short-circuit before judge
        tool_name_match = self._tool_names_match(predicted_calls, expected_calls)

        if not tool_name_match:
            return PalantirFdeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                num_expected=len(expected_calls),
                num_predicted=len(predicted_calls),
                tool_name_match=False,
                structure_valid=True,
                judge_score=None,
                predicted_calls=predicted_calls,
            )

        # 6. Layer 3: LLM judge for parameter quality (JUDGE-01, EVAL-05)
        judge_pass = await self._judge_parameters(predicted_calls, expected_calls)
        reward = 1.0 if judge_pass else 0.0

        return PalantirFdeEvalVerifyResponse(
            **body.model_dump(),
            reward=reward,
            num_expected=len(expected_calls),
            num_predicted=len(predicted_calls),
            tool_name_match=True,
            structure_valid=True,
            judge_score=judge_pass,
            predicted_calls=predicted_calls,
        )


if __name__ == "__main__":
    PalantirFdeEvalResourcesServer.run_webserver()
