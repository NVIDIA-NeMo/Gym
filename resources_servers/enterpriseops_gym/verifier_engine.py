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
"""EnterpriseOps-Gym verifier engine, ported from ``benchmark/verifier.py`` + ``benchmark/executor.py``.

PARITY WARNING: the extraction, comparison, judge-prompt, and result-collapse semantics below
are intentionally line-for-line ports of EnterpriseOps-Gym (Apache 2.0), including its quirks:

- ``_extract_value_from_sql_result`` has finicky single-row/single-column unwrapping rules.
- ``compare_values`` uses loose Python comparisons (``0 != "0"``).
- Verifier results are collapsed into a dict keyed by verifier *name*; tasks reuse names, so
  duplicate-named verifiers OVERWRITE each other and only the LAST one per name is scored
  (executor.py:474 upstream). The published leaderboard numbers use this behavior.
- Verifiers referencing an unknown ``gym_name`` are skipped entirely (not counted).

Do not "fix" these here — parity with the upstream benchmark is the point. The strict,
every-verifier-counts variant is computed separately for RL reward shaping.

Deviations from upstream (behavior-preserving): verifiers execute concurrently instead of
sequentially (they are independent reads; collapse is applied in config order afterwards),
and the LLM judge call is injected as an async callable instead of a LangChain client.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from resources_servers.enterpriseops_gym.mcp_client import MCPGymClient


logger = logging.getLogger(__name__)


JudgeFn = Callable[[str, str], Awaitable[str]]


# Verbatim EOG judge system prompt (benchmark/verifier.py::_compare_with_llm).
JUDGE_SYSTEM_PROMPT = """You are an AI judge evaluating the quality and accuracy of an AI assistant's response.
Compare the database query result with the AI's response and rate how well they match.
Provide a score from 1-10 where:
- 1-3: Poor match, incorrect or missing information
- 4-6: Partial match, some correct information
- 7-8: Good match, mostly correct
- 9-10: Excellent match, fully accurate

Respond with ONLY a JSON object in this format:
{
  "score": <number 1-10>,
  "reasoning": "<brief explanation>"
}"""


def extract_value_from_sql_result(result: dict) -> Any:
    """Extract the actual value from SQL query result (verbatim EOG port)."""
    if not result:
        return None

    # If the result itself is not successful, check for error content
    if not result.get("success"):
        # Try to extract error message from MCP response format
        result_data = result.get("result", {})
        if isinstance(result_data, dict) and "content" in result_data:
            content = result_data["content"]
            if isinstance(content, list) and len(content) > 0:
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "Error")
        return None

    result_data = result.get("result", {})

    # Handle different result formats from MCP sql-runner
    if isinstance(result_data, dict):
        # If result has 'data' field (common format)
        if "data" in result_data:
            data = result_data["data"]
            if isinstance(data, list) and len(data) > 0:
                # If single row with single column, return the value directly
                if len(data) == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                    return list(data[0].values())[0]
                # If single row with multiple columns, return the row dict
                elif len(data) == 1:
                    return data[0]
                # Multiple rows, return the full result
                else:
                    return data
            return data

        # If result has 'rows' field
        elif "rows" in result_data:
            rows = result_data["rows"]
            if isinstance(rows, list) and len(rows) > 0:
                # Single value from single row
                if len(rows) == 1 and isinstance(rows[0], dict) and len(rows[0]) == 1:
                    return list(rows[0].values())[0]
                # Single row as list
                elif len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1:
                    return rows[0][0]
                # Single row (dict or list)
                elif len(rows) == 1:
                    return rows[0]
                # Multiple rows
                else:
                    return rows
            return rows

        # If result has 'content' field (MCP error format)
        elif "content" in result_data:
            content = result_data["content"]
            if isinstance(content, list) and len(content) > 0:
                # Extract text from content array
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", result_data)
            return content

        # Direct result format (nested 'result' field)
        elif "result" in result_data:
            return result_data["result"]

    # Return as-is if we can't extract anything specific
    return result_data


def compare_values(actual: Any, expected: Any, comparison_type: str) -> Dict[str, Any]:
    """Compare actual vs expected values (verbatim EOG port)."""
    try:
        if comparison_type == "equals":
            passed = actual == expected
        elif comparison_type == "greater_than":
            passed = actual > expected
        elif comparison_type == "less_than":
            passed = actual < expected
        elif comparison_type == "contains":
            passed = expected in str(actual)
        else:
            return {
                "passed": False,
                "details": f"Unknown comparison type: {comparison_type}",
            }

        return {
            "passed": passed,
            "details": f"Comparison {comparison_type}: {actual} vs {expected}",
        }

    except Exception as e:
        return {"passed": False, "details": f"Comparison error: {e}"}


def extract_llm_content(model_response: Dict[str, Any]) -> str:
    """Extract text content from LLM response (verbatim EOG port)."""
    if "content" in model_response:
        return str(model_response["content"])
    elif "text" in model_response:
        return str(model_response["text"])
    elif "response" in model_response:
        return str(model_response["response"])

    return str(model_response)


def extract_json_from_judge_response(response_text: str) -> str:
    """Strip markdown code fences from a judge response (verbatim EOG parsing)."""
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    return response_text


class VerifierEngine:
    """Multi-gym verifier engine. Supports: database_state, response_check, tool_execution."""

    def __init__(self, gym_clients: Dict[str, MCPGymClient], judge_fn: Optional[JudgeFn] = None) -> None:
        self.gym_clients = gym_clients
        self.judge_fn = judge_fn

    def _get_mcp_client_for_gym(self, gym_name: Optional[str] = None) -> MCPGymClient:
        """Get the appropriate MCP client for a gym, or default (first), as EOG does."""
        if gym_name and gym_name in self.gym_clients:
            return self.gym_clients[gym_name]
        return next(iter(self.gym_clients.values()))

    async def execute_verifier(
        self,
        verifier: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: Optional[str],
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single verifier (EOG dispatch semantics)."""
        verifier_type = verifier.get("verifier_type")
        validation_config = verifier.get("validation_config") or {}

        if verifier_type == "database_state":
            return await self._execute_database_state_verifier(validation_config, database_id, context, gym_name)
        elif verifier_type == "response_check":
            return await self._execute_response_check_verifier(
                validation_config, model_response, database_id, context, gym_name
            )
        elif verifier_type == "tool_execution":
            return self._execute_tool_execution_verifier(validation_config, model_response)
        else:
            return {
                "passed": False,
                "error": f"Unsupported verifier type: {verifier_type}",
            }

    async def _execute_database_state_verifier(
        self,
        validation_config: Dict[str, Any],
        database_id: Optional[str],
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        sql_query = validation_config.get("query")
        expected_value = validation_config.get("expected_value")
        comparison_type = validation_config.get("comparison_type", "equals")

        if not sql_query:
            return {"passed": False, "error": "No SQL query provided"}

        client = self._get_mcp_client_for_gym(gym_name)
        result = await client.run_sql(sql_query, database_id, context)

        if not result["success"]:
            return {
                "passed": False,
                "error": f"SQL query failed: {result.get('error')}",
                "query": sql_query,
            }

        actual_value = extract_value_from_sql_result(result)
        comparison_result = compare_values(actual_value, expected_value, comparison_type)

        return {
            "passed": comparison_result["passed"],
            "expected": expected_value,
            "actual": actual_value,
            "comparison_type": comparison_type,
            "query": sql_query,
            "details": comparison_result.get("details"),
        }

    async def _execute_response_check_verifier(
        self,
        validation_config: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: Optional[str],
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        sql_query = validation_config.get("sql_query")
        comparison_prompt = validation_config.get("comparison_prompt")
        minimum_comparison_value = validation_config.get("minimum_comparison_value", 7)

        if not sql_query or not comparison_prompt:
            return {"passed": False, "error": "Missing sql_query or comparison_prompt"}

        client = self._get_mcp_client_for_gym(gym_name)
        sql_result = await client.run_sql(sql_query, database_id, context)

        if not sql_result["success"]:
            return {
                "passed": False,
                "error": f"SQL query failed: {sql_result.get('error')}",
            }

        llm_response_text = extract_llm_content(model_response)
        return await self._compare_with_llm(sql_result, llm_response_text, comparison_prompt, minimum_comparison_value)

    def _execute_tool_execution_verifier(
        self, validation_config: Dict[str, Any], model_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool execution verifier (verbatim EOG port)."""
        selected_tools = validation_config.get("selected_tools", [])
        minimum_tool_calls = validation_config.get("minimum_tool_calls", 1)

        tools_called = []
        if "tool_calls" in model_response and model_response["tool_calls"]:
            tools_called = [tc["name"] for tc in model_response["tool_calls"]]

        missing_tools = [tool for tool in selected_tools if tool not in tools_called]
        passed = len(missing_tools) == 0 and len(tools_called) >= minimum_tool_calls

        return {
            "passed": passed,
            "selected_tools": selected_tools,
            "tools_called": tools_called,
            "missing_tools": missing_tools,
            "minimum_tool_calls": minimum_tool_calls,
            "actual_tool_calls": len(tools_called),
        }

    async def _compare_with_llm(
        self,
        sql_result: Dict[str, Any],
        llm_response: str,
        comparison_prompt: str,
        minimum_score: int,
    ) -> Dict[str, Any]:
        """Use LLM as judge to compare SQL result with LLM response (verbatim EOG prompts/parsing)."""
        sql_result_str = json.dumps(sql_result.get("result", {}), indent=2)

        user_prompt = f"""Database Query Result:
{sql_result_str}

AI Assistant Response:
{llm_response}

Comparison Task:
{comparison_prompt}

Please provide your judgment as JSON."""

        try:
            if self.judge_fn is None:
                raise RuntimeError("No judge model configured (set judge_model_server in the resources server config)")

            response_text = await self.judge_fn(JUDGE_SYSTEM_PROMPT, user_prompt)
            response_text = extract_json_from_judge_response(response_text)

            judge_result = json.loads(response_text)
            score = judge_result.get("score", 0)
            reasoning = judge_result.get("reasoning", "")

            passed = score >= minimum_score

            return {
                "passed": passed,
                "score": score,
                "minimum_score": minimum_score,
                "reasoning": reasoning,
                "sql_result": sql_result_str,
                "llm_response": llm_response,
            }

        except Exception as e:
            logger.error(f"LLM judge comparison failed: {e}")
            return {"passed": False, "error": f"Judge comparison failed: {e}"}

    async def run_verifiers(
        self,
        verifiers: List[Dict[str, Any]],
        model_response: Dict[str, Any],
        gym_databases: Dict[str, Optional[str]],
        gym_contexts: Dict[str, Optional[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run all verifiers and collapse results with EOG executor semantics.

        Returns:
            (verification_results, all_verifier_results)

            verification_results: EOG-parity dict keyed by verifier name — duplicate names
                overwrite in config order, so only the LAST verifier per name is scored.
                Verifiers with an unknown gym_name are skipped (absent), as upstream.
            all_verifier_results: every verifier's result (strict view), in config order.
        """

        async def _run_one(index: int, verifier: Dict[str, Any]) -> Dict[str, Any]:
            verifier_name = verifier.get("name") or f"verifier_{index + 1}"
            gym_name = verifier.get("gym_name")

            # Gym resolution, mirroring executor._run_verifiers:
            # - gym_name set and known -> that gym's database/context
            # - gym_name set and unknown -> skip the verifier entirely (not counted)
            # - gym_name unset -> upstream falls back to the legacy single-gym database
            #   (None for multi-gym configs); we use the single gym when unambiguous.
            skipped = False
            database_id: Optional[str] = None
            context: Optional[Dict[str, Any]] = None
            if gym_name:
                if gym_name in gym_databases:
                    database_id = gym_databases[gym_name]
                    context = gym_contexts.get(gym_name)
                else:
                    logger.warning(f"Gym '{gym_name}' not found in gym_servers_config!")
                    skipped = True
            elif len(gym_databases) == 1:
                only_gym = next(iter(gym_databases))
                database_id = gym_databases[only_gym]
                context = gym_contexts.get(only_gym)

            if skipped:
                return {
                    "index": index,
                    "name": verifier_name,
                    "gym_name": gym_name,
                    "skipped": True,
                    "result": {"passed": False, "error": f"Gym '{gym_name}' not found in configuration"},
                }

            result = await self.execute_verifier(verifier, model_response, database_id, context, gym_name=gym_name)
            return {
                "index": index,
                "name": verifier_name,
                "gym_name": gym_name,
                "skipped": False,
                "result": result,
            }

        all_verifier_results = await asyncio.gather(*(_run_one(i, v) for i, v in enumerate(verifiers)))

        # EOG-parity collapse: dict keyed by name, insertion/overwrite in config order,
        # skipped verifiers omitted entirely.
        verification_results: Dict[str, Any] = {}
        for entry in all_verifier_results:
            if entry["skipped"]:
                continue
            verification_results[entry["name"]] = entry["result"]

        return verification_results, list(all_verifier_results)
