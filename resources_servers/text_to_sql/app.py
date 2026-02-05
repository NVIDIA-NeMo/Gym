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
"""
Text-to-SQL LLM-as-judge resources server.

Compares a model's generated SQL query to an expected query using an LLM judge.
Supports multiple SQL dialects: MySQL, PostgreSQL, SQLite (more to come - TODO).
"""

import asyncio
import re
from contextlib import nullcontext
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
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
from resources_servers.text_to_sql.prompts import SQL_JUDGE_PROMPT_TEMPLATE, SQL_JUDGE_SYSTEM_MESSAGE


class FailureCode(str, Enum):
    """Enumeration of possible failure reasons."""

    NONE = "none"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    JUDGE_EVALUATION_FAILED = "judge_evaluation_failed"
    UNKNOWN_ERROR = "unknown_error"


# Supported SQL dialects
# TODO: Add more dialects (Oracle, SQL Server, BigQuery, Snowflake)
SUPPORTED_DIALECTS = {"mysql", "postgresql", "sqlite"}


def extract_sql_from_response(text: str) -> Optional[str]:
    """Extract SQL query from model response.

    Attempts to extract SQL in the following order:
    1. SQL wrapped in ```sql ... ``` code blocks
    2. SQL wrapped in ``` ... ``` code blocks
    3. Raw SQL statements starting with SELECT/INSERT/UPDATE/DELETE/WITH

    Returns:
        Extracted SQL query or None if no SQL found.
    """
    if not text:
        return None

    # Try to extract from ```sql ... ``` code block
    sql_block_pattern = r"```sql\s*([\s\S]*?)\s*```"
    matches = re.findall(sql_block_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Try to extract from ``` ... ``` code block
    generic_block_pattern = r"```\s*([\s\S]*?)\s*```"
    matches = re.findall(generic_block_pattern, text)
    if matches:
        # Check if the content looks like SQL
        for match in reversed(matches):
            content = match.strip()
            if re.match(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s", content, re.IGNORECASE):
                return content

    # Try to find raw SQL statement
    sql_pattern = r"((?:SELECT|INSERT|UPDATE|DELETE|WITH)\s+[\s\S]*?(?:;|$))"
    matches = re.findall(sql_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip(";") + ";"

    return None


def _extract_question_text(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Extract the question text from the last user message."""
    last_text: Optional[str] = None
    for m in params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
    return (last_text or "").strip()


def _extract_dialect_from_prompt(text: str) -> Optional[str]:
    """Extract SQL dialect from a structured prompt."""
    if not text:
        return None
    match = re.search(r"^\s*DIALECT:\s*([a-zA-Z0-9_+-]+)\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def _normalize_dialect(dialect: Optional[str]) -> Optional[str]:
    if not dialect:
        return None
    return dialect.strip().lower()


def _extract_judge_response_text(response: NeMoGymResponse) -> str:
    """Extract the judge's assistant text across all output messages."""
    texts: list[str] = []
    for o in response.output or []:
        if getattr(o, "type", None) != "message":
            continue
        role = getattr(o, "role", None)
        if role is not None and role != "assistant":
            continue
        content = getattr(o, "content", None)
        if isinstance(content, list):
            msg_texts: list[str] = []
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    msg_texts.append(t.strip())
            if msg_texts:
                texts.append("\n".join(msg_texts))
        elif isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return "\n".join(texts).strip()


class TextToSqlResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the Text-to-SQL judge server."""

    name: str = "text_to_sql"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: str = SQL_JUDGE_SYSTEM_MESSAGE
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"

    # Swap check: Run second judge pass with swapped expected/generated to detect positional bias
    check_twice_swap: bool = True
    # Reward when the second (swap) pass fails
    reward_if_swap_fails: float = 0.0


class TextToSqlRunRequest(BaseRunRequest):
    """Run/verify request payload for text-to-SQL tasks."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    sql: str  # Ground truth SQL query (required)
    sql_dialect: str  # SQL dialect: mysql, postgresql, sqlite (required)
    sql_context: str = ""  # Database schema (CREATE/INSERT statements)
    sql_prompt: Optional[str] = None  # Natural language question (optional, extracted from input if not provided)
    metadata: Optional[dict[str, Any]] = None


class TextToSqlVerifyRequest(TextToSqlRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    """Record of a single judge evaluation."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    verdict_label: Optional[str] = None


class TextToSqlVerifyResponse(BaseVerifyResponse):
    """Verification response for text-to-SQL tasks."""

    uuid: Optional[str | int] = None
    sql: str  # Ground truth SQL query
    model_output: str
    extracted_sql: Optional[str] = None
    sql_dialect: str  # SQL dialect used
    sql_context: str  # Database schema provided
    sql_prompt: Optional[str] = None  # May be extracted from input
    judge_passed: bool = False
    failure_reason: Optional[FailureCode] = None
    judge_evaluations: list[JudgeEvaluation] = []
    metadata: Optional[dict[str, Any]] = None


class TextToSqlResourcesServer(SimpleResourcesServer):
    """Text-to-SQL judge verifier using an LLM to compare SQL queries."""

    config: TextToSqlResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = None

        self._judge_prompt_template = SQL_JUDGE_PROMPT_TEMPLATE

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: TextToSqlVerifyRequest) -> TextToSqlVerifyResponse:
        """Verify model response by comparing generated SQL with expected using LLM judge."""
        # These are required fields, validated by Pydantic
        expected_sql = body.sql
        sql_context = body.sql_context

        # Normalize and validate dialect
        sql_dialect = _normalize_dialect(body.sql_dialect)
        if not sql_dialect:
            raise ValueError("SQL dialect is required but was not provided")
        if sql_dialect not in SUPPORTED_DIALECTS:
            raise ValueError(f"Unsupported SQL dialect '{sql_dialect}'. Supported: {sorted(SUPPORTED_DIALECTS)}")

        # Extract question from request field or from user message
        sql_prompt = body.sql_prompt or _extract_question_text(body.responses_create_params)

        # Get model output text directly from response
        generated = body.response.output_text or ""
        if not generated:
            raise ValueError("No assistant response found/extracted to verify")

        reward = 0.0
        failure_reason = None
        judge_passed = False
        judge_evaluations = []
        extracted_sql = None

        try:
            # Extract SQL from model output
            extracted_sql = extract_sql_from_response(generated)

            if not extracted_sql:
                failure_reason = FailureCode.NO_SQL_EXTRACTED
                reward = 0.0
            else:
                # Run LLM judge evaluation
                first_equal, first_eval = await self._generate_judge_evaluation(
                    sql_prompt=sql_prompt,
                    sql_context=sql_context,
                    expected_sql=expected_sql,
                    generated_sql=extracted_sql,
                    sql_dialect=sql_dialect,
                )
                judge_evaluations.append(first_eval)

                if first_equal:
                    if self.config.check_twice_swap:
                        # Run swap check
                        second_equal, second_eval = await self._generate_judge_evaluation(
                            sql_prompt=sql_prompt,
                            sql_context=sql_context,
                            expected_sql=extracted_sql,
                            generated_sql=expected_sql,
                            sql_dialect=sql_dialect,
                        )
                        judge_evaluations.append(second_eval)

                        if second_equal:
                            judge_passed = True
                            reward = 1.0
                            failure_reason = FailureCode.NONE
                        else:
                            reward = self.config.reward_if_swap_fails
                            failure_reason = FailureCode.JUDGE_EVALUATION_FAILED
                    else:
                        judge_passed = True
                        reward = 1.0
                        failure_reason = FailureCode.NONE
                else:
                    failure_reason = FailureCode.JUDGE_EVALUATION_FAILED
                    reward = 0.0

        except Exception as e:
            failure_reason = FailureCode.UNKNOWN_ERROR
            reward = 0.0
            print(f"DEBUG: Unknown error in verify: {type(e).__name__} {e}", flush=True)

        payload = body.model_dump()
        payload.pop("sql", None)
        payload.pop("sql_dialect", None)
        payload.pop("sql_context", None)
        payload.pop("sql_prompt", None)

        return TextToSqlVerifyResponse(
            **payload,
            reward=reward,
            sql=expected_sql,
            model_output=generated,
            extracted_sql=extracted_sql,
            sql_dialect=sql_dialect,
            sql_context=sql_context,
            sql_prompt=sql_prompt,
            judge_passed=judge_passed,
            failure_reason=failure_reason,
            judge_evaluations=judge_evaluations,
        )

    async def _generate_judge_evaluation(
        self,
        *,
        sql_prompt: str,
        sql_context: str,
        expected_sql: str,
        generated_sql: str,
        sql_dialect: str,
    ) -> tuple[bool, JudgeEvaluation]:
        """Run a single judge evaluation."""
        cfg = self.config
        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)

        user_prompt = self._judge_prompt_template.format(
            sql_prompt=sql_prompt,
            sql_context=sql_context,
            first_answer=expected_sql,
            second_answer=generated_sql,
            sql_dialect=sql_dialect,
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        ctx = self._judge_endpoint_max_concurrency or nullcontext()
        async with ctx:
            try:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )

                judge_response = NeMoGymResponse.model_validate(await response.json())

            except asyncio.TimeoutError:
                print(
                    "DEBUG: TextToSqlResourcesServer: Judge model server timeout",
                    flush=True,
                )
                raise RuntimeError("Judge model server timeout")
            except Exception as e:
                print(
                    f"DEBUG: TextToSqlResourcesServer: judge model server HTTP POST error: {type(e).__name__} {e}",
                    flush=True,
                )
                raise e

        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        verdict_label = None
        is_equal = False

        # Extract text from judge response
        text = _extract_judge_response_text(judge_response)

        # Check text for verdict labels
        if text:
            eq_pos = text.find(equal_label)
            neq_pos = text.find(not_equal_label)

            if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
                verdict_label = equal_label
                is_equal = True
            elif neq_pos >= 0:
                verdict_label = not_equal_label

        eval_record.verdict_label = verdict_label
        return is_equal, eval_record


if __name__ == "__main__":
    TextToSqlResourcesServer.run_webserver()
