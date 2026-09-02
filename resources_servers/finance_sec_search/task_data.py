# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the finance_sec_search server.

Wire-identical to math_with_judge's two-field core: ``FinanceAgentRunRequest`` (app.py:321)
requires ``question`` and ``expected_answer``, so this schema reuses the family parent and only
re-annotates ``question``. Quirk: verify() (app.py:1300) never reads ``body.question`` — it
re-extracts the question from the last user message in ``responses_create_params.input`` when
filling the judge prompt — but the field stays required because the wire 422s without it. Scoring
compares the agent's submit_final_result payload against ``expected_answer`` via an LLM judge,
falling back to case-insensitive substring match when no judge_model_server is configured.

Only ``data/example.jsonl`` contains task rows; ``example_questions.jsonl`` is a pre-conversion
question source (no responses_create_params) and ``example_rollouts.jsonl`` is rollout output.
"""

from pydantic import Field

from resources_servers.math_with_judge.task_data import TaskData as MathWithJudgeTaskData


class TaskData(MathWithJudgeTaskData):
    question: str = Field(
        description=(
            "The SEC-filings research question. Wire-required but unread by verify(), which re-extracts "
            "the question from the last user message in responses_create_params.input."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_answer: str = Field(
        description="Ground-truth answer the judge (or substring fallback) compares the submitted final result to.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
