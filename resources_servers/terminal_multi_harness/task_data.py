# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the terminal_multi_harness server.

Rows pair an ``expected_action`` (the same discriminated union family as
single_step_tool_use_with_argument_comparison, whose message/function_call variants are imported
here) with harness-specific verification context: which ``harness``'s tool conventions to grade
under, the ``declared_tools`` schemas used to validate tool-call arguments, and an optional
per-row similarity ``threshold``. This server's batch variant differs from the parent's — it adds
``ordered`` and allows an empty ``calls`` list — so it is redeclared locally, mirroring
``common/verification_utils.py``.

Committed rows also carry ``uuid`` and ``metadata`` ({harness, example_kind}), which are NOT
fields of today's wire request model (``TerminalMultiHarnessRunRequest`` uses pydantic's default
``extra='ignore'`` and silently drops them); they are declared here as optional provenance so the
schema stops that silent drop from going unnoticed.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field

from resources_servers.single_step_tool_use_with_argument_comparison.task_data import (
    FunctionCallAction,
    MessageAction,
)


class FunctionCallBatchAction(BaseModel):
    """This server's batch variant: adds ``ordered`` and (unlike the parent's) allows empty calls."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function_call_batch"]
    calls: List[FunctionCallAction]
    ordered: bool = True


ExpectedAction: TypeAlias = Annotated[
    Union[MessageAction, FunctionCallAction, FunctionCallBatchAction],
    Field(discriminator="type"),
]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    harness: str = Field(
        default="generic",
        description=(
            "Which agent harness's tool conventions to grade under (e.g. 'codex', 'generic'); "
            "all committed rows use 'codex'."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_action: ExpectedAction = Field(
        description=(
            "Action verify() compares the response against, discriminated on 'type': message {content}, "
            "function_call {name, arguments}, or function_call_batch {calls, ordered}."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    declared_tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Tool definitions whose schemas validate the model's tool-call arguments; when None, "
            "verify() falls back to responses_create_params.tools. Empty list for message-only rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Per-row similarity-threshold override; wire-accepted but absent from committed rows.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    uuid: Optional[str] = Field(
        default=None,
        description="Row identifier. Not a wire request field today: silently dropped by extra='ignore'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Provenance dict ({harness, example_kind} in committed rows). Not a wire request field "
            "today: silently dropped by extra='ignore'."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
