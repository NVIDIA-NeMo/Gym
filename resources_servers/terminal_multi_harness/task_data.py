# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the terminal_multi_harness server.

Rows pair an ``expected_action`` (a discriminated union of message / function_call /
function_call_batch actions) with harness-specific verification context: which ``harness``'s tool
conventions to grade under, the ``declared_tools`` schemas used to validate tool-call arguments,
and an optional per-row similarity ``threshold``. The action models mirror
``common/verification_utils.py`` field-for-field; the batch variant adds ``ordered`` and allows
an empty ``calls`` list.

Committed rows also carry ``uuid`` and ``metadata`` ({harness, example_kind}), which are NOT
fields of today's wire request model (``TerminalMultiHarnessRunRequest`` uses pydantic's default
``extra='ignore'`` and silently drops them); they are declared here as optional provenance so the
schema stops that silent drop from going unnoticed.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field


class MessageAction(BaseModel):
    """The expected action is an assistant chat message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["message"]
    content: str


class FunctionCallAction(BaseModel):
    """The expected action is a single tool call."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function_call"]
    name: str
    arguments: str = Field(description="JSON-encoded object string of the expected tool-call arguments.")


class FunctionCallBatchAction(BaseModel):
    """The expected action is a batch of tool calls; ``ordered`` controls order sensitivity."""

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
