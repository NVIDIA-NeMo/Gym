# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the single_step_tool_use_with_argument_comparison server.

Rows carry exactly one task-owned field, ``expected_action``: the discriminated union of
message / function_call / function_call_batch actions that verify() compares the model's response
against. The action models mirror ``common/verification_utils.py`` field-for-field (including the
``min_length=1`` constraint on batch calls) but are redeclared here so this module stays a
dependency-light leaf importable without the server's requirements.

This file is the parent of the expected-action schema family: ``terminal_multi_harness`` imports
the message/function_call variants (its batch variant differs), and ``swe_pivot``'s untyped
``expected_action`` dict carries the same function_call shape.
"""

from typing import Annotated, List, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field


class MessageAction(BaseModel):
    """The expected action is an assistant chat message (any chat message earns full credit)."""

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
    """The expected action is a batch of parallel tool calls (compared order-insensitively)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function_call_batch"]
    calls: List[FunctionCallAction] = Field(min_length=1)


ExpectedAction: TypeAlias = Annotated[
    Union[MessageAction, FunctionCallAction, FunctionCallBatchAction],
    Field(discriminator="type"),
]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_action: ExpectedAction = Field(
        description=(
            "Action verify() compares the response against, discriminated on 'type': "
            "message {content}, function_call {name, arguments}, or function_call_batch {calls}."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
