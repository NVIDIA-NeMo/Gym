# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the swe_pivot server.

Single-step pivot rows carved out of SWE-agent trajectories. All task data is top-level (no
verifier_metadata anywhere). verify() resolves the expected function call through a deliberate
three-way fallback chain — ``expected_action`` (this server's native format), then
``expected_answer`` as a JSON-encoded string (terminal_pivot-style rows), then
``metadata['expected_action']`` — so every link in that chain is wire-Optional and stays Optional
here. ``expected_action`` is an UNTYPED dict on the wire (``SwePivotRunRequest`` at app.py:488,
``extra='allow'``); its expected shape is the ``function_call`` variant of
``resources_servers.single_step_tool_use_with_argument_comparison.task_data.FunctionCallAction``
({type: 'function_call', name, arguments}), but verify() only soft-``.get``s it and rewards 0.0
on anything else, so this schema keeps it an open dict to match the wire.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Row identifier; used only for logging and echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Fallback expected action as a JSON-encoded string of "
            "{type: 'function_call', name, arguments} (terminal_pivot-style rows); absent in "
            "committed swe_pivot rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Expected tool call: {type: 'function_call', name: str, arguments: str} where arguments "
            "is a JSON-encoded object string. Untyped dict on the wire; only type='function_call' is "
            "handled by verify()."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Open provenance dict, echoed into the verify response. May carry a second-fallback "
            "'expected_action' key; committed rows carry {category: 'edit'|'bash'}, which is unread."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    pivot_tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Pivot classification tags (e.g. ['P1']). Not declared on the wire model; survives via "
            "extra='allow' but is never read by verify()."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
