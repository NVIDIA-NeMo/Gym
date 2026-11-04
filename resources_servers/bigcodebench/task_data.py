# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the bigcodebench server.

Rows carry all task data inside the legacy ``verifier_metadata`` bucket (the wire model types it
only as ``Optional[Dict[str, Any]]``); the schema is written flat with ``legacy_location``
annotations per the protocol. Required-ness follows verify()'s hard reads: ``test`` and
``entry_point`` are indexed with ``meta[...]`` (KeyError if absent), while ``task_id`` /
``code_prompt`` are read via ``.get`` with defaults and ``split`` is never read at all.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="BigCodeBench task identifier, e.g. 'BigCodeBench/15'; echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    test: str = Field(
        description=(
            "unittest test-suite source code (can be thousands of chars) executed against the calibrated "
            "model solution in the sandboxed BCB venv."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    entry_point: str = Field(
        description="Name of the function under test (e.g. 'task_func') that the test suite imports.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    code_prompt: str = Field(
        default="",
        description=(
            "Imports + function-signature prefix prepended (with a trailing 'pass' stub) before the model's "
            "code so the entry_point exists even if the model returned only a function body."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    split: Optional[str] = Field(
        default=None,
        description="Source split of the task (e.g. 'hard'); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
