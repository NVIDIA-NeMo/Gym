# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the terminus_judge server.

All task fields are top-level (no ``verifier_metadata``) and mirror ``TerminusJudgeRunRequest``
(app.py:206, ``extra="allow"``) minus ``threshold``, which is a verify-time request override of
the string-similarity threshold and not a dataset column. ``expected_answer`` is a JSON-encoded
string holding a terminus command batch; ``verify()`` parses it and validates it against
``TERMINUS_1_SCHEMA``/``TERMINUS_2_SCHEMA`` (schemas.py) selected by ``metadata['harness']``.
"""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Row identifier; used for logging and echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "JSON-encoded terminus command batch (object with analysis/plan/commands[...]); parsed and "
            "schema-validated by verify(). Falls back to metadata['expected_answer'] when empty."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Untyped dict ({'category': str, 'harness': str} in committed data). verify() reads "
            "metadata['harness'] ('terminus_1'|'terminus_2') to select the command-batch schema, and "
            "metadata['expected_answer'] as a fallback gold answer."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
