# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the multichallenge server (multi-turn conversation rubric judging).

Mirrors ``MultiChallengeRunRequest`` (app.py): every task field is Optional with a ``None``
default and the wire model carries ``extra="allow"``. ``verify()`` is fully defensive — it falls
back to ``metadata["messages"]``/``metadata["rubric"]`` when ``context``/``rubric`` are absent,
and reads rubric items as untyped dicts via ``.get`` (``question``, ``pass_criteria`` default
``"YES"``, ``weight`` default ``1.0``). Rubric items are therefore left as open dicts here.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Source dataset row identifier; passed through, not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_id: Optional[int] = Field(
        default=None,
        description="Numeric task identifier; passed through, not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    rubric: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Rubric items judged one by one; each dict is read defensively via .get: "
            "question (str, default ''), pass_criteria (str, default 'YES'), weight (float, default 1.0)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    context: Optional[str] = Field(
        default=None,
        description="Rendered conversation context shown to the judge (falls back to metadata['messages']).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Open provenance dict; verify() only reads it as a fallback source for 'messages' and 'rubric'. "
            "Committed rows carry only {topic: str, challenge: str}."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
