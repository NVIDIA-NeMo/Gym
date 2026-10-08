# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the workplace_assistant server.

Stateful workplace tool-use environment (email/calendar/analytics/project-management/CRM tool
envs seeded per session). Task fields ride at the row top level with no verifier_metadata, and
all four are required by ``WorkbenchVerifyRequest`` (app.py:44). verify() replays both the
predicted and the ``ground_truth`` actions in fresh tool environments and compares the resulting
state (utils.is_correct); each ground-truth item is hard-indexed as ``action["name"]`` and
``json.loads(action["arguments"])`` — ``arguments`` is a JSON-ENCODED STRING here, unlike
xlam_fc where it is a dict. The wire annotation admits ``ground_truth: str`` as a union arm, but
every committed row is a list of {name, arguments} dicts. ``id``/``category``/``environment_name``
are wire-required grouping/routing metadata never read by verify logic.
"""

from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ground_truth: Union[List[Dict[str, str]], str] = Field(
        description=(
            "Expected tool actions replayed against a fresh tool env for state comparison. Each item is "
            "{'name': tool function name, 'arguments': JSON-encoded string of the kwargs object} — "
            "arguments stays a str on the wire and is json.loads'd by the verifier. The bare-str union "
            "arm exists on the wire annotation but appears in no committed row."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: int = Field(
        description="Row identifier; wire-required but never read by verify logic.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: str = Field(
        description=(
            "Task grouping label (e.g. 'workplace_assistant_email'); wire-required routing/grouping "
            "metadata, unread by verify logic."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    environment_name: str = Field(
        description="Environment label ('workplace_assistant' in all committed rows); wire-required, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
