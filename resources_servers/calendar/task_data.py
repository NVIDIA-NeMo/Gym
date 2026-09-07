# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the calendar server.

A single top-level task field; there is no verifier_metadata. The wire model
(``CalendarRunRequest``, app.py) types ``exp_cal_state`` only as ``dict[str, Any]``, but
``utils.py`` hard-indexes ``duration``/``min_time``/``max_time``/``constraint`` on every event
value, so that de-facto per-event schema is typed here (``ExpCalEvent``). The same rows are
committed for ``environments/calendar`` and ``environments/calendar_v2``, which route to this
server; this one schema covers all three data files.
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExpCalEvent(BaseModel):
    """Expected placement constraints for one calendar event.

    Mirrors the dict shape hard-indexed by ``grade_assistant_response`` / ``is_constraint_satisfied``
    in ``resources_servers/calendar/utils.py``.
    """

    model_config = ConfigDict(extra="allow")

    event_id: Optional[int] = Field(
        default=None,
        description=(
            "Event id duplicated from the enclosing key; matching is done on the KEY vs the model "
            "output's event_id, so this field itself is never read by the grader."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    duration: int = Field(
        description="Required event duration in minutes; must equal the scheduled event's duration exactly.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    constraint: Optional[str] = Field(
        description=(
            "Natural-language time constraint parsed by the grader: 'before <time>', 'after <time>', "
            "'at <time>' (event must START at exactly that time), or 'between <time> and <time>' "
            "(e.g. 'between 10am and 11:45am'). The KEY must be present (hard-indexed); null means only "
            "min_time/max_time bounds apply. Unrecognized constraint strings fall through to True (no-op)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    min_time: str = Field(
        description="Earliest allowed start, 'HH:MM' 24h format (e.g. '10:00').",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_time: str = Field(
        description="Latest allowed end, 'HH:MM' 24h format (e.g. '16:00').",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    exp_cal_state: Dict[str, ExpCalEvent] = Field(
        description=(
            "Expected calendar state keyed by STRINGIFIED event index ('0', '1', ...) matched against "
            "str(event['event_id']) parsed from the model's JSON output. An empty dict is meaningful: "
            "'no change expected' scores reward 1 when the model schedules nothing."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
