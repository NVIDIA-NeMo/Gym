# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re


def time_to_minutes(time_str: str) -> int:
    time_str = time_str.strip()

    if "am" in time_str or "pm" in time_str:
        # Assume 12hr time format
        if ":" not in time_str:
            if "am" in time_str:
                hour = int(time_str.replace("am", ""))
                return hour * 60 if hour != 12 else 0
            elif "pm" in time_str:
                hour = int(time_str.replace("pm", ""))
                return (hour * 60 if hour != 12 else 0) + 12 * 60
        else:
            # Handle format like "10:30am" or "2:15pm"
            if "am" in time_str:
                time_part = time_str.replace("am", "")
                hour, minute = map(int, time_part.split(":"))
                return (hour * 60 if hour != 12 else 0) + minute
            elif "pm" in time_str:
                time_part = time_str.replace("pm", "")
                hour, minute = map(int, time_part.split(":"))
                return ((hour * 60 if hour != 12 else 0) + 12 * 60) + minute

        raise ValueError(f"Unable to parse time: {time_str}")
    else:
        # Assume 24hr time format
        hours, minutes = map(int, time_str.split(":"))
        return hours * 60 + minutes


def minutes_to_time(minutes: int, format: str = "12hr") -> str:
    if format == "12hr":
        hours = minutes // 60
        mins = minutes % 60
        period = "am" if hours < 12 else "pm"
        display_hours = hours if hours <= 12 else hours - 12
        if display_hours == 0:
            display_hours = 12
        if mins == 0:
            return f"{display_hours}{period}"
        return f"{display_hours}:{mins:02d}{period}"
    elif format == "24hr":
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    else:
        raise ValueError(f"Invalid time format: {format}")


def extract_json_list(text: str) -> list:
    """Extract the first JSON list containing objects from a string."""
    pattern = r"\[(?:[^\[\]]|\{[^}]*\})*\{(?:[^\[\]]|\{[^}]*\})*\}(?:[^\[\]]|\{[^}]*\})*\]"

    # Try to find a match
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    return None


def is_event_conflicting(events, check_event, exclude_event=None):
    """Check if an event conflicts with existing events. Returns conflicting event or None."""
    # Parse start time
    event_start = time_to_minutes(check_event["start_time"])
    event_end = event_start + check_event["duration"]

    for existing in events:
        if exclude_event and existing == exclude_event:
            continue

        # Parse existing event time
        existing_start = time_to_minutes(existing["start_time"])
        existing_end = existing_start + existing["duration"]

        # Check for overlap
        if not (event_end <= existing_start or event_start >= existing_end):
            return True

    return False


def is_constraint_satisfied(event, exp_event):
    # Duration is incorrect
    if event["duration"] != exp_event["duration"]:
        return False

    # Event is outside the time window
    min_time = time_to_minutes(exp_event["min_time"])
    max_time = time_to_minutes(exp_event["max_time"])
    event_start = time_to_minutes(event["start_time"])
    event_end = event_start + event["duration"]
    if event_start < min_time or event_end > max_time:
        return False

    # Event is not scheduled at the correct time
    constraint = exp_event["constraint"]
    if constraint is None:
        return True
    elif constraint.startswith("before "):
        # Event must end before the specified time
        time_str = constraint.replace("before ", "")
        constraint_time = time_to_minutes(time_str)
        return event_end <= constraint_time

    elif constraint.startswith("after "):
        # Event must start after the specified time
        time_str = constraint.replace("after ", "")
        constraint_time = time_to_minutes(time_str)
        return event_start >= constraint_time

    elif constraint.startswith("between "):
        # Event must be between two times
        parts = constraint.replace("between ", "").split(" and ")
        if len(parts) != 2:
            raise ValueError(f"Invalid 'between' constraint format: {constraint}")

        time_x = time_to_minutes(parts[0])
        time_y = time_to_minutes(parts[1])

        # Event should start at or after time_x and end at or before time_y
        return event_start >= time_x and event_end <= time_y

    elif constraint.startswith("at "):
        # Event must start at the specified time
        time_str = constraint.replace("at ", "")
        constraint_time = time_to_minutes(time_str)
        return event_start == constraint_time
    return True


def grade_assistant_response(assistant_response, exp_cal_state, allow_no_json_list=False):
    # invalid response
    if "<think>" in assistant_response:
        return 0, "think_found"

    # valid response
    elif len(exp_cal_state) == 0:
        # No change expected
        return 1, "pass"
    else:
        # events were expected to be scheduled/changed
        try:
            cal_state = extract_json_list(assistant_response)
            if cal_state is None or len(cal_state) == 0:
                if allow_no_json_list:
                    return 1, "pass"
                else:
                    return 0, "no_json_list"

            events_dict = {}
            for event in cal_state:
                events_dict[str(event["event_id"])] = event

            # Assistant response has a different number of events than the expected calendar state
            if len(events_dict) != len(exp_cal_state):
                return 0, "different_number_of_events"

            # There are events with time conflicts
            for event in cal_state:
                if is_event_conflicting(cal_state, event, exclude_event=event):
                    return 0, "conflicting_events"

            # Events are not correctly scheduled
            for event_id in exp_cal_state.keys():
                if not is_constraint_satisfied(events_dict[event_id], exp_cal_state[event_id]):
                    return 0, "constraint_violated"
        except Exception:
            return 0, "error_in_grading"

    return 1, "pass"
