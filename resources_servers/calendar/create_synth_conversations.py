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
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from datasets import load_dataset
from openai import OpenAI
from prompts import GENERATE_EVENTS_PROMPT, USER_AGENT_PROMPT
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from utils import minutes_to_time


class Model:
    def __init__(self, model: str = "openai/gpt-oss-120b", response_json: bool = False, endpoint="nims"):
        """Initialize OpenAI API with the environment variable and other necessary parameters."""
        if endpoint == "nims":
            api_key = os.getenv("NIMS_API_KEY")
            base_url = "https://integrate.api.nvidia.com/v1"
        elif endpoint == "vllm":
            api_key = "EMPTY"
            base_url = "http://localhost:8000/v1"
        if not api_key:
            raise ValueError("API_KEY is not set in the .env file.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.response_json = response_json
        self.max_completion_tokens = 8192

    def generate(self, prompt: Any, temp: float = 0.6, reasoning_effort: str = None):
        """Generate a response using the NIMS model."""
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) and "role" in item and item["role"] in ["user", "assistant"] for item in prompt
        ):
            pass
        else:
            raise ValueError(
                "Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=temp,
            top_p=0.95,
            max_completion_tokens=self.max_completion_tokens,
            extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort is not None else None,
        )
        if self.response_json:
            return json.loads(response.choices[0].message.content)
        return response.choices[0].message.content


def _validate_events(events: list[dict], min_time: int = 600, max_time: int = 960) -> bool:
    """Validate events by checking for overlaps and ensuring all events are within the time window.

    Returns True if all events are valid (no overlaps and within time range), False otherwise.

    Args:
        events: List of event dictionaries with start_time and duration
        min_time: Minimum time in minutes from midnight
        max_time: Maximum time in minutes from midnight

    Returns:
        True if events are valid, False otherwise
    """

    def parse_event_times(event: dict) -> tuple[int, int]:
        """Parse event and return (start_time, end_time) in minutes from midnight."""
        start_time = event["start_time"]
        if isinstance(start_time, str):
            parts = start_time.split(":")
            start_time = int(parts[0]) * 60 + int(parts[1])

        duration = event["duration"]
        if isinstance(duration, str):
            duration = int(duration)

        end_time = start_time + duration
        return start_time, end_time

    # Check all events for time window validity and overlaps
    for i in range(len(events)):
        start_i, end_i = parse_event_times(events[i])

        # Check if event is outside the allowed time range
        if start_i < min_time or end_i > max_time:
            return False

        # Check for overlaps with other events
        for j in range(i + 1, len(events)):
            start_j, end_j = parse_event_times(events[j])

            # Two events overlap if one starts before the other ends
            # and ends after the other starts
            if not (end_i <= start_j or start_i >= end_j):
                return False  # Overlap detected

    return True  # All events are valid


def _generate_random_constraint(event: dict, min_time: int = 600, max_time: int = 960) -> str:
    # Parse start_time - could be string like "10:00" or int like 600 (minutes)
    start_time = event["start_time"]
    if isinstance(start_time, str):
        parts = start_time.split(":")
        start_time = int(parts[0]) * 60 + int(parts[1])

    # Parse duration - could be string or int
    duration = event["duration"]
    if isinstance(duration, str):
        duration = int(duration)

    end_time = start_time + duration

    def round_to_nice_time(minutes: int) -> int:
        """Round time to nearest 15-minute increment for cleaner constraints."""
        # Round to nearest 15 minutes
        return round(minutes / 15) * 15

    # Define constraint types
    constraint_types = ["before", "after", "between", "at"]
    constraint_type = random.choice(constraint_types)

    if constraint_type == "before":
        # Choose a time after the event ends
        # Add 15-90 minutes buffer after event ends
        buffer = random.randint(15, 90)
        time_x = end_time + buffer
        # Cap at max_time
        time_x = min(time_x, max_time)
        time_x = round_to_nice_time(time_x)  # Round to nice time
        return f"before {minutes_to_time(time_x)}"

    elif constraint_type == "after":
        # Choose a time before the event starts
        # Subtract 15-90 minutes buffer before event starts
        buffer = random.randint(15, 90)
        time_x = start_time - buffer
        time_x = max(time_x, min_time)
        time_x = round_to_nice_time(time_x)  # Round to nice time
        return f"after {minutes_to_time(time_x)}"

    elif constraint_type == "between":
        # Choose times that encompass the event
        # Start boundary: before event starts
        buffer_before = random.randint(30, 120)
        time_x = max(start_time - buffer_before, min_time)  # Not before min_time
        time_x = round_to_nice_time(time_x)  # Round to nice time

        # End boundary: after event ends
        buffer_after = random.randint(30, 120)
        time_y = min(end_time + buffer_after, max_time)  # Not after max_time
        time_y = round_to_nice_time(time_y)  # Round to nice time

        return f"between {minutes_to_time(time_x)} and {minutes_to_time(time_y)}"

    elif constraint_type == "at":
        # Specify the exact start time of the event
        return f"at {minutes_to_time(start_time)}"
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}")


def _check_constraint_satisfied(event: dict) -> bool:
    """Check if an event satisfies its constraint. Returns True if satisfied, False otherwise."""
    # Parse event times
    start_time = event["start_time"]
    if isinstance(start_time, str):
        parts = start_time.split(":")
        start_time = int(parts[0]) * 60 + int(parts[1])

    duration = event["duration"]
    if isinstance(duration, str):
        duration = int(duration)

    end_time = start_time + duration

    def parse_time_from_constraint(time_str: str) -> int:
        """Parse time string like '10am', '10:30am', '2pm' to minutes from midnight."""
        time_str = time_str.strip()

        # Handle format like "10am" or "10pm"
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

    constraint = event["constraint"]

    # Parse constraint and check
    if constraint.startswith("before "):
        # Event must end before the specified time
        time_str = constraint.replace("before ", "")
        constraint_time = parse_time_from_constraint(time_str)
        return end_time <= constraint_time

    elif constraint.startswith("after "):
        # Event must start after the specified time
        time_str = constraint.replace("after ", "")
        constraint_time = parse_time_from_constraint(time_str)
        return start_time >= constraint_time

    elif constraint.startswith("between "):
        # Event must be between two times
        parts = constraint.replace("between ", "").split(" and ")
        if len(parts) != 2:
            raise ValueError(f"Invalid 'between' constraint format: {constraint}")

        time_x = parse_time_from_constraint(parts[0])
        time_y = parse_time_from_constraint(parts[1])

        # Event should start at or after time_x and end at or before time_y
        return start_time >= time_x and end_time <= time_y

    elif constraint.startswith("at "):
        # Event must start at the specified time
        time_str = constraint.replace("at ", "")
        constraint_time = parse_time_from_constraint(time_str)
        return start_time == constraint_time

    else:
        raise ValueError(f"Unknown constraint format: {constraint}")


class Calendar:
    def __init__(self, min_time: int, max_time: int):
        self.min_time_24hr = minutes_to_time(min_time, format="24hr")
        self.max_time_24hr = minutes_to_time(max_time, format="24hr")
        self.events = {}

    def get_current_state(self) -> dict:
        """Return a copy of the current calendar state."""
        return {k: v.copy() for k, v in self.events.items()}

    def add_event(self, event_id: int, duration: int):
        self.events[event_id] = {
            "event_id": event_id,
            "duration": duration,
            "constraint": None,
            "min_time": self.min_time_24hr,
            "max_time": self.max_time_24hr,
        }

    def add_constraint(self, event_id: int, constraint: str):
        self.events[event_id]["constraint"] = constraint


def modify_constraint(constraint: str) -> str:
    """Modify the constraint to be more specific. This is because the LLM may not understand the constraint if it's not specific enough."""
    if "before" in constraint:
        return constraint.replace("before", "ends at or before")
    elif "after" in constraint:
        return constraint.replace("after", "starts at or after")
    elif "between" in constraint:
        return constraint.replace("between", "scheduled between")
    elif "at" in constraint:
        return constraint.replace("at", "scheduled at")
    else:
        return constraint


def _construct_user_agent_prompts_cal_states(
    events: list[dict], persona: str, model: Model, min_time: int = 540, max_time: int = 960
) -> tuple[list[str], list[dict]]:
    """Construct user agent prompts to converse with the assistant. We follow the following strategy:

    a. Randomize the order of the events.
    b. Start with a greeting and set the context of the conversation. Ask for help with organizing the day.
    c. Mention an event from the list. Ask to add it to the calendar. (LLM assisted). Maintain a list of events that have been mentioned.
    d. Append a fixed assistant message saying that the event has been added to the calendar.
    e. With prob 0.5, mention a constraint for an event that has been mentioned.(LLM assisted). Mark this constraint as satisfied. With prob 0.5, continue the conversation based on the persona (without asking any questions or scheduling any events).
    f. Append a fixed assistant message saying that the constraint has been added to the calendar if a constraint has been mentioned. If no constraint has been mentioned, append a fixed assistant messsage saying somethign generic to acknowledge the user's message (e.g. that sounds interesting!).
    g. Repeat (c), (d), (e), (f) until all events are mentioned.
    """
    # Fixed assistant response templates
    EVENT_ADDED_RESPONSES = [
        "I've added {event_name} to your calendar.",
        "Done! {event_name} has been scheduled.",
        "Got it, I've added {event_name} to your calendar.",
        "{event_name} has been added to your calendar.",
    ]

    CONSTRAINT_ADDED_RESPONSES = [
        "I'll make sure to schedule that {constraint}.",
        "Noted! I'll ensure it's scheduled {constraint}.",
        "Got it, I'll schedule that {constraint}.",
        "Perfect, I'll make sure that happens {constraint}.",
    ]

    AGENDA_OPTIONS = {
        "add_event": 'Ask to add the following event to calendar: "{event_name}". Mention the duration as {duration} minutes. Don\'t mention the exact time, just the duration. Additionally, mention the event id as {event_id} in brackets (e.g. (event id: 2))',
        "add_constraint": "Ask for the \"{event_name}\" event to be scheduled with the following constraint: {constraint}. (Note: The event is already in the calendar. You're just providing a time constraint for the event. -- Don't mention this explicitly. The LLM already knows that the event is in the calendar). Additionally, mention the event id as {event_id} in brackets (e.g. (event id: 2))",
        "continue_conversation": "Continue the conversation based on your persona. DO NOT ask any questions or try to schedule any events. Stating things about your work/day/feelings/hopes etc are fine. Don't talk about doing anything before/after the event as that might lead the assistant to schedule more events.",
    }
    calendar = Calendar(min_time=min_time, max_time=max_time)
    calendar_states = []
    # Smalltalk factor is the probability of continuing the conversation based on the persona (randomly chosen between 0.6 and 1)
    smalltalk_factor = random.uniform(0.6, 1.0)
    constraint_eagerness = random.uniform(0.6, 1.0)

    # Randomize event order
    shuffled_events = events.copy()

    # Initialize conversation history
    conversation = []
    user_intents = []

    # Track mentioned events and constraints
    mentioned_events = []  # Events that have been added
    mentioned_constraints = set()  # Constraint indices that have been mentioned

    # Helper function to get last N messages for conversation history
    def get_conversation_history(n=4):
        recent_messages = conversation[-n:] if len(conversation) >= n else conversation
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages])
        return history_str if history_str else "No previous messages"

    # Helper function to generate user message with LLM
    def generate_user_message(agenda: str, reasoning_effort: str = None) -> str:
        history = get_conversation_history(4)
        prompt = USER_AGENT_PROMPT.format(persona=persona, agenda=agenda, partial_conversation_history=history)
        return model.generate(prompt, reasoning_effort=reasoning_effort)["USER_MESSAGE"]

    # Step c-g: Add events with interleaved constraints
    for _, event in enumerate(shuffled_events):
        # Step c: Generate user message to add event
        add_event_agenda = AGENDA_OPTIONS["add_event"].format(
            event_name=event["event_name"], duration=event["duration"], event_id=event["event_id"]
        )
        user_message = generate_user_message(add_event_agenda)
        user_intents.append("add_event")
        conversation.append({"role": "user", "content": user_message})
        calendar.add_event(event["event_id"], event["duration"])
        calendar_states.append(calendar.get_current_state())

        # Step d: Fixed assistant response confirming event added
        assistant_response = random.choice(EVENT_ADDED_RESPONSES).format(
            event_name=event["event_name"],
        )
        conversation.append({"role": "assistant", "content": assistant_response})

        # Mark event as mentioned
        mentioned_events.append(event)
        unmentioned_constraints = [idx for idx, e in enumerate(mentioned_events) if idx not in mentioned_constraints]

        # Step e-f: With 80% probability, add constraint
        # But only if we have mentioned events and not all constraints are mentioned yet
        if (len(unmentioned_constraints) > 0) and (random.random() < constraint_eagerness):
            constraint_idx = random.choice(unmentioned_constraints)
            constraint_event = mentioned_events[constraint_idx]
            constraint = modify_constraint(constraint_event["constraint"])

            add_constraint_agenda = AGENDA_OPTIONS["add_constraint"].format(
                event_name=constraint_event.get("short_hand_name", constraint_event["event_name"]),
                # event_name=constraint_event["event_name"],
                event_id=constraint_event["event_id"],
                constraint=constraint,
            )
            user_message = generate_user_message(add_constraint_agenda)
            conversation.append({"role": "user", "content": user_message})
            user_intents.append("add_constraint")

            calendar.add_constraint(constraint_event["event_id"], constraint_event["constraint"])
            calendar_states.append(calendar.get_current_state())

            # Fixed assistant response for constraint
            assistant_response = random.choice(CONSTRAINT_ADDED_RESPONSES).format(
                constraint=constraint_event["constraint"]
            )
            conversation.append({"role": "assistant", "content": assistant_response})

            mentioned_constraints.add(constraint_idx)
        if (random.random() < smalltalk_factor) and (len(unmentioned_constraints) > 0):
            # Continue conversation based on persona
            continue_agenda = AGENDA_OPTIONS["continue_conversation"]
            user_message = generate_user_message(continue_agenda, reasoning_effort="high")
            conversation.append({"role": "user", "content": user_message})
            user_intents.append("continue_conversation")

            # Calendar needs to be mentioned in each response acc to sys prompt.
            calendar_states.append(calendar.get_current_state())

            # Generic acknowledgment
            assistant_response = "Okay."
            conversation.append({"role": "assistant", "content": assistant_response})

    # Ensure all constraints are mentioned
    unmentioned_constraints = [idx for idx in range(len(mentioned_events)) if idx not in mentioned_constraints]

    for constraint_idx in unmentioned_constraints:
        constraint_event = mentioned_events[constraint_idx]
        constraint = modify_constraint(constraint_event["constraint"])
        add_constraint_agenda = AGENDA_OPTIONS["add_constraint"].format(
            event_name=constraint_event.get("short_hand_name", constraint_event["event_name"]),
            # event_name=constraint_event["event_name"],
            event_id=constraint_event["event_id"],
            constraint=constraint,
        )
        user_message = generate_user_message(add_constraint_agenda)
        conversation.append({"role": "user", "content": user_message})
        user_intents.append("add_constraint")
        calendar.add_constraint(constraint_event["event_id"], constraint_event["constraint"])
        calendar_states.append(calendar.get_current_state())

        # Fixed assistant response for constraint
        assistant_response = random.choice(CONSTRAINT_ADDED_RESPONSES).format(
            constraint=constraint_event["constraint"]
        )
        conversation.append({"role": "assistant", "content": assistant_response})

        mentioned_constraints.add(constraint_idx)

    # Extract only user messages
    user_prompts = [msg["content"] for msg in conversation if msg["role"] == "user"]
    calendar_states.append(calendar.get_current_state())

    return user_prompts, user_intents, calendar_states, smalltalk_factor, constraint_eagerness


def _populate_event_times(events: list[dict], min_time: int, max_time: int) -> list[dict]:
    """Populate start_time and duration fields for events with only event_name and short_hand_name.

    Strategy:
    1. Assign random durations [30,45,60,75,90] to all events
    2. Compute total_duration (sum of all event durations)
    3. Compute time_window = max_time - min_time
    4. Compute free_time = time_window - total_duration
    5. If free_time < 0, resample durations until free_time >= 0
    6. Compute target_free_time = min(15 * int(n_events / 2), free_time)
    7. Compute unallocated_time = free_time - target_free_time
    8. Cycle through events and add +15 to durations until unallocated_time == 0
    9. Pack all events together starting from min_time
    10. Randomly distribute gaps by pushing events forward

    Args:
        events: List of event dictionaries with event_name and short_hand_name
        min_time: Minimum time in minutes from midnight
        max_time: Maximum time in minutes from midnight

    Returns:
        List of events with populated start_time and duration fields
    """
    if not events:
        return events

    # Helper function to format time to HH:MM string
    def format_time(minutes: int) -> str:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    # Possible durations in 15-min increments (30, 45, 60, 75, 90 minutes)
    possible_durations = [30, 45, 60, 75, 90]
    n_events = len(events)

    # Step 1: Assign random durations to all events
    for event in events:
        event["duration"] = random.choice(possible_durations)

    # Step 2: Compute total_duration
    total_duration = sum(event["duration"] for event in events)

    # Step 3: Compute time_window
    time_window = max_time - min_time

    # Step 4: Compute free_time
    free_time = time_window - total_duration

    # Step 5: If free_time < 0, resample durations until free_time >= 0
    max_resampling_attempts = 100
    attempt = 0
    while free_time < 0 and attempt < max_resampling_attempts:
        for event in events:
            event["duration"] = random.choice(possible_durations)
        total_duration = sum(event["duration"] for event in events)
        free_time = time_window - total_duration
        attempt += 1

    # If still negative after max attempts, return empty list
    if free_time < 0:
        return []

    # Step 6: Compute target_free_time
    target_free_time = min(15 * int(n_events / 2), free_time)

    # Step 7: Compute unallocated_time
    unallocated_time = free_time - target_free_time

    # Step 8: Cycle through events and add +15 to durations until unallocated_time == 0
    event_idx = 0
    while unallocated_time > 0:
        events[event_idx]["duration"] += 15
        unallocated_time -= 15
        event_idx = (event_idx + 1) % n_events

    # Step 9: Pack all events together starting from min_time
    current_time = min_time
    for event in events:
        event["start_time"] = format_time(current_time)
        current_time += event["duration"]

    # Step 10: Randomly distribute gaps
    # For (target_free_time // 15) iterations, pick a random event and push all events after it by 15 min
    num_gap_insertions = target_free_time // 15
    for _ in range(num_gap_insertions):
        # Pick a random event index (0 to n-2, since pushing after last event doesn't create a gap)
        if n_events > 1:
            gap_position = random.randint(0, n_events - 2)
            # Push all events after gap_position by 15 minutes
            for i in range(gap_position + 1, n_events):
                # Parse current start time
                parts = events[i]["start_time"].split(":")
                start_minutes = int(parts[0]) * 60 + int(parts[1])
                # Add 15 minutes
                new_start_minutes = start_minutes + 15
                events[i]["start_time"] = format_time(new_start_minutes)

    return events


def process_persona(
    persona: str,
    model: Model,
    sample_id: int,
    max_retries: int = 3,
    n_events: int = 7,
    min_time: int = 600,
    max_time: int = 960,
) -> Optional[dict]:
    """Process a single persona and generate calendar events with constraints.

    Args:
        persona: The persona description
        model: The Model instance to use
        max_retries: Maximum number of retries if events overlap
        n_events: Number of events to generate

    Returns:
        Dictionary with persona and events, or None if failed after retries
    """
    min_time_24hr = minutes_to_time(min_time, format="24hr")
    max_time_24hr = minutes_to_time(max_time, format="24hr")
    for _ in range(max_retries):
        try:
            prompt = GENERATE_EVENTS_PROMPT.format(persona=persona, n_events=n_events)
            events = model.generate(prompt, reasoning_effort="high")
            events = _populate_event_times(events, min_time=min_time, max_time=max_time)

            # Validate events (check for overlaps and time window)
            if not _validate_events(events, min_time=min_time, max_time=max_time):
                print("validation check failed. Retrying...")
                print(f"Events: {events}")
                exit(0)
                continue

            # Generate constraints and validate
            for event in events:
                event["constraint"] = _generate_random_constraint(event, min_time=min_time, max_time=max_time)
                event["min_time"] = min_time_24hr
                event["max_time"] = max_time_24hr
                assert _check_constraint_satisfied(event), (
                    f"Event {event['event_name']} does not satisfy constraint: {event['constraint']}"
                )

            # Generate user agent prompts
            random.shuffle(events)
            for event_id, event in enumerate(events):
                event["event_id"] = event_id
            user_prompts, user_intents, calendar_states, smalltalk_factor, constraint_eagerness = (
                _construct_user_agent_prompts_cal_states(events, persona, model, min_time=min_time, max_time=max_time)
            )

            return {
                "sample_id": sample_id,
                "persona": persona,
                "events": events,
                "expected_calendar_states": calendar_states,
                "user_prompts": user_prompts,
                "user_intents": user_intents,
                "smalltalk_factor": smalltalk_factor,
                "constraint_eagerness": constraint_eagerness,
            }
        except Exception as e:
            print(f"Error processing persona: {e}")
            continue

    return None


def get_personas(offset, n_samples, ds_name="nvidia/Nemotron-Personas-USA", n_traits=None):
    personas = []
    if ds_name == "nvidia/Nemotron-Personas-USA":
        ds = load_dataset(ds_name, split=f"train[{offset}:{offset + n_samples}]")
        for sample in ds:
            persona = f"PERSONA: {sample['professional_persona']}"
            personas.append(persona)
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    return personas


def main():
    parser = argparse.ArgumentParser(description="Generate calendar dataset from PersonaHub")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start from")
    parser.add_argument("--n-workers", type=int, default=100, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="./data/train.json", help="Output file path")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Model to use")
    parser.add_argument("--endpoint", type=str, default="vllm", choices=["vllm", "nims"], help="Endpoint to use")
    parser.add_argument("--min-time", type=int, default=600, help="Minimum time for events (default: 10am)")
    parser.add_argument("--max-time", type=int, default=960, help="Maximum time for events (default: 4pm)")
    parser.add_argument("--n-events", type=int, default=7, help="Number of events to generate")
    parser.add_argument(
        "--ds-name",
        type=str,
        default="nvidia/Nemotron-Personas-USA",
        choices=["nvidia/Nemotron-Personas-USA"],
        help="Dataset to use",
    )
    args = parser.parse_args()

    personas = get_personas(args.offset, args.n_samples, ds_name=args.ds_name)
    model = Model(args.model, response_json=True, endpoint=args.endpoint)

    ds_out = []

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Generating conversations...", total=len(personas))

        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            future_to_persona = {
                executor.submit(
                    process_persona,
                    persona,
                    model,
                    sample_id=i + args.offset,
                    n_events=args.n_events,
                    min_time=args.min_time,
                    max_time=args.max_time,
                )
                for i, persona in enumerate(personas)
            }

            for future in as_completed(future_to_persona):
                result = future.result()
                if result is not None:
                    ds_out.append(result)
                progress.update(task, advance=1)

    print(f"\nSuccessfully generated {len(ds_out)} out of {args.n_samples} samples")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ds_out, f, indent=2)

    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
