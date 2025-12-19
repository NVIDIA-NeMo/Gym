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

GENERATE_EVENTS_PROMPT = """I will give you a PERSONA. I want you to give me a **packed** calendar with exactly {n_events} events for someone with this persona either for their personal calendar or an event that they are organizing

I want the output in a JSON array format. With each entry in the list consisting of the following fields:

{{

"event_name": "name of the event/meeting (e.g. Online museum tour: Roman artifacts related to early Christianity)",

"short_hand_name": "short hand way of referring to the meeting (e.g. the museum tour)",

}}

Additional Info

1. The calendar is for a single day

2. Only provide the JSON array in your response and nothing else.

3. Ensure that the shorthand names uniquely refer to a single entry in the calendar. There shouldn't be any ambiguity in terms of which name refers to which entry. (e.g. if you have a shorthand name called "museum tour", it might be confusing if you had multiple museum tours in the calendar)

4. Feel free to include 1:1 meetings with other people (use your imagination for names)

5. Don't include events that implicitly mention time of day (e.g. "lunch break", "afternoon coffee", "end of day meeting", "morning briefing", etc.)


PERSONA: {persona}"""


USER_AGENT_PROMPT = """
ROLE: You are roleplaying as a user with the following persona.
PERSONA: {persona}

TASK: Generate the next user message in a conversation with a calendar assistant.

PARTIAL CONVERSATION HISTORY:
{partial_conversation_history}

YOUR GOAL FOR THIS MESSAGE:
{agenda}

REQUIREMENTS:
1. Stay in character with your persona
2. Sound natural and conversational (contractions, casual language OK)
3. Only include information relevant to the AGENDA
4. Don't mention any days of the week in your messages.
6. Respond ONLY with valid JSON in this exact format:
{{
"USER_MESSAGE": "your message here"
}}
"""

SYSTEM_PROMPT = """You are a helpful assistant. If asked to help with organizing the user’s calendar, adhere to the following rules

1. Calendar Format: Only print the calendar in a json list format with each entry containing the following fields: “event_id” (unique integer id), “event_name” (string), “start_time”(string in 24-hour HH:MM format), “duration” (integer - in minutes)
2. Initial State: Assume the calendar is empty at the start of the conversation.
3. User Constraints: Honor all time-based constraints specified by the user for an event (e.g., "before 11 am," "after 2 pm", "at 11:15am"). These constraints are permanent and **must be maintained during any rescheduling**.
4. Conflict Resolution: Ensure that there are no conflicts (overlapping events) in the calendar. If a new event conflicts with an existing one, automatically reschedule events to the next available slot that respects all active constraints. Do not ask for user confirmation.
5. Interpret constraints like "before 11 am" as "ends at or before 11 am", "after 2 pm" as "starts at or after 2 pm".
6. General Queries: For any requests not related to the calendar, respond as a helpful assistant.
7. Ensure that all events are between {start_time_str} and {end_time_str}.
8. Display the calendar in the json list format in every responses. e.g.[{{"event_id": 0, "event_name": "event name", "start_time": "10:00", "duration": 60}}, {{"event_id": 1, "event_name": "event name", "start_time": "11:00", "duration": 60}}]"""
