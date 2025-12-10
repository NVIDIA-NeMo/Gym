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
from asyncio import run

from nemo_gym.server_utils import ServerClient


async def main():
    """
    Demo client for the Calendar Resource Server.

    The server verifies whether an assistant correctly schedules calendar events
    by checking if the assistant's response contains a valid JSON list of events
    that satisfy all time constraints and have no conflicts.
    """
    server_client = ServerClient.load_from_global_config()

    print("=" * 80)
    print("Calendar Resource Server Demo")
    print("=" * 80)
    print("\nThis demo shows how the resource server grades assistant responses")
    print("for calendar scheduling tasks with time constraints.\n")

    # Example 1: Valid response - correctly scheduled events
    print("\n" + "=" * 80)
    print("Example 1: Valid Calendar Response (Should get reward=1)")
    print("=" * 80)

    exp_cal_state_1 = {
        "1": {
            "event_id": 1,
            "event_name": "Team Meeting",
            "duration": 60,
            "constraint": "at 10am",
            "min_time": "10am",
            "max_time": "4pm",
        },
        "2": {
            "event_id": 2,
            "event_name": "Lunch",
            "duration": 60,
            "constraint": "after 12pm",
            "min_time": "10am",
            "max_time": "4pm",
        },
    }

    assistant_response_1 = """
    I've scheduled both events for you:
    [
        {"event_id": 1, "event_name": "Team Meeting", "start_time": "10am", "duration": 60},
        {"event_id": 2, "event_name": "Lunch", "start_time": "12pm", "duration": 60}
    ]
    """

    print(f"\nExpected calendar state: {json.dumps(exp_cal_state_1, indent=2)}")
    print(f"\nAssistant response: {assistant_response_1}")

    # Call the server's verify endpoint
    response = await server_client.post(
        server_name="calendar",
        url_path="/verify",
        json={
            "responses_create_params": {"input": []},
            "response": {
                "id": "response_1",
                "created_at": 1.0,
                "model": "demo_model",
                "object": "response",
                "output": [
                    {
                        "id": "message_1",
                        "content": [
                            {
                                "annotations": [],
                                "text": assistant_response_1,
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "exp_cal_state": exp_cal_state_1,
        },
    )
    result_1 = await response.json()
    print(f"\n✓ Reward: {result_1['reward']} (1=success, 0=failure)")

    # Example 2: Invalid response - constraint violation
    print("\n" + "=" * 80)
    print("Example 2: Constraint Violation (Should get reward=0)")
    print("=" * 80)

    assistant_response_2 = """
    Here's your schedule:
    [
        {"event_id": 1, "event_name": "Team Meeting", "start_time": "11am", "duration": 60},
        {"event_id": 2, "event_name": "Lunch", "start_time": "12pm", "duration": 60}
    ]
    """

    print(f"\nExpected calendar state: {json.dumps(exp_cal_state_1, indent=2)}")
    print(f"\nAssistant response: {assistant_response_2}")
    print("\nNote: Team Meeting should be 'at 10am' but was scheduled at 11am")

    response = await server_client.post(
        server_name="calendar",
        url_path="/verify",
        json={
            "responses_create_params": {"input": []},
            "response": {
                "id": "response_2",
                "created_at": 1.0,
                "model": "demo_model",
                "object": "response",
                "output": [
                    {
                        "id": "message_2",
                        "content": [
                            {
                                "annotations": [],
                                "text": assistant_response_2,
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "exp_cal_state": exp_cal_state_1,
        },
    )
    result_2 = await response.json()
    print(f"\n✗ Reward: {result_2['reward']} (Expected 0 due to constraint violation)")

    # Example 3: Invalid response - time conflict
    print("\n" + "=" * 80)
    print("Example 3: Time Conflict (Should get reward=0)")
    print("=" * 80)

    exp_cal_state_3 = {
        "1": {
            "event_id": 1,
            "event_name": "Meeting",
            "duration": 90,
            "constraint": None,
            "min_time": "10am",
            "max_time": "4pm",
        },
        "2": {
            "event_id": 2,
            "event_name": "Call",
            "duration": 60,
            "constraint": None,
            "min_time": "10am",
            "max_time": "4pm",
        },
    }

    assistant_response_3 = """
    [
        {"event_id": 1, "event_name": "Meeting", "start_time": "10am", "duration": 90},
        {"event_id": 2, "event_name": "Call", "start_time": "10:30am", "duration": 60}
    ]
    """

    print(f"\nExpected calendar state: {json.dumps(exp_cal_state_3, indent=2)}")
    print(f"\nAssistant response: {assistant_response_3}")
    print("\nNote: Meeting (10am-11:30am) conflicts with Call (10:30am-11:30am)")

    response = await server_client.post(
        server_name="calendar",
        url_path="/verify",
        json={
            "responses_create_params": {"input": []},
            "response": {
                "id": "response_3",
                "created_at": 1.0,
                "model": "demo_model",
                "object": "response",
                "output": [
                    {
                        "id": "message_3",
                        "content": [
                            {
                                "annotations": [],
                                "text": assistant_response_3,
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "exp_cal_state": exp_cal_state_3,
        },
    )
    result_3 = await response.json()
    print(f"\n✗ Reward: {result_3['reward']} (Expected 0 due to time conflict)")

    # Example 4: Invalid response - contains thinking tags
    print("\n" + "=" * 80)
    print("Example 4: Invalid Format - Thinking Tags (Should get reward=0)")
    print("=" * 80)

    assistant_response_4 = """
    <think>Let me schedule these events...</think>
    [
        {"event_id": 1, "event_name": "Team Meeting", "start_time": "10am", "duration": 60},
        {"event_id": 2, "event_name": "Lunch", "start_time": "12pm", "duration": 60}
    ]
    """

    print(f"\nAssistant response: {assistant_response_4}")
    print("\nNote: Response contains <think> tags which are not allowed")

    response = await server_client.post(
        server_name="calendar",
        url_path="/verify",
        json={
            "responses_create_params": {"input": []},
            "response": {
                "id": "response_4",
                "created_at": 1.0,
                "model": "demo_model",
                "object": "response",
                "output": [
                    {
                        "id": "message_4",
                        "content": [
                            {
                                "annotations": [],
                                "text": assistant_response_4,
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "exp_cal_state": exp_cal_state_1,
        },
    )
    result_4 = await response.json()
    print(f"\n✗ Reward: {result_4['reward']} (Expected 0 due to thinking tags)")

    # Example 5: Complex constraints - between
    print("\n" + "=" * 80)
    print("Example 6: Complex Constraint - Between (Should get reward=1)")
    print("=" * 80)

    exp_cal_state_5 = {
        "1": {
            "event_id": 1,
            "event_name": "Workout",
            "duration": 45,
            "constraint": "between 11am and 1pm",
            "min_time": "10am",
            "max_time": "4pm",
        }
    }

    assistant_response_5 = """
    [{"event_id": 1, "event_name": "Workout", "start_time": "11:30am", "duration": 45}]
    """

    print(f"\nExpected calendar state: {json.dumps(exp_cal_state_5, indent=2)}")
    print(f"\nAssistant response: {assistant_response_5}")
    print("\nNote: Event must be between 11am and 1pm (11:30am-12:15pm satisfies this)")

    response = await server_client.post(
        server_name="calendar",
        url_path="/verify",
        json={
            "responses_create_params": {"input": []},
            "response": {
                "id": "response_5",
                "created_at": 1.0,
                "model": "demo_model",
                "object": "response",
                "output": [
                    {
                        "id": "message_5",
                        "content": [
                            {
                                "annotations": [],
                                "text": assistant_response_5,
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "exp_cal_state": exp_cal_state_5,
        },
    )
    result_5 = await response.json()
    print(f"\n✓ Reward: {result_5['reward']} (Expected 1)")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("- The server grades assistant responses based on calendar scheduling correctness")
    print("- Responses must contain valid JSON lists of events (unless no events expected)")
    print("- Events must satisfy all time constraints (at, before, after, between)")
    print("- Events must not have time conflicts")
    print("- Events must be within the specified time window")
    print("- Thinking tags (<think>) are not allowed in responses")


if __name__ == "__main__":
    run(main())
