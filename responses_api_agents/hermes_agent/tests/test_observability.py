# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from responses_api_agents.hermes_agent.observability import normalize_hermes_messages


def test_normalize_hermes_messages_preserves_conversation_order():
    items = normalize_hermes_messages(
        [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "working",
                "reasoning": "checking the workspace",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "terminal", "arguments": {"command": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        id_prefix="root",
    )

    assert [item.type for item in items] == [
        "message",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[1].summary[0].text == "checking the workspace"
    assert items[3].arguments == '{"command": "pwd"}'
    assert items[4].call_id == "call-1"
    assert items[5].content[0].text == "done"


def test_normalize_hermes_messages_keeps_user_images_in_order():
    image_uri = "data:image/png;base64,image-bytes"
    items = normalize_hermes_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": image_uri, "detail": "high"}},
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_uri}}]},
            # Hermes also reads a bare-URL image_url and a Responses input_image part.
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image_uri},
                    {"type": "input_image", "image_url": image_uri, "detail": "low"},
                ],
            },
        ]
    )
    first, second, third = [item.model_dump(mode="json")["content"] for item in items]
    assert [part["type"] for part in first] == ["input_text", "input_image", "input_text"]
    assert first[0]["text"] == "before"
    assert first[1]["image_url"] == image_uri
    assert first[1]["detail"] == "high"
    assert first[2]["text"] == "after"
    assert second[0]["image_url"] == image_uri
    assert second[0]["detail"] == "auto"
    assert [(part["type"], part["image_url"], part["detail"]) for part in third] == [
        ("input_image", image_uri, "auto"),
        ("input_image", image_uri, "low"),
    ]
