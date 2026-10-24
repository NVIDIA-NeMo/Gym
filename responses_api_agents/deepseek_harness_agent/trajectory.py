# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project DSH's committed root-session messages; retain raw notifications separately."""

from typing import Any

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
    accumulate_response_usage,
)


def convert_events(
    notifications: list[dict[str, Any]], session_id: str
) -> tuple[list[NeMoGymResponseOutputItem], NeMoGymResponseUsage | None]:
    output = []
    usage = None
    for notification in notifications:
        payload = notification["payload"]
        if notification["method"] != "session.event" or payload.get("sessionId") != session_id:
            continue
        event = payload["event"]
        data = event["data"]
        if event["type"] not in {"assistant/message", "tool/result"}:
            continue
        for index, block in enumerate(data["message"]["content"]):
            item_id = f"dsh_{session_id}_{event['seq']}_{index}"
            if block["type"] == "text":
                output.append(
                    NeMoGymResponseOutputMessage(
                        id=item_id,
                        content=[NeMoGymResponseOutputText(text=block["text"], annotations=[])],
                    )
                )
            elif block["type"] == "reasoning":
                output.append(
                    NeMoGymResponseReasoningItem(
                        id=item_id, summary=[NeMoGymSummary(type="summary_text", text=block["text"])]
                    )
                )
            elif block["type"] == "tool-call":
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        id=item_id, call_id=block["id"], name=block["name"], arguments=block["arguments"]
                    )
                )
            elif block["type"] == "tool-result":
                if any(part["type"] != "text" for part in block["content"]):
                    raise ValueError("DSH currently supports text tool results only")
                texts = [part["text"] for part in block["content"]]
                output.append(NeMoGymFunctionCallOutput(call_id=block["toolCallId"], output="\n".join(texts)))
            else:
                raise ValueError(f"Unsupported DSH output block: {block['type']}")
        if counts := data.get("usage"):
            # DSH's inputTokens excludes cache reads and writes; Gym's includes them.
            input_tokens = counts["inputTokens"] + counts.get("cacheReadTokens", 0) + counts.get("cacheWriteTokens", 0)
            usage = accumulate_response_usage(
                usage,
                NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    output_tokens=counts["outputTokens"],
                    total_tokens=input_tokens + counts["outputTokens"],
                    input_tokens_details={"cached_tokens": counts.get("cacheReadTokens")},
                    output_tokens_details={"reasoning_tokens": counts.get("reasoningTokens")},
                ),
            )
    return output, usage
