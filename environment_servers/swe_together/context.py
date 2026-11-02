# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic reference view; private reasoning never enters the simulator."""

from nemo_gym.core import NeMoGymResponse


INCREMENTAL_NOTICE = (
    "\n\nIMPORTANT: Work incrementally. After completing each distinct "
    "sub-task (e.g., implementing one feature, fixing one bug, making one "
    "significant change), STOP and report what you did and what you plan "
    "to do next. Wait for user feedback before proceeding to the next "
    "sub-task. Do NOT implement everything in one go."
)


def project_turn(response: NeMoGymResponse, *, context_chars: int = 3000) -> tuple[str, str]:
    """Project ordered public output into activity and the final assistant report.

    Reasoning and tool arguments stay out of the simulator's allowed view.
    The ordinary Responses output is already scoped to this activation.
    """
    events = []
    for item in response.output:
        if item.type == "message":
            text = "\n".join(part.text for part in item.content if part.type == "output_text")
            if text.strip():
                events.append(("text", text.strip()))
        elif item.type == "function_call":
            events.append(("tool", item.name))
    final = next((i for i in range(len(events) - 1, -1, -1) if events[i][0] == "text"), None)
    activity = []
    for i, (kind, value) in enumerate(events):
        if kind == "tool":
            activity.append(f"[{i + 1}] tool_call({value})")
        elif i != final:
            activity.append(f"[{i + 1}] thinking: {value[:300]}" + ("…" if len(value) > 300 else ""))
    report = (
        f"[{final + 1}] agent: {events[final][1][:3000]}" if final is not None else "(no agent narration this turn)"
    )
    budget = max(500, context_chars) * 2
    return ("\n".join(activity) or "(no intermediate steps)")[-budget:], report
