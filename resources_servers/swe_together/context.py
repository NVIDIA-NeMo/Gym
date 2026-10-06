# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic reference view; private reasoning never enters the simulator."""

import json

from nemo_gym.interactive_agent_types import AgentActivationObservation


INCREMENTAL_NOTICE = (
    "\n\nIMPORTANT: Work incrementally. After completing each distinct "
    "sub-task (e.g., implementing one feature, fixing one bug, making one "
    "significant change), STOP and report what you did and what you plan "
    "to do next. Wait for user feedback before proceeding to the next "
    "sub-task. Do NOT implement everything in one go."
)


def project_turn(
    observation: AgentActivationObservation, *, raw_history: list[str], context_chars: int = 3000
) -> tuple[str, str]:
    """Match the pinned OpenCode wrapper's activity/final-report partition."""
    budget = max(500, context_chars)
    events = []
    step = 0
    opened = False
    for event in observation.events:
        if event.kind == "step_start":
            step += 1
            opened = True
        elif event.kind == "step_finish":
            opened = False
        elif opened and event.kind == "text" and (event.text or "").strip():
            events.append(("text", step, event.text.strip()))
        elif opened and event.kind == "tool_use":
            events.append(("tool", step, (event.name or "?", event.result)))
    if not events:
        # Upstream's raw fallback includes previous activations. Preserve it as
        # a separate, explicitly supplied public log; adapters redact reasoning.
        tail = "\n".join(raw_history)[-budget:] if raw_history else "(nothing yet)"
        return tail, tail
    final = next((i for i in range(len(events) - 1, -1, -1) if events[i][0] == "text"), None)
    activity = []
    for i, (kind, step, payload) in enumerate(events):
        if kind == "text" and i != final:
            snippet = payload if len(payload) <= 300 else payload[:300] + "…"
            activity.append(f"[{step}] thinking: {snippet}")
        elif kind == "tool":
            activity.append(f"[{step}] tool_call({payload[0]})")
    trajectory = "\n".join(activity) or "(no intermediate steps)"
    if len(trajectory) > budget * 2:
        trajectory = "…[earlier steps elided]…\n" + trajectory[-budget * 2 :]
    report = "(no agent narration this turn)"
    if final is not None:
        _, step, text = events[final]
        report = f"[{step}] agent: {text[:3000]}"
    else:
        for kind, step, payload in reversed(events):
            if kind == "tool" and payload[1]:
                result = json.dumps(payload[1]) if isinstance(payload[1], dict) else str(payload[1])
                report = f"[{step}] result: {result[:500]}"
                break
    return trajectory, report
