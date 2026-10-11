# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Togetherbench/SWE-Together, Apache-2.0, revision 891d19eb4b3a64a47c3d49bbd066a311e0133254.


import json
from pathlib import Path


def load_analysis(task_dir: Path) -> dict:
    analysis_path = task_dir / "analysis.json"
    if analysis_path.exists():
        return json.loads(analysis_path.read_text())
    return {}


def load_user_messages(task_dir: Path, analysis: dict) -> list[str]:
    """Load the recorded user messages from whichever artifact has them."""
    candidates = analysis.get("user_messages")
    if isinstance(candidates, list):
        return [msg for msg in candidates if isinstance(msg, str) and not msg.startswith("[Request interrupted")]

    session_path = task_dir / "original_session.json"
    if not session_path.exists():
        return []

    # Prefixes that mark Claude Code system/tooling messages, not genuine user turns
    _SYSTEM_PREFIXES = (
        "[Request interrupted",
        "<local-command-caveat>",
        "<command-name>",
        "<command-message>",
        "<command-args>",
        "<local-command-stdout>",
        "<task-",
        "Base directory",
    )

    session = json.loads(session_path.read_text())
    return [
        msg.get("content", "")
        for msg in session.get("messages", [])
        if msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and not msg["content"].startswith(_SYSTEM_PREFIXES)
    ]


def _compute_message_guidance(gt_count: int) -> tuple[int, int]:
    """Compute a suggested message range based on ground-truth count.

    Returns (low, high) — a guidance range of [GT*0.5, GT*1.5] that is
    injected into the user sim's system prompt as a soft target. No hard
    cap is enforced; the user sim decides based on the session context.
    """
    import math

    low = max(1, math.ceil(gt_count * 0.5))
    high = max(low + 1, math.ceil(gt_count * 1.5))
    return low, high


def session_analysis(task_dir: Path) -> tuple[str, list[str]]:
    messages = load_user_messages(task_dir, load_analysis(task_dir))
    gt_count = len(messages)
    msg_low, msg_high = _compute_message_guidance(gt_count)
    guidance_note = (
        f"\n\n## Message Guidance (auto-generated)\n"
        f"The real user sent {gt_count} messages in the original session. "
        f"Aim for **{msg_low}–{msg_high} messages** total. "
        f"This is a soft target — send fewer if the agent handles everything "
        f"well, send more if it needs correction. Do NOT treat any cap in "
        f"the session analysis above as a hard limit; use this range instead.\n\n"
        f"## Trigger Interpretation (auto-generated)\n"
        f"The agent works incrementally and reports after each sub-task. "
        f"When evaluating trigger conditions from the session analysis above, "
        f"apply them broadly:\n"
        f"- If a trigger says 'ONLY if agent has X but not Y', also fire "
        f"if the agent has completed both X and Y but Y has issues.\n"
        f"- If the agent reports completing a sub-task, check whether the "
        f"next ground-truth message in sequence is relevant and send it.\n"
        f"- Do NOT skip a turn just because the agent already moved past "
        f"the exact intermediate state described in the trigger. The agent "
        f"may have done it incorrectly.\n"
        f"- Prioritize sending ground-truth messages in order. If the agent's "
        f"progress maps to turn N in the session analysis, send turn N's "
        f"message even if the trigger condition is not a perfect match."
    )
    return (task_dir / "user_simulation_prompt.md").read_text() + guidance_note, messages
