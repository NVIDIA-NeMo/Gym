# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The instruction a check grades, and how the judge is shown it (`judge.yaml` `instruction`)."""

from .contracts import InstructionContext, ResolvedCheck, SessionEvidence


def captured_instruction(check: ResolvedCheck, evidence: SessionEvidence) -> InstructionContext | None:
    """Prefer observed message/result carriers; never infer an instruction from its rubric."""
    instruction = check.instruction
    placement = check.provenance.get("placement") or {}
    at = instruction.at if instruction else placement.get("at")
    steps = evidence.steps
    carrier = instruction.carrier if instruction else None
    if at == "system_prompt":
        candidates = [step.text for step in steps if step.role == "system"]
        matching = [text for text in candidates if instruction and instruction.text in text]
        if matching:
            carrier = matching[0]
    elif at == "task_message":
        carrier = next((step.text for step in steps if step.role == "user"), carrier)
    elif at == "user_turn":
        index = evidence.continuation_start
        if index is not None and index < len(steps) and steps[index].role == "user":
            carrier = steps[index].text
    elif at == "tool_output":
        receipt = next((event for event in evidence.events if event.id == check.scope.event_id), None)
        if receipt is not None and steps[receipt.step_index].role == "tool":
            carrier = steps[receipt.step_index].output
    # A user message that arrives mid-session is itself the instruction.
    text = instruction.text if instruction else carrier if at == "user_turn" else None
    if not text:
        return None
    return InstructionContext(text=text, at=at, carrier=carrier)


def _window(carrier: str, index: int, length: int, settings: dict) -> str:
    """The text around the instruction in its carrier: whole up to whole_chars, else from before_chars before it to
    after_chars after it, each end moved to a line break within snap_chars, with the instruction marked [[ ]]."""
    if len(carrier) <= settings["whole_chars"]:
        first, last = 0, len(carrier)
    else:
        first = max(0, index - settings["before_chars"])
        last = min(len(carrier), index + length + settings["after_chars"])
        if first > 0:
            newline = carrier.find("\n", first, index)
            if newline != -1 and newline - first < settings["snap_chars"]:
                first = newline + 1
        if last < len(carrier):
            newline = carrier.rfind("\n", index + length, last)
            if newline != -1 and last - newline < settings["snap_chars"]:
                last = newline
    return (
        ("[...]\n" if first > 0 else "")
        + carrier[first:index]
        + "[["
        + carrier[index : index + length]
        + "]]"
        + carrier[index + length : last]
        + ("\n[...]" if last < len(carrier) else "")
    )


def instruction_parts(config: dict, instruction: InstructionContext) -> dict[str, str]:
    """The `instruction`, `where` and `weigh` parts of the instruction prompt: the carrier marked, whole, unmarked
    or bare (`judge.yaml` `instruction`)."""
    settings = config["instruction"]
    text, at, carrier = instruction.text, instruction.at, instruction.carrier or ""
    where = settings["where"]["tool"].replace("{tool}", at[5:]) if at.startswith("tool:") else settings["where"][at]
    kind, around = "bare", None
    if carrier.strip():
        index, length = carrier.find(text), len(text)
        if index < 0 and text.strip():
            index, length = carrier.find(text.strip()), len(text.strip())
        if index < 0:
            kind = "unmarked"
            around = (
                carrier if len(carrier) <= settings["whole_chars"] else carrier[: settings["whole_chars"]] + " [...]"
            )
        elif carrier.strip() == carrier[index : index + length].strip():
            kind = "whole"
        else:
            kind, around = "marked", _window(carrier, index, length, settings)
    line = settings[kind].replace("{where}", where)
    if around is not None:
        line = line.replace("{around}", around)
    return {"instruction": text, "where": line, "weigh": settings["weigh"] if around is not None else ""}
