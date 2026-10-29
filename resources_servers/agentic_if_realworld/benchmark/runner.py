# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First-complete selection and independent judge recovery, with crash-safe journals."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from .core import PHASES, Dataset, digest, judge, read, unscored, write
from .transport import verified_response


async def run_cell(
    dataset: Dataset,
    identifier: str,
    *,
    directory: Path,
    settings: dict,
    generate: Callable[[dict, int, int, str], Awaitable[dict]],
    verify: Callable[..., Awaitable[dict]],
    resume: bool = False,
    validate_only: bool = False,
) -> dict | None:
    """Validate saved state before resuming; validation never creates or dispatches work."""
    fingerprint = judge.sha(
        {
            "dataset": dataset.report,
            "settings": settings,
            "policy": judge.VERSION,
            "phases": PHASES,
        }
    )
    path = directory / "cells" / (judge.sha(identifier) + ".json")
    if path.exists():
        if not resume:
            raise ValueError("Existing journal requires explicit resume")
        state = read(path)
        if state["item_id"] != identifier or state["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration fingerprint mismatch")
    else:
        if validate_only:
            return None
        state = {"item_id": identifier, "fingerprint": fingerprint, "attempts": []}
        write(path, state)
    for attempt in state["attempts"]:
        if attempt["status"] == "started":
            raise ValueError("Ambiguous in-flight generation; reconcile receipt before resume")
        if "receipt_path" in attempt:
            receipt = read(Path(attempt["receipt_path"]))
            if digest(Path(attempt["receipt_path"])) != attempt["receipt_sha256"]:
                raise ValueError("Saved generation transport receipt changed")
            if receipt["status"] == "started":
                raise ValueError("Ambiguous in-flight generation receipt; reconcile receipt before resume")
            if judge.sha(receipt["request_body"]) != receipt["request_sha256"]:
                raise ValueError("Saved generation request binding changed")
            if any(receipt["request_body"].get(key) != value for key, value in dataset.request(identifier).items()):
                raise ValueError("Saved generation input differs from frozen task")
            for raw in receipt["raw_attempts"]:
                if digest(Path(raw["path"])) != raw["sha256"]:
                    raise ValueError("Raw transport binding changed")
            if receipt["status"] == "generated":
                verified_response(receipt)
    if "selected" in state:
        if judge.sha(state["selected"]) != state["selected_sha256"]:
            raise ValueError("Saved selected response changed")
        if (
            "receipt_path" in state["selected"]
            and digest(Path(state["selected"]["receipt_path"])) != state["selected"]["receipt_sha256"]
        ):
            raise ValueError("Saved selected transport receipt changed")
    if "result" in state and judge.sha(state["result"]) != state["result_sha256"]:
        raise ValueError("Saved result changed")
    if state.get("result", {}).get("scoring_status") == "scored":
        result = await verify(state["selected"], dataset.criteria[identifier], cached_only=True)
        if result != state["result"]:
            raise ValueError("Saved result differs from validated cached judge result")
        return result
    if validate_only:
        if "selected" in state:
            await verify(state["selected"], dataset.criteria[identifier], validate_only=True)
        return None
    for stage, (seed, cap) in enumerate(PHASES):
        if "selected" in state:
            break
        if stage < len(state["attempts"]):
            attempt = state["attempts"][stage]
            if attempt["status"] == "started":
                raise ValueError("Ambiguous in-flight generation; reconcile receipt before resume")
            if attempt.get("terminal"):
                break
            continue
        receipt_id = judge.sha({"fingerprint": fingerprint, "item_id": identifier, "stage": stage})
        state["attempts"].append({"status": "started", "seed": seed, "cap": cap, "receipt_id": receipt_id})
        write(path, state)
        row = dataset.prepared_row(identifier)
        attempt = await generate(row, seed, cap, receipt_id)
        state["attempts"][stage].update(attempt, status="settled")
        if attempt.get("complete"):
            state["selected"] = attempt
            state["selected_sha256"] = judge.sha(attempt)
        write(path, state)
        if attempt.get("terminal"):
            break
    if "selected" not in state:
        result = unscored(identifier, "incomplete_delivery_budget_exhausted")
    else:
        selected = state["selected"]
        if judge.sha(selected) != state["selected_sha256"]:
            raise ValueError("Saved selected response changed")
        # Judge failures cannot re-enter the generation branch.
        result = await verify(selected, dataset.criteria[identifier])
    state["result"] = result
    state["result_sha256"] = judge.sha(result)
    write(path, state)
    return result
