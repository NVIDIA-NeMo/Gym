# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stopping eligibility for a single instruction-ordering evaluation."""

from typing import Literal


def ordering_outcome(
    graded: dict, events: list[dict], *, probe_namespace: str = "sibling", expected_criterion_id: str | None = None
) -> Literal["ordering_failed", "ordering_passed"] | None:
    """A semantic violation is decisive; no violation alone does not earn a pass.

    The shared verifier decides whether advice is premature, including advice
    selected in the same response as a probe. Only returned, identity-validated
    probe results can turn a nonviolating prefix into a procedural pass.
    """
    if graded.get("mask_sample") is not False:
        raise ValueError("Prefix verification is unscorable")
    items = (graded.get("if_result") or {}).get("items", [])
    if len(items) != 1 or not isinstance(items[0].get("id"), str) or not items[0]["id"]:
        raise ValueError("Prefix verification requires exactly one named ordering rubric")
    if expected_criterion_id is not None and items[0]["id"] != expected_criterion_id:
        raise ValueError("Prefix verification requires exactly the ordering rubric")
    passed = items[0].get("passed")
    if not isinstance(passed, bool) or graded.get("reward") != float(passed):
        raise ValueError("Prefix verification has inconsistent judgment/reward")
    if not passed:
        return "ordering_failed"
    if any(e.get("verified_probe", {}).get("namespace") == probe_namespace for e in events):
        return "ordering_passed"
    return None
