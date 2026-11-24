# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation of native weighted scores, shared by both SaaS harnesses."""

import math


SCORING_POLICY = "native_weighted_final_state_v1"


def score_validation_errors(verify: dict, *, require_summary: bool = False) -> list[str]:
    """Reject invalid scores and contradictory summaries without inventing a grade.

    Native task scripts print their weighted fraction rounded to three decimals.
    Older result envelopes may contain only status and score; when summary fields
    are present, validate the complete point accounting as well.
    """
    errors = []
    score = verify.get("score")
    valid_score = type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
    if not valid_score:
        errors.append("native score must be a finite number in [0, 1]")
    if valid_score and verify.get("status") == "PASS" and score != 1:
        errors.append("native PASS requires full credit")
    fields = ("earned", "total", "all_pass")
    if not require_summary and not any(name in verify for name in fields):
        return errors
    if not all(name in verify for name in fields):
        return errors + ["native point summary is incomplete"]
    earned, total, all_pass = (verify[name] for name in fields)
    if type(earned) is not int or type(total) is not int or total <= 0 or not 0 <= earned <= total:
        errors.append("native earned/total points are invalid")
        return errors
    if type(all_pass) is not bool:
        errors.append("native all_pass must be boolean")
    checks = verify.get("checks")
    if not isinstance(checks, list) or not checks:
        return errors + ["native summary needs nonempty check results"]
    if any(
        not isinstance(check, dict)
        or type(check.get("weight")) is not int
        or check["weight"] < 0
        or type(check.get("passed")) is not bool
        for check in checks
    ):
        return errors + ["native check weights/statuses are invalid"]
    if total != sum(check["weight"] for check in checks):
        errors.append("native total differs from printed check weights")
    if earned != sum(check["weight"] for check in checks if check["passed"]):
        errors.append("native earned points differ from printed passed checks")
    if all_pass != all(check["passed"] for check in checks):
        errors.append("native all_pass contradicts check results")
    if valid_score and not math.isclose(score, earned / total, rel_tol=0, abs_tol=0.000500000001):
        errors.append("native score differs from its weighted point fraction")
    if verify.get("status") in {"PASS", "FAIL"} and (verify["status"] == "PASS") != all_pass:
        errors.append("native status contradicts all_pass")
    return errors
