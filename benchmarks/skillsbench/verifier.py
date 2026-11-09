# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize inconclusive verifier builds without changing official rewards."""

import json
import re
from pathlib import Path


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TRANSIENT_TRANSFER = re.compile(
    r"Attempted read from closed stream|Connection reset|Connection timed out|Read timed out|"
    r"Temporary failure in name resolution|Received close_notify during handshake|"
    r"(?:status(?: code)?\s*[:=]?\s*|HTTP\s+)(?:429|500|502|503|504)\b",
    re.IGNORECASE,
)


def verifier_infrastructure_failure(trial_dir: Path) -> str | None:
    """Find a transient Maven transfer error in an actual failed pytest assertion.

    Inspect structured verifier failures, not the agent's transcript or arbitrary
    captured stdout. A missing artifact (404), compile error or wrong answer is
    still a task failure. This is deliberately a narrow detector for observed
    failures, not a guarantee that every verifier infrastructure fault is found.
    """
    report_path = trial_dir / "verifier" / "ctrf.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    for test in report.get("results", {}).get("tests", []):
        if test.get("status") != "failed":
            continue
        for line in (test.get("trace") or "").splitlines():
            line = _ANSI.sub("", line)
            # Pytest's E prefix identifies the raised assertion's diagnostic,
            # excluding source listings and successful/recovered download logs.
            if not re.match(r"^E\s+", line) or "Could not transfer artifact" not in line:
                continue
            if _TRANSIENT_TRANSFER.search(line):
                return f"Transient dependency transfer failure in verifier test {test.get('name', '<unknown>')}"
    return None
