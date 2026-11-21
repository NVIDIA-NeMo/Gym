# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one Gym input row per slot in a supplied RoboDojo panel."""

import json
import os
import tempfile
from pathlib import Path

from responses_api_agents.robodojo_agent.cases import load_panel


HERE = Path(__file__).resolve().parent


def prepare(
    panel_manifest: str | None = None,
    output_fpath: str | Path = HERE / "data/panel.jsonl",
    case_id: str | None = None,
) -> Path:
    source = panel_manifest or os.environ.get("ROBODOJO_PANEL")
    if not source:
        raise ValueError("Set ROBODOJO_PANEL to the frozen panel JSON")
    panel = load_panel(Path(source))
    selected = case_id or os.environ.get("ROBODOJO_CASE_ID")
    cases = [case for case in panel.cases if selected is None or case.case_id == selected]
    if not cases:
        raise ValueError(f"Case {selected!r} is absent from the frozen panel")
    output = Path(output_fpath)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as file:
        temporary = Path(file.name)
        for case in cases:
            row = {"case_id": case.case_id, "responses_create_params": {"input": []}}
            file.write(json.dumps(row) + "\n")
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
