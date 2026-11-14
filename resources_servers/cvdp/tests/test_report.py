# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("flags", [{"mask_sample": True}, {"evaluation_completed": False}])
@pytest.mark.parametrize("include_pass", [False, True])
def test_report_refuses_inconclusive_rows(tmp_path, monkeypatch, flags, include_pass):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from resources_servers.cvdp.scripts.cvdp_report import build_collateral

    measured = {"task_id": "task_0001", "category": "cid003", "difficulty": "easy", "container_exit_code": 0}
    rows = ([measured] if include_pass else []) + [measured | flags]
    source = tmp_path / "rollouts.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="require measured rollouts"):
        build_collateral(str(source), tmp_path / "report")
    assert not (tmp_path / "report/prompt_response.jsonl").exists()
    assert not (tmp_path / "report/run.log").exists()


def test_report_accepts_legacy_measured_rows(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from resources_servers.cvdp.scripts.cvdp_report import build_collateral

    source = tmp_path / "rollouts.jsonl"
    source.write_text(
        json.dumps({"task_id": "task_0001", "category": "cid003", "difficulty": "easy", "container_exit_code": 0})
        + "\n"
    )
    result = build_collateral(str(source), tmp_path / "report")
    assert result["task_0001"]["errors"] == 0
    assert [test["result"] for test in result["task_0001"]["tests"]] == [0]
