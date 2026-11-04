# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.labbench2_pdf.verifier import (
    call_judge,
    deterministic_match,
    extract_final_answer,
    main,
)


@pytest.mark.parametrize(
    ("ideal", "answer"),
    [
        ("Ciprofloxacin", "The answer is ciprofloxacin."),
        ("1.5000000", "1.5"),
        ("alpha (beta)", "alpha beta"),
    ],
)
def test_deterministic_match_accepts_equivalent_answers(ideal: str, answer: str) -> None:
    assert deterministic_match(ideal, answer)[0] is True


def test_deterministic_match_rejects_different_answers() -> None:
    assert deterministic_match("control", "treatment")[0] is False


def test_extract_final_answer_uses_last_non_tool_agent_message() -> None:
    trajectory = {
        "steps": [
            {"source": "agent", "message": "planning", "tool_calls": [{"name": "shell"}]},
            {"source": "environment", "message": "result"},
            {"source": "agent", "message": "final answer", "tool_calls": []},
        ]
    }

    assert extract_final_answer(trajectory) == "final answer"


def test_call_judge_reports_unavailable_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JUDGE_API_KEY", "OPENAI_API_KEY", "JUDGE_BASE_URL", "OPENAI_BASE_URL", "JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)

    result = call_judge("litqa3", "question", "expected", "different")

    assert result["score"] == 0.0
    assert result["result"] == "unavailable"
    assert result["judge_available"] is False


def test_main_writes_harbor_reward_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_dir = tmp_path / "app"
    tests_dir = tmp_path / "tests"
    logs_dir = tmp_path / "logs" / "verifier"
    agent_logs_dir = tmp_path / "logs" / "agent"
    for path in (app_dir, tests_dir, agent_logs_dir):
        path.mkdir(parents=True)
    (app_dir / "answer.txt").write_text("42\n", encoding="utf-8")
    (tests_dir / "gold_metadata.json").write_text(
        json.dumps({"tag": "tableqa2", "question": "Q?", "ideal": "42", "item_id": "id"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HARBOR_APP_DIR", str(app_dir))
    monkeypatch.setenv("HARBOR_TESTS_DIR", str(tests_dir))
    monkeypatch.setenv("HARBOR_LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("HARBOR_AGENT_LOGS_DIR", str(agent_logs_dir))

    main()

    assert (logs_dir / "reward.txt").read_text(encoding="utf-8") == "1.0\n"
    reward = json.loads((logs_dir / "reward.json").read_text(encoding="utf-8"))
    details = json.loads((logs_dir / "details.json").read_text(encoding="utf-8"))
    assert reward == {"reward": 1.0}
    assert details["execution_status"] == "completed"
    assert details["fallback_exact"] is True
