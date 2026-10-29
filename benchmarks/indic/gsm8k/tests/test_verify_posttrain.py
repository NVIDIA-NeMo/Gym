# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.indic.gsm8k.verify_posttrain import PROMPT, read_jsonl, verify_run


@pytest.fixture
def campaign(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "campaign"
    work = run / "results/model/kn"
    work.mkdir(parents=True)
    (run / "data").mkdir()
    prompt_path = run / "source" / PROMPT
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text('user: "Solve {question}"\n', encoding="utf-8")
    task = {"question": "ಪ್ರಶ್ನೆ", "expected_answer": "Work #### 18", "uuid": "source-id"}
    model = {"name": "test-model", "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1}}
    row = {
        **task,
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "responses_create_params": {
            "input": [{"role": "user", "content": "Solve ಪ್ರಶ್ನೆ"}],
            "max_output_tokens": 120000,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "response": {
            "status": "completed",
            "usage": {"output_tokens": 32},
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Maybe #### 99"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "ಉತ್ತರ\u2028#### 18\u2029"}]},
            ],
        },
        "reward": 1.0,
        "strict_match": 1.0,
        "flexible_extract": 1.0,
    }
    transport = {
        "event": "transport_request",
        "request_payload": {
            "model": model["name"],
            "max_tokens": 120000,
            **model["sampling"],
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    for path, value in (
        (run / "models.json", {"model": model}),
        (run / "data/kn.jsonl", task),
        (work / "rollouts.jsonl", row),
        (work / "transport.jsonl", transport),
    ):
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return run, work


def test_unicode_line_separators_do_not_split_rollouts(campaign: tuple[Path, Path]) -> None:
    run, work = campaign
    # This is the real failure mode: splitlines splits inside the JSON string.
    assert len((work / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    report = verify_run(run, model_key="model", language="kn")
    assert report["valid"] is True
    assert report["observed"] == report["expected"] == 1
    assert report["strict_accuracy_percent"] == report["flexible_accuracy_percent"] == 100.0
    assert report["reasoning_responses"] == 1
    assert report["actual_requests"] == 1
    assert (work / "DONE").is_file()
    assert json.loads((work / "summary.json").read_text()) == report


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [("uuid", "wrong-source", "Prompt/source mismatch"), ("reward", 0.0, "Scorer mismatch")],
)
def test_invalid_rollout_clears_stale_success(
    campaign: tuple[Path, Path], field: str, value: object, error: str
) -> None:
    run, work = campaign
    (work / "DONE").touch()
    row = read_jsonl(work / "rollouts.jsonl")[0]
    row[field] = value
    (work / "rollouts.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=error):
        verify_run(run, model_key="model", language="kn")
    assert not (work / "DONE").exists()
    assert json.loads((work / "summary.json").read_text())["valid"] is False


def test_actual_request_settings_are_checked(campaign: tuple[Path, Path]) -> None:
    run, work = campaign
    transport = read_jsonl(work / "transport.jsonl")[0]
    transport["request_payload"]["chat_template_kwargs"]["enable_thinking"] = False
    (work / "transport.jsonl").write_text(json.dumps(transport) + "\n")
    with pytest.raises(ValueError, match="Actual HTTP thinking/stopping mismatch"):
        verify_run(run, model_key="model", language="kn")
    assert not (work / "DONE").exists()


def test_truncated_json_still_fails(campaign: tuple[Path, Path]) -> None:
    run, work = campaign
    (work / "DONE").touch()
    (work / "rollouts.jsonl").write_text('{"response": "truncated\n')
    with pytest.raises(json.JSONDecodeError):
        verify_run(run, model_key="model", language="kn")
    assert not (work / "DONE").exists()
