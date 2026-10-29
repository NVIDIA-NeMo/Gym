# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from benchmarks.indic.milu_1_5.prepare import SOURCE_REVISION
from benchmarks.indic.milu_1_5.rescore import rescore_shard
from nemo_gym.core import NeMoGymResponse


@pytest.fixture(params=["gym", "lm_eval"])
def saved(request, tmp_path):
    options = [{letter: f"choice {letter}"} for letter in "ABCD"]
    originals, rows = [], []
    for index, (text, gold, reward) in enumerate([(r"\boxed{A}", "A", 1), ("B", "B", 0)]):
        if request.param == "gym":
            response = NeMoGymResponse(
                id="r",
                created_at=0,
                model="test",
                object="response",
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                output=[
                    {"type": "reasoning", "id": "t", "summary": [{"type": "summary_text", "text": r"\boxed{D}"}]},
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                ],
            ).model_dump(mode="json")
            originals.append(
                {
                    "_ng_task_index": index,
                    "expected_answer": gold,
                    "options": options,
                    "metadata": {
                        "source_row_index": index,
                        "language": "hi",
                        "source_revision": SOURCE_REVISION,
                    },
                }
            )
            rows.append({"_ng_task_index": index, "expected_answer": gold, "reward": reward, "response": response})
        else:
            rows.append(
                {
                    "doc_id": index,
                    "doc": {
                        **{f"option{i + 1}": f"choice {letter}" for i, letter in enumerate("ABCD")},
                        "target": f"option{index + 1}",
                        "language": "Hindi",
                    },
                    "target": gold,
                    "resps": [[text]],
                    "exact_match": reward,
                }
            )
    rollouts = tmp_path / "responses.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in originals))
    return {
        "rollouts": rollouts,
        "inputs": inputs,
        "input_format": request.param,
        "language": "hi",
        "expected_count": 2,
    }


def test_saved_formats_have_identical_scores_and_preserve_sources(saved, tmp_path):
    source = saved["rollouts"].read_bytes()
    output = tmp_path / "scores.jsonl"
    result = rescore_shard(**saved, output=output)
    assert result["accuracy"] == {"strict_boxed": 50, "normalized": 100, "reference": 50}
    assert result["rows"] == 2
    assert result["extracted"] == {"strict_boxed": 1, "normalized": 2}
    assert result["rollouts_sha256"] == hashlib.sha256(source).hexdigest()
    assert saved["rollouts"].read_bytes() == source
    scores = [json.loads(line) for line in output.read_text().splitlines()]
    assert scores[1]["strict_boxed_answer"] is None
    assert scores[1]["normalized_answer"] == "B"


@pytest.mark.parametrize("corruption", ["incomplete", "duplicate", "wrong_reward"])
def test_invalid_saved_runs_do_not_publish_scores(saved, tmp_path, corruption):
    rows = [json.loads(line) for line in saved["rollouts"].read_text().splitlines()]
    if corruption == "incomplete":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    else:
        rows[1]["reward" if saved["input_format"] == "gym" else "exact_match"] = 1
    saved["rollouts"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "scores.jsonl"
    output.write_text("previous complete scores\n")
    with pytest.raises(ValueError):
        rescore_shard(**saved, output=output)
    assert output.read_text() == "previous complete scores\n"
    assert list(tmp_path.glob(".scores.jsonl.*")) == []


def test_source_overwrite_is_rejected(saved):
    with pytest.raises(ValueError, match="overwrite"):
        rescore_shard(**saved, output=saved["rollouts"])
