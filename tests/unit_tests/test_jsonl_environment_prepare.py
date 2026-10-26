# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


# Valid rows that Arrow's JSON reader (behind datasets.load_dataset) cannot load: a null entry in a list of
# objects (apache/arrow#51305), and a metadata value whose JSON type changes between rows.
INSTRUCTION_FOLLOWING_ROWS = [
    {"id": 1, "instruction_id_list": ["a:b", "c:d"], "kwargs": [None, {"keywords": ["x"]}], "prompt": "café"},
    {"id": 2, "instruction_id_list": ["e:f"], "kwargs": [{"num_words": 3, "relation": "less than"}], "prompt": "q"},
]
REASONING_GYM_ROWS = [
    {"question": "q1", "answer": "1", "metadata": {"solution": 1}},
    {"question": "q2", "answer": "[1, 2]", "metadata": {"solution": [1, 2]}},
]


@pytest.mark.parametrize(
    ("module_name", "artifact", "rows"),
    [
        ("environments.instruction_following.prepare", "instruction_following.jsonl", INSTRUCTION_FOLLOWING_ROWS),
        ("environments.reasoning_gym.prepare", "data/train.jsonl", REASONING_GYM_ROWS),
    ],
)
def test_prepare_copies_the_raw_jsonl_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module_name: str,
    artifact: str,
    rows: list[dict],
) -> None:
    # Not json.dumps' default formatting, so re-serializing rows would change the bytes.
    content = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows)
    source = tmp_path / "cache" / artifact
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")
    download = Mock(return_value=str(source))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "environment" / "prepare.py"))
    output = tmp_path / "environment" / "data" / "train.jsonl"

    module.prepare("train")

    assert output.read_text(encoding="utf-8") == content
    # Copied, not moved: the shared Hugging Face cache keeps its file.
    assert source.read_text(encoding="utf-8") == content
    download.assert_called_once_with(repo_id=module.REPO_ID, filename=artifact, repo_type="dataset")
    assert capsys.readouterr().out == f"Wrote {len(rows)} rows to {output}\n"
