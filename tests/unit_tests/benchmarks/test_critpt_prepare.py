# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

from benchmarks.critpt import prepare


SOURCE_ROWS = [
    {"problem_id": "Challenge_1", "problem_description": "Find E.", "code_template": "def answer():\n    pass\n"},
]


def test_prepare_downloads_the_pinned_revision(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return SOURCE_ROWS

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "benchmark.jsonl")

    output_fpath = prepare.prepare()

    assert calls == [((prepare.HF_REPO_ID,), {"split": "train", "revision": prepare.HF_REVISION})]
    assert len(prepare.HF_REVISION) == 40
    rows = [json.loads(line) for line in output_fpath.read_text().splitlines()]
    assert rows == [
        {
            "problem_id": "Challenge_1",
            "problem": "Find E.",
            "code_template": "def answer():\n    pass\n",
            "uuid": "Challenge_1",
        }
    ]
