# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Smoke tests for benchmarks/hotpotqa_closedbook/prepare.py.  The real
# load_dataset() pulls the HotpotQA distractor split off the HuggingFace
# Hub which is not available in unit-test environments, so these tests
# patch `datasets.load_dataset` to return a fixture and verify the JSONL
# emission shape.

import json
from unittest.mock import patch

from benchmarks.hotpotqa_closedbook import prepare as prepare_module


_FIXTURE = [
    {
        "id": "5a8b57f25542995d1e6f1371",  # pragma: allowlist secret
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        # The Skills format includes context + supporting_facts; closed-book
        # prepare drops them, so we just include them on the fixture to verify
        # they don't leak into the output.
        "context": {"title": ["A"], "sentences": [["x"]]},
        "supporting_facts": {"title": ["A"], "sent_id": [0]},
    },
    {
        "id": "5a754ab35542993748c89819",  # pragma: allowlist secret
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "type": "comparison",
        "level": "medium",
        "context": {"title": [], "sentences": []},
        "supporting_facts": {"title": [], "sent_id": []},
    },
]


class TestPrepare:
    def test_emits_expected_fields(self, tmp_path) -> None:
        out_fpath = tmp_path / "hotpotqa_closedbook_benchmark.jsonl"

        with (
            patch.object(prepare_module, "load_dataset", return_value=_FIXTURE),
            patch.object(prepare_module, "OUTPUT_FPATH", out_fpath),
            patch.object(prepare_module, "DATA_DIR", tmp_path),
        ):
            result = prepare_module.prepare()

        assert result == out_fpath
        assert out_fpath.exists()

        rows = [json.loads(line) for line in out_fpath.open()]
        assert len(rows) == len(_FIXTURE)
        assert {row["id"] for row in rows} == {f["id"] for f in _FIXTURE}

        for row, fixture in zip(rows, _FIXTURE):
            # Closed-book emits exactly these fields — no context, no
            # supporting_facts.
            assert set(row) == {"id", "question", "expected_answer", "type", "level"}
            assert row["question"] == fixture["question"]
            assert row["expected_answer"] == fixture["answer"]
            assert row["type"] == fixture["type"]
            assert row["level"] == fixture["level"]

    def test_renames_answer_to_expected_answer(self, tmp_path) -> None:
        # Critical for parity with the resource server's verify request.
        out_fpath = tmp_path / "hotpotqa_closedbook_benchmark.jsonl"

        with (
            patch.object(prepare_module, "load_dataset", return_value=_FIXTURE),
            patch.object(prepare_module, "OUTPUT_FPATH", out_fpath),
            patch.object(prepare_module, "DATA_DIR", tmp_path),
        ):
            prepare_module.prepare()

        for row in (json.loads(line) for line in out_fpath.open()):
            assert "answer" not in row
            assert "expected_answer" in row

    def test_atomic_write(self, tmp_path) -> None:
        # On success the .tmp file must not survive.
        out_fpath = tmp_path / "hotpotqa_closedbook_benchmark.jsonl"

        with (
            patch.object(prepare_module, "load_dataset", return_value=_FIXTURE),
            patch.object(prepare_module, "OUTPUT_FPATH", out_fpath),
            patch.object(prepare_module, "DATA_DIR", tmp_path),
        ):
            prepare_module.prepare()

        assert not (tmp_path / "hotpotqa_closedbook_benchmark.jsonl.tmp").exists()
