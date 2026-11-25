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
"""Network-free tests for prepare_opencode.py, run beside prepare.py on one synthetic encrypted CSV.

The user-only rows must ask exactly the questions prepare.py's rows ask: the same samples in the same order,
with a byte-identical user turn and nothing else in the request.
"""

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pandas
import pytest
from pydantic import TypeAdapter

from benchmarks.browsecomp import prepare as browsecomp_prepare
from benchmarks.browsecomp import prepare_opencode
from nemo_gym.data import validate_jsonl_rows
from nemo_gym.server_runtime import BaseRunRequest
from resources_servers.opencode_search.task_data import TaskData


def _encrypt(plaintext: str, password: str) -> str:
    """BrowseComp's encryption, written apart from prepare.decrypt: XOR with the repeated SHA-256, then base64."""
    data = plaintext.encode()
    digest = hashlib.sha256(password.encode()).digest()
    key = (digest * (len(data) // len(digest) + 1))[: len(data)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def _synthetic_source(num_rows: int = 1266) -> pandas.DataFrame:
    """The upstream CSV's columns, holding synthetic text encrypted under a per-row canary."""
    rows = []
    for index in range(num_rows):
        canary = f"synthetic-canary-{index % 3}"
        # Non-ASCII text, quotes and a newline, so the byte comparison covers their encodings.
        question = f'Synthetic question {index}: which café is "first"?\nAnswer briefly.'
        rows.append(
            {
                "problem": _encrypt(question, canary),
                "answer": _encrypt(f"answer-{index}", canary),
                "problem_topic": "Synthetic",
                "canary": canary,
            }
        )
    return pandas.DataFrame(rows)


SOURCE = _synthetic_source()


def _read_source(url: str) -> pandas.DataFrame:
    assert url == browsecomp_prepare.BROWSECOMP_CSV_URL
    return SOURCE.copy()


@pytest.fixture
def run_prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[ModuleType], list[dict]]:
    """Run a prepare script on the synthetic CSV, writing under tmp_path, and return its rows."""
    monkeypatch.setattr(pandas, "read_csv", _read_source)
    monkeypatch.delenv("BROWSECOMP_RUN_FULL", raising=False)

    def run(module: ModuleType) -> list[dict]:
        monkeypatch.setattr(module, "DATA_DIR", tmp_path)
        monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / module.OUTPUT_FPATH.name)
        return [json.loads(line) for line in module.prepare().read_text().splitlines()]

    return run


def _user_turn(row: dict) -> bytes:
    """The row's only user message as serialized bytes, key order included."""
    (message,) = [message for message in row["responses_create_params"]["input"] if message["role"] == "user"]
    return json.dumps(message, ensure_ascii=False).encode()


@pytest.mark.parametrize("run_full, num_rows", [(None, 400), ("0", 400), ("1", 1266), ("True", 1266), ("YES", 1266)])
def test_same_samples_in_the_same_order_with_byte_identical_user_turns(
    run_prepare: Callable[[ModuleType], list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    run_full: str | None,
    num_rows: int,
) -> None:
    if run_full is not None:
        monkeypatch.setenv("BROWSECOMP_RUN_FULL", run_full)

    rows, reference = run_prepare(prepare_opencode), run_prepare(browsecomp_prepare)

    assert len(rows) == len(reference) == num_rows
    assert [_user_turn(row) for row in rows] == [_user_turn(row) for row in reference]
    assert [(row["question"], row["ground_truth"]) for row in rows] == [
        (row["question"], row["ground_truth"]) for row in reference
    ]


def test_rows_hold_only_the_user_turn(run_prepare: Callable[[ModuleType], list[dict]]) -> None:
    for row in run_prepare(prepare_opencode):
        params = row["responses_create_params"]
        # No tools, instructions or sampling fields: the session harness supplies its own.
        assert set(params) == {"input"}
        assert [message["role"] for message in params["input"]] == ["user"]


def test_rows_validate_as_opencode_search_rollout_inputs(run_prepare: Callable[[ModuleType], list[dict]]) -> None:
    rows = run_prepare(prepare_opencode)
    for row in rows:
        BaseRunRequest.model_validate(row)

    lines = [json.dumps(row) for row in rows]
    report = validate_jsonl_rows(
        "opencode_search", TypeAdapter(TaskData), "browsecomp_opencode_benchmark.jsonl", lines
    )

    assert report.rows == len(rows) == 400
    assert report.clean, report.summary()
