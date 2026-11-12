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

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.gsm8k import prepare as prepare_module


@pytest.mark.parametrize(
    "question,answer,expected_answer",
    [
        ("Integer answer", "1,234", "1234"),
        ("Decimal answer", "1.5", "1.5"),
        *[(question, "0", str(answer)) for question, answer in prepare_module.FIXES.items()],
    ],
)
def test_prepare_emits_string_answers(tmp_path: Path, question: str, answer: str, expected_answer: str) -> None:
    source = tmp_path / "original_test.jsonl"
    source.write_text(json.dumps({"question": question, "answer": f"Calculation: <<1+1=2>>2.\n#### {answer}"}) + "\n")
    output = tmp_path / "gsm8k_benchmark.jsonl"

    with (
        patch.object(prepare_module.urllib.request, "urlretrieve"),
        patch.object(prepare_module, "DATA_DIR", tmp_path),
        patch.object(prepare_module, "OUTPUT_FPATH", output),
    ):
        assert prepare_module.prepare() == output

    assert json.loads(output.read_text()) == {
        "question": question,
        "expected_answer": expected_answer,
        "reference_solution": "Calculation: 2.\n",
    }
