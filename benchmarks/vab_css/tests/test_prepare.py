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

import pytest

from benchmarks.vab_css import prepare as prepare_module
from resources_servers.vab_css.prepare_data import INSTRUCTIONS


EXAMPLE_TASKS = Path(__file__).resolve().parents[3] / "resources_servers" / "vab_css" / "data" / "example_tasks"


@pytest.fixture
def output_fpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "vab_css_benchmark.jsonl"
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", path)
    return path


def test_prepare_writes_one_row_per_task_without_agent_ref(output_fpath: Path) -> None:
    written = prepare_module.prepare(data_dir=str(EXAMPLE_TASKS), expected_tasks=None, min_distance=0)

    rows = [json.loads(line) for line in written.read_text().splitlines()]
    assert written == output_fpath
    assert [row["task_id"] for row in rows] == ["example-cinema", "example-menu"]
    assert all("agent_ref" not in row for row in rows)
    assert rows[0]["responses_create_params"]["input"][0]["content"] == INSTRUCTIONS


def test_prepare_requires_data_dir(output_fpath: Path) -> None:
    with pytest.raises(ValueError, match="needs prepare_script_args.data_dir"):
        prepare_module.prepare()
    assert not output_fpath.exists()


def test_prepare_rejects_a_split_of_the_wrong_size(output_fpath: Path) -> None:
    with pytest.raises(ValueError, match="has 2 VAB-CSS tasks after upstream's filter, expected 165"):
        prepare_module.prepare(data_dir=str(EXAMPLE_TASKS), min_distance=0)
    assert not output_fpath.exists()


def test_prepare_fails_when_the_filter_keeps_nothing(output_fpath: Path) -> None:
    # Both synthetic corruptions are below upstream's 0.2 distance filter.
    with pytest.raises(ValueError, match="passes the filter"):
        prepare_module.prepare(data_dir=str(EXAMPLE_TASKS), expected_tasks=None)
    assert not output_fpath.exists()
