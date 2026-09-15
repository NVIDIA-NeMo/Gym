# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end cover for the preparer, with the Hub download stubbed.

`format_row` and `build_parser` were already tested, but the pipeline around
them was not: the pinned remote paths, the toxicity-cliff CSV join, output
writing, `--limit`, and the missing-join warning. A regression in any of those
produces a silently wrong dataset, which no scorer test would catch.

The download is monkeypatched rather than hit, so these stay offline and fast.
"""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_moldetox  # noqa: E402


CONFIG = "task3_smiles_gen_single"

QA_ROWS = [
    {
        "question": "Q1",
        "answer": {"answer": "CCO"},
        "endpoint": "e1",
        "dataset_name": "d1",
        "source_index": 10,
        "id": 1,
    },
    {
        "question": "Q2",
        "answer": {"answer": "CCC"},
        "endpoint": "e2",
        "dataset_name": "d2",
        "source_index": 11,
        "id": 2,
    },
    {
        "question": "Q3",
        "answer": {"answer": "CCN"},
        "endpoint": "e3",
        "dataset_name": "d3",
        "source_index": 99,
        "id": 3,
    },
]
CLIFF_CSV = "orig_test_index,toxic_smiles\n10,CBr\n11,CI\n"


@pytest.fixture
def stub_hub(monkeypatch, tmp_path):
    """Serve the two files the preparer fetches, and record what it asked for."""
    requested: list[str] = []

    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in QA_ROWS) + "\n")
    csv_path = tmp_path / "cliff.csv"
    csv_path.write_text(CLIFF_CSV)

    def fake_download(repo_id, filename, repo_type, revision):
        requested.append(filename)
        assert repo_id == prepare_moldetox.DATASET
        assert revision == prepare_moldetox.REVISION
        return str(csv_path if filename.endswith(".csv") else qa)

    monkeypatch.setattr(prepare_moldetox, "hf_hub_download", fake_download)
    return requested


def _run(stub, tmp_path, *extra):
    out = tmp_path / "out.jsonl"
    sys.argv = ["prepare_moldetox.py", "--output", str(out), "--config", CONFIG, *extra]
    prepare_moldetox.main()
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_fetches_the_pinned_paths_for_the_requested_config(stub_hub, tmp_path) -> None:
    """The remote layout is a convention, not an API; a typo here yields a 404 at run time."""
    _run(stub_hub, tmp_path)
    assert f"MolDeTox_QA/test/test_{CONFIG}.jsonl" in stub_hub
    assert "toxicitycliff_test.csv" in stub_hub


def test_writes_every_row_with_both_turns_and_the_cliff_join(stub_hub, tmp_path) -> None:
    rows = _run(stub_hub, tmp_path)
    assert len(rows) == len(QA_ROWS)

    first = rows[0]
    assert [m["role"] for m in first["responses_create_params"]["input"]] == ["system", "user"]
    assert first["responses_create_params"]["input"][1]["content"] == "Q1"
    assert first["verifier_metadata"]["toxic_smiles"] == "CBr"
    assert first["verifier_metadata"]["answer"] == "CCO"
    assert first["verifier_metadata"]["scoring_mode"] == "smiles"


def test_unjoined_row_is_kept_with_a_null_and_counted(stub_hub, tmp_path, capsys) -> None:
    """Source index 99 has no cliff entry. Dropping it silently would shrink the split."""
    rows = _run(stub_hub, tmp_path)
    assert rows[2]["verifier_metadata"]["toxic_smiles"] is None
    assert "WARNING: 1 rows had no toxicitycliff join" in capsys.readouterr().err


def test_limit_truncates(stub_hub, tmp_path) -> None:
    assert len(_run(stub_hub, tmp_path, "--limit", "2")) == 2


def test_no_system_prompt_flag_reaches_the_written_rows(stub_hub, tmp_path) -> None:
    """The flag exists to make the divergence measurable, so it has to survive `main`."""
    rows = _run(stub_hub, tmp_path, "--no-system-prompt")
    assert all(len(r["responses_create_params"]["input"]) == 1 for r in rows)
    assert all(r["responses_create_params"]["input"][0]["role"] == "user" for r in rows)
