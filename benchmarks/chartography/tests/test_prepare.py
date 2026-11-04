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
"""Network-free tests for the Chartography prepare script, on a synthetic two-task source directory."""

import base64
import json
from pathlib import Path

import pytest

from benchmarks.chartography.prepare import (
    SOURCE_DIR_ENV,
    ChartographySourceError,
    build_gym_rows,
    domain_slug,
    prepare,
    read_source_table,
)
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16

RECORDS = [
    {
        "task_id": "t1",
        "prompt": "At a flow of 35 L/s, what head does the 225 mm impeller deliver? Give the value in m.",
        "golden_answer": "Head: 34.5 m (acceptable range 33.5 to 35.5 m) ",
        "chart_path": "charts/one.png",
        "domain_combined": "STEM - Mechanical Engineering",
        "source_type": "Expert-created",
        "source_url": None,
    },
    {
        "task_id": "t2",
        "prompt": "Which plant has the higher output?",
        "golden_answer": "South plant",
        # Uppercase extension: the MIME type comes from the file's bytes.
        "chart_path": "charts/two.JPG",
        "domain_combined": "Manufacturing/Supply Chain",
        "source_type": "Sourced online",
        "source_url": "https://example.com/chart",
    },
]


def _write_source(root: Path, records=RECORDS, fmt: str = "jsonl") -> Path:
    (root / "charts").mkdir(parents=True)
    (root / "charts" / "one.png").write_bytes(PNG)
    (root / "charts" / "two.JPG").write_bytes(JPEG)
    if fmt == "jsonl":
        (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    else:
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(records), root / "data.parquet")
    return root


class TestRows:
    def test_prompt_then_chart_in_one_user_message(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        rows = build_gym_rows(read_source_table(source), source)
        assert len(rows) == 2
        (message,) = rows[0]["responses_create_params"]["input"]
        assert message["role"] == "user"
        text, image = message["content"]
        assert text == {"type": "input_text", "text": RECORDS[0]["prompt"]}
        assert image["type"] == "input_image" and image["detail"] == "auto"
        assert image["image_url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()

    def test_mime_type_follows_file_bytes(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        rows = build_gym_rows(read_source_table(source), source)
        image = rows[1]["responses_create_params"]["input"][0]["content"][1]
        assert image["image_url"].startswith("data:image/jpeg;base64,")

    def test_verifier_metadata(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        meta = build_gym_rows(read_source_table(source), source)[0]["verifier_metadata"]
        assert meta["task_id"] == "t1"
        assert meta["prompt"] == RECORDS[0]["prompt"]
        assert meta["golden_answer"] == "Head: 34.5 m (acceptable range 33.5 to 35.5 m)"
        assert meta["domain"] == "stem_mechanical_engineering"
        assert meta["domain_combined"] == "STEM - Mechanical Engineering"
        assert meta["chart_path"] == "charts/one.png"

    def test_rows_are_valid_gym_requests(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        for row in build_gym_rows(read_source_table(source), source):
            NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])

    def test_parquet_and_jsonl_sources_agree(self, tmp_path: Path) -> None:
        a = _write_source(tmp_path / "a", fmt="jsonl")
        b = _write_source(tmp_path / "b", fmt="parquet")
        assert build_gym_rows(read_source_table(a), a) == build_gym_rows(read_source_table(b), b)


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("Manufacturing / Supply Chain", "manufacturing_supply_chain"),
        ("Manufacturing/Supply Chain", "manufacturing_supply_chain"),
        ("STEM - Civil/Environmental Engineering", "stem_civil_environmental_engineering"),
        ("Finance / Investing", "finance_investing"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_domain_slug(variant, expected) -> None:
    assert domain_slug(variant) == expected


class TestInvalidSources:
    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("golden_answer", "  ", "no golden_answer"),
            ("prompt", "", "no prompt"),
            ("task_id", None, "no task_id"),
            ("chart_path", "charts/missing.png", "missing chart"),
            ("chart_path", "../outside.png", "outside"),
        ],
    )
    def test_bad_record(self, tmp_path: Path, field, value, match) -> None:
        records = [dict(RECORDS[0], **{field: value})]
        source = _write_source(tmp_path / "src", records)
        (tmp_path / "outside.png").write_bytes(PNG)
        with pytest.raises(ChartographySourceError, match=match):
            build_gym_rows(read_source_table(source), source)

    def test_duplicate_task_id(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path, [RECORDS[0], dict(RECORDS[1], task_id="t1")])
        with pytest.raises(ChartographySourceError, match="duplicate task_id"):
            build_gym_rows(read_source_table(source), source)

    def test_unsupported_image(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        (source / "charts" / "one.png").write_bytes(b"not an image")
        with pytest.raises(ChartographySourceError, match="unsupported image format"):
            build_gym_rows(read_source_table(source), source)

    def test_no_table(self, tmp_path: Path) -> None:
        with pytest.raises(ChartographySourceError, match="neither data.parquet nor data.jsonl"):
            read_source_table(tmp_path)


class TestPrepare:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path / "src")
        out = prepare(source_dir=str(source), output_fpath=str(tmp_path / "out" / "bench.jsonl"))
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["verifier_metadata"]["task_id"] for r in rows] == ["t1", "t2"]
        assert not (tmp_path / "out" / "bench.jsonl.tmp").exists()

    def test_limit(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path / "src")
        out = prepare(source_dir=str(source), limit=1, output_fpath=str(tmp_path / "bench.jsonl"))
        assert len(out.read_text().splitlines()) == 1

    def test_source_from_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _write_source(tmp_path / "src")
        monkeypatch.setenv(SOURCE_DIR_ENV, str(source))
        out = prepare(output_fpath=str(tmp_path / "bench.jsonl"))
        assert len(out.read_text().splitlines()) == 2

    def test_no_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SOURCE_DIR_ENV, raising=False)
        with pytest.raises(ChartographySourceError, match="No Chartography source given"):
            prepare()
