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

from benchmarks.chartography import prepare as prepare_module
from benchmarks.chartography.prepare import (
    SOURCE_DIR_ENV,
    ChartographySourceError,
    build_gym_rows,
    domain_slug,
    prepare,
    read_source_table,
    release_fingerprint,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.chartography.app import ChartographyVerifyRequest
from resources_servers.chartography.task_data import TaskData
from resources_servers.chartography.verifier_fixture import create_chartography_server, invoke_chartography


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


@pytest.fixture
def pinned_to(monkeypatch: pytest.MonkeyPatch):
    """Pin the release checks to a synthetic source, so a full preparation of it passes."""

    def pin(source: Path) -> None:
        records = read_source_table(source)
        monkeypatch.setattr(prepare_module, "EXPECTED_NUM_TASKS", len(records))
        monkeypatch.setattr(prepare_module, "RELEASE_SHA256", release_fingerprint(records, source))

    return pin


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

    def test_chart_symlinked_into_a_cache_is_accepted(self, tmp_path: Path) -> None:
        # A Hugging Face cache snapshot holds symlinks into a blobs/ folder outside the snapshot directory.
        source = _write_source(tmp_path / "snapshot")
        blob = tmp_path / "blobs" / "0123abcd"
        blob.parent.mkdir()
        blob.write_bytes(PNG)
        (source / "charts" / "one.png").unlink()
        (source / "charts" / "one.png").symlink_to(blob)
        rows = build_gym_rows(read_source_table(source), source)
        assert rows[0]["responses_create_params"]["input"][0]["content"][1]["image_url"].endswith(
            base64.b64encode(PNG).decode()
        )

    @pytest.mark.asyncio
    async def test_prepared_row_validates_and_is_graded(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        row = build_gym_rows(read_source_table(source), source)[0]
        TaskData.model_validate(row["verifier_metadata"])
        judge_output = json.dumps(
            {"scores": [{"criterion_id": "criterion_1", "extracted_answer": "34.8 m", "score": 1, "rationale": "r"}]}
        )
        request = ChartographyVerifyRequest(
            **row, response=_response("Head: 34.8 m"), fixture_judge_output=judge_output
        )
        server = create_chartography_server()
        result = await invoke_chartography(server, request)
        assert result.reward == 1.0 and not result.mask_sample
        sent = server.server_client.post.await_args.kwargs["json"].messages[0]["content"]
        assert f"Golden answer: {RECORDS[0]['golden_answer'].strip()}\n" in sent


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "resp",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }
    )


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
            ("chart_path", "charts/../../outside.png", "outside"),
            ("chart_path", "/etc/hostname", "outside"),
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


class TestReleasePin:
    def test_published_release_is_pinned(self) -> None:
        assert prepare_module.EXPECTED_NUM_TASKS == 100
        assert len(prepare_module.RELEASE_SHA256) == 64

    def test_fingerprint_ignores_format_and_row_order(self, tmp_path: Path) -> None:
        a = _write_source(tmp_path / "a", fmt="jsonl")
        b = _write_source(tmp_path / "b", fmt="parquet")
        records = read_source_table(a)
        assert release_fingerprint(records, a) == release_fingerprint(read_source_table(b), b)
        assert release_fingerprint(records, a) == release_fingerprint(list(reversed(records)), a)

    @pytest.mark.parametrize("edit", ["golden_answer", "chart"])
    def test_fingerprint_changes_with_any_edit(self, tmp_path: Path, edit: str) -> None:
        source = _write_source(tmp_path)
        before = release_fingerprint(read_source_table(source), source)
        if edit == "chart":
            (source / "charts" / "one.png").write_bytes(PNG + b"\x00")
            records = read_source_table(source)
        else:
            records = [dict(RECORDS[0], golden_answer="Head: 34.6 m"), RECORDS[1]]
        assert release_fingerprint(records, source) != before

    def test_full_run_needs_the_release_task_count(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path / "src")
        with pytest.raises(ChartographySourceError, match="has 2 tasks; the release has 100"):
            prepare(source_dir=str(source), output_fpath=str(tmp_path / "bench.jsonl"))
        assert not (tmp_path / "bench.jsonl").exists()

    def test_full_run_rejects_a_modified_release(self, tmp_path: Path, pinned_to) -> None:
        source = _write_source(tmp_path / "src")
        pinned_to(source)
        (source / "charts" / "two.JPG").write_bytes(JPEG + b"\x00")
        with pytest.raises(ChartographySourceError, match="does not match the published Chartography release"):
            prepare(source_dir=str(source), output_fpath=str(tmp_path / "bench.jsonl"))
        assert not (tmp_path / "bench.jsonl").exists()


class TestPrepare:
    def test_writes_jsonl(self, tmp_path: Path, pinned_to) -> None:
        source = _write_source(tmp_path / "src")
        pinned_to(source)
        out = prepare(source_dir=str(source), output_fpath=str(tmp_path / "out" / "bench.jsonl"))
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["verifier_metadata"]["task_id"] for r in rows] == ["t1", "t2"]
        assert not (tmp_path / "out" / "bench.jsonl.tmp").exists()

    def test_limit_skips_the_release_checks(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path / "src")
        out = prepare(source_dir=str(source), limit=1, output_fpath=str(tmp_path / "bench.jsonl"))
        assert len(out.read_text().splitlines()) == 1

    def test_source_from_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_to) -> None:
        source = _write_source(tmp_path / "src")
        pinned_to(source)
        monkeypatch.setenv(SOURCE_DIR_ENV, str(source))
        out = prepare(output_fpath=str(tmp_path / "bench.jsonl"))
        assert len(out.read_text().splitlines()) == 2

    def test_no_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SOURCE_DIR_ENV, raising=False)
        with pytest.raises(ChartographySourceError, match="No Chartography source given"):
            prepare()
