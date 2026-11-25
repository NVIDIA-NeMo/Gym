# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the Hub parquet reader of the benchmark preparation scripts (synthetic shards)."""

import fnmatch
import threading
from pathlib import Path

import pytest


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import huggingface_hub  # noqa: E402

from benchmarks import hf_parquet  # noqa: E402


REPO, REVISION = "org/dataset", "a" * 40
ROOT = f"datasets/{REPO}@{REVISION}"
CARD = """---
# --- section markers in comments are not the end of the header ---
configs:
- config_name: alpha
  data_files:
  - split: test
    path: alpha/test-*
  - split: dev
    path: alpha/dev-*
- config_name: beta
  data_files:
  - split: test.clean
    path: [beta/test.clean-*]
- config_name: empty
  data_files:
  - split: test
    path: empty/test-*
---
# card
"""


def _write_shard(path: Path, first: int, rows: int, group_size: int) -> None:
    table = pa.table(
        {
            "audio": [{"bytes": f"wav{i}".encode(), "path": f"clip{i}.wav"} for i in range(first, first + rows)],
            "text": [f"text {i}" for i in range(first, first + rows)],
            "id": [f"id{i}" for i in range(first, first + rows)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=group_size)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch) -> dict:
    """A local stand-in for the Hub file system and downloads; records opened files, closed files, row groups read."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text(CARD)
    # Shards are created out of name order on purpose: reading must follow sorted names, not creation order.
    _write_shard(repo_dir / "alpha" / "test-00001-of-00002.parquet", first=6, rows=6, group_size=3)
    _write_shard(repo_dir / "alpha" / "test-00000-of-00002.parquet", first=0, rows=6, group_size=3)
    _write_shard(repo_dir / "alpha" / "dev-00000-of-00001.parquet", first=100, rows=2, group_size=2)
    _write_shard(repo_dir / "beta" / "test.clean-00000-of-00001.parquet", first=50, rows=2, group_size=2)
    (repo_dir / "chunks.jsonl").write_text('{"id": "a", "n": 1}\n\n{"id": "b", "n": 2}\n')
    state = {"opened": [], "closed": [], "groups": [], "block_sizes": []}

    class FakeStream:
        def __init__(self, path: Path) -> None:
            self.handle = path.open("rb")

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self.close()

        def close(self) -> None:
            state["closed"].append(self.handle.name)
            self.handle.close()

    class FakeFileSystem:
        def glob(self, pattern: str) -> list[str]:
            names = [f"{ROOT}/{p.relative_to(repo_dir)}" for p in repo_dir.rglob("*") if p.is_file()]
            return sorted(name for name in names if fnmatch.fnmatch(name, pattern))

        def open(self, path: str, mode: str, block_size: int | None = None):
            assert mode == "rb" and path.startswith(ROOT + "/")
            state["opened"].append(path)
            state["block_sizes"].append(block_size)
            return FakeStream(repo_dir / path[len(ROOT) + 1 :])

    class RecordingParquetFile(pq.ParquetFile):
        def iter_batches(self, batch_size=65536, row_groups=None, columns=None, use_threads=True, **kwargs):
            state["groups"].append((tuple(row_groups), use_threads))
            return super().iter_batches(
                batch_size=batch_size, row_groups=row_groups, columns=columns, use_threads=use_threads, **kwargs
            )

    def fake_download(repo, filename, *, repo_type, revision):
        assert (repo, repo_type, revision) == (REPO, "dataset", REVISION)
        return str(repo_dir / filename)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFileSystem)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pq, "ParquetFile", RecordingParquetFile)
    return state


def test_config_files_follow_the_card_split_and_sorted_order(hub: dict) -> None:
    assert hf_parquet.config_files(REPO, REVISION, "alpha", "test") == [
        "alpha/test-00000-of-00002.parquet",
        "alpha/test-00001-of-00002.parquet",
    ]
    assert hf_parquet.config_files(REPO, REVISION, "alpha", "dev") == ["alpha/dev-00000-of-00001.parquet"]
    assert hf_parquet.config_files(REPO, REVISION, "beta", "test.clean") == ["beta/test.clean-00000-of-00001.parquet"]


def test_config_files_reject_unknown_config_and_empty_split(hub: dict) -> None:
    with pytest.raises(ValueError, match="expected one config 'gamma'"):
        hf_parquet.config_files(REPO, REVISION, "gamma", "test")
    with pytest.raises(ValueError, match="expected one config None, found 3"):
        hf_parquet.config_files(REPO, REVISION, None, "test")  # no config named: the card must hold exactly one
    with pytest.raises(ValueError, match="no files for config 'empty'"):
        hf_parquet.config_files(REPO, REVISION, "empty", "test")
    with pytest.raises(ValueError, match="no files for config 'alpha' split 'validation'"):
        hf_parquet.config_files(REPO, REVISION, "alpha", "validation")


def test_iter_rows_reads_every_row_in_shard_order_with_the_audio_dict(hub: dict) -> None:
    files = hf_parquet.config_files(REPO, REVISION, "alpha", "test")
    rows = list(hf_parquet.iter_rows(REPO, REVISION, files))
    assert [row["id"] for row in rows] == [f"id{i}" for i in range(12)]
    assert rows[7] == {"audio": {"bytes": b"wav7", "path": "clip7.wav"}, "text": "text 7", "id": "id7"}
    assert hub["opened"] == [f"{ROOT}/{name}" for name in files]
    assert hub["closed"] and len(hub["closed"]) == len(files), "every shard is closed"
    assert hub["groups"] == [((0,), False), ((1,), False), ((0,), False), ((1,), False)], "single-threaded groups"
    assert set(hub["block_sizes"]) == {hf_parquet.BLOCK_SIZE}


def test_iter_rows_limit_stops_inside_the_first_row_groups(hub: dict) -> None:
    files = hf_parquet.config_files(REPO, REVISION, "alpha", "test")
    rows = list(hf_parquet.iter_rows(REPO, REVISION, files, limit=4))
    assert [row["id"] for row in rows] == ["id0", "id1", "id2", "id3"]
    assert hub["groups"] == [((0,), False), ((1,), False)], "only the row groups that hold the rows are read"
    assert hub["opened"] == [f"{ROOT}/{files[0]}"], "the second shard is never opened"
    assert len(hub["closed"]) == 1


def test_iter_rows_limit_spans_shards_and_zero_reads_nothing(hub: dict) -> None:
    files = hf_parquet.config_files(REPO, REVISION, "alpha", "test")
    assert [row["id"] for row in hf_parquet.iter_rows(REPO, REVISION, files, limit=7)] == [f"id{i}" for i in range(7)]
    assert len(hub["opened"]) == 2 and len(hub["closed"]) == 2
    hub["opened"].clear()
    assert list(hf_parquet.iter_rows(REPO, REVISION, files, limit=0)) == []
    assert hub["opened"] == []
    assert len(list(hf_parquet.iter_rows(REPO, REVISION, files, limit=1000))) == 12


def test_iter_rows_leaves_no_background_threads_or_open_shards(hub: dict) -> None:
    before = threading.active_count()
    files = hf_parquet.config_files(REPO, REVISION, "alpha", "test")
    rows = hf_parquet.iter_rows(REPO, REVISION, files)
    next(rows)
    rows.close()  # a consumer that stops early closes the shard it was reading
    assert len(hub["closed"]) == 1
    assert threading.active_count() == before


def test_iter_json_rows_skips_blank_lines(hub: dict) -> None:
    assert list(hf_parquet.iter_json_rows(REPO, REVISION, ["chunks.jsonl"])) == [
        {"id": "a", "n": 1},
        {"id": "b", "n": 2},
    ]
