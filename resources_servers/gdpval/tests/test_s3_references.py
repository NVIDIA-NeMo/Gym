# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference deliverables downloaded from S3 one task at a time."""

import asyncio

import pytest

from resources_servers.gdpval import persisted_layout
from resources_servers.gdpval.s3_references import S3ReferenceCache, is_s3_uri


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.downloads: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        assert Bucket == "evals"
        yield {"Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)]}

    def download_file(self, bucket, key, filename):
        self.downloads.append(key)
        with open(filename, "wb") as handle:
            handle.write(self.objects[key])


URI = "s3://evals/datasets/anchors/kimi_k26"
TASK = "7b08cd4d"


def _objects() -> dict[str, bytes]:
    base = f"datasets/anchors/kimi_k26/task_{TASK}/repeat_0"
    return {
        f"{base}/Report.xlsx": b"xlsx",
        f"{base}/Report.pdf": b"%PDF",
        f"{base}/finish_params.json": b'{"paths": ["Report.xlsx"]}',
        f"{base}/history.json": b"[]",
        f"{base}/history.pkl": b"pickle",
        f"{base}/reference_files/abc/Ref File.xlsx": b"ref",
        "datasets/anchors/kimi_k26/task_other/repeat_0/Other.xlsx": b"other",
    }


def test_is_s3_uri():
    assert is_s3_uri(URI)
    assert not is_s3_uri("/gdpval/refs/kimi_k26")


async def test_only_the_judged_task_is_downloaded_without_transcripts(tmp_path):
    s3 = FakeS3(_objects())
    cache = S3ReferenceCache(tmp_path / "cache", client_factory=lambda: s3)

    root = await cache.local_root(URI, TASK)

    repeat = persisted_layout.repeat_dir(root, TASK, 0)
    assert sorted(p.relative_to(repeat).as_posix() for p in repeat.rglob("*") if p.is_file()) == [
        "Report.pdf",
        "Report.xlsx",
        "finish_params.json",
        "reference_files/abc/Ref File.xlsx",
    ]
    assert not any("task_other" in key or "history" in key for key in s3.downloads)


async def test_a_task_is_downloaded_once_even_when_requested_concurrently(tmp_path):
    s3 = FakeS3(_objects())
    cache = S3ReferenceCache(tmp_path / "cache", client_factory=lambda: s3)

    roots = await asyncio.gather(*(cache.local_root(URI, TASK) for _ in range(3)))

    assert len(set(roots)) == 1
    assert len(s3.downloads) == 4


async def test_a_task_the_reference_lacks_has_no_directory_and_is_listed_once(tmp_path):
    s3 = FakeS3(_objects())
    listings = []
    paginate = s3.paginate
    s3.paginate = lambda **kwargs: listings.append(kwargs) or paginate(**kwargs)
    cache = S3ReferenceCache(tmp_path / "cache", client_factory=lambda: s3)

    for _ in range(2):
        root = await cache.local_root(URI, "missing")

    assert persisted_layout.repeat_dirs(persisted_layout.task_dir(root, "missing")) == []
    assert len(listings) == 1


async def test_a_failed_download_leaves_nothing_cached(tmp_path):
    s3 = FakeS3(_objects())
    s3.download_file = lambda *args: (_ for _ in ()).throw(OSError("connection reset"))
    cache = S3ReferenceCache(tmp_path / "cache", client_factory=lambda: s3)

    with pytest.raises(OSError):
        await cache.local_root(URI, TASK)

    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []
