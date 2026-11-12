# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retention must never delete caches outside the authorized experiment."""

from scripts.pants.replay_cache import LIMIT, REF, plan_removals


def cache(number, key, size=1, ref=REF, version="v1"):
    return dict(
        id=number,
        key=key,
        ref=ref,
        version=version,
        size_in_bytes=size,
        created_at=f"2026-11-{number:02}T00:00:00Z",
        last_accessed_at=f"2026-11-{number:02}T00:00:00Z",
    )


def test_replay_namespace_and_uv_generation_retention_preserve_unrelated_caches():
    current = "pants-full-v2-uv-replay-abc-baseline-Linux-"
    entries = [
        cache(1, current + "a" * 64),
        cache(2, current + "b" * 64),
        cache(3, "pants-full-v2-uv-live-baseline-Linux-" + "c" * 64),
        cache(4, current + "d" * 64, ref="refs/heads/main"),
        cache(5, "unrelated"),
    ]
    removed, remaining = plan_removals(entries, namespace="replay-abc")
    assert {r["cache"]["id"] for r in removed} == {1, 3}
    assert [r["id"] for r in remaining] == [2]


def test_distinct_shards_and_versions_keep_newest_results():
    stem = "pants-full-results-v1-replay-abc-Linux-X64-core-0-contract-deps-"
    entries = [
        cache(1, stem + "a" * 40),
        cache(2, stem + "b" * 40),
        cache(3, stem.replace("core-0", "core-1") + "a" * 40),
        cache(4, stem + "a" * 40, version="v2"),
    ]
    removed, remaining = plan_removals(entries, contract="contract", namespace="replay-abc")
    assert [r["cache"]["id"] for r in removed] == [1]
    assert {r["id"] for r in remaining} == {2, 3, 4}


def test_optional_engine_archive_removed_before_only_result_archive():
    result = cache(1, "pants-full-results-v1-r-core-0-c-deps-" + "a" * 40, size=LIMIT // 2)
    engine = cache(2, "pants-full-engine-v4-r-session-c-deps-" + "b" * 40, size=LIMIT)
    removed, remaining = plan_removals([result, engine], enforce_limit=True)
    assert removed[0]["cache"]["id"] == 2
    assert remaining == [result]
