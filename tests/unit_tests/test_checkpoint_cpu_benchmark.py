# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from scripts.benchmark_checkpoint_cpu import (
    _worker_counts,
    run_checkpoint_benchmark,
    run_checkpoint_overlap_benchmark,
    run_control_benchmark,
)


def test_cpu_benchmark_worker_distribution_preserves_total_and_skew() -> None:
    counts = _worker_counts(100, 8, 0.45)

    assert counts[0] == 45
    assert sum(counts) == 100
    assert max(counts[1:]) - min(counts[1:]) <= 1


@pytest.mark.asyncio
async def test_cpu_control_benchmark_runs_and_bulk_acknowledges_every_result() -> None:
    result = await run_control_benchmark(
        runs=32,
        concurrency=8,
        ack_batch_size=8,
        ack_concurrency=2,
        ack_coalesce_ms=10,
        delay_ms=0,
        data_connections=8,
        control_connections=4,
        transport="asgi",
    )

    assert result["runs"] == 32
    assert result["acknowledged"] == 32
    assert result["ack_calls"] == 4
    assert result["minimum_ack_batch_size"] == 8
    assert result["maximum_ack_batch_size"] == 8


@pytest.mark.asyncio
async def test_cpu_checkpoint_benchmark_cuts_commits_and_restores(tmp_path) -> None:
    long_run_root = tmp_path / ("descriptive-checkpoint-benchmark-root-" + "x" * 80)
    result = await run_checkpoint_benchmark(
        long_run_root,
        workers=2,
        cuts=8,
        hot_worker_fraction=0.5,
        prefix_tokens=16,
        staging_key_bytes=8,
        timeout_s=30,
    )

    assert result["prepare_records"] == 8
    assert result["journal_files"] == 2
    assert result["artifact_reference_wire_bytes_total"] < result["journal_bytes"]
    assert result["generation_cuts_restored"] == 8
    assert result["restored_lineage_files"] == 8


@pytest.mark.asyncio
async def test_cpu_overlap_benchmark_interrupts_coalescing_and_drains_prepare() -> None:
    result = await run_checkpoint_overlap_benchmark(
        runs=64,
        agents=4,
        hot_agent_fraction=0.5,
        completion_batch_size=4,
        completion_interval_ms=1,
        ack_batch_size=8,
        ack_concurrency=2,
        normal_ack_coalesce_ms=1_000,
        checkpoint_ack_coalesce_ms=5,
        progress_interval_s=0.01,
        timeout_s=30,
    )

    assert result["acknowledged"] == 64
    assert result["ack_calls"] >= 8
    assert result["checkpoint_flush_latency_seconds"] < 0.5
    assert result["final_running"] == 0
    assert result["final_parked_without_boundary"] == 0
    assert result["final_completed_unacknowledged"] == 0
    assert result["maximum_ack_pending"] > 0


@pytest.mark.asyncio
async def test_cpu_checkpoint_benchmark_handles_mixed_inventory(tmp_path) -> None:
    result = await run_checkpoint_benchmark(
        tmp_path,
        workers=2,
        cuts=12,
        hot_worker_fraction=0.5,
        prefix_tokens=16,
        staging_key_bytes=8,
        timeout_s=30,
        mixed_inventory=True,
    )

    assert result["inventory_counts"] == {
        "active_prefix": 2,
        "durable_completed": 2,
        "durable_failure": 2,
        "no_generation": 2,
        "pre_generation": 2,
        "response_egress": 2,
    }
    assert result["prepare_records"] == 2
    assert result["generation_cuts_restored"] == 2
    assert result["restored_prefix_coordinates_match"] is True
