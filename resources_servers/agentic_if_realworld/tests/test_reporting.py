# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from benchmarks.agentic_if_realworld.rollout_driver import write_reports
from resources_servers.agentic_if_realworld.app import FixtureVerifier, VerifyRequest
from resources_servers.agentic_if_realworld.benchmark.core import read
from resources_servers.agentic_if_realworld.benchmark.synthetic import delivery


def test_complete_accuracy_stays_null_until_all_tasks_scored(tmp_path):
    verifier = FixtureVerifier()
    try:
        output = tmp_path / "rollouts.jsonl"
        write_reports(output, [], verifier.dataset)
        metrics = read(tmp_path / "rollouts_aggregate_metrics.json")[0]["key_metrics"]
        assert metrics["composite/complete_accuracy"] is None
        assert metrics["not_scored"] == 1 and metrics["scoring_coverage"] == 0
        body = VerifyRequest.model_validate(delivery(verifier.dataset, verifier.receipt_dir, "OK.."))
        result = asyncio.run(verifier.verify(body)).model_dump()
        write_reports(output, [result], verifier.dataset)
        metrics = read(tmp_path / "rollouts_aggregate_metrics.json")[0]["key_metrics"]
        assert metrics["composite/complete_accuracy"] == 0
        assert metrics["fail"] == 1 and metrics["scoring_coverage"] == 1
    finally:
        verifier.work.cleanup()
