# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil

import pytest
from full_ci_aggregate import merge_execution_outputs
from full_ci_report import report
from test_full_ci_aggregate import coverage, execute, fixture


def legacy_case(root):
    core, _ = execute(root / "core", cache_source="hit_locally")
    checkout, batches, collection = fixture(root / "sandbox")
    value = json.loads(collection.read_text())
    value["marker"] = "sandbox"
    value["shards"] = 1
    collection.write_text(json.dumps(value))
    for index, batch in enumerate(batches):
        batch["mode"] = "forced"
        batch["targets"] = [target.replace("core-tests", "sandbox-tests") for target in batch["targets"]]
        path = batch["directory"] / "result-info.json"
        path.write_text(json.dumps({"run_id": 1, "info": {batch["targets"][0]: {"source": "ran"}}}))
        coverage(batch["directory"] / "coverage/.coverage", [10 + index])
    sandbox = root / "sandbox/output"
    merge_execution_outputs(
        checkout=checkout,
        batches=batches,
        collection_json=collection,
        output=sandbox,
        group="sandbox",
        shard=0,
        env=dict(os.environ),
    )
    artifacts = root / "artifacts"
    items, jobs = [], []
    for engine in ("baseline", "pants"):
        directory = artifacts / f"full-ci-{engine}-legacy-core-0"
        (directory / "junit").mkdir(parents=True)
        for group, source in (("core", core), ("sandbox", sandbox)):
            shutil.copy2(source / f"junit/{group}.xml", directory / f"junit/{group}.xml")
            if engine == "pants":
                shutil.copytree(source, directory / "native" / group)
        coverage(directory / ".coverage.legacy-core", [1, 2, 3, 10, 11])
        (directory / "manifest.json").write_text(json.dumps({"job": "legacy-core", "shard": 0, "shards": 1}))
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "source_sha": "sample",
                    "exit_code": 0,
                    "command_exit_code": 0,
                    "implementation": "native Python test targets" if engine == "pants" else "unchanged CI commands",
                    "execution_mode": "incremental" if engine == "pants" else "forced",
                    "pants_result_cache": "incremental-native-tests" if engine == "pants" else None,
                }
            )
        )
        items.append({"engine": engine, "job": "legacy-core", "shard": 0})
        jobs.append(
            {
                "name": f"{engine} / legacy-core / 0",
                "conclusion": "success",
                "started_at": "2026-11-05T00:00:00Z",
                "completed_at": "2026-11-05T00:01:00Z",
            }
        )
    route = root / "route.json"
    route.write_text(
        json.dumps(
            {
                "source_sha": "sample",
                "experiment_sha": "controller",
                "execution_mode": "incremental",
                "run_full": "true",
                "run_servers": "false",
                "separate_coverage": False,
                "matrix": {"include": items},
            }
        )
    )
    job_file = root / "jobs.json"
    job_file.write_text(json.dumps({"jobs": jobs}))
    return argparse.Namespace(route=route, jobs=job_file, run=None, artifacts=artifacts, output=root / "summary")


def test_legacy_core_counts_both_groups_and_the_combined_gate_without_extra_job(tmp_path):
    args = legacy_case(tmp_path)
    report(args)
    value = json.loads((args.output / "results.json").read_text())
    assert value["valid"]
    assert value["test_accounting"]["pants/legacy-core-0"]["counts"] == {
        "forced": 3,
        "executed": 0,
        "cached": 1,
        "collection_skips": 2,
        "total": 6,
    }
    assert value["coverage"]["pants/legacy-core-0"]["executed_line_count"] == 5
    assert value["timing"]["pants"]["job_count"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["missing-sandbox-proof", "missing-copied-report", "lost-sandbox-coverage", "failed-gate", "extra-coverage-job"],
)
def test_legacy_core_fails_closed_for_partial_groups_or_failed_gate(tmp_path, mutation):
    args = legacy_case(tmp_path)
    directory = args.artifacts / "full-ci-pants-legacy-core-0"
    if mutation == "missing-sandbox-proof":
        (directory / "native/sandbox/test-accounting.json").unlink()
    elif mutation == "missing-copied-report":
        (directory / "junit/sandbox.xml").unlink()
    elif mutation == "lost-sandbox-coverage":
        (directory / ".coverage.legacy-core").unlink()
        coverage(directory / ".coverage.legacy-core", [1, 2, 3])
    elif mutation == "failed-gate":
        value = json.loads((directory / "metadata.json").read_text())
        value["exit_code"] = 1
        (directory / "metadata.json").write_text(json.dumps(value))
    else:
        value = json.loads(args.jobs.read_text())
        value["jobs"].append(dict(value["jobs"][-1], name="pants / combined coverage"))
        args.jobs.write_text(json.dumps(value))
    with pytest.raises(SystemExit):
        report(args)
    assert not json.loads((args.output / "results.json").read_text())["valid"]
