# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from coverage import CoverageData
from full_ci_aggregate import merge_execution_outputs, verify_accounting
from full_ci_junit import compare_reports, summarize_reports
from full_ci_report import report


def coverage(path: Path, lines: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = CoverageData(basename=str(path))
    data.add_lines({"nemo_gym/example.py": set(lines)})
    data.write()


def fixture(root: Path, *, cache_source: str = "ran") -> tuple[Path, list[dict], Path]:
    checkout = root / "checkout"
    (checkout / ".venv/bin").mkdir(parents=True)
    python = checkout / ".venv/bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    collection = root / "collection.json"
    files = ["tests/unit_tests/test_a.py", "tests/unit_tests/test_b.py"]
    collection.write_text(
        json.dumps(
            {
                "valid": True,
                "source_sha": "sample",
                "marker": "not sandbox",
                "files": {name: [name + "::test_case"] for name in files},
                "assignments": dict.fromkeys(files, 0),
            }
        )
    )
    collection.with_suffix(".xml").write_text(
        '<testsuite><testcase file="tests/optional.py" classname="tests.optional" name="module">'
        '<skipped message="optional module"/></testcase></testsuite>'
    )
    coverage(collection.with_suffix(".coverage"), [1])
    batches = []
    for index, name in enumerate(files):
        directory = root / f"batch-{index}"
        (directory / "junit").mkdir(parents=True)
        (directory / "junit/test.xml").write_text(
            f'<testsuite><testcase file="{name}" classname="test_{index}" name="test_case"/></testsuite>'
        )
        address = f"//{name}:../../core-tests"
        (directory / "result-info.json").write_text(
            json.dumps({"run_id": 1, "info": {address: {"source": cache_source if index == 0 else "ran"}}})
        )
        coverage(directory / "coverage/.coverage", [index + 2])
        batches.append(
            {
                "id": str(index),
                "mode": "cacheable" if index == 0 else "forced",
                "directory": directory,
                "targets": [address],
            }
        )
    return checkout, batches, collection


def execute(root: Path, *, cache_source: str = "ran") -> tuple[Path, dict]:
    checkout, batches, collection = fixture(root, cache_source=cache_source)
    output = root / "output"
    result = merge_execution_outputs(
        checkout=checkout,
        batches=batches,
        collection_json=collection,
        output=output,
        group="core",
        shard=0,
        env=dict(os.environ),
    )
    return output, result


def test_cold_warm_and_changed_runs_preserve_all_cases_and_coverage(tmp_path):
    reports = []
    for lane, cache_source in (("cold", "ran"), ("warm", "hit_locally"), ("changed", "ran")):
        output, result = execute(tmp_path / lane, cache_source=cache_source)
        expected_hit = int(lane == "warm")
        assert result["counts"] == {
            "executed": 1 - expected_hit,
            "cached": expected_hit,
            "forced": 1,
            "collection_skips": 1,
            "total": 3,
        }
        data = CoverageData(basename=str(output / ".coverage.core-0"))
        data.read()
        assert data.lines("nemo_gym/example.py") == [1, 2, 3]
        current = summarize_reports(output / "junit")
        assert verify_accounting(output, report=current, source_sha="sample")["valid"]
        reports.append(current)
    assert all(compare_reports(reports[0], current)["equal"] for current in reports)


def test_collection_skip_is_added_once_and_real_case_multiplicity_is_retained(tmp_path):
    checkout, batches, collection = fixture(tmp_path, cache_source="hit_locally")
    path = batches[0]["directory"] / "junit/test.xml"
    tree = ElementTree.parse(path)
    original = tree.getroot().find("testcase")
    tree.getroot().append(ElementTree.fromstring(ElementTree.tostring(original)))
    tree.getroot().append(ElementTree.parse(collection.with_suffix(".xml")).getroot().find("testcase"))
    tree.write(path)
    output = tmp_path / "output"
    result = merge_execution_outputs(
        checkout=checkout,
        batches=batches,
        collection_json=collection,
        output=output,
        group="core",
        shard=0,
        env=dict(os.environ),
    )
    assert result["counts"] == {"executed": 0, "cached": 3, "forced": 1, "collection_skips": 0, "total": 4}
    current = summarize_reports(output / "junit")
    assert current["outcomes"] == {"passed": 3, "skipped": 1}
    assert next(case for case in current["cases"] if case["classname"] == "test_0")["count"] == 2


@pytest.mark.parametrize(
    "mutation", ["missing-target", "duplicate-target", "forced-hit", "missing-coverage", "memoized"]
)
def test_incomplete_or_ambiguous_batch_inputs_are_rejected(tmp_path, mutation):
    checkout, batches, collection = fixture(tmp_path)
    if mutation == "missing-target":
        batches.pop()
    elif mutation == "duplicate-target":
        batches.append(batches[0])
    elif mutation == "missing-coverage":
        (batches[0]["directory"] / "coverage/.coverage").unlink()
    else:
        batch = batches[1] if mutation == "forced-hit" else batches[0]
        path = batch["directory"] / "result-info.json"
        value = json.loads(path.read_text())
        value["info"][batch["targets"][0]]["source"] = "hit_locally" if mutation == "forced-hit" else "memoized"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        merge_execution_outputs(
            checkout=checkout,
            batches=batches,
            collection_json=collection,
            output=tmp_path / "output",
            group="core",
            shard=0,
            env=dict(os.environ),
        )


@pytest.mark.parametrize(
    "mutation", ["missing-coverage", "status", "cases", "union", "missing-accounting", "missing-digest"]
)
def test_retained_proof_is_checked_again_not_trusted(tmp_path, mutation):
    output, _ = execute(tmp_path, cache_source="hit_locally")
    path = output / "test-accounting.json"
    value = json.loads(path.read_text())
    if mutation == "missing-coverage":
        (output / "execution-batches/0/.coverage").unlink()
    elif mutation == "missing-accounting":
        path.unlink()
    else:
        if mutation == "status":
            value["batches"][0]["status"] = "executed"
        elif mutation == "cases":
            value["batches"][0]["cases"][0]["count"] = 2
        elif mutation == "missing-digest":
            value["batches"][0]["files"].pop("execution-batches/0/.coverage")
        else:
            # Even a self-consistent hash cannot conceal omitted cached coverage.
            raw = output / ".coverage.core-0"
            raw.unlink()
            coverage(raw, [1, 3])
            value["combined_files"][raw.name] = hashlib.sha256(raw.read_bytes()).hexdigest()
        path.write_text(json.dumps(value))
    assert not verify_accounting(output, report=summarize_reports(output / "junit"), source_sha="sample")["valid"]


def test_full_report_requires_incremental_provenance_and_keeps_full_suite_parity(tmp_path):
    native, _ = execute(tmp_path / "proof", cache_source="hit_locally")
    artifacts = tmp_path / "artifacts"
    items = []
    jobs = []
    for engine in ("baseline", "pants"):
        directory = artifacts / f"full-ci-{engine}-core-0"
        directory.mkdir(parents=True)
        if engine == "pants":
            shutil.copytree(native, directory / "native")
        shutil.copytree(native / "junit", directory / "junit")
        shutil.copy2(native / ".coverage.core-0", directory / ".coverage.core-0")
        (directory / "manifest.json").write_text(json.dumps({"job": "core", "shard": 0}))
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "exit_code": 0,
                    "command_exit_code": 0,
                    "source_sha": "sample",
                    "implementation": "native Python test targets" if engine == "pants" else "unchanged CI commands",
                    "execution_mode": "incremental" if engine == "pants" else "forced",
                    "pants_result_cache": "incremental-native-tests" if engine == "pants" else None,
                }
            )
        )
        items.append({"engine": engine, "job": "core", "shard": 0})
        for job in ("core / 0", "combined coverage"):
            jobs.append(
                {
                    "name": f"{engine} / {job}",
                    "conclusion": "success",
                    "started_at": "2026-11-05T00:00:00Z",
                    "completed_at": "2026-11-05T00:01:00Z",
                }
            )
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "source_sha": "sample",
                "experiment_sha": "controller",
                "run_full": "true",
                "run_servers": "false",
                "matrix": {"include": items},
            }
        )
    )
    job_file = tmp_path / "jobs.json"
    job_file.write_text(json.dumps({"jobs": jobs}))
    args = argparse.Namespace(route=route, jobs=job_file, run=None, artifacts=artifacts, output=tmp_path / "summary")
    report(args)
    result = json.loads((args.output / "results.json").read_text())
    assert result["valid"]
    assert result["test_accounting"]["pants/core-0"]["counts"]["cached"] == 1
    routing = json.loads(route.read_text())
    routing["execution_mode"] = "forced"
    route.write_text(json.dumps(routing))
    with pytest.raises(SystemExit):
        report(args)
    assert any(
        "execution mode differs" in error for error in json.loads((args.output / "results.json").read_text())["errors"]
    )
    routing["execution_mode"] = "incremental"
    route.write_text(json.dumps(routing))
    (artifacts / "full-ci-pants-core-0/native/test-accounting.json").unlink()
    with pytest.raises(SystemExit):
        report(args)
    assert not json.loads((args.output / "results.json").read_text())["valid"]
