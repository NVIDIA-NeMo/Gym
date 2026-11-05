# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from coverage import CoverageData
from full_ci_aggregate import merge_execution_outputs, normalize_coverage, verify_accounting
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


@pytest.mark.parametrize("arcs", [False, True])
def test_absolute_collection_paths_become_portable_without_losing_contexts_or_raw_evidence(tmp_path, arcs):
    checkout, batches, collection = fixture(tmp_path)
    source = checkout / "nemo_gym/example.py"
    source.parent.mkdir()
    source.write_text("one = 1\ntwo = 2\nthree = 3\n")
    inputs = [collection.with_suffix(".coverage"), *(batch["directory"] / "coverage/.coverage" for batch in batches)]
    contexts = ["collection", "cached-test", "forced-test"]
    line_sets = [{1, 2}, {2}, {3}]
    arc_sets = [{(-1, 1), (1, 2)}, {(2, -1)}, {(-1, 3), (3, -1)}]
    for index, path in enumerate(inputs):
        path.unlink()
        data = CoverageData(basename=str(path))
        data.set_context(contexts[index])
        name = str(source.resolve()) if index == 0 else "nemo_gym/example.py"
        if arcs:
            data.add_arcs({name: arc_sets[index]})
        else:
            data.add_lines({name: line_sets[index]})
        data.write()
    original_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
    output = tmp_path / "output"
    merge_execution_outputs(
        checkout=checkout,
        batches=batches,
        collection_json=collection,
        output=output,
        group="core",
        shard=0,
        env=dict(os.environ),
    )
    raw = output / ".coverage.core-0"
    data = CoverageData(basename=str(raw))
    data.read()
    assert data.measured_files() == {"nemo_gym/example.py"}
    assert data.measured_contexts() == set(contexts)
    if arcs:
        assert set(data.arcs("nemo_gym/example.py")) == set.union(*arc_sets)
    else:
        assert {line: set(names) for line, names in data.contexts_by_lineno("nemo_gym/example.py").items()} == {
            1: {"collection"},
            2: {"collection", "cached-test"},
            3: {"forced-test"},
        }
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs] == original_hashes
    original = CoverageData(basename=str(output / "execution-collection/collection.coverage"))
    original.read()
    assert original.measured_files() == {str(source.resolve())}
    final_checkout = tmp_path / "final-checkout"
    shutil.copytree(source.parent, final_checkout / "nemo_gym")
    shutil.rmtree(checkout)
    # The legacy gate combines portable inputs then rebases them to its existing
    # sampled workspace. Preserve the native artifact hash/accounting unchanged.
    legacy_raw = tmp_path / ".coverage.legacy-core"
    shutil.copy2(raw, legacy_raw)
    normalize_coverage(legacy_raw, final_checkout, absolute=True)
    # Exercise the actual later gate after the source checkout used for collection
    # no longer exists. No path aliases or ignore-errors mask a missing source.
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--fail-under=100"],
        cwd=final_checkout,
        env=dict(os.environ, COVERAGE_FILE=str(legacy_raw)),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "100%" in result.stdout
    assert verify_accounting(output, report=summarize_reports(output / "junit"), source_sha="sample")["valid"]


def test_portable_coverage_does_not_rewrite_lookalike_prefixes_or_external_symlinks(tmp_path):
    checkout, batches, collection = fixture(tmp_path)
    outside = tmp_path / "checkout-other/nemo_gym/outside.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("outside = 1\n")
    link = checkout / "nemo_gym/link.py"
    link.parent.mkdir()
    link.symlink_to(outside)
    raw = collection.with_suffix(".coverage")
    raw.unlink()
    data = CoverageData(basename=str(raw))
    data.add_lines({str(outside): {1}, str(link): {1}})
    data.write()
    output = tmp_path / "output"
    merge_execution_outputs(
        checkout=checkout,
        batches=batches,
        collection_json=collection,
        output=output,
        group="core",
        shard=0,
        env=dict(os.environ),
    )
    final = CoverageData(basename=str(output / ".coverage.core-0"))
    final.read()
    assert str(outside) in final.measured_files()
    assert str(link) in final.measured_files()
    assert "nemo_gym/link.py" not in final.measured_files()


@pytest.mark.parametrize("name", ["nemo_gym/missing.py", "../outside.py", "nemo_gym/link.py"])
def test_legacy_rebase_keeps_unverified_names_for_the_original_report_policy(tmp_path, name):
    checkout = tmp_path / "checkout"
    (checkout / "nemo_gym").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("outside = 1\n")
    (checkout / "nemo_gym/link.py").symlink_to(outside)
    raw = tmp_path / ".coverage"
    data = CoverageData(basename=str(raw))
    data.add_lines({name: {1}})
    data.write()
    normalize_coverage(raw, checkout, absolute=True)
    normalized = CoverageData(basename=str(raw))
    normalized.read()
    assert normalized.measured_files() == {name}
    assert normalized.lines(name) == [1]
    if name == "nemo_gym/missing.py":
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "report"],
            cwd=checkout,
            env=dict(os.environ, COVERAGE_FILE=str(raw)),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "No source for code" in result.stdout
