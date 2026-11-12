# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve full-suite evidence while accounting for native execution and reuse."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal, TypedDict
from xml.etree import ElementTree

from full_ci_junit import Case, Report, summarize_reports


def normalize_coverage(raw: Path, checkout: Path, *, absolute: bool = False) -> None:
    """Rebase verified source paths without losing contexts, arcs or line unions.

    Import coverage only in the sampled interpreter, which owns its data format.
    Native outputs use portable paths; legacy report configs require those paths
    rebased to the final checkout after the temporary collection checkout is gone.
    """
    from coverage import CoverageData

    root = checkout.resolve()
    prefixes = {checkout.absolute(), root}

    def map_path(name: str) -> str:
        path = Path(name)
        if absolute:
            if path.is_absolute():
                return name
            source = root / path
            if source.is_file() and source.resolve().is_relative_to(root):
                return str(source.resolve())
            # Generated helpers can be absent and intentionally omitted by the
            # sampled report config. Keep unknown names for its normal rules.
            return name
        if path.is_absolute() and path.is_file() and path.resolve().is_relative_to(root):
            for prefix in prefixes:
                if path.is_relative_to(prefix):
                    return path.relative_to(prefix).as_posix()
        return name

    original = CoverageData(basename=str(raw))
    original.read()
    # Resolve the mapping once before writing the replacement database.
    mapped_paths = {name: map_path(name) for name in original.measured_files()}
    with tempfile.TemporaryDirectory(prefix="coverage-relative-", dir=raw.parent) as temporary:
        portable = CoverageData(basename=str(Path(temporary) / "data"))
        portable.update(original, map_path=mapped_paths.__getitem__)
        portable.write()
        Path(portable.data_filename()).replace(raw)


class Batch(TypedDict):
    id: str
    mode: Literal["forced", "cacheable"]
    directory: Path
    targets: list[str]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counter(cases: list[Case]) -> Counter[tuple[str, str, str, str]]:
    return Counter({(c["file"], c["classname"], c["name"], c["outcome"]): c["count"] for c in cases})


def _status(mode: str, provenance: dict, targets: list[str]) -> str:
    if mode not in {"forced", "cacheable"} or not targets or len(targets) != len(set(targets)):
        raise ValueError("Invalid native batch mode or targets")
    info = provenance.get("info")
    if not isinstance(info, dict) or list(info) != [min(targets)] or "run_id" not in provenance:
        raise ValueError("Expected one authentic Pants result-info batch matching its target representative")
    if not isinstance(info[min(targets)], dict):
        raise ValueError("Malformed Pants batch provenance")
    source = info[min(targets)].get("source")
    if source not in {"ran", "hit_locally", "hit_remotely"}:
        raise ValueError(f"Unsupported or ambiguous Pants result source: {source}")
    if mode == "forced" and source != "ran":
        raise ValueError("Forced batch unexpectedly reused a result")
    return "forced" if mode == "forced" else "executed" if source == "ran" else "cached"


def _safe_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping execution evidence: {name}")
    return path


def _selected_files(collection: dict, group: str, shard: int) -> set[str]:
    if not collection.get("valid"):
        raise ValueError("Global collection did not succeed")
    marker = "not sandbox" if group == "core" else "sandbox"
    if group not in {"core", "sandbox"} or collection.get("marker") != marker:
        raise ValueError("Global collection does not match native group")
    return {path for path in collection["files"] if group == "sandbox" or collection["assignments"][path] == shard}


def _coverage_lines(path: Path) -> dict[str, int]:
    """Read normalized repository line sets without installing coverage in the reporter."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        names = {}
        for number, name in database.execute("SELECT id, path FROM file"):
            name = name.replace("\\", "/")
            if name.startswith("nemo_gym/") or "/nemo_gym/" in name:
                names[number] = "nemo_gym/" + name.split("nemo_gym/", 1)[1]
        result = {name: 0 for name in names.values()}
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "line_bits" in tables:
            for number, bitmap in database.execute("SELECT file_id, numbits FROM line_bits"):
                if number in names:
                    result[names[number]] |= int.from_bytes(bitmap, "little")
        if "arc" in tables:
            for number, start, end in database.execute("SELECT file_id, fromno, tono FROM arc"):
                if number in names:
                    for line in (start, end):
                        if line > 0:
                            result[names[number]] |= 1 << line
        return result


def merge_execution_outputs(
    *,
    checkout: Path,
    batches: list[Batch],
    collection_json: Path,
    output: Path,
    group: str,
    shard: int,
    env: dict[str, str],
) -> dict:
    """Combine every executed/reused batch and collection evidence exactly once.

    Each input directory must contain one Pants pytest batch, its verbatim
    result-info.json, junit/*.xml and coverage/.coverage. Cache provenance is
    read from Pants, never inferred from elapsed time or requested cache mode.
    """
    collection = json.loads(collection_json.read_text())
    selected = _selected_files(collection, group, shard)
    addresses = [target for batch in batches for target in batch["targets"]]
    paths = [target.split(":", 1)[0].removeprefix("//") for target in addresses]
    if len(paths) != len(set(paths)) or set(paths) != selected:
        raise ValueError("Native batches omitted or duplicated globally selected test files")
    combined = ElementTree.Element("testsuites")
    records = []
    raw_inputs = [collection_json.with_suffix(".coverage")]
    ids: set[str] = set()
    output.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        identifier = batch["id"]
        if not identifier or Path(identifier).name != identifier or identifier in {".", ".."} or identifier in ids:
            raise ValueError("Batch identifiers must be unique directory names")
        ids.add(identifier)
        source = batch["directory"]
        provenance = json.loads((source / "result-info.json").read_text())
        status = _status(batch["mode"], provenance, batch["targets"])
        report = summarize_reports(source / "junit")
        if not report["valid"] or len(report["reports"]) != 1:
            raise ValueError(f"Incomplete batch JUnit: {identifier}: {report['errors']}")
        raw = source / "coverage/.coverage"
        if not raw.is_file() or not raw.stat().st_size:
            raise ValueError(f"Missing batch coverage: {identifier}")
        destination = output / "execution-batches" / identifier
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source / "junit", destination / "junit")
        shutil.copy2(source / "result-info.json", destination / "result-info.json")
        shutil.copy2(raw, destination / ".coverage")
        raw_inputs.append(destination / ".coverage")
        files = list((destination / "junit").rglob("*.xml")) + [
            destination / "result-info.json",
            destination / ".coverage",
        ]
        for path in files:
            if path.suffix == ".xml":
                tree = ElementTree.parse(path).getroot()
                combined.extend([tree] if tree.tag == "testsuite" else list(tree))
        records.append(
            {
                "id": identifier,
                "mode": batch["mode"],
                "status": status,
                "targets": batch["targets"],
                "testcase_count": report["testcase_count"],
                "cases": report["cases"],
                "files": {path.relative_to(output).as_posix(): _digest(path) for path in files},
            }
        )
    # Collection emits genuine module skips; normal collected test items are not
    # executed cases. Retain only skips not already emitted by a selected batch.
    collection_suite = ElementTree.Element("testsuite", name="collection", errors="0", failures="0")
    seen = Counter(
        tuple(case.get(key, "") for key in ("file", "classname", "name"))
        for case in combined.iter("testcase")
        if case.find("skipped") is not None
    )
    for case in ElementTree.parse(collection_json.with_suffix(".xml")).iter("testcase"):
        if case.find("skipped") is None:
            raise ValueError("Collect-only JUnit unexpectedly contains an executed or failed testcase")
        identity = tuple(case.get(key, "") for key in ("file", "classname", "name"))
        if seen[identity]:
            seen[identity] -= 1
        else:
            collection_suite.append(case)
    if len(collection_suite):
        collection_suite.set("tests", str(len(collection_suite)))
        collection_suite.set("skipped", str(len(collection_suite)))
        combined.append(collection_suite)
    junit = output / "junit" / f"{group}.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(combined).write(junit, encoding="utf-8", xml_declaration=True)
    collection_output = output / "execution-collection"
    collection_output.mkdir()
    for suffix in (".json", ".xml", ".coverage"):
        shutil.copy2(collection_json.with_suffix(suffix), collection_output / ("collection" + suffix))
    raw_inputs[0] = collection_output / "collection.coverage"
    coverage = output / (f".coverage.core-{shard}" if group == "core" else ".coverage.sandbox")
    subprocess.run(
        [str(checkout / ".venv/bin/python"), "-m", "coverage", "combine", "--keep", *map(str, raw_inputs)],
        cwd=checkout,
        env=dict(env, COVERAGE_FILE=str(coverage.resolve())),
        capture_output=True,
        text=True,
        check=True,
    )
    # Older main revisions record absolute collection paths. The disposable
    # checkout is removed before the final gate, so use the sampled coverage
    # API to merge verified checkout paths into portable names, retaining every
    # context/arc and leaving the original collection and batch evidence intact.
    subprocess.run(
        [
            str(checkout / ".venv/bin/python"),
            str(Path(__file__).resolve()),
            "normalize-coverage",
            "--coverage",
            str(coverage.resolve()),
            "--checkout",
            str(checkout.absolute()),
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = summarize_reports(junit.parent, expected_reports=[junit.name])
    totals = {
        status: sum(r["testcase_count"] for r in records if r["status"] == status)
        for status in ("executed", "cached", "forced")
    }
    totals["collection_skips"] = len(collection_suite)
    totals["total"] = report["testcase_count"]
    result = {
        "schema_version": 1,
        "source_sha": collection["source_sha"],
        "group": group,
        "shard": shard,
        "batches": records,
        "counts": totals,
        "collection_files": {p.relative_to(output).as_posix(): _digest(p) for p in collection_output.iterdir()},
        "combined_files": {p.relative_to(output).as_posix(): _digest(p) for p in (junit, coverage)},
    }
    (output / "test-accounting.json").write_text(json.dumps(result, indent=2) + "\n")
    validation = verify_accounting(output, report=report, source_sha=collection["source_sha"])
    if not validation["valid"]:
        raise ValueError("Invalid native accounting: " + "; ".join(validation["errors"]))
    return result


def verify_accounting(root: Path, *, report: Report, source_sha: str) -> dict:
    """Validate retained execution/cache provenance against the complete JUnit."""
    errors = []
    counts = {status: 0 for status in ("executed", "cached", "forced", "collection_skips", "total")}
    try:
        accounting = json.loads((root / "test-accounting.json").read_text())
        if accounting.get("schema_version") != 1 or accounting.get("source_sha") != source_sha:
            raise ValueError("Accounting version or sampled source differs")
        collection_names = {f"execution-collection/collection{suffix}" for suffix in (".json", ".xml", ".coverage")}
        coverage_name = (
            f".coverage.core-{accounting['shard']}" if accounting["group"] == "core" else ".coverage.sandbox"
        )
        if set(accounting["collection_files"]) != collection_names or set(accounting["combined_files"]) != {
            f"junit/{accounting['group']}.xml",
            coverage_name,
        }:
            raise ValueError("Accounting omitted required collection or combined evidence digests")
        for fields in [
            accounting["collection_files"],
            accounting["combined_files"],
            *(r["files"] for r in accounting["batches"]),
        ]:
            for name, digest in fields.items():
                if _digest(_safe_file(root, name)) != digest:
                    raise ValueError(f"Execution evidence digest mismatch: {name}")
        collection = json.loads((root / "execution-collection/collection.json").read_text())
        if collection.get("source_sha") != source_sha:
            raise ValueError("Collection source differs from accounting")
        selected = _selected_files(collection, accounting["group"], accounting["shard"])
        paths = []
        cases: Counter[tuple[str, str, str, str]] = Counter()
        ids = set()
        coverage_inputs = [root / "execution-collection/collection.coverage"]
        for batch in accounting["batches"]:
            identifier = batch["id"]
            if identifier in ids or Path(identifier).name != identifier or identifier in {".", ".."}:
                raise ValueError("Duplicate or invalid batch identifier")
            ids.add(identifier)
            directory = root / "execution-batches" / identifier
            provenance = json.loads(_safe_file(root, f"execution-batches/{identifier}/result-info.json").read_text())
            status = _status(batch["mode"], provenance, batch["targets"])
            current = summarize_reports(directory / "junit")
            expected_files = {
                f"execution-batches/{identifier}/result-info.json",
                f"execution-batches/{identifier}/.coverage",
                *(f"execution-batches/{identifier}/junit/{name}" for name in current["reports"]),
            }
            if set(batch["files"]) != expected_files:
                raise ValueError(f"Batch evidence digest inventory differs: {identifier}")
            if not current["valid"] or len(current["reports"]) != 1 or current["cases"] != batch["cases"]:
                raise ValueError(f"Batch testcase evidence differs: {identifier}")
            if status != batch["status"] or current["testcase_count"] != batch["testcase_count"]:
                raise ValueError(f"Batch execution status/count differs: {identifier}")
            if not _safe_file(root, f"execution-batches/{identifier}/.coverage").stat().st_size:
                raise ValueError(f"Empty coverage: {identifier}")
            coverage_inputs.append(directory / ".coverage")
            counts[status] += current["testcase_count"]
            cases.update(_counter(current["cases"]))
            paths.extend(target.split(":", 1)[0].removeprefix("//") for target in batch["targets"])
        if len(paths) != len(set(paths)) or set(paths) != selected:
            raise ValueError("Accounting omitted or duplicated selected test files")
        complete = _counter(report["cases"])
        missing = complete - cases
        if cases - complete or any(key[3] not in {"skipped", "xfailed"} for key in missing):
            raise ValueError("Batch cases do not partition the complete JUnit")
        collected = summarize_reports(root / "execution-collection", expected_reports=["collection.xml"])
        # Empty collect-only XML is valid when no module is skipped.
        if any(case["outcome"] not in {"skipped", "xfailed"} for case in collected["cases"]):
            raise ValueError("Collect-only evidence contains executed or failed cases")
        if missing - _counter(collected["cases"]):
            raise ValueError("Unaccounted cases were not authentic collection skips")
        counts["collection_skips"] = sum(missing.values())
        counts["total"] = sum(complete.values())
        if counts != accounting["counts"] or not report["valid"]:
            raise ValueError("Accounting totals differ from complete JUnit")
        expected_lines: dict[str, int] = {}
        for path in coverage_inputs:
            for name, bitmap in _coverage_lines(path).items():
                expected_lines[name] = expected_lines.get(name, 0) | bitmap
        if _coverage_lines(root / coverage_name) != expected_lines:
            raise ValueError("Combined coverage does not contain exactly the executed and reused batch union")
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, ElementTree.ParseError) as error:
        errors.append(str(error))
    return {"valid": not errors, "errors": errors, "counts": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebase raw coverage using the sampled coverage version")
    parser.add_argument("command", choices=["normalize-coverage"])
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--absolute", action="store_true")
    args = parser.parse_args()
    normalize_coverage(args.coverage.resolve(), args.checkout, absolute=args.absolute)
