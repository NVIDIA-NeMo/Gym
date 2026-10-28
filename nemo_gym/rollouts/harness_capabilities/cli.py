# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone command: python -m nemo_gym.harness_capabilities inspect --bundle FILE."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from string import ascii_letters, digits
from typing import Iterator

from . import __version__
from .checker import NAMES, PROFILE, EvidenceScope, inspect_record
from .contracts import SCHEMA_VERSION


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def json_rows(path: Path) -> Iterator[tuple[int, dict]]:
    """Read JSONL strictly; malformed/truncated rows are checker errors."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line, parse_constant=_reject_constant)
                except (ValueError, RecursionError) as exc:
                    raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{number}: expected an object")
                yield number, value


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def inspect_bundle(
    bundle: Path,
    *,
    output: Path,
    profile: str = PROFILE,
    scope: EvidenceScope = EvidenceScope(),
    capture_dir: Path | None = None,
) -> tuple[Path, dict]:
    """Write an immutable content-addressed report set after a complete read.

    A checker error never replaces an earlier report or leaves a current-looking
    summary. Files are hashed before and after reading to detect changing input.
    """
    if profile != PROFILE:
        raise ValueError("unknown artifact profile")
    if bundle.is_dir():
        candidates = [
            bundle / "rollouts.jsonl",
            bundle / "artifacts/rollouts.jsonl",
            bundle / "evaluator_rollouts.jsonl",
            bundle / "artifacts/evaluator_rollouts.jsonl",
        ]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError(
                "bundle must contain exactly one rollouts.jsonl or evaluator_rollouts.jsonl; pass a file explicitly"
            )
        bundle = matches[0]
    sources = [bundle]
    if capture_dir is not None:
        if not capture_dir.is_dir():
            raise ValueError("capture directory does not exist")
        sources.extend(sorted(capture_dir.glob("*.capture.*")))
    hashes = {str(path.resolve()): digest_file(path) for path in sources}
    checker_sources = sorted(Path(__file__).parent.glob("*.py"))
    checker_hash = hashlib.sha256("".join(digest_file(p) for p in checker_sources).encode()).hexdigest()
    manifest = {
        "sources": hashes,
        "checker_sha256": checker_hash,
        "profile": profile,
        "applicability": asdict(scope),
    }
    report_id = hashlib.sha256(_json(manifest).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".capabilities-", dir=output))
    totals = {key: {"records": 0, "fulfilled": 0, "not_fulfilled": 0, "not_applicable": 0} for key in NAMES}
    token_availability = {}
    passing_records = 0
    count = 0
    try:
        with (temporary / "evidence_results.jsonl").open("w") as handle:
            for line, record in json_rows(bundle):
                result = inspect_record(record, source=f"{bundle.name}:{line}", scope=scope)
                count += 1
                passing_records += result["verdict"] == "fulfilled"
                for key, verdict in result["evidence"].items():
                    totals[key]["records"] += 1
                    totals[key][verdict["verdict"]] += 1
                for key, values in result["token_availability"].items():
                    target = token_availability.setdefault(key, {"available": 0, "calls": 0})
                    for metric, value in values.items():
                        target[metric] += value
                handle.write(json.dumps(result, allow_nan=False) + "\n")
        if count == 0:
            raise ValueError("no rollout records")
        if any(digest_file(path) != hashes[str(path.resolve())] for path in sources):
            raise ValueError("source changed during inspection")
        verdicts = {
            key: {
                **counts,
                "verdict": "not_fulfilled"
                if counts["not_fulfilled"]
                else "not_applicable"
                if counts["not_applicable"] == count
                else "fulfilled",
                "scenario_coverage": {"observed": 0, "passed": 0},
            }
            for key, counts in totals.items()
        }
        # Require the individual P0 checks to pass on every rollout.
        passed = passing_records == count
        summary = {
            "schema_version": "harness-evidence/v1",
            "checker_version": __version__,
            "decoder_version": SCHEMA_VERSION,
            "checker_status": "completed",
            "profile": profile,
            "scope": "retained Gym artifacts; no behavioral qualification",
            "scope_closure": "not_independently_witnessed",
            "verdict": "fulfilled" if passed else "not_fulfilled",
            "is_behavioral_qualification": False,
            "records": count,
            "evidence": verdicts,
            "token_availability": token_availability,
            "delivery_surface": "rollout_jsonl",
            "limits": [
                "retained artifacts only; no live qualification or health certification",
                "TE-6 checks shipped Gym reward/resolution; extended verifier provenance is not certified",
                "sandbox rows are reported under TE-6; record presence is enforced only with require_sandbox",
                "all applicable individual P0 checks must pass; TE labels only group results",
                "all current checks are P0, including ownership and call-to-step checks",
            ],
            **manifest,
        }
        (temporary / "evidence_summary.json").write_text(_json(summary))
        report = [
            "# Harness artifact conformance",
            "",
            f"Profile: `{profile}`. Gate: **{summary['verdict']}**. Records: {count}.",
            "",
            "These checks establish retained artifact requirements. They do not qualify the RFC's behavioral scenarios.",
            "",
            "| Evidence | Artifact verdict | Passing / applicable records |",
            "|---|---|---|",
        ]
        report.extend(
            f"| {key}: {NAMES[key]} | {value['verdict']} | {value['fulfilled']}/{count - value['not_applicable']} |"
            for key, value in verdicts.items()
        )
        report.extend(
            [
                "",
                "Field-level failure locations are in `evidence_results.jsonl`; payloads are never copied into this report.",
                "",
            ]
        )
        (temporary / "evidence_report.md").write_text("\n".join(report))
        destination = output / report_id
        if destination.exists():
            if any((destination / path.name).read_bytes() != path.read_bytes() for path in temporary.iterdir()):
                raise ValueError("existing content-addressed report differs")
            shutil.rmtree(temporary)
        else:
            os.rename(temporary, destination)
        return destination, summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def inspect_matrix(
    harnesses: dict[str, Path], *, output: Path, scope: EvidenceScope = EvidenceScope()
) -> tuple[Path, dict]:
    """Check each harness bundle and publish a reproducible harness × TE matrix."""
    if not harnesses:
        raise ValueError("at least one harness is required")
    rows = {}
    for name, bundle in harnesses.items():
        if not name or any(c not in ascii_letters + digits + "_-" for c in name):
            raise ValueError("harness names must be alphanumeric with hyphens or underscores")
        path, summary = inspect_bundle(bundle, output=output / name, scope=scope)
        rows[name] = {"report": str(path / "evidence_summary.json"), **summary}
    matrix = {
        "schema_version": "harness-evidence-matrix/v1",
        "profile": PROFILE,
        "harnesses": rows,
        "is_behavioral_qualification": False,
    }
    matrix_id = hashlib.sha256(_json(matrix).encode()).hexdigest()
    destination = output / matrix_id
    temporary = Path(tempfile.mkdtemp(prefix=".matrix-", dir=output))
    try:
        (temporary / "harness_evidence.json").write_text(_json(matrix))
        table = [
            "# Harness × P0 trajectory evidence",
            "",
            "Retained artifacts only; not live harness certification.",
            "",
            "| Harness | " + " | ".join(NAMES) + " | P0 |",
            "|---|" + "---|" * (len(NAMES) + 1),
        ]
        labels = {
            "fulfilled": "PASS",
            "not_fulfilled": "FAIL",
            "not_applicable": "N/A",
        }
        for name, summary in rows.items():
            table.append(
                "| "
                + name
                + " | "
                + " | ".join(labels[summary["evidence"][key]["verdict"]] for key in NAMES)
                + " | "
                + labels[summary["verdict"]]
                + " |"
            )
        table.extend(
            [
                "",
                *[f"- {key}: {name}" for key, name in NAMES.items()],
                "",
                "P0 requires every applicable individual P0 check to pass on every record; TE labels only group results.",
                "TE-2 PASS means saved counts have valid types and ranges; availability is reported separately.",
                "See each evidence_summary.json for input hashes, applicability, token availability and limitations.",
            ]
        )
        (temporary / "harness_evidence.md").write_text("\n".join(table) + "\n")
        if destination.exists():
            if any((destination / p.name).read_bytes() != p.read_bytes() for p in temporary.iterdir()):
                raise ValueError("existing matrix differs")
            shutil.rmtree(temporary)
        else:
            os.rename(temporary, destination)
        return destination, matrix
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--bundle", required=True, type=Path)
    inspect.add_argument(
        "--capture-dir",
        type=Path,
        help="include capture sidecars in input provenance; checks use designated rollout fields",
    )
    inspect.add_argument("--profile", choices=[PROFILE], default=PROFILE)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--harness", action="append", required=True, metavar="NAME=PATH")
    for command in (inspect, matrix):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument(
            "--no-tools", action="store_true", help="this harness/benchmark pair does not dispatch tools"
        )
        command.add_argument("--no-verifier", action="store_true", help="this pair has no verifier")
        command.add_argument("--no-steps", action="store_true", help="this pair has no policy step structure")
        command.add_argument(
            "--require-sandbox", action="store_true", help="require saved sandbox outcomes for this pair"
        )
    args = parser.parse_args(argv)
    scope = EvidenceScope(
        tools=not args.no_tools,
        verifier=not args.no_verifier,
        steps=not args.no_steps,
        require_sandbox=args.require_sandbox,
    )
    if args.command == "inspect":
        return run_inspection(
            bundle=args.bundle, output=args.output, profile=args.profile, capture_dir=args.capture_dir, scope=scope
        )
    try:
        harnesses = {}
        for value in args.harness:
            name, separator, path = value.partition("=")
            if not separator or not path or name in harnesses:
                raise ValueError("each harness needs a unique NAME=PATH")
            harnesses[name] = Path(path)
        destination, result = inspect_matrix(harnesses, output=args.output, scope=scope)
        print(destination / "harness_evidence.md")
        return 0 if all(row["verdict"] == "fulfilled" for row in result["harnesses"].values()) else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        print(f"checker_error ({type(exc).__name__}); no matrix published")
        return 2


def run_inspection(
    *,
    bundle: Path,
    output: Path,
    profile: str = PROFILE,
    capture_dir: Path | None = None,
    scope: EvidenceScope = EvidenceScope(),
) -> int:
    """Inspect retained rollouts and print the report location; return 0, 1, or 2."""
    try:
        destination, summary = inspect_bundle(
            bundle, output=output, profile=profile, capture_dir=capture_dir, scope=scope
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        print(f"checker_error ({type(exc).__name__}); no report published")
        return 2
    print(f"{summary['verdict']}: {destination / 'evidence_summary.json'}")
    return 0 if summary["verdict"] == "fulfilled" else 1
