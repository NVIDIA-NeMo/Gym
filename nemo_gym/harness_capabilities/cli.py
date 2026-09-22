# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone command: python -m nemo_gym.harness_capabilities inspect --bundle FILE."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import __version__
from .checker import NAMES, PENDING_VALIDATORS, PROFILES, inspect_record
from .reader import digest_file, hydrate_record, json_rows
from .schemas import SCHEMA_VERSION, SCHEMAS


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def inspect_bundle(
    bundle: Path,
    *,
    output: Path,
    profile: str,
    capture_dir: Path | None = None,
) -> tuple[Path, dict]:
    """Write an immutable content-addressed report set after a complete read.

    A checker error never replaces an earlier report or leaves a current-looking
    summary. Files are hashed before and after reading to detect changing input.
    """
    if profile not in PROFILES:
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
    registry_hash = hashlib.sha256(_json({"schemas": SCHEMAS, "profiles": PROFILES}).encode()).hexdigest()
    checker_hash = hashlib.sha256(
        "".join(digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))).encode()
    ).hexdigest()
    manifest = {
        "sources": hashes,
        "registry_sha256": registry_hash,
        "checker_sha256": checker_hash,
        "profile": profile,
    }
    report_id = hashlib.sha256(_json(manifest).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".capabilities-", dir=output))
    totals = {capability: {"observed_records": 0, "passed_records": 0} for capability in NAMES}
    identities: set[object] = set()
    count = 0
    try:
        with (temporary / "capability_results.jsonl").open("w") as handle:
            for line, raw in json_rows(bundle):
                record = hydrate_record(raw, capture_dir=capture_dir)
                identity = (record.get("ng_model_call_capture") or {}).get("rollout_id") or record.get(
                    "_ng_rollout_id"
                )
                if identity in identities:
                    record.setdefault("_capability_reader_issues", []).append("duplicate rollout identity in input")
                identities.add(identity)
                result = inspect_record(record, source=f"{bundle.name}:{line}")
                count += 1
                for capability, verdict in result["capabilities"].items():
                    totals[capability]["observed_records"] += 1
                    totals[capability]["passed_records"] += verdict["verdict"] == "fulfilled"
                handle.write(json.dumps(result, allow_nan=False) + "\n")
        if count == 0:
            raise ValueError("no rollout records")
        if any(digest_file(path) != hashes[str(path.resolve())] for path in sources):
            raise ValueError("source changed during inspection")
        verdicts = {
            capability: {
                **counts,
                "verdict": ("fulfilled" if counts["passed_records"] == count else "not_fulfilled"),
                "behavioral_qualification": False,
                "validator_status": ("incomplete" if capability in PENDING_VALIDATORS else "artifact_subset"),
                "scenario_coverage": {"observed": 0, "passed": 0},
            }
            for capability, counts in totals.items()
        }
        passed = all(verdicts[capability]["verdict"] == "fulfilled" for capability in PROFILES[profile])
        summary = {
            "schema_version": "harness-capabilities/v1",
            "checker_version": __version__,
            "decoder_version": SCHEMA_VERSION,
            "checker_status": "completed",
            "profile": profile,
            "scope": "retained Gym artifacts; no behavioral qualification",
            "scope_closure": "not_independently_witnessed",
            "verdict": "fulfilled" if passed else "not_fulfilled",
            "is_behavioral_qualification": False,
            "records": count,
            "capabilities": verdicts,
            **manifest,
        }
        (temporary / "capability_summary.json").write_text(_json(summary))
        report = [
            "# Harness artifact conformance",
            "",
            f"Profile: `{profile}`. Gate: **{summary['verdict']}**. Records: {count}.",
            "",
            "These checks establish retained artifact requirements. They do not qualify the RFC's behavioral scenarios.",
            "",
            "| Capability | Artifact verdict | Passing records |",
            "|---|---|---|",
        ]
        report.extend(
            f"| {key}: {NAMES[key]} | {value['verdict']} | {value['passed_records']}/{count} |"
            for key, value in verdicts.items()
        )
        report.extend(
            [
                "",
                "Field-level failure locations are in `capability_results.jsonl`; payloads are never copied into this report.",
                "",
            ]
        )
        (temporary / "capability_report.md").write_text("\n".join(report))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--bundle", required=True, type=Path)
    inspect.add_argument("--capture-dir", type=Path)
    inspect.add_argument("--output", required=True, type=Path)
    inspect.add_argument("--profile", choices=PROFILES, default="gym-artifacts-p0/v1")
    args = parser.parse_args(argv)
    return run_inspection(bundle=args.bundle, output=args.output, profile=args.profile, capture_dir=args.capture_dir)


def run_inspection(
    *, bundle: Path, output: Path, profile: str = "gym-artifacts-p0/v1", capture_dir: Path | None = None
) -> int:
    """Inspect retained rollouts and print the report location; return 0, 1, or 2."""
    try:
        destination, summary = inspect_bundle(
            bundle,
            output=output,
            profile=profile,
            capture_dir=capture_dir,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
    ) as exc:
        # Do not print exception details: malformed input may contain secrets.
        print(f"checker_error ({type(exc).__name__}); no report published")
        return 2
    print(f"{summary['verdict']}: {destination / 'capability_summary.json'}")
    return 0 if summary["verdict"] == "fulfilled" else 1
