# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare, validate and grade one BioML-bench task inside the upstream task image.

Runs with the image's ``biomlb`` conda Python, which has the pinned BioML-bench package at
``/biomlbench``. That package's CLI imports an agent registry the image does not ship, so this
calls the library functions behind ``biomlbench prepare`` and ``biomlbench grade-sample``.
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path


SUBMISSION_NAMES = ("submission.csv", "submission.h5ad")


def _task(task_id: str, data_dir: str):
    from biomlbench.registry import registry

    return registry.set_data_dir(Path(data_dir)).get_task(task_id)


def checksum_mismatches(actual: dict, expected: dict) -> list[str]:
    """Files that differ from the pins, as ``public/<name>`` or ``private/<name>``.

    A file must exist on both sides. A ``null`` pin only requires the file: some upstream preparers
    write sample submissions with unseeded random values.
    """
    mismatches = []
    for side in ("public", "private"):
        got, want = actual.get(side) or {}, expected.get(side) or {}
        for name in set(got) | set(want):
            if name not in got or name not in want or (want[name] is not None and got[name] != want[name]):
                mismatches.append(f"{side}/{name}")
    return sorted(mismatches)


def prepare(args: argparse.Namespace) -> None:
    from biomlbench.data import download_and_prepare_dataset, generate_checksums

    task = _task(args.task_id, args.data_dir)
    # The image's checksums.yaml files no longer match what its preparers write; task.json holds the pins.
    download_and_prepare_dataset(task, keep_raw=False, skip_verification=True)
    actual = {"public": generate_checksums(task.public_dir), "private": generate_checksums(task.private_dir)}
    expected = json.loads(Path(args.task_json).read_text()).get("checksums") if args.task_json else None
    if expected is not None and (mismatches := checksum_mismatches(actual, expected)):
        raise SystemExit(f"Prepared {task.id} differs from its pinned checksums: {', '.join(mismatches)}")
    public = Path(args.public_dir)
    public.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task.public_dir, public, dirs_exist_ok=True)
    print(json.dumps({"task_id": task.id, "checksums": actual}))


def submission_path(sample_submission: Path, submission_dir: Path) -> Path | None:
    """The agent's submission: ``submission.<ext>``, the sample submission's extension first.

    Only a regular file counts: a symlink could point the grader at the answers.
    """
    preferred = "submission" + Path(sample_submission).suffix.lower()
    for name in (preferred, *SUBMISSION_NAMES):
        candidate = submission_dir / name
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None


def validate(submission: Path, sample_submission: Path) -> tuple[bool, str]:
    """Check a submission against the public sample submission; the agent's sandbox has no answers.

    Upstream's checker runs the grader on the private answers. This checks the format from public
    data: the columns, the row count, the ids in the first column, and no missing values.
    """
    import pandas as pd

    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."
    suffix = submission.suffix.lower()
    if suffix == ".h5ad":
        import anndata

        anndata.read_h5ad(submission)
        return True, "Submission is valid."
    if suffix != ".csv":
        return False, f"Submission invalid! Submission file must be one of ['.csv', '.h5ad'], got {suffix}."
    sample = pd.read_csv(sample_submission)
    try:
        got = pd.read_csv(submission)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        return False, f"Submission invalid! The file cannot be read as CSV: {e}"
    if list(got.columns) != list(sample.columns):
        return False, f"Submission invalid! Expected columns {list(sample.columns)}, got {list(got.columns)}."
    if len(got) != len(sample):
        return False, f"Submission invalid! Expected {len(sample)} rows, got {len(got)}."
    key = sample.columns[0]
    if set(got[key].astype(str)) != set(sample[key].astype(str)):
        return False, f"Submission invalid! The values of column {key!r} differ from the sample submission."
    if got.isna().any().any():
        return False, "Submission invalid! The submission has missing values."
    return True, "Submission is valid."


def serve_validation(args: argparse.Namespace) -> None:
    """Answer ``POST /validate`` and ``GET /health`` as upstream's grading server does."""
    from flask import Flask, jsonify, request

    sample = next(Path(args.public_dir).glob("sample_submission.*"))
    app = Flask(__name__)

    @app.route("/validate", methods=["POST"])
    def validate_route():
        upload = request.files["file"]
        suffix = Path(upload.filename or sample.name).suffix or sample.suffix
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"submission{suffix}"
            upload.save(path)
            try:
                _, message = validate(path, sample)
            except Exception as e:  # noqa: BLE001 - report an unreadable upload to the agent, as upstream does
                return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500
        return jsonify({"result": message})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "running"}), 200

    app.run(host="127.0.0.1", port=args.port)


def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def rewards(report: dict) -> dict[str, float]:
    """Gym's reward file from a BioML-bench task report; ``reward`` is the any-medal flag.

    Harbor rejects non-finite values; a constant prediction scores NaN on correlation metrics.
    """
    valid = bool(report.get("valid_submission"))
    percentile = _finite(report.get("leaderboard_percentile")) if valid else None
    score = _finite(report.get("score")) if valid else None
    result = {
        "reward": float(bool(report.get("any_medal"))),
        "gold_medal": float(bool(report.get("gold_medal"))),
        "silver_medal": float(bool(report.get("silver_medal"))),
        "bronze_medal": float(bool(report.get("bronze_medal"))),
        "above_median": float(bool(report.get("above_median"))),
        "valid_submission": float(valid),
        # Penalized scoring, as in the paper's main table: no valid submission is percentile 0.
        "leaderboard_percentile": percentile if percentile is not None else 0.0,
    }
    if score is not None:
        result["score"] = score
    return result


def grade(args: argparse.Namespace) -> None:
    from biomlbench.grade import grade_submission

    task = _task(args.task_id, args.data_dir)
    submission = submission_path(task.sample_submission, Path(args.submission_dir))
    with tempfile.TemporaryDirectory() as empty:
        # No regular submission file: grade a path that does not exist, as upstream does for a missing file.
        missing = Path(empty) / ("submission" + Path(task.sample_submission).suffix.lower())
        # Upstream's grader turns an invalid submission into a None score and raises on anything else;
        # an exception fails the verifier, so Gym does not score an infrastructure failure as 0.
        report = grade_submission(submission or missing, task).to_dict()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "report.json").write_text(json.dumps(report, default=str, indent=2))
    out.write_text(json.dumps(rewards(report)))
    print(out.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Prepare the task; copy its public split to --public-dir.")
    p.add_argument("--task-id", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--public-dir", required=True)
    p.add_argument("--task-json", help="task.json with the pinned checksums to verify.")
    p.set_defaults(handler=prepare)
    g = commands.add_parser("grade", help="Grade the submission; write Gym's reward.json.")
    g.add_argument("--task-id", required=True)
    g.add_argument("--data-dir", required=True)
    g.add_argument("--submission-dir", required=True)
    g.add_argument("--output", required=True)
    g.set_defaults(handler=grade)
    v = commands.add_parser("serve-validation", help="Serve the submission checker on localhost.")
    v.add_argument("--public-dir", required=True)
    v.add_argument("--port", type=int, default=5000)
    v.set_defaults(handler=serve_validation)
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
