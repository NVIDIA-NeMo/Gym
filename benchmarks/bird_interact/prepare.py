# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a BIRD-Interact release, Lite or Full, for the bird_interact Environment Server.

Downloads the release from Hugging Face (``birdsql/bird-interact-<release>``): its rows and, for each database, the
schema, column meanings and knowledge base that the server reads from ``database_dir``. The databases themselves are
not downloaded; they are in the sandbox image that ``Dockerfile`` builds.

The public rows withhold the ground truth: ``sol_sql``, ``external_knowledge`` and ``test_cases``, for the task and
for its follow-up. The authors send it on request, and ``prepare`` merges it in by ``instance_id``. It keeps the rows
that the server can grade: those with their ground truth, and without custom test cases for the task or its follow-up
(BIRD-Interact's management tasks).
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from environment_servers.bird_interact.task_data import BirdInteractTask


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
RELEASES = ("lite", "full")
MODES = ("c_interact", "a_interact")
ROWS_FILE = "bird_interact_data.jsonl"
# The fields that the public rows leave empty, for the task and for its follow-up.
GROUND_TRUTH_FIELDS = ("sol_sql", "external_knowledge", "test_cases")


def database_dir(release: str) -> Path:
    """The release's download: its rows, and a directory of schema and knowledge files for each database."""
    return DATA_DIR / release / "databases"


def download_release(release: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=f"birdsql/bird-interact-{release}", repo_type="dataset", local_dir=database_dir(release)
        )
    )


def merge_ground_truth(row: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """``row`` with the ground truth of the task and of its follow-up taken from ``truth``."""
    merged = {**row, **{field: truth.get(field, []) for field in GROUND_TRUTH_FIELDS}}
    if isinstance(truth.get("follow_up"), dict):
        follow_up = dict(merged.get("follow_up") or {})
        follow_up.update({field: truth["follow_up"].get(field, []) for field in GROUND_TRUTH_FIELDS})
        merged["follow_up"] = follow_up
    return merged


def select_tasks(
    rows: list[dict[str, Any]], ground_truth: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """The rows the server can grade, with their ground truth merged in, and why each other row was left out."""
    kept: list[dict[str, Any]] = []
    left_out: Counter[str] = Counter()
    for row in rows:
        truth = ground_truth.get(row["instance_id"])
        if truth is None:
            left_out["no ground truth"] += 1
            continue
        merged = merge_ground_truth(row, truth)
        follow_up = merged.get("follow_up")
        if not merged["sol_sql"] or (follow_up is not None and not follow_up.get("sol_sql")):
            left_out["no ground truth"] += 1
        elif merged["test_cases"] or (follow_up is not None and follow_up.get("test_cases")):
            left_out["custom test cases"] += 1
        else:
            BirdInteractTask.model_validate(merged)
            kept.append(merged)
    return kept, left_out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(release: str = "lite", mode: str = "c_interact", ground_truth_jsonl: str | None = None) -> Path:
    """Write the release's rows for ``mode`` to ``data/<release>/<mode>.jsonl`` and return that path.

    The rows are the same for both modes; each mode has its own file, so each run has its own metrics.
    """
    if release not in RELEASES:
        raise ValueError(f"release must be one of {', '.join(RELEASES)}, not {release!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if ground_truth_jsonl is None:
        raise ValueError(
            "BIRD-Interact withholds its ground truth. Pass the file the authors send with "
            "+prepare_script_args.ground_truth_jsonl=<path>; see benchmarks/bird_interact/README.md."
        )
    rows = _read_jsonl(download_release(release) / ROWS_FILE)
    ground_truth = {truth["instance_id"]: truth for truth in _read_jsonl(Path(ground_truth_jsonl))}
    kept, left_out = select_tasks(rows, ground_truth)
    output = DATA_DIR / release / f"{mode}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in kept))
    reasons = ", ".join(f"{count} with {reason}" for reason, count in sorted(left_out.items())) or "none"
    print(f"BIRD-Interact {release}: kept {len(kept)} of {len(rows)} tasks; left out: {reasons}.")
    return output
