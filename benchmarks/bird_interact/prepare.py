# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a BIRD-Interact release, Lite or Full, for the bird_interact Environment Server.

Downloads the release from Hugging Face (``birdsql/bird-interact-<release>``): its rows and, for each database, the
schema, column meanings and knowledge base that the server reads from ``database_dir``. The databases themselves are
not downloaded; they are in the sandbox image that ``Dockerfile`` builds.

The public rows withhold the ground truth: ``sol_sql``, ``external_knowledge`` and ``test_cases``, for the task and
for its follow-up. The authors send it on request, and ``prepare`` merges it in by ``instance_id``. It keeps the rows
that the server can grade: query tasks, whose two phases are both ``Query``, with their ground truth. As in
BIRD-Interact, a query phase is graded by comparing its result with the solution's even when it has test cases.
BIRD-Interact's management tasks, which test cases grade, are left out: the server does not run test cases yet. It reads
the file the authors sent from ``data/`` (``GROUND_TRUTH_FILES``), which git ignores, unless ``ground_truth_jsonl``
names another.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from environment_servers.bird_interact.task_data import BirdInteractTask
from nemo_gym.data import materialize_task


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
RELEASES = ("lite", "full")
MODES = ("c_interact", "a_interact")
ROWS_FILE = "bird_interact_data.jsonl"
# The fields that the public rows leave empty, for the task and for its follow-up.
GROUND_TRUTH_FIELDS = ("sol_sql", "external_knowledge", "test_cases")
# The ground truth of each release, as the authors send it, in data/. The ground truth must not be published, and
# data/.gitignore keeps it out of the repository.
GROUND_TRUTH_FILES = {
    "lite": "bird_interact_gt_kg_testcases_1008.jsonl",
    "full": "bird_interact_full_gt_kg_testcases_08022.jsonl",
}
# The revision of each release's Hugging Face dataset that the ground truth files belong to, so a later edit upstream
# cannot pair new rows with old ground truth.
RELEASE_REVISIONS = {
    "lite": "f7881a9c2b9630cc4fc13b0c39279740b0a2fd87",  # pragma: allowlist secret
    "full": "5d78b0722433c3821e1ebda5d8c39d24070049d7",  # pragma: allowlist secret
}
# A phase's category when its row names none, as in the ADK.
QUERY_CATEGORY = "Query"


def database_dir(release: str) -> Path:
    """The release's download: its rows, and a directory of schema and knowledge files for each database."""
    return DATA_DIR / release / "databases"


def download_release(release: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=f"birdsql/bird-interact-{release}",
            repo_type="dataset",
            revision=RELEASE_REVISIONS[release],
            local_dir=database_dir(release),
        )
    )


def _phase_truth(truth: dict[str, Any]) -> dict[str, Any]:
    """A phase's ground truth fields. The authors' Lite file writes most follow-ups' ``sol_sql`` as one string, not a
    list of statements; it becomes a list of that one statement. A missing or ``null`` field becomes an empty list."""
    fields = {field: truth.get(field) or [] for field in GROUND_TRUTH_FIELDS}
    if isinstance(fields["sol_sql"], str):
        fields["sol_sql"] = [fields["sol_sql"]]
    return fields


def merge_ground_truth(row: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """``row`` with the ground truth of the task and of its follow-up taken from ``truth``."""
    merged = {**row, **_phase_truth(truth)}
    if isinstance(truth.get("follow_up"), dict):
        follow_up = dict(merged.get("follow_up") or {})
        follow_up.update(_phase_truth(truth["follow_up"]))
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
        categories = {merged.get("category") or QUERY_CATEGORY, (follow_up or {}).get("category") or QUERY_CATEGORY}
        if not merged["sol_sql"] or (follow_up is not None and not follow_up.get("sol_sql")):
            left_out["no ground truth"] += 1
        elif categories != {QUERY_CATEGORY}:
            left_out["management task"] += 1
        else:
            BirdInteractTask.model_validate(merged)
            kept.append(merged)
    return kept, left_out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    """The ground truth file's entries by ``instance_id``. Raises a ``ValueError`` that names the file, and the line
    when one is wrong, so a bad file is found before the download."""
    if not path.is_file():
        raise ValueError(f"No ground truth file at {path}")
    ground_truth: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            truth = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}, line {number}: not JSON: {error}") from error
        if not isinstance(truth, dict) or not isinstance(truth.get("instance_id"), str):
            raise ValueError(f"{path}, line {number}: the entry has no instance_id")
        ground_truth[truth["instance_id"]] = truth
    return ground_truth


def prepare(release: str = "lite", mode: str = "c_interact", ground_truth_jsonl: str | None = None) -> Path:
    """Write the release's rows for ``mode`` to ``data/<release>/<mode>.jsonl`` and return that path.

    The rows are the same for both modes; each mode has its own file, so each run has its own metrics. An empty ground
    truth file only downloads the release: no rows are written, and an earlier file stays as it was.
    """
    if release not in RELEASES:
        raise ValueError(f"release must be one of {', '.join(RELEASES)}, not {release!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if ground_truth_jsonl is None:
        default = DATA_DIR / GROUND_TRUTH_FILES[release]
        if not default.is_file():
            raise ValueError(
                f"BIRD-Interact withholds its ground truth. Put the file the authors send at {default}, or pass it "
                "with +prepare_script_args.ground_truth_jsonl=<path>; see benchmarks/bird_interact/README.md."
            )
        ground_truth_jsonl = str(default)
    ground_truth = read_ground_truth(Path(ground_truth_jsonl))
    rows = _read_jsonl(download_release(release) / ROWS_FILE)
    output = DATA_DIR / release / f"{mode}.jsonl"
    if not ground_truth:
        print(f"BIRD-Interact {release}: downloaded; {ground_truth_jsonl} is empty, so no rows were written.")
        return output
    if not ground_truth.keys() & {row["instance_id"] for row in rows}:
        raise ValueError(
            f"{ground_truth_jsonl} has the ground truth of none of {release}'s tasks: is it another release's?"
        )
    kept, left_out = select_tasks(rows, ground_truth)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Materialized for the mode's taskset, so manifest validation reads each row as the server's task input.
    taskset = f"bird_interact:{mode}"
    output.write_text("".join(json.dumps(materialize_task(row, taskset=taskset)) + "\n" for row in kept))
    reasons = ", ".join(f"{count} with {reason}" for reason, count in sorted(left_out.items())) or "none"
    print(f"BIRD-Interact {release}: kept {len(kept)} of {len(rows)} tasks; left out: {reasons}.")
    return output
