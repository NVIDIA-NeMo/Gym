# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and merge deterministic parallel shards of the official BioMysteryBench run."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.biomysterybench.prepare import BENCHMARK_DIR, RELEASES


REVERIFIED_RESULT_FIELDS = (
    "reward",
    "extracted_answer",
    "expected_answer",
    "verdict",
    "judge_output",
    "invalid_judge_response",
    "cheat_detected",
    "cheat_evidence",
)
LEGACY_AGENT_FAILURE_NORMALIZATION = "legacy_anyterminal_missing_agent_failed"
OFFICIAL_REPEATS = 5
DEFAULT_EXPECTED = BENCHMARK_DIR / "data" / RELEASES["official-99"].output_filename
BIOMYSTERY_AGENT_REF = {
    "type": "responses_api_agents",
    "name": "biomysterybench_claude_code",
}


class ComparisonError(ValueError):
    """Raised when shard input is invalid."""


def validate_official_expected_dataset(rows: list[dict[str, Any]]) -> None:
    release = RELEASES["official-99"]
    ids = [row.get("id") for row in rows]
    split_counts = Counter(row.get("human_solvable") for row in rows)
    revisions = {row.get("dataset_revision") for row in rows}
    if len(rows) != release.expected_task_count or len(set(ids)) != len(ids):
        raise ComparisonError("expected dataset does not contain 99 unique tasks")
    if dict(split_counts) != release.expected_split_counts:
        raise ComparisonError(f"unexpected task split: {dict(split_counts)}")
    if revisions != {release.revision}:
        raise ComparisonError(f"unexpected dataset revision: {revisions}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ComparisonError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ComparisonError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _rewrite_data_dir(row: dict[str, Any], data_root: Path | None) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if data_root is None:
        return result
    task_id = result.get("id")
    params = result.get("responses_create_params")
    if isinstance(task_id, str) and isinstance(params, dict):
        metadata = params.get("metadata")
        if isinstance(metadata, dict):
            metadata["data_dir"] = str((data_root / task_id).resolve())
    return result


def _normalize_legacy_policy_evidence(row: dict[str, Any]) -> None:
    """Upgrade an unambiguous pre-``agent_failed`` AnyTerminal success row.

    Early AnyTerminal rows did not serialize ``agent_failed`` on success.  A
    missing value is only safe to normalize when the response is nonempty,
    the agent recorded a positive runtime, and every failure/masking signal
    available in that schema is explicitly false.  The normalization is
    tagged in the merged artifact so the original policy files stay immutable
    and the schema migration remains visible to auditors.
    """

    agent_metrics = row.get("agent_metrics")
    response = row.get("response")
    if not isinstance(agent_metrics, dict) or not isinstance(response, dict):
        return
    if row.get("agent_failed") is not None or agent_metrics.get("agent_failed") is not None:
        return
    if not isinstance(response.get("output"), list) or not response["output"]:
        return
    agent_run_time = agent_metrics.get("agent_run_time")
    if isinstance(agent_run_time, bool) or not isinstance(agent_run_time, (int, float)) or agent_run_time <= 0:
        return
    normalizations = row.get("_ng_policy_evidence_normalizations")
    if normalizations is not None and not isinstance(normalizations, list):
        return
    legacy_failure_fields = ("mask_sample", "agent_timed_out", "container_timed_out", "sandbox_failed")
    if any(row.get(field) is not False for field in legacy_failure_fields):
        return
    if any(agent_metrics.get(field) is not False for field in legacy_failure_fields):
        return

    row["agent_failed"] = False
    agent_metrics["agent_failed"] = False
    if normalizations is None:
        normalizations = row["_ng_policy_evidence_normalizations"] = []
    if LEGACY_AGENT_FAILURE_NORMALIZATION not in normalizations:
        normalizations.append(LEGACY_AGENT_FAILURE_NORMALIZATION)


def prepare_shards(
    expected_path: Path,
    output_dir: Path,
    *,
    shard_count: int,
    existing_rollouts_path: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ComparisonError("shard_count must be at least 1")
    expected_rows = _read_jsonl(expected_path)
    validate_official_expected_dataset(expected_rows)
    if shard_count > len(expected_rows):
        raise ComparisonError(f"shard_count={shard_count} exceeds task count {len(expected_rows)}")

    task_to_official = {str(row["id"]): index for index, row in enumerate(expected_rows)}
    shard_rows: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    shard_local_index: dict[str, tuple[int, int]] = {}
    for official_index, source_row in enumerate(expected_rows):
        shard_index = official_index % shard_count
        local_index = len(shard_rows[shard_index])
        row = _rewrite_data_dir(source_row, data_root)
        shard_rows[shard_index].append(row)
        shard_local_index[str(row["id"])] = (shard_index, local_index)

    seeded_rows: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    seen: set[tuple[str, int]] = set()
    if existing_rollouts_path is not None:
        for line_number, source_row in enumerate(_read_jsonl(existing_rollouts_path), start=1):
            task_id = source_row.get("id")
            rollout_index = source_row.get("_ng_rollout_index")
            if task_id not in task_to_official:
                raise ComparisonError(f"existing rollout row {line_number}: unknown task id {task_id!r}")
            if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
                raise ComparisonError(
                    f"existing rollout row {line_number} ({task_id}): invalid _ng_rollout_index {rollout_index!r}"
                )
            if not 0 <= rollout_index < OFFICIAL_REPEATS:
                raise ComparisonError(
                    f"existing rollout row {line_number} ({task_id}): rollout index {rollout_index} is outside "
                    f"0..{OFFICIAL_REPEATS - 1}"
                )
            key = (str(task_id), rollout_index)
            if key in seen:
                raise ComparisonError(f"existing rollouts contain duplicate key {key!r}")
            seen.add(key)
            shard_index, local_index = shard_local_index[str(task_id)]
            row = _rewrite_data_dir(source_row, data_root)
            row["_ng_task_index"] = local_index
            seeded_rows[shard_index].append(row)

    # ``gym eval run --resume`` requires both the rollout output and its full
    # materialized-input companion. Without the latter Gym starts a fresh run
    # and clears the output, which would discard seed rows. Materialize the
    # exact five official repeats here with shard-local task indices.
    materialized_rows: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    preprocessed_rows: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for shard_index, rows in enumerate(shard_rows):
        for local_index, source_row in enumerate(rows):
            for rollout_index in range(OFFICIAL_REPEATS):
                row = copy.deepcopy(source_row)
                row.setdefault("agent_ref", copy.deepcopy(BIOMYSTERY_AGENT_REF))
                row["_ng_task_index"] = local_index
                row["_ng_rollout_index"] = rollout_index
                materialized_rows[shard_index].append(row)

                # ``gym eval run`` checks for this collated benchmark artifact
                # before it reaches resume handling. Produce the same logical
                # rows as TrainDataProcessor (dataset repetition + agent_ref)
                # so an already-prepared deterministic shard can skip the
                # generic dataset downloader. That downloader only supports
                # train/validation DatasetConfig sources, not a benchmark's
                # prepare_script-based BenchmarkDatasetConfig.
                preprocessed_row = copy.deepcopy(source_row)
                preprocessed_row.setdefault("agent_ref", copy.deepcopy(BIOMYSTERY_AGENT_REF))
                preprocessed_rows[shard_index].append(preprocessed_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_shards: list[dict[str, Any]] = []
    for shard_index, rows in enumerate(shard_rows):
        shard_dir = output_dir / f"shard-{shard_index:02d}"
        _write_jsonl(shard_dir / "dataset.jsonl", rows)
        _write_jsonl(shard_dir / "preprocessed_datasets" / "benchmark.jsonl", preprocessed_rows[shard_index])
        _write_jsonl(shard_dir / "rollouts.jsonl", seeded_rows[shard_index])
        _write_jsonl(shard_dir / "rollouts_materialized_inputs.jsonl", materialized_rows[shard_index])
        official_indices = [task_to_official[str(row["id"])] for row in rows]
        manifest_shards.append(
            {
                "shard_index": shard_index,
                "task_count": len(rows),
                "expected_rollout_count": len(rows) * OFFICIAL_REPEATS,
                "seeded_rollout_count": len(seeded_rows[shard_index]),
                "official_task_indices": official_indices,
                "task_ids": [row["id"] for row in rows],
            }
        )

    manifest = {
        "expected_dataset": str(expected_path.resolve()),
        "data_root": str(data_root.resolve()) if data_root is not None else None,
        "shard_count": shard_count,
        "task_count": len(expected_rows),
        "expected_rollout_count": len(expected_rows) * OFFICIAL_REPEATS,
        "seeded_rollout_count": len(seen),
        "shards": manifest_shards,
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def merge_shards(
    expected_path: Path,
    shards_dir: Path,
    output_path: Path,
    *,
    rollout_name: str = "rollouts.jsonl",
    policy_name: str | None = None,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    expected_rows = _read_jsonl(expected_path)
    validate_official_expected_dataset(expected_rows)
    task_to_official = {str(row["id"]): index for index, row in enumerate(expected_rows)}

    merged_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if Path(rollout_name).name != rollout_name:
        raise ComparisonError(f"rollout_name must be a filename, got {rollout_name!r}")
    if policy_name is not None and Path(policy_name).name != policy_name:
        raise ComparisonError(f"policy_name must be a filename, got {policy_name!r}")
    for shard_path in sorted(shards_dir.glob(f"shard-*/{rollout_name}")):
        policy_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        if policy_name is not None:
            policy_path = shard_path.with_name(policy_name)
            if not policy_path.is_file():
                raise ComparisonError(f"missing policy evidence for {shard_path}: {policy_path}")
            for policy_line_number, policy_row in enumerate(_read_jsonl(policy_path), start=1):
                policy_id = policy_row.get("id")
                policy_rollout_index = policy_row.get("_ng_rollout_index")
                policy_key = (str(policy_id), policy_rollout_index)
                if policy_key in policy_by_key:
                    raise ComparisonError(
                        f"duplicate policy rollout key {policy_key!r} in {policy_path}:{policy_line_number}"
                    )
                policy_by_key[policy_key] = policy_row
        for line_number, source_row in enumerate(_read_jsonl(shard_path), start=1):
            task_id = source_row.get("id")
            rollout_index = source_row.get("_ng_rollout_index")
            if task_id not in task_to_official:
                raise ComparisonError(f"{shard_path}:{line_number}: unknown task id {task_id!r}")
            if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
                raise ComparisonError(
                    f"{shard_path}:{line_number} ({task_id}): invalid _ng_rollout_index {rollout_index!r}"
                )
            key = (str(task_id), rollout_index)
            if key in merged_by_key:
                raise ComparisonError(f"duplicate rollout key {key!r} while merging {shard_path}")
            if policy_name is None:
                row = copy.deepcopy(source_row)
            else:
                policy_row = policy_by_key.get(key)
                if policy_row is None:
                    raise ComparisonError(f"{shard_path}:{line_number}: no policy evidence for rollout key {key!r}")
                if source_row.get("response") != policy_row.get("response"):
                    raise ComparisonError(f"{shard_path}:{line_number}: reverified response differs for {key!r}")
                row = copy.deepcopy(policy_row)
                _normalize_legacy_policy_evidence(row)
                # ``resolved`` is a convenience field from the original
                # verifier and is not emitted by stateless re-verification.
                # Drop it rather than retaining a stale pre-reverify value.
                row.pop("resolved", None)
                for field in REVERIFIED_RESULT_FIELDS:
                    if field not in source_row:
                        raise ComparisonError(f"{shard_path}:{line_number}: reverified row lacks {field!r}")
                    row[field] = copy.deepcopy(source_row[field])
            row["_ng_task_index"] = task_to_official[str(task_id)]
            merged_by_key[key] = row

    expected_keys = {
        (str(row["id"]), rollout_index) for row in expected_rows for rollout_index in range(OFFICIAL_REPEATS)
    }
    extra_keys = sorted(set(merged_by_key) - expected_keys)
    missing_keys = sorted(expected_keys - set(merged_by_key))
    if extra_keys:
        raise ComparisonError(f"merged shards contain {len(extra_keys)} unexpected keys; first: {extra_keys[:5]}")
    if require_complete and missing_keys:
        raise ComparisonError(f"merged shards are missing {len(missing_keys)} keys; first: {missing_keys[:5]}")

    merged = sorted(
        merged_by_key.values(),
        key=lambda row: (int(row["_ng_task_index"]), int(row["_ng_rollout_index"])),
    )
    _write_jsonl(output_path, merged)
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--shards", type=int, default=16)
    prepare_parser.add_argument("--existing-rollouts", type=Path, default=None)
    prepare_parser.add_argument("--data-root", type=Path, default=None)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    merge_parser.add_argument("--shards-dir", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--rollout-name", default="rollouts.jsonl")
    merge_parser.add_argument(
        "--policy-name",
        default=None,
        help="join reverified result fields onto immutable policy rows from this sibling filename",
    )
    merge_parser.add_argument("--require-complete", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_shards(
                args.expected,
                args.output_dir,
                shard_count=args.shards,
                existing_rollouts_path=args.existing_rollouts,
                data_root=args.data_root,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            rows = merge_shards(
                args.expected,
                args.shards_dir,
                args.output,
                rollout_name=args.rollout_name,
                policy_name=args.policy_name,
                require_complete=args.require_complete,
            )
            print(f"Merged {len(rows)} rollout rows into {args.output}")
    except (ComparisonError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
