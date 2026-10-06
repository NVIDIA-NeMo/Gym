# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy completed NOOA generation artifacts into canonical GDP judge-only layout."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath


JUDGE_GYM_COMMIT = "183ef8601aad3a3c5b933065b05c8cb87442560e"
REFERENCE_EFB_COMMIT = "3737282b3890c83cc50306955f59e326362d9c7c"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Artifact root must be an ordinary directory: {root}")
    paths = sorted(root.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
        raise ValueError(f"Artifact tree contains links or nonregular files: {root}")
    files = [p for p in paths if p.is_file()]
    if any(p.stat().st_nlink != 1 for p in files):
        raise ValueError(f"Artifact tree contains hard-linked files: {root}")
    return files


def prepare_judging(
    *,
    source: Path,
    generation_roots: list[Path],
    reference_models: Path,
    output: Path,
    expected_tasks: int = 220,
    excluded_task_ids: set[str] | None = None,
) -> Path:
    """Require exact task coverage and checked artifact bytes; never overwrite a bundle.

    This supports one selected generation attempt per task. Duplicate receipts
    require an explicit selection of generation roots, rather than silently
    choosing the latest or highest-scoring attempt. Explicit exclusions keep the
    canonical dataset intact and create no candidate directory or finish marker.
    Native judge-only scoring records these missing candidates as skipped.
    No inference or judging runs.
    """
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"Choose a new judging bundle: {output}")
    source_digest = _digest(source)
    reference_digest = _digest(reference_models)
    with source.open("rb") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    tasks = {row["task_id"]: row for row in rows}
    if len(rows) != expected_tasks or len(tasks) != len(rows):
        raise ValueError(f"Expected {expected_tasks} unique canonical tasks, found {len(rows)} rows/{len(tasks)} IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id) for task_id in tasks):
        raise ValueError("Canonical task IDs must be safe directory names")
    excluded = set(excluded_task_ids or ())
    if excluded - set(tasks):
        raise ValueError("Excluded task IDs must belong to the canonical dataset")
    if excluded == set(tasks):
        raise ValueError("At least one genuine export is required for judging")
    refs = json.loads(reference_models.read_text())
    if not refs or any(
        not isinstance(ref, dict)
        or not isinstance(ref.get("deliverables_dir"), str)
        or not Path(ref["deliverables_dir"]).is_absolute()
        or type(ref.get("elo")) not in (int, float)
        for ref in refs.values()
    ):
        raise ValueError("Reference manifest requires absolute deliverables_dir and numeric elo for every reference")
    selected = {}
    episodes = set()
    for root in generation_roots:
        for path in sorted(root.glob("gdp-*/generation.json")):
            if path.is_symlink():
                raise ValueError(f"Generation receipt must not be a symlink: {path}")
            receipt = json.loads(path.read_text())
            task_id = receipt["task_id"]["task_id"]
            episode = receipt["episode_id"]
            key = (episode["rollout_id"], episode["attempt"])
            payload = receipt["verify_request"]
            if receipt.get("schema_version") != 1 or task_id not in tasks or payload["task_id"] != task_id:
                raise ValueError(f"Unknown or mismatched generation identity: {path}")
            if task_id in excluded:
                raise ValueError(f"An excluded task has a generation receipt: {path}")
            if task_id in selected or key in episodes:
                raise ValueError(f"Duplicate task or episode: {path}")
            for field in ("prompt", "sector", "occupation", "rubric_json", "rubric_pretty", "reference_file_urls"):
                expected = tasks[task_id].get(field)
                if field == "reference_file_urls" and isinstance(expected, str):
                    expected = json.loads(expected)
                if expected is not None and payload.get(field) != expected:
                    raise ValueError(f"Generation metadata differs from canonical {task_id}/{field}")
            artifact_dir = Path(payload["deliverables_dir"])
            if not artifact_dir.is_absolute() or artifact_dir.parent.resolve() != path.parent.resolve():
                raise ValueError(f"Deliverables must belong to their generation receipt: {path}")
            files = _regular_files(artifact_dir)
            finish = artifact_dir / "finish_params.json"
            if not finish.is_file() or json.loads(finish.read_text()).get("submission_method") != (
                "nooa_final_response_output_directory"
            ):
                raise ValueError(f"Missing NOOA completion marker: {path}")
            artifacts_path = path.parent / "artifacts.json"
            artifacts = json.loads(artifacts_path.read_text())
            names = [item["name"] for item in artifacts]
            actual = {
                p.relative_to(artifact_dir).as_posix()
                for p in files
                if p != finish and p.relative_to(artifact_dir).parts[0] != "reference_files"
            }
            if len(set(names)) != len(names) or set(names) != actual:
                raise ValueError(f"Export manifest does not exactly cover submitted files: {path}")
            for item in artifacts:
                name = item["name"]
                relative = PurePosixPath(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != name
                    or "\\" in name
                    or "\x00" in name
                ):
                    raise ValueError(f"Noncanonical submitted artifact path: {name!r}")
                candidate = artifact_dir / name
                if (
                    not candidate.resolve().is_relative_to(artifact_dir.resolve())
                    or candidate.stat().st_size != item["size"]
                    or (_digest(candidate) != item["sha256"])
                ):
                    raise ValueError(f"Artifact changed since generation: {candidate}")
            hashes = {str(p.relative_to(artifact_dir)): _digest(p) for p in files}
            selected[task_id] = (
                path,
                receipt,
                artifact_dir,
                hashes,
                artifacts_path,
                _digest(path),
                _digest(artifacts_path),
            )
            episodes.add(key)
    required = set(tasks) - excluded
    if set(selected) != required:
        raise ValueError(
            f"Generation coverage must match explicit selection; missing={sorted(required - set(selected))}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        entries = []
        for task_id in tasks:
            if task_id in excluded:
                continue
            path, receipt, artifact_dir, hashes, artifacts_path, receipt_digest, artifact_digest = selected[task_id]
            destination = staging / "deliverables" / f"task_{task_id}" / "repeat_0"
            shutil.copytree(artifact_dir, destination)
            if any(_digest(destination / name) != digest for name, digest in hashes.items()):
                raise ValueError(f"Artifact bytes changed during copy: {task_id}")
            provenance = staging / "generation" / f"task_{task_id}"
            provenance.mkdir(parents=True)
            shutil.copyfile(path, provenance / "generation.json")
            shutil.copyfile(artifacts_path, provenance / "artifacts.json")
            if _digest(provenance / "generation.json") != receipt_digest or (
                _digest(provenance / "artifacts.json") != artifact_digest
            ):
                raise ValueError(f"Generation receipt changed during copy: {task_id}")
            entries.append(
                {
                    "task_id": receipt["task_id"],
                    "generation_episode_id": receipt["episode_id"],
                    "judging_repeat_index": 0,
                    "source_receipt": str(path.absolute()),
                    "source_receipt_sha256": receipt_digest,
                    "source_artifacts_sha256": artifact_digest,
                    "files_sha256": hashes,
                }
            )
        shutil.copyfile(source, staging / "gdpval_benchmark.jsonl")
        shutil.copyfile(reference_models, staging / "reference-models.json")
        if _digest(staging / "gdpval_benchmark.jsonl") != source_digest or (
            _digest(staging / "reference-models.json") != reference_digest
        ):
            raise ValueError("Canonical input or reference manifest changed during copy")
        overlay = {
            "gdpval_resources_server": {"resources_servers": {"gdpval": {"reference_models": refs}}},
            "gdpval_stirrup_agent": {
                "responses_api_agents": {
                    "stirrup_agent": {
                        "persist_deliverables_dir": str(output / "deliverables"),
                        "datasets": [
                            {
                                "name": "gdpval",
                                "type": "benchmark",
                                "jsonl_fpath": str(output / "gdpval_benchmark.jsonl"),
                                "prepare_script": "benchmarks/gdpval/prepare.py",
                                "num_repeats": 1,
                            }
                        ],
                    }
                }
            },
        }
        (staging / "judge_data.yaml").write_text(json.dumps(overlay, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "judge_gym_commit": JUDGE_GYM_COMMIT,
            "reference_recipe_efb_commit": REFERENCE_EFB_COMMIT,
            "canonical_dataset_sha256": source_digest,
            "reference_models_sha256": reference_digest,
            "tasks": entries,
            "task_count": len(entries),
            "canonical_task_count": len(tasks),
            "excluded_task_ids": sorted(excluded),
            "policy_inference_performed": False,
            "judging_performed": False,
        }
        (staging / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output


def main() -> None:
    """Prepare a new immutable-input bundle; generation roots are Resources artifact roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, action="append", required=True)
    parser.add_argument("--reference-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=220)
    parser.add_argument("--exclude-task-id", action="append", default=[])
    args = parser.parse_args()
    print(
        prepare_judging(
            source=args.source,
            generation_roots=args.generation_root,
            reference_models=args.reference_models,
            output=args.output,
            expected_tasks=args.expected_tasks,
            excluded_task_ids=set(args.exclude_task_id),
        )
    )


if __name__ == "__main__":
    main()
