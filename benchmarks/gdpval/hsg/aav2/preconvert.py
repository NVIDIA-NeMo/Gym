#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare candidate copies with native Office renders and media proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from benchmarks.gdpval.hsg.aav2.completion import read_rows, rollout_complete, task_ids
from benchmarks.gdpval.hsg.aav2.media import build
from resources_servers.gdpval.preconvert import preconvert_dir


def office(root: Path) -> None:
    ok, failed, errors = preconvert_dir(root, max_concurrent=4)
    print(f"Office: converted={ok}, failed={failed}", flush=True)
    for error in errors:
        print(f"Office render skipped: {error}", flush=True)


def check_reference_inputs(
    dataset: Path, source: Path, candidate: Path, missing_task_ids: set[str] | None = None
) -> None:
    """Check declared benchmark inputs against the existing conversion receipt."""
    receipt = candidate.with_name(candidate.name + ".media.json")
    entries = (
        {entry["source"]: entry for entry in json.loads(receipt.read_text())["entries"]} if receipt.exists() else {}
    )
    for row in read_rows(dataset):
        if row["task_id"] in (missing_task_ids or set()):
            continue
        inputs = row.get("reference_files") or []
        if isinstance(inputs, str):
            inputs = json.loads(inputs)
        for name in inputs:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid benchmark input path: {name}")
            if relative.parts[0] != "reference_files":
                relative = Path("reference_files") / relative
            relative = Path(f"task_{row['task_id']}") / "repeat_0" / relative
            entry = entries.get(relative.as_posix())
            if entry is None:
                raise ValueError(f"benchmark input has no conversion receipt: {relative}")
            for root, path_key, hash_key in (
                (source, "source", "source_sha256"),
                (candidate, "output", "output_sha256"),
            ):
                path = root / entry[path_key]
                if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"missing prepared benchmark input: {path}")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != entry[hash_key]:
                        raise ValueError(f"benchmark input changed since preparation: {path}")


def prepare(run_dir: Path) -> Path:
    run_dir = run_dir.resolve(strict=True)
    settings = json.loads((run_dir / "run.json").read_text())
    if Path(settings["RUN_DIR"]).resolve() != run_dir:
        raise ValueError("run.json does not belong to this run directory")
    dataset = Path(settings["DATASET"])
    source = run_dir / "deliverables"
    missing_ids = (
        set(json.loads(settings.get("MISSING_EVAL_TASK_IDS", "[]")))
        if settings.get("COUNT_EVAL_MISSING_AS_LOSS") == "true"
        else set()
    )
    rollout_complete(dataset, source, missing_ids)
    output = run_dir / "prepared"
    if output.exists() or output.is_symlink():
        candidate = output / "candidate"
        if output.is_symlink() or candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("prepared output must contain a real candidate directory")
        rollout_complete(dataset, candidate, missing_ids)
        check_reference_inputs(dataset, source, candidate, missing_ids)
        print("Reusing completed candidate preconversion", flush=True)
        return output
    task_names = {f"task_{task_id}" for task_id in task_ids(dataset) - missing_ids}
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=run_dir) as temporary:
        staged = Path(temporary) / "prepared"
        candidate = staged / "candidate"
        print("Preparing candidate", flush=True)
        build(source, candidate, prepare_zip=office, top_level_names=task_names | {"FILTER_MANIFEST.txt"})
        office(candidate)
        rollout_complete(dataset, candidate, missing_ids)
        check_reference_inputs(dataset, source, candidate, missing_ids)
        os.rename(staged, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(prepare(args.run_dir))
