#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare candidate copies with native Office renders and media proxies."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from benchmarks.gdpval.hsg.aav2.completion import rollout_complete, task_ids
from benchmarks.gdpval.hsg.aav2.media import build
from resources_servers.gdpval.preconvert import preconvert_dir


def office(root: Path) -> None:
    ok, failed, errors = preconvert_dir(root, max_concurrent=4)
    print(f"Office: converted={ok}, failed={failed}", flush=True)
    for error in errors:
        print(f"Office render skipped: {error}", flush=True)


def prepare(run_dir: Path) -> Path:
    run_dir = run_dir.resolve(strict=True)
    settings = json.loads((run_dir / "run.json").read_text())
    if Path(settings["RUN_DIR"]).resolve() != run_dir:
        raise ValueError("run.json does not belong to this run directory")
    dataset = Path(settings["DATASET"])
    source = run_dir / "deliverables"
    rollout_complete(dataset, source)
    output = run_dir / "prepared"
    if output.exists() or output.is_symlink():
        candidate = output / "candidate"
        if output.is_symlink() or candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("prepared output must contain a real candidate directory")
        rollout_complete(dataset, candidate)
        print("Reusing completed candidate preconversion", flush=True)
        return output
    task_names = {f"task_{task_id}" for task_id in task_ids(dataset)}
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=run_dir) as temporary:
        staged = Path(temporary) / "prepared"
        candidate = staged / "candidate"
        print("Preparing candidate", flush=True)
        build(source, candidate, prepare_zip=office, top_level_names=task_names | {"FILTER_MANIFEST.txt"})
        office(candidate)
        rollout_complete(dataset, candidate)
        os.rename(staged, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(prepare(args.run_dir))
