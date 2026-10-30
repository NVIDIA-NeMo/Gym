# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restrict a recipe run to named tasks.

`gym eval run --only-tasks` applies to Harbor task targets only; recipe runs read a prepared dataset file. This
script filters each benchmark dataset of a resolved config to the rows whose task id matches one of the patterns,
writes the filtered files and a config overlay that points the run at them, and prints the overlay's path. The run
adds it as a final `--config` with `++use_cached_prepared_benchmarks=true` so the filtered files are not prepared over.

    gym env resolve <configs...> > resolved.yaml
    OVERLAY=$(python recipes/nooa-atom/only_tasks.py resolved.yaml <out dir> 'code-from-image,fp8-*')
"""

from __future__ import annotations

import json
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterator

import yaml


ID_FIELDS = ("task_id", "task_name", "instance_id")


def task_ids(row: dict[str, Any]) -> Iterator[str]:
    """Every name a dataset row can be selected by."""
    for field in ID_FIELDS:
        value = row.get(field)
        if isinstance(value, dict):
            value = value.get("task_id")
        if isinstance(value, str) and value:
            yield value
            if "/" in value:
                yield value.rsplit("/", 1)[-1]


def matches(row: dict[str, Any], patterns: list[str]) -> bool:
    return any(fnmatchcase(name, pattern) for name in task_ids(row) for pattern in patterns)


def benchmark_datasets(config: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """(Hydra key of the dataset entry, dataset entry) for every benchmark dataset a resolved config declares."""
    for instance, block in config.items():
        if not isinstance(block, dict) or "resources_servers" not in block:
            continue
        for server, server_config in (block["resources_servers"] or {}).items():
            for index, dataset in enumerate((server_config or {}).get("datasets") or []):
                if isinstance(dataset, dict) and dataset.get("type") == "benchmark":
                    yield f"{instance}.resources_servers.{server}.datasets.{index}", dataset


def filter_datasets(config: dict[str, Any], out_dir: Path, patterns: list[str]) -> dict[str, Any]:
    """Write the matching rows of every benchmark dataset under out_dir.

    Returns a config overlay that re-declares each affected `datasets` list whole with the filtered files (a list
    in a later config replaces the list; a dotted CLI override cannot address a list element).
    """
    overlay: dict[str, Any] = {}
    for key, dataset in benchmark_datasets(config):
        instance, _, server, _, index = key.split(".")
        source = Path(dataset["jsonl_fpath"])
        if not source.exists():
            raise SystemExit(f"only_tasks: dataset file {source} is missing; run `gym eval prepare` first")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = [row for row in rows if matches(row, patterns)]
        if not kept:
            raise SystemExit(f"only_tasks: no task in {source} matches {patterns}")
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / source.name
        target.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
        print(f"only_tasks: {len(kept)} of {len(rows)} rows of {source} -> {target}", file=sys.stderr)
        datasets = overlay.setdefault(instance, {}).setdefault("resources_servers", {}).setdefault(server, {})
        if "datasets" not in datasets:
            datasets["datasets"] = [dict(entry) for entry in config[instance]["resources_servers"][server]["datasets"]]
        datasets["datasets"][int(index)]["jsonl_fpath"] = str(target)
    if not overlay:
        raise SystemExit("only_tasks: the resolved config declares no benchmark dataset")
    return overlay


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    config = yaml.safe_load(Path(argv[1]).read_text(encoding="utf-8"))
    patterns = [p.strip() for p in argv[3].split(",") if p.strip()]
    if not isinstance(config, dict) or not patterns:
        raise SystemExit(__doc__)
    out_dir = Path(argv[2])
    overlay = filter_datasets(config, out_dir, patterns)
    overlay_path = out_dir / "only-tasks.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    print(overlay_path)


if __name__ == "__main__":
    main(sys.argv)
