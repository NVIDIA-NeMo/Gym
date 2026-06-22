#!/usr/bin/env python3
"""Download BiomniBench-DA and materialize Harbor task trees for harbor_agent.

Uses Gym's project venv (run from Gym repo root):

  source .venv/bin/activate
  python responses_api_agents/harbor_agent/scripts/prepare_biomnibench_da.py --smoke

Smoke mode materializes two tasks (da-1-3, da-1-4) for docker and singularity profiles.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HARBOR_AGENT_ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = HARBOR_AGENT_ROOT.parents[1]
DATA_ROOT = HARBOR_AGENT_ROOT / "data" / "biomnibench_da"
SOURCE_DIR = DATA_ROOT / "source"
MATERIALIZER = HARBOR_AGENT_ROOT / "scripts" / "materialize_biomnibench_da.py"
DATASET_ID = "phylobio/BiomniBench-DA"
SMOKE_TASKS = ("da-1-3", "da-1-4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Materialize a tiny docker+singularity slice (da-1-3, da-1-4).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip hf download when source/ already exists.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=SOURCE_DIR,
        help=f"HF dataset directory (default: {SOURCE_DIR}).",
    )
    parser.add_argument(
        "--write-rollout-jsonl",
        type=Path,
        default=None,
        help="Write rollout input JSONL with instance_id rows (default: none).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hf_download(local_dir: Path) -> None:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        DATASET_ID,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=GYM_ROOT)


def run_materializer(
    *,
    local_dir: Path,
    output_dir: Path,
    environment_type: str,
    tasks: list[str],
    n_repeats: int,
    overwrite: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(MATERIALIZER),
        "--local-dir",
        str(local_dir),
        "--output-dir",
        str(output_dir),
        "--environment-type",
        environment_type,
        "--tasks",
        *tasks,
        "--n-repeats",
        str(n_repeats),
        "--partition",
        "all",
        "--include-singletons",
        "--include-uncovered",
    ]
    if overwrite:
        cmd.append("--overwrite")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=GYM_ROOT)
    registry = json.loads((output_dir / "registry.json").read_text(encoding="utf-8"))
    return [task["name"] for task in registry[0]["tasks"]]


def write_rollout_jsonl(path: Path, task_names: list[str]) -> None:
    rows = [
        {
            "instance_id": f"biomnibench_da::{name}",
            "responses_create_params": {"input": []},
            "agent_ref": {"name": "harbor_agent"},
        }
        for name in sorted(task_names)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote rollout JSONL ({len(rows)} rows) -> {path}")


def main() -> None:
    args = parse_args()
    if not args.smoke:
        raise SystemExit("Only --smoke is implemented for now. Use materialize_biomnibench_da.py for full runs.")

    if not args.skip_download and not args.local_dir.is_dir():
        hf_download(args.local_dir)
    elif not args.local_dir.is_dir():
        raise SystemExit(f"Missing dataset at {args.local_dir}. Run without --skip-download.")

    tasks = list(SMOKE_TASKS)
    docker_out = DATA_ROOT / "tasks_smoke_docker"
    sing_out = DATA_ROOT / "tasks_smoke_singularity"

    docker_task_names = run_materializer(
        local_dir=args.local_dir,
        output_dir=docker_out,
        environment_type="docker",
        tasks=tasks,
        n_repeats=1,
        overwrite=args.overwrite,
    )
    run_materializer(
        local_dir=args.local_dir,
        output_dir=sing_out,
        environment_type="singularity",
        tasks=tasks,
        n_repeats=1,
        overwrite=args.overwrite,
    )

    rollout_path = args.write_rollout_jsonl or (DATA_ROOT / "biomnibench_da_smoke.jsonl")
    write_rollout_jsonl(rollout_path, docker_task_names)
    print(f"Docker tasks: {docker_out}")
    print(f"Singularity tasks: {sing_out}")


if __name__ == "__main__":
    main()
