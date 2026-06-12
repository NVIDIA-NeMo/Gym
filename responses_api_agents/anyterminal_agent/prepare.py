# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare the anyterminal_agent input dataset from Terminal Bench tasks.

    python prepare.py                                 # download tasks + build dataset + build SIFs
    python prepare.py --limit 5                       # first 5 tasks (smoke test)
    python prepare.py --task-name gpt2-codegolf       # single task
    python prepare.py --no-build-sif                  # skip SIF builds (pull docker:// at runtime)
    python prepare.py --sif-dir PATH                  # build SIFs into a custom directory

Prerequisites:
  - Harbor CLI on PATH (for dataset download).
  - `apptainer` on PATH for SIF builds (skip with --no-build-sif).

Schema anyterminal_agent expects: each row has `responses_create_params.metadata` with
`instance_id`, `task_name`, `docker_image`, `problem_statement`, and `task_dir`
(absolute path to the task definition directory on the host).
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


_THIS_DIR = Path(__file__).parent

DEFAULT_TASKS_CACHE = Path.home() / ".cache" / "harbor" / "tasks"
DEFAULT_DATASET_NAME = "terminal-bench"
DEFAULT_DATASET_VERSION = "2.0"


def _load_task_config(task_dir: Path) -> dict:
    """Read task.toml and instruction.md from a task directory."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            sys.exit("Python 3.11+ or `pip install tomli` is required to read task.toml files")

    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"

    if not config_path.exists():
        return {}

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    docker_image = (config.get("environment") or {}).get("docker_image", "ubuntu:22.04")
    problem_statement = instruction_path.read_text() if instruction_path.exists() else ""
    agent_timeout = (config.get("agent") or {}).get("timeout_sec", None)
    verifier_timeout = (config.get("verifier") or {}).get("timeout_sec", None)

    # Parse last WORKDIR from Dockerfile (if present).
    workdir = None
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        for line in dockerfile.read_text().splitlines():
            if line.strip().upper().startswith("WORKDIR"):
                workdir = line.strip().split(None, 1)[1] if len(line.strip().split()) > 1 else None

    return {
        "docker_image": docker_image,
        "problem_statement": problem_statement,
        "agent_timeout_sec": agent_timeout,
        "verifier_timeout_sec": verifier_timeout,
        "workdir": workdir,
    }


def _find_task_dirs(cache_dir: Path, dataset_name: str) -> list[Path]:
    """Locate all task directories under the harbor cache for a given dataset version."""
    tasks_dir = cache_dir / f"{dataset_name}"
    if tasks_dir.exists():
        return sorted(p for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.toml").exists())

    # Fallback: flat structure
    if cache_dir.exists():
        return sorted(p for p in cache_dir.iterdir() if p.is_dir() and (p / "task.toml").exists())

    return []


def _to_gym_row(task_dir: Path, task_cfg: dict) -> dict:
    task_name = task_dir.name
    return {
        "responses_create_params": {
            "input": [],
            "model": "model",  # placeholder; overridden at collect time via ng_collect_rollouts
            "metadata": {
                "instance_id": f"terminal_bench::{task_name}",
                "task_name": task_name,
                "docker_image": task_cfg.get("docker_image", "ubuntu:22.04"),
                "problem_statement": task_cfg.get("problem_statement", ""),
                "task_dir": str(task_dir.resolve()),
                "agent_timeout_sec": str(task_cfg["agent_timeout_sec"])
                if task_cfg.get("agent_timeout_sec") is not None
                else None,
                "verifier_timeout_sec": str(task_cfg["verifier_timeout_sec"])
                if task_cfg.get("verifier_timeout_sec") is not None
                else None,
                "workdir": task_cfg.get("workdir"),
            },
        },
    }


def harbor_download(tasks_cache: Path, dataset_name: str, version: str) -> None:
    """Download tasks via the Harbor CLI, skipping if already present."""
    from shutil import which

    tasks_dir = tasks_cache / f"{dataset_name}"
    if tasks_dir.exists() and any(tasks_dir.iterdir()):
        print(f"Tasks already present at {tasks_dir}, skipping download.", flush=True)
        return
    if not which("harbor"):
        sys.exit("`harbor` CLI not found on PATH.\nInstall it or download tasks manually.")
    print(f"Downloading {dataset_name}@{version} via Harbor CLI...", flush=True)
    proc = subprocess.run(
        ["harbor", "datasets", "download", f"{dataset_name}@{version}", "--output-dir", str(tasks_cache)], check=False
    )
    if proc.returncode != 0:
        sys.exit(f"harbor datasets download failed (exit {proc.returncode})")


def build_dataset(
    output: Path,
    tasks_cache: Path,
    dataset_name: str,
    limit: int | None,
    task_name: str | None,
) -> list[str]:
    task_dirs = _find_task_dirs(tasks_cache, dataset_name)
    if not task_dirs:
        sys.exit(f"No task directories found under {tasks_cache} after download — check harbor output above.")

    if task_name:
        names = [task_name] if isinstance(task_name, str) else task_name
        task_dirs = [d for d in task_dirs if d.name in names]
        missing = set(names) - {d.name for d in task_dirs}
        if missing:
            sys.exit(f"Tasks not found under {tasks_cache}: {', '.join(sorted(missing))}")
    elif limit:
        task_dirs = task_dirs[:limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for td in task_dirs:
        cfg = _load_task_config(td)
        if not cfg:
            print(f"  [skip] {td.name}: task.toml missing or empty", flush=True)
            continue
        rows.append(json.dumps(_to_gym_row(td, cfg)))

    output.write_text("\n".join(rows) + ("\n" if rows else ""))
    ids = [json.loads(r)["responses_create_params"]["metadata"]["task_name"] for r in rows]
    print(f"Wrote {len(ids)} rows -> {output}", flush=True)
    return ids


def _build_one_sif(task_name: str, docker_image: str, sif_dir: Path, force: bool) -> tuple[str, bool, str]:
    sif_path = sif_dir / f"{task_name}.sif"
    if sif_path.exists() and not force:
        return task_name, True, "exists"
    proc = subprocess.run(
        ["apptainer", "build", "--force", str(sif_path), f"docker://{docker_image}"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        return task_name, False, proc.stderr.strip()[-500:]
    return task_name, True, "built"


def build_images(task_rows: list[dict], sif_dir: Path, jobs: int, force: bool) -> None:
    from shutil import which

    if not which("apptainer"):
        sys.exit("`apptainer` not found on PATH. Install it or omit --sif-dir to skip SIF builds.")
    sif_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(task_rows)} SIF(s) into {sif_dir} with {jobs} worker(s)...", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _build_one_sif,
                r["responses_create_params"]["metadata"]["task_name"],
                r["responses_create_params"]["metadata"]["docker_image"],
                sif_dir,
                force,
            ): r
            for r in task_rows
        }
        for done in as_completed(futures):
            name, ok, detail = done.result()
            print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)
            if not ok:
                failures.append(name)
    if failures:
        print(f"\n{len(failures)} SIF build(s) failed:", flush=True)
        for name in failures:
            print(f"  - {name}", flush=True)
        sys.exit(1)
    print(f"All SIFs ready. Use: tb_sif_dir='{sif_dir}'", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=_THIS_DIR / "data" / "terminal_bench.jsonl")
    p.add_argument("--tasks-cache", type=Path, default=DEFAULT_TASKS_CACHE)
    p.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--version", default=DEFAULT_DATASET_VERSION)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task-name", nargs="+", default=None, metavar="TASK")
    p.add_argument("--sif-dir", type=Path, default=_THIS_DIR / "data" / "sifs")
    p.add_argument("--build-sif", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Rebuild SIFs that already exist")
    args = p.parse_args()

    # Download tasks via the Harbor CLI
    harbor_download(args.tasks_cache, args.dataset_name, args.version)

    # Build the dataset JSONL
    build_dataset(args.output, args.tasks_cache, args.dataset_name, args.limit, args.task_name)

    # Build the SIFs
    if args.build_sif:
        task_rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
        build_images(task_rows, args.sif_dir, args.jobs, args.force)


if __name__ == "__main__":
    main()
