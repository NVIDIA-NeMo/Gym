#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and materialize BiomniBench-DA as Harbor tasks for NeMo Gym harbor_agent.

Downloads the upstream ``phylobio/BiomniBench-DA`` dataset from HuggingFace (unless
already present locally), then materializes it into Harbor task trees with an
OpenAI-compatible LLM judge, upstream-faithful content otherwise. Supports two
deployment profiles via ``--environment-type``:

- ``docker``: bind-mount source task data at ``/app/data`` via docker-compose.yaml
- ``singularity``: copy data into ``environment/files/data`` + ``setup.sh`` staging

Examples::

  # Download (if needed) + materialize an example, docker profile
  python environments/biomnibench_da/prepare.py \\
    --download \\
    --build-docker-image \\
    --environment-type docker \\
    --tasks da-10-1 \\
    --include-singletons --include-uncovered \\
    --output-dir environments/biomnibench_da/data/example \\
    --overwrite

  # Materialize the full default (train+test) split, singularity profile,
  # assuming the dataset was already downloaded to --local-dir
  python environments/biomnibench_da/prepare.py \\
    --local-dir environments/biomnibench_da/data/source \\
    --environment-type singularity \\
    --output-dir environments/biomnibench_da/data/tasks_singularity \\
    --overwrite

Also writes ``<output-dir>/rollout_input.jsonl`` — one ``ng_collect_rollouts`` row per
materialized task (``instance_id`` form ``<dataset-name>::<task-name>``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "phylobio/BiomniBench-DA"
TASK_DIR_RE = re.compile(r"^da-(\d+)-(\d+)$")
DEFAULT_SPLIT_SEED = "trace2skill-biomnibench-da"
DEFAULT_TRAIN_FRACTION = 0.2
DEFAULT_CONTAINER_DATA_DIR = "/app/data"
DEFAULT_DOCKER_IMAGE = "biomnibench-da-runtime:smoke"
DEFAULT_ROLLOUT_INPUT_NAME = "rollout_input.jsonl"
DEFAULT_AGENT_NAME = "harbor_agent"
ENV_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_DIR = ENV_ROOT / "data" / "source"
DEFAULT_DOCKERFILE = str(ENV_ROOT / "docker" / "biomnibench-da-runtime.Dockerfile")

SINGULARITY_SETUP_SH = """#!/bin/bash
# BiomniBench-DA Singularity bootstrap: Harbor server deps + task data staging.
set -e
if ! python3 -c "import uvicorn, fastapi" 2>/dev/null; then
  echo "[harbor] Installing server dependencies (Python/uvicorn)..." >&2
  if python3 -m pip install uvicorn fastapi 2>/dev/null; then
    :
  elif python3 -m pip install --user uvicorn fastapi 2>/dev/null; then
    :
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq 2>/dev/null && apt-get install -y -qq python3-uvicorn python3-fastapi python3-pydantic 2>/dev/null || true
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache py3-uvicorn 2>/dev/null || true
  fi
  if ! python3 -c "import uvicorn, fastapi" 2>/dev/null && command -v pip3 >/dev/null 2>&1; then
    pip3 install --break-system-packages uvicorn fastapi 2>/dev/null || pip3 install uvicorn fastapi 2>/dev/null || true
  fi
fi
if [ -d "${HARBOR_STAGING:-}/data" ]; then
  mkdir -p /app/data
  cp -r "${HARBOR_STAGING}/data/." /app/data/
fi
"""

SINGLETON_THRESHOLD = 1
MIN_TRAIN_FOR_SKILL = 1
MIN_TEST_FOR_EVAL = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download {DATASET_ID} from HuggingFace into --local-dir before materializing "
        "(skipped automatically if --local-dir already exists and is non-empty).",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=DATASET_ID,
        help=f"HuggingFace dataset repo id to download (default: {DATASET_ID}).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local BiomniBench-DA download dir, from `hf download` or --download (default: {DEFAULT_LOCAL_DIR}).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--partition", choices=["all", "train", "test"], default="all")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Reuse an existing split_manifest.json (from a prior train run).",
    )
    parser.add_argument("--stratify-by", default="task_type", choices=["task_type", "category", "difficulty"])
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="Only include these task IDs (e.g. da-9-1 da-12-2).")
    parser.add_argument(
        "--papers", nargs="*", default=None, help="Only include tasks from these papers (e.g. da-9 da-12)."
    )
    parser.add_argument(
        "--max-data-mb", type=int, default=None, help="Exclude tasks whose environment/data/ exceeds this size in MB."
    )
    parser.add_argument(
        "--include-singletons", action="store_true", help="Include singleton task_types (default: excluded)."
    )
    parser.add_argument(
        "--include-uncovered", action="store_true", help="Include task_types that land entirely in one partition."
    )
    parser.add_argument(
        "--judge-model", default=None, help="Override JUDGE_MODEL (defaults to ${JUDGE_MODEL} env var passthrough)."
    )
    parser.add_argument(
        "--storage-mb-override",
        type=int,
        default=None,
        help="Override storage_mb for all tasks (e.g. 40960 for heavy-data tasks).",
    )
    parser.add_argument("--dataset-name", default="biomnibench_da")
    parser.add_argument(
        "--rollout-input-fpath",
        type=Path,
        default=None,
        help=f"ng_collect_rollouts input JSONL path (default: <output-dir>/{DEFAULT_ROLLOUT_INPUT_NAME}).",
    )
    parser.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help=f"agent_ref.name written into rollout input rows (default: {DEFAULT_AGENT_NAME}).",
    )
    parser.add_argument(
        "--environment-type",
        choices=["docker", "singularity"],
        default="docker",
        help="docker: bind-mount data via docker-compose.yaml. "
        "singularity: stage data under environment/files/ for HPC.",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help=f"Pre-built runtime image (Harbor [environment].docker_image; default: {DEFAULT_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--build-docker-image",
        action="store_true",
        help="Build --docker-image before materializing Docker tasks.",
    )
    parser.add_argument(
        "--data-mount-root",
        type=Path,
        default=None,
        help="Host root for docker bind mounts (default: --local-dir).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_dataset(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
    """Download the BiomniBench-DA dataset repo from HuggingFace into local_dir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Missing dependency `huggingface_hub`. Install with: pip install huggingface_hub")

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir} ...", file=sys.stderr)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    print("Download complete.", file=sys.stderr)


def requested_download_patterns(args: argparse.Namespace) -> list[str] | None:
    patterns = [f"{task}/**" for task in args.tasks or []]
    patterns.extend(f"{paper}-*/**" for paper in args.papers or [])
    return sorted(set(patterns)) or None


def build_docker_image(image: str) -> None:
    subprocess.run(
        ["docker", "build", "-t", image, "-f", DEFAULT_DOCKERFILE, str(ENV_ROOT)],
        check=True,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_rollout_input_rows(
    registry_tasks: list[dict[str, Any]],
    *,
    dataset_name: str,
    agent_name: str,
) -> list[dict[str, Any]]:
    """Build ng_collect_rollouts rows for each materialized Harbor task."""
    return [
        {
            "instance_id": f"{dataset_name}::{task['name']}",
            "responses_create_params": {"input": []},
            "agent_ref": {"name": agent_name},
        }
        for task in registry_tasks
    ]


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    forbidden = {cwd, cwd.parent, Path.home().resolve(), Path("/")}
    if path == Path(".") or resolved in forbidden:
        raise SystemExit(f"Refusing unsafe --output-dir {path!s}.")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"{path} exists and is not empty. Pass --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stable_key(seed: str, paper: str) -> str:
    return hashlib.sha256(f"{seed}\0{paper}".encode("utf-8")).hexdigest()


def base_paper(task_id: str) -> str:
    m = TASK_DIR_RE.match(task_id)
    if not m:
        raise ValueError(f"Bad task_id: {task_id}")
    return f"da-{m.group(1)}"


def paper_sort_key(task_id: str) -> tuple[int, int]:
    m = TASK_DIR_RE.match(task_id)
    if not m:
        return (10**9, 0)
    return (int(m.group(1)), int(m.group(2)))


def validate_options(args: argparse.Namespace) -> None:
    if not args.docker_image:
        raise SystemExit(f"--docker-image is required (default: {DEFAULT_DOCKER_IMAGE}).")
    if args.data_mount_root is not None and not args.data_mount_root.is_dir():
        raise SystemExit(f"--data-mount-root does not exist: {args.data_mount_root}")


def is_docker_bind(args: argparse.Namespace) -> bool:
    return args.environment_type == "docker"


def is_singularity_copy(args: argparse.Namespace) -> bool:
    return args.environment_type == "singularity"


def source_task_data_dir(source_task_id: str, args: argparse.Namespace) -> Path:
    root = (args.data_mount_root or args.local_dir).resolve()
    return root / source_task_id / "environment" / "data"


def _yaml_quote(value: str) -> str:
    return json.dumps(value)


def write_docker_compose_yaml(
    path: Path,
    *,
    image: str,
    data_mount: tuple[Path, str] | None = None,
    skills_mount: tuple[Path, str] | None = None,
    read_only: bool = True,
) -> None:
    """Write a self-contained compose override for the shared prebuilt runtime image.

    Harbor's ``DockerEnvironment`` picks ``environment/docker-compose.yaml`` over its
    own built-in prebuilt-image template whenever the file exists (see
    ``harbor.environments.docker.docker.DockerEnvironment._docker_compose_path``), so
    this file must declare everything Harbor's own templates would otherwise provide:

    - ``image``/``command``: mirrors ``docker-compose-prebuilt.yaml`` (long-running
      container for the shared prebuilt image, since Harbor never merges the two
      files).
    - ``${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}`` and the agent-logs
      equivalent: Harbor's ``DockerEnvironment.is_mounted`` is ``True``, so
      ``Verifier.verify()`` skips downloading ``/logs/verifier`` after running
      ``tests/test.sh`` and instead assumes it is already bind-mounted to
      ``trial_dir/verifier`` on the host (see
      ``harbor.models.trial.paths.EnvironmentPaths``). Without this mount, the judge
      still runs and scores the trial, but ``reward.txt``/``reward.json`` never reach
      the host and the trial fails with ``RewardFileNotFoundError``.
    - ``TEST_DIR`` env var, ``network_mode``, and CPU/memory limits: passed through by
      Harbor as compose-file env vars (``DockerEnvironmentEnvVars.to_env_dict()``), not
      OS env vars, so they only take effect if the compose file references them.

    These come from ``DockerEnvironmentEnvVars`` (see ``harbor/environments/docker/
    docker.py``) and are injected into the ``docker compose`` subprocess env by Harbor
    itself, so referencing them here as ``${...}`` is resolved the same way Harbor's
    own templates resolve them -- no extra plumbing needed on our side.
    """
    volume_lines: list[str] = [
        "      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}",
        "      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}",
    ]
    for mount in (data_mount, skills_mount):
        if mount is None:
            continue
        host_path, container_path = mount
        volume_lines.extend(
            [
                "      - type: bind",
                f"        source: {_yaml_quote(str(host_path.resolve()))}",
                f"        target: {_yaml_quote(container_path)}",
                f"        read_only: {'true' if read_only else 'false'}",
            ]
        )
    if data_mount is None and skills_mount is None:
        return
    lines = [
        "services:",
        "  main:",
        f"    image: {_yaml_quote(image)}",
        "    pull_policy: never",
        '    command: ["sh", "-c", "sleep infinity"]',
        "    network_mode: ${NETWORK_MODE:-bridge}",
        "    environment:",
        "      - TEST_DIR=${TEST_DIR}",
        "    volumes:",
        *volume_lines,
        "    deploy:",
        "      resources:",
        "        limits:",
        "          cpus: ${CPUS}",
        "          memory: ${MEMORY}",
    ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_singularity_setup_sh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SINGULARITY_SETUP_SH, encoding="utf-8")
    path.chmod(0o755)


def stage_singularity_data(dst_env: Path) -> None:
    """Relocate environment/data -> environment/files/data for Singularity staging."""
    src_data = dst_env / "data"
    files_data = dst_env / "files" / "data"
    if not src_data.is_dir():
        return
    files_data.parent.mkdir(parents=True, exist_ok=True)
    if files_data.exists():
        shutil.rmtree(files_data)
    shutil.copytree(src_data, files_data)
    shutil.rmtree(src_data)
    write_singularity_setup_sh(dst_env / "files" / "setup.sh")


def materialize_environment(
    source_dir: Path,
    dst_env: Path,
    source_task_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Create task environment/ for docker bind mounts or singularity file staging."""
    env_info: dict[str, Any] = {
        "environment_type": args.environment_type,
        "container_data_dir": DEFAULT_CONTAINER_DATA_DIR,
        "source_data_dir": None,
        "docker_image": args.docker_image,
    }

    data_mount: tuple[Path, str] | None = None

    if is_docker_bind(args):
        data_dir = source_task_data_dir(source_task_id, args)
        if not data_dir.is_dir():
            raise SystemExit(f"Missing data directory for bind mount: {data_dir} (source task {source_task_id})")
        env_info["source_data_dir"] = str(data_dir.resolve())
        data_mount = (data_dir, DEFAULT_CONTAINER_DATA_DIR)
        dst_env.mkdir(parents=True, exist_ok=True)
    else:
        src_env = source_dir / "environment"
        if src_env.is_dir():
            shutil.copytree(src_env, dst_env, dirs_exist_ok=True)
        stage_singularity_data(dst_env)
        compose_path = dst_env / "docker-compose.yaml"
        if compose_path.exists():
            compose_path.unlink()

    if is_docker_bind(args):
        compose_path = dst_env / "docker-compose.yaml"
        if data_mount:
            write_docker_compose_yaml(
                compose_path,
                image=args.docker_image,
                data_mount=data_mount,
                skills_mount=None,
                read_only=True,
            )
        elif compose_path.exists():
            compose_path.unlink()

    return env_info


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def discover_tasks(local_dir: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(local_dir.iterdir()):
        if not task_dir.is_dir() or not TASK_DIR_RE.match(task_dir.name):
            continue
        toml_path = task_dir / "task.toml"
        if not toml_path.exists():
            print(f"WARNING: skipping {task_dir.name} (no task.toml)", file=sys.stderr)
            continue
        parsed = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        meta = parsed.get("metadata", {})
        data_dir = task_dir / "environment" / "data"
        data_mb = dir_size_mb(data_dir) if data_dir.is_dir() else 0.0
        tasks[task_dir.name] = {
            "path": task_dir,
            "metadata": meta,
            "task_type": str(meta.get("task_type", "")),
            "category": str(meta.get("category", "")),
            "difficulty": str(meta.get("difficulty", "")),
            "paper": base_paper(task_dir.name),
            "parsed_toml": parsed,
            "data_mb": round(data_mb, 1),
        }
    if not tasks:
        raise SystemExit(f"No da-*/task.toml found under {local_dir}.")
    return tasks


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def assign_partitions(tasks: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    if args.split_manifest:
        manifest = read_json(args.split_manifest)
        assignments: dict[str, str] = {}
        for item in manifest.get("task_assignments", []):
            assignments[str(item["task_id"])] = str(item["partition"])
        for task_id in tasks:
            if task_id not in assignments:
                assignments[task_id] = "test"
        return assignments

    papers: dict[str, list[str]] = defaultdict(list)
    for task_id, info in tasks.items():
        papers[info["paper"]].append(task_id)

    def paper_stratum(paper: str) -> str:
        field = args.stratify_by
        vals = Counter(tasks[t]["metadata"].get(field, "unknown") for t in papers[paper])
        return vals.most_common(1)[0][0]

    train_papers: set[str] = set()
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for paper in papers:
        by_stratum[paper_stratum(paper)].append(paper)
    for _stratum, plist in sorted(by_stratum.items()):
        ordered = sorted(plist, key=lambda p: stable_key(args.split_seed, p))
        n_train = max(1, round(len(ordered) * args.train_fraction)) if len(ordered) > 1 else 0
        train_papers.update(ordered[:n_train])

    return {task_id: ("train" if info["paper"] in train_papers else "test") for task_id, info in tasks.items()}


def filter_tasks(
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0, "total": 0})
    for task_id, info in tasks.items():
        tt = info["task_type"] or "unknown"
        part = assignments[task_id]
        type_counts[tt][part] += 1
        type_counts[tt]["total"] += 1

    excluded_types: set[str] = set()
    for tt, counts in type_counts.items():
        if not args.include_singletons and counts["total"] <= SINGLETON_THRESHOLD:
            excluded_types.add(tt)
            continue
        if not args.include_uncovered:
            if counts["train"] < MIN_TRAIN_FOR_SKILL or counts["test"] < MIN_TEST_FOR_EVAL:
                excluded_types.add(tt)

    if excluded_types:
        excl_detail = ", ".join(
            f"{tt}(n={type_counts[tt]['total']},tr={type_counts[tt]['train']},te={type_counts[tt]['test']})"
            for tt in sorted(excluded_types)
        )
        print(f"Excluding {len(excluded_types)} task_types: {excl_detail}")

    return {task_id: info for task_id, info in tasks.items() if (info["task_type"] or "unknown") not in excluded_types}


def select_tasks(
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    args: argparse.Namespace,
) -> list[str]:
    selected = sorted(tasks.keys(), key=paper_sort_key)
    if args.partition != "all":
        selected = [t for t in selected if assignments[t] == args.partition]
    if args.tasks is not None:
        allowed = set(args.tasks)
        selected = [t for t in selected if t in allowed]
    if args.papers is not None:
        allowed_papers = set(args.papers)
        selected = [t for t in selected if tasks[t]["paper"] in allowed_papers]
    if args.max_data_mb is not None:
        cap = args.max_data_mb
        skipped = [t for t in selected if tasks[t]["data_mb"] > cap]
        if skipped:
            detail = ", ".join(f"{t}({tasks[t]['data_mb']:.0f}MB)" for t in skipped)
            print(f"Skipping {len(skipped)} tasks exceeding --max-data-mb {cap}: {detail}")
        selected = [t for t in selected if tasks[t]["data_mb"] <= cap]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


# --------------------------------------------------------------------------- #
# Patched LLM judge — upstream-faithful rubric scoring, OpenAI-compatible endpoint
# --------------------------------------------------------------------------- #
def patched_judge_source() -> str:
    return r'''#!/usr/bin/env python3
"""BiomniBench-DA LLM judge — upstream-faithful, OpenAI-compatible endpoint.

Verbatim port of the upstream judge (phylobio/BiomniBench-DA da-*/tests/llm_judge.py):
same rubric-level parsing, same prompt, same A/B/C -> points scoring (total 0-100), and
the same reward.json ({"score": <int>}) + evaluation.json outputs. The judge reads the
agent-authored `/logs/verifier/trace.md` and `/logs/verifier/answer.txt` (copied from
/app by test.sh), exactly as the agent is instructed to produce in instruction.md.

Deviations from upstream are confined to integration plumbing:
  (1) the model call goes through an OpenAI-compatible chat completions endpoint
      instead of the Anthropic SDK, wrapped with retry/backoff on transient errors;
  (2) a `reward.txt` is written with the score normalized to 0.0-1.0 so Harbor (which
      prefers reward.txt over reward.json) surfaces a properly-scaled NeMo Gym reward;
  (3) gold metadata is folded into evaluation.json for downstream rollout analysis.
The rubric scoring math itself (level letters -> rubric-defined points -> integer sum)
is byte-for-byte the upstream algorithm.

Environment variables (set in [verifier.env] of task.toml):
  OPENAI_API_KEY   — API key for the judge endpoint
  OPENAI_BASE_URL  — base URL (e.g. https://inference-api.nvidia.com/v1)
  JUDGE_MODEL      — judge model name
Optional: JUDGE_MAX_TOKENS (default 8192), JUDGE_TIMEOUT_SEC, and retry knobs
  JUDGE_MAX_RETRIES / JUDGE_RETRY_BASE_SEC / JUDGE_RETRY_MAX_SEC.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _int_env(name: str, default: int) -> int:
    """Read an int env var, tolerating unset/empty/garbage values.

    JUDGE_MAX_TOKENS is passed through from task.toml as "${JUDGE_MAX_TOKENS}";
    when that shell var is unset the substitution can yield "" (or junk), which
    would crash a bare int(). Fall back to the default in that case.
    """
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def parse_rubric_levels(rubric_text: str) -> dict[str, dict[str, int]]:
    """Parse the rubric into {criterion_<N>: {"A": pts, "B": pts, "C": pts}}.

    Supports the current rubric format (single `Levels: A=X B=Y C=0` header per
    criterion) and the legacy format (per-line `[A] (X points): ...`).
    """
    out: dict[str, dict[str, int]] = {}
    parts = re.split(r"^Criterion\s+(\d+)\s*:", rubric_text, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        n = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        levels: dict[str, int] = {}
        # Current format: single "Levels: A=X B=Y C=0" header (supports negatives)
        m = re.search(
            r"Levels:\s*((?:[A-Z]=-?\d+\s*)+)",
            body,
        )
        if m:
            for lm in re.finditer(r"([A-Z])=(-?\d+)", m.group(1)):
                levels[lm.group(1).upper()] = int(lm.group(2))
        # Legacy fallback: per-line "[A] (N points)"
        if not levels:
            for lm in re.finditer(r"\[([A-Z])\]\s*\(\s*(-?\d+)\s*points?\s*\)", body):
                levels[lm.group(1).upper()] = int(lm.group(2))
        if levels:
            out[f"criterion_{n}"] = levels
    return out


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retryable_exception_types() -> tuple[type, ...]:
    """Transient OpenAI SDK exception types available in the installed version."""
    try:
        import openai
    except ImportError:
        return tuple()
    types: list[type] = []
    for name in (
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    ):
        exc_type = getattr(openai, name, None)
        if isinstance(exc_type, type):
            types.append(exc_type)
    return tuple(types)


def _is_retryable(exc: Exception, retryable_types: tuple[type, ...]) -> bool:
    if retryable_types and isinstance(exc, retryable_types):
        return True
    # Some OpenAI-compatible gateways surface 429/5xx as a generic APIStatusError.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _RETRYABLE_STATUS


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honor a server-provided Retry-After header (seconds) if present."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _create_with_retry(client, **kwargs):
    """chat.completions.create with backoff on rate-limit / transient errors.

    The judge runs once per trial under high concurrency, so a single 429 from the
    gateway otherwise marks the trial judge-unavailable. Tunable via verifier env
    (set in task.toml [verifier.env]):
      JUDGE_MAX_RETRIES    extra attempts after the first (default 6)
      JUDGE_RETRY_BASE_SEC base backoff seconds, doubled each attempt (default 2.0)
      JUDGE_RETRY_MAX_SEC  per-wait backoff cap in seconds (default 60.0)
    """
    max_retries = int(os.environ.get("JUDGE_MAX_RETRIES", "6"))
    base = float(os.environ.get("JUDGE_RETRY_BASE_SEC", "2.0"))
    cap = float(os.environ.get("JUDGE_RETRY_MAX_SEC", "60.0"))
    retryable_types = _retryable_exception_types()

    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc, retryable_types):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(cap, base * (2 ** attempt))
            delay += random.uniform(0.0, base)  # jitter to avoid thundering herd
            attempt += 1
            print(
                f"[judge] transient error ({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{max_retries} in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)


def call_judge(prompt: str) -> str:
    """Send the upstream judge prompt to an OpenAI-compatible endpoint; return text.

    This is the ONLY upstream deviation in the model call: it goes through an
    OpenAI-compatible chat completions endpoint (OPENAI_API_KEY / OPENAI_BASE_URL /
    JUDGE_MODEL) instead of the Anthropic SDK, and is wrapped in retry/backoff.
    Prompt construction, JSON parsing, and rubric scoring stay upstream-verbatim.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Missing dependency `openai`. Install with: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("JUDGE_MODEL", "") or os.environ.get("MODEL_NAME", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    if not model:
        raise SystemExit("JUDGE_MODEL not set.")

    # max_retries=0: our _create_with_retry loop owns retry/backoff so the SDK's
    # built-in retries don't compound the wait. timeout caps a single attempt.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        max_retries=0,
        timeout=float(os.environ.get("JUDGE_TIMEOUT_SEC", "120")),
    )

    response = _create_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_int_env("JUDGE_MAX_TOKENS", 8192),
    )
    return response.choices[0].message.content or ""


def main() -> None:
    logs_dir = Path("/logs/verifier")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Read rubric from tests directory
    rubric_path = Path("/tests/rubric.txt")
    if rubric_path.exists():
        rubric = rubric_path.read_text()
    else:
        print("ERROR: rubric.txt not found in /tests/")
        (logs_dir / "reward.json").write_text(json.dumps({"score": 0}, indent=2))
        (logs_dir / "reward.txt").write_text("0.0\n")
        return

    gold = read_json(Path("/tests/gold_metadata.json"))

    # Read agent outputs (copied to /logs/verifier/ by test.sh)
    trace_path = logs_dir / "trace.md"
    answer_path = logs_dir / "answer.txt"

    trace_content = ""
    answer_content = ""

    if trace_path.exists():
        trace_content = trace_path.read_text()
    else:
        print("Warning: trace.md not found")

    if answer_path.exists():
        answer_content = answer_path.read_text()
    else:
        print("Warning: answer.txt not found")

    # If no output files exist, score is 0
    if not trace_content and not answer_content:
        print("No output files found. Score: 0")
        (logs_dir / "reward.json").write_text(json.dumps({"score": 0}, indent=2))
        (logs_dir / "reward.txt").write_text("0.0\n")
        (logs_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "schema_version": "biomnibench_da_reward.v1",
                    "total_score": 0,
                    "reward": 0.0,
                    "scoring_method": "llm_rubric_levels_openai",
                    "criteria": {},
                    "reasoning": "No trace.md or answer.txt produced by the agent.",
                    "judge_model": os.environ.get("JUDGE_MODEL"),
                    "answer_source": "none",
                    **{k: gold.get(k) for k in gold},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    prompt = f"""You are an expert evaluator for a data analysis task.

Evaluate the agent's work using the following rubric:

{rubric}

Here is the agent's analysis trace:

<trace>
{trace_content if trace_content else "[No trace file provided]"}
</trace>

Here is the agent's final answer:

<answer>
{answer_content if answer_content else "[No answer file provided]"}
</answer>

For each criterion in the rubric, choose ONE level: A, B, or C — based purely on which level description best describes the agent's work. Do not output numerical points; the score for each level is computed automatically from the rubric.

You MUST respond with a JSON object in exactly this format:
{{
  "criteria": {{
    "criterion_1": {{"level": "A", "reason": "<one-sentence explanation>"}},
    "criterion_2": {{"level": "B", "reason": "<one-sentence explanation>"}},
    ...
  }},
  "overall_reasoning": "<short summary>"
}}

Each "level" value must be exactly the single character "A", "B", or "C". Only output the JSON object, nothing else."""

    # The ONLY upstream deviation: OpenAI-compatible endpoint + retry (see call_judge).
    response_text = call_judge(prompt)
    print(f"Raw response (first 1000 chars): {response_text[:1000]}...")

    # Parse JSON from response
    try:
        # Try to find JSON object in response: opening brace -> matching close brace
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)

        criteria = result.get("criteria", {})
        reasoning = result.get("overall_reasoning", result.get("reasoning", "No reasoning provided"))

        # The LLM produces only level letters (A / B / C). Map each letter to
        # its rubric-defined point value programmatically. This eliminates
        # judge arithmetic noise entirely.
        try:
            criterion_levels = parse_rubric_levels(rubric)  # {criterion_n: {"A": pts, ...}}
        except Exception as parse_err:
            print(f"NOTE: failed to parse rubric levels: {parse_err}")
            criterion_levels = {}

        for k, c in list(criteria.items()):
            if not isinstance(c, dict):
                continue
            allowed = criterion_levels.get(k) or {}
            level = (c.get("level") or "").strip().upper()
            if level in allowed:
                c["score"] = allowed[level]
            elif "score" in c:
                # Legacy fallback: LLM gave a numeric score; snap to nearest allowed
                try:
                    stated = int(c.get("score", 0))
                except (TypeError, ValueError):
                    stated = 0
                if allowed:
                    target = min(allowed.values(), key=lambda v: abs(v - stated))
                    c["score"] = target
            else:
                c["score"] = 0  # missing level + missing score -> no credit

        # Total = sum of programmatically-derived per-criterion scores.
        if criteria:
            criterion_sum = 0
            for c in criteria.values():
                if not isinstance(c, dict):
                    continue
                try:
                    criterion_sum += int(c.get("score", 0))
                except (TypeError, ValueError):
                    pass
            total_score = criterion_sum
        else:
            total_score = int(result.get("total_score", result.get("score", 0)))

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Response was: {response_text}")

        # Try to extract total score from text
        score_match = re.search(r'"total_score"\s*:\s*(\d+)', response_text)
        if not score_match:
            score_match = re.search(r'"score"\s*:\s*(\d+)', response_text)

        if score_match:
            total_score = int(score_match.group(1))
        else:
            total_score = 0

        criteria = {}
        reasoning = f"Failed to parse full response: {str(e)}"

    # Clamp score to valid range (upstream 0-100 integer scale)
    total_score = max(0, min(100, total_score))
    # Normalized 0.0-1.0 reward for NeMo Gym / Harbor (reward.txt takes precedence).
    normalized_reward = total_score / 100.0

    print(f"Total Score: {total_score}/100  (reward={normalized_reward:.3f})")
    print(f"Criteria: {json.dumps(criteria, indent=2)}")
    print(f"Reasoning: {reasoning}")

    # reward.txt: normalized 0-1 scalar. Harbor reads this in preference to
    # reward.json, so the Gym `reward` is correctly scaled to [0, 1].
    (logs_dir / "reward.txt").write_text(f"{normalized_reward}\n", encoding="utf-8")
    # reward.json: upstream-faithful single integer key (0-100) as an artifact.
    (logs_dir / "reward.json").write_text(json.dumps({"score": total_score}, indent=2))

    # Rich evaluation artifact (not parsed by Harbor) for rollout analysis.
    evaluation_data = {
        "schema_version": "biomnibench_da_reward.v1",
        "total_score": total_score,
        "reward": normalized_reward,
        "scoring_method": "llm_rubric_levels_openai",
        "criteria": criteria,
        "reasoning": reasoning,
        "judge_model": os.environ.get("JUDGE_MODEL"),
        "answer_source": "file" if answer_content else ("trace" if trace_content else "none"),
        "question_group_key": gold.get("question_group_key", ""),
        "partition": gold.get("partition", ""),
        "repeat_index": gold.get("repeat_index"),
        "n_repeats": gold.get("n_repeats"),
        "task_name": gold.get("task_name", ""),
        "source_task_id": gold.get("source_task_id", ""),
        "biomnibench_task_type": gold.get("biomnibench_task_type", ""),
        "biomnibench_category": gold.get("biomnibench_category", ""),
        "biomnibench_difficulty": gold.get("biomnibench_difficulty", ""),
    }
    (logs_dir / "evaluation.json").write_text(
        json.dumps(evaluation_data, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
'''


def patched_test_sh() -> str:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        "mkdir -p /logs/verifier\n"
        "# Copy agent outputs into the verifier dir for evaluation (upstream behavior).\n"
        "cp /app/trace.md /logs/verifier/trace.md 2>/dev/null || echo 'Warning: trace.md not found'\n"
        "cp /app/answer.txt /logs/verifier/answer.txt 2>/dev/null || echo 'Warning: answer.txt not found'\n"
        "# openai is pre-installed in biomnibench-da-runtime; pip fallback for copy-mode Dockerfiles\n"
        "if ! python3 -c 'import openai' 2>/dev/null; then\n"
        "  python3 -m pip install --break-system-packages -q openai\n"
        "fi\n"
        "python3 /tests/llm_judge.py\n"
    )


# --------------------------------------------------------------------------- #
# Task materialization
# --------------------------------------------------------------------------- #
def rewrite_task_toml(
    parsed: dict[str, Any],
    task_meta: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append('version = "1.0"')
    lines.append("")

    meta = parsed.get("metadata", {})
    lines.append("[metadata]")
    for k, v in meta.items():
        lines.append(f"{k} = {json.dumps(str(v))}")
    lines.append(f"question_group_key = {json.dumps(task_meta['question_group_key'])}")
    lines.append(f"source_task_id = {json.dumps(task_meta['source_task_id'])}")
    lines.append(f"partition = {json.dumps(task_meta['partition'])}")
    lines.append(f"repeat_index = {task_meta['repeat_index']}")
    lines.append(f"n_repeats = {task_meta['n_repeats']}")
    lines.append(f"biomnibench_task_type = {json.dumps(task_meta['task_type'])}")
    lines.append(f"biomnibench_category = {json.dumps(task_meta['category'])}")
    lines.append(f"biomnibench_difficulty = {json.dumps(task_meta['difficulty'])}")
    lines.append("")

    agent = parsed.get("agent", {})
    lines.append("[agent]")
    lines.append(f"timeout_sec = {agent.get('timeout_sec', 3600.0)}")
    perm = agent.get("permission")
    if perm:
        if isinstance(perm, str):
            lines.append(f"permission = {json.dumps(perm)}")
        elif isinstance(perm, dict):
            lines.append(
                "permission = { " + ", ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in perm.items()) + " }"
            )
    lines.append("")

    verifier = parsed.get("verifier", {})
    lines.append("[verifier]")
    lines.append(f"timeout_sec = {verifier.get('timeout_sec', 900.0)}")
    lines.append("")
    lines.append("[verifier.env]")
    lines.append('OPENAI_API_KEY = "${JUDGE_API_KEY}"')
    lines.append('OPENAI_BASE_URL = "${JUDGE_BASE_URL}"')
    if args.judge_model:
        lines.append(f"JUDGE_MODEL = {json.dumps(args.judge_model)}")
    else:
        lines.append('JUDGE_MODEL = "${JUDGE_MODEL}"')
    lines.append("")

    env = parsed.get("environment", {})
    lines.append("[environment]")
    lines.append(f"build_timeout_sec = {env.get('build_timeout_sec', 600.0)}")
    if args.docker_image and (is_docker_bind(args) or is_singularity_copy(args)):
        lines.append(f"docker_image = {json.dumps(args.docker_image)}")
    if args.storage_mb_override:
        lines.append(f"storage_mb = {args.storage_mb_override}")
    else:
        lines.append(f"storage_mb = {env.get('storage_mb', 20480)}")
    lines.append(f"memory_mb = {env.get('memory_mb', 16384)}")
    lines.append(f"cpus = {env.get('cpus', 2)}")
    lines.append(f"gpus = {env.get('gpus', 0)}")
    lines.append(f"allow_internet = {str(env.get('allow_internet', True)).lower()}")
    lines.append("")
    lines.append("[environment.env]")
    lines.append('OPENAI_API_KEY = "${OPENAI_API_KEY}"')
    lines.append('OPENAI_BASE_URL = "${OPENAI_BASE_URL}"')
    lines.append('ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"')
    lines.append('ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"')

    return "\n".join(lines) + "\n"


def materialize_task(
    source_dir: Path,
    output_dir: Path,
    task_meta: dict[str, Any],
    parsed_toml: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_name = task_meta["task_name"]
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    src_instruction = source_dir / "instruction.md"
    if src_instruction.exists():
        shutil.copy2(src_instruction, task_dir / "instruction.md")

    env_info = materialize_environment(
        source_dir,
        task_dir / "environment",
        task_meta["source_task_id"],
        args,
    )

    toml_text = rewrite_task_toml(parsed_toml, task_meta, args)
    (task_dir / "task.toml").write_text(toml_text, encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    src_tests = source_dir / "tests"
    if src_tests.is_dir():
        for name in src_tests.iterdir():
            if name.name in {"llm_judge.py", "test.sh"}:
                continue
            dst = tests_dir / name.name
            if name.is_dir():
                shutil.copytree(name, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(name, dst)

    src_rubric = source_dir / "tests" / "rubric.txt"
    if src_rubric.exists():
        shutil.copy2(src_rubric, tests_dir / "rubric.txt")

    judge_path = tests_dir / "llm_judge.py"
    judge_path.write_text(patched_judge_source(), encoding="utf-8")
    judge_path.chmod(0o755)

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(patched_test_sh(), encoding="utf-8")
    test_sh.chmod(0o755)

    gold_meta = {
        "task_name": task_name,
        "source_task_id": task_meta["source_task_id"],
        "question_group_key": task_meta["question_group_key"],
        "partition": task_meta["partition"],
        "repeat_index": task_meta["repeat_index"],
        "n_repeats": task_meta["n_repeats"],
        "biomnibench_task_type": task_meta["task_type"],
        "biomnibench_category": task_meta["category"],
        "biomnibench_difficulty": task_meta["difficulty"],
    }
    write_json(tests_dir / "gold_metadata.json", gold_meta)

    return {
        "name": task_name,
        "path": task_name,
        "source_task_id": task_meta["source_task_id"],
        "question_group_key": task_meta["question_group_key"],
        "partition": task_meta["partition"],
        "repeat_index": task_meta["repeat_index"],
        "n_repeats": task_meta["n_repeats"],
        "biomnibench_task_type": task_meta["task_type"],
        "biomnibench_category": task_meta["category"],
        "biomnibench_difficulty": task_meta["difficulty"],
        **env_info,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    validate_options(args)
    if args.n_repeats < 1:
        raise SystemExit("--n-repeats must be >= 1.")

    local_dir_has_tasks = args.local_dir.is_dir() and any(args.local_dir.glob("da-*"))
    download_patterns = requested_download_patterns(args)
    if args.download and (download_patterns or not local_dir_has_tasks):
        download_dataset(args.hf_repo_id, args.local_dir, download_patterns)
    elif not args.local_dir.is_dir():
        raise SystemExit(f"--local-dir does not exist: {args.local_dir} (pass --download to fetch it).")

    if args.build_docker_image:
        build_docker_image(args.docker_image)

    prepare_output_dir(args.output_dir, args.overwrite)

    all_tasks = discover_tasks(args.local_dir)
    print(f"Discovered {len(all_tasks)} tasks across {len(set(t['paper'] for t in all_tasks.values()))} papers")

    assignments = assign_partitions(all_tasks, args)
    filtered = filter_tasks(all_tasks, assignments, args)
    selected_ids = select_tasks(filtered, assignments, args)

    print(f"Selected {len(selected_ids)} tasks for partition={args.partition}")

    registry_tasks: list[dict[str, Any]] = []
    answers_rows: list[dict[str, Any]] = []

    for task_id in selected_ids:
        info = filtered[task_id]
        for repeat_idx in range(1, args.n_repeats + 1):
            task_meta = {
                "task_name": f"{task_id}-r{repeat_idx:03d}",
                "source_task_id": task_id,
                "question_group_key": task_id,
                "partition": assignments[task_id],
                "repeat_index": repeat_idx,
                "n_repeats": args.n_repeats,
                "task_type": info["task_type"],
                "category": info["category"],
                "difficulty": info["difficulty"],
            }
            reg = materialize_task(info["path"], args.output_dir, task_meta, info["parsed_toml"], args)
            registry_tasks.append(reg)
            answers_rows.append(task_meta)

    registry = [
        {
            "name": args.dataset_name,
            "version": "1.0",
            "description": "BiomniBench-DA materialized as Harbor tasks (upstream-faithful).",
            "metrics": [{"type": "mean"}],
            "tasks": [{"name": t["name"], "path": t["path"]} for t in registry_tasks],
        }
    ]
    write_json(args.output_dir / "registry.json", registry)
    write_jsonl(args.output_dir / "answers.jsonl", answers_rows)

    rollout_input_fpath = args.rollout_input_fpath or (args.output_dir / DEFAULT_ROLLOUT_INPUT_NAME)
    rollout_rows = build_rollout_input_rows(
        registry_tasks,
        dataset_name=args.dataset_name,
        agent_name=args.agent_name,
    )
    write_jsonl(rollout_input_fpath, rollout_rows)

    manifest_rows = []
    for task_id in sorted(all_tasks.keys(), key=paper_sort_key):
        info = all_tasks[task_id]
        manifest_rows.append(
            {
                "task_id": task_id,
                "paper": info["paper"],
                "partition": assignments.get(task_id, "test"),
                "task_type": info["task_type"],
                "category": info["category"],
                "difficulty": info["difficulty"],
                "excluded": task_id not in filtered,
            }
        )
    split_manifest = {
        "schema_version": "biomnibench_da_split_manifest.v1",
        "dataset_id": DATASET_ID,
        "split_seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "stratify_by": args.stratify_by,
        "task_assignments": manifest_rows,
        "counts": dict(sorted(Counter(assignments[t] for t in filtered).items())),
        "excluded_count": len(all_tasks) - len(filtered),
    }
    write_json(args.output_dir / "split_manifest.json", split_manifest)

    manifest = {
        "schema_version": "biomnibench_da_materialization_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "environments/biomnibench_da/prepare.py",
        "generation_command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "variant": "upstream_faithful",
        "storage": {
            "environment_type": args.environment_type,
            "container_data_dir": DEFAULT_CONTAINER_DATA_DIR,
            "data_mount_root": str((args.data_mount_root or args.local_dir).resolve()),
            "docker_image": args.docker_image,
            "dockerfile": DEFAULT_DOCKERFILE if is_docker_bind(args) else None,
        },
        "dataset": {
            "id": DATASET_ID,
            "tasks_discovered": len(all_tasks),
            "tasks_after_filter": len(filtered),
            "tasks_selected": len(selected_ids),
            "n_repeats": args.n_repeats,
            "harbor_tasks": len(registry_tasks),
        },
        "counts": {
            "by_partition": dict(sorted(Counter(t["partition"] for t in registry_tasks).items())),
            "by_task_type": dict(sorted(Counter(t["biomnibench_task_type"] for t in registry_tasks).items())),
            "by_category": dict(sorted(Counter(t["biomnibench_category"] for t in registry_tasks).items())),
            "by_difficulty": dict(sorted(Counter(t["biomnibench_difficulty"] for t in registry_tasks).items())),
        },
        "tasks": registry_tasks,
        "rollout_input": {
            "path": str(rollout_input_fpath),
            "agent_name": args.agent_name,
            "rows": len(rollout_rows),
        },
    }
    write_json(args.output_dir / "materialization_manifest.json", manifest)

    print(f"\nWrote {len(registry_tasks)} Harbor tasks to {args.output_dir}")
    print(f"Tasks selected: {len(selected_ids)}; repeats: {args.n_repeats}")
    print(f"Partition counts: {manifest['counts']['by_partition']}")
    print(f"Task type counts: {manifest['counts']['by_task_type']}")
    print(f"Rollout input: {rollout_input_fpath} ({len(rollout_rows)} rows)")
    print("Judge: upstream logic, OpenAI-compatible endpoint only")
    if is_docker_bind(args):
        print(f"Data: bind mount -> {DEFAULT_CONTAINER_DATA_DIR} via docker-compose.yaml")
        print(f"Runtime image: {args.docker_image}")
    else:
        print("Data: copied to environment/files/data with setup.sh staging for Singularity")
        print(f"Runtime image: {args.docker_image}")


if __name__ == "__main__":
    main()
