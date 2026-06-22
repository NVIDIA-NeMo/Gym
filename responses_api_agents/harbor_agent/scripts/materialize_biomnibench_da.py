#!/usr/bin/env python3
"""Materialize BiomniBench-DA as Harbor tasks for NeMo Gym harbor_agent.

Upstream-faithful task content with an OpenAI-compatible LLM judge. Supports two
deployment profiles via ``--environment-type``:

- ``docker``: bind-mount source task data at ``/app/data`` via docker-compose.yaml
- ``singularity``: copy data into ``environment/files/data`` + ``setup.sh`` staging

Examples::

  python responses_api_agents/harbor_agent/scripts/materialize_biomnibench_da.py \\
    --local-dir responses_api_agents/harbor_agent/data/biomnibench_da/source \\
    --environment-type docker \\
    --tasks da-1-3 da-1-4 \\
    --output-dir responses_api_agents/harbor_agent/data/biomnibench_da/tasks_smoke_docker \\
    --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
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
HARBOR_AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKERFILE = str(HARBOR_AGENT_ROOT / "docker" / "biomnibench-da-runtime.Dockerfile")

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
        "--local-dir", type=Path, required=True, help="Local BiomniBench-DA download (from hf download)."
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
        "--data-mount-root",
        type=Path,
        default=None,
        help="Host root for docker bind mounts (default: --local-dir).",
    )
    return parser.parse_args()


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
    data_mount: tuple[Path, str] | None = None,
    skills_mount: tuple[Path, str] | None = None,
    read_only: bool = True,
) -> None:
    volume_lines: list[str] = []
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
    if not volume_lines:
        return
    text = "services:\n  main:\n    volumes:\n" + "\n".join(volume_lines) + "\n"
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
# Patched LLM judge — endpoint swap only (Gemini SDK -> OpenAI-compatible)
# --------------------------------------------------------------------------- #
def patched_judge_source() -> str:
    return r'''#!/usr/bin/env python3
"""BiomniBench-DA LLM judge — OpenAI-compatible endpoint.

Drop-in replacement for the upstream Gemini-based llm_judge.py.
Reads the rubric, agent trajectory, and instruction, then calls an
OpenAI-compatible chat completions endpoint to score the trajectory.

Environment variables (set in [verifier.env] of task.toml):
  OPENAI_API_KEY   — API key for the judge endpoint
  OPENAI_BASE_URL  — base URL (e.g. https://inference-api.nvidia.com/v1)
  JUDGE_MODEL      — model name (e.g. openai/openai/gpt-5.5)
"""
from __future__ import annotations

import json
import os
import re
import sys
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


def truncate(text: str, limit: int = 60000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 30].rstrip() + "\n...[truncated]"


def format_trajectory(trajectory: dict[str, Any]) -> str:
    steps = trajectory.get("steps") or []
    parts: list[str] = []
    for i, step in enumerate(steps):
        source = step.get("source", "unknown")
        msg = str(step.get("message") or "").strip()
        tool_calls = step.get("tool_calls") or []
        if msg:
            parts.append(f"[Step {i+1} | {source}]\n{truncate(msg, 8000)}")
        for tc in tool_calls:
            name = tc.get("name") or tc.get("tool") or "unknown_tool"
            args_str = json.dumps(tc.get("arguments") or tc.get("args") or {}, ensure_ascii=False)
            result = str(tc.get("result") or tc.get("output") or "")
            parts.append(f"[Tool call: {name}]\nArgs: {truncate(args_str, 2000)}\nResult: {truncate(result, 4000)}")
    return "\n\n".join(parts) if parts else "(no trajectory steps found)"


def extract_answer_file_content() -> str:
    for candidate in [Path("/app/answer.txt"), Path("/app/output.txt"), Path("/app/results.txt")]:
        if candidate.exists():
            return read_text(candidate)
    return ""


def call_judge(rubric: str, instruction: str, trajectory_text: str, answer: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Missing dependency `openai`. Install with: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("JUDGE_MODEL", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    if not model:
        raise SystemExit("JUDGE_MODEL not set.")

    client = OpenAI(api_key=api_key, base_url=base_url or None)

    system_prompt = """You are an expert biomedical research evaluator. You will be given:
1. A research TASK (the instruction given to an AI agent)
2. A RUBRIC with specific criteria for evaluating the agent's work
3. The agent's TRAJECTORY (its step-by-step analytical process)
4. The agent's FINAL ANSWER (if any)

Score the agent's work against each rubric criterion. Then provide an overall score from 0.0 to 1.0.

You MUST respond with valid JSON in exactly this format:
{
  "dimension_scores": {
    "<dimension_name>": {"score": <float 0-1>, "rationale": "<brief explanation>"}
  },
  "overall_score": <float 0-1>,
  "overall_rationale": "<summary of strengths and weaknesses>"
}"""

    user_prompt = f"""## TASK

{truncate(instruction, 8000)}

## RUBRIC

{truncate(rubric, 16000)}

## AGENT TRAJECTORY

{truncate(trajectory_text, 40000)}

## FINAL ANSWER

{truncate(answer, 8000)}

Score the agent's analytical trajectory against the rubric. Respond with JSON only."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
    )

    content = response.choices[0].message.content or ""
    json_match = re.search(r"\{.*\}", content, re.S)
    if not json_match:
        return {
            "overall_score": 0.0,
            "overall_rationale": f"Could not parse judge response as JSON: {content[:500]}",
            "dimension_scores": {},
            "raw_response": content[:2000],
        }
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {
            "overall_score": 0.0,
            "overall_rationale": f"Invalid JSON in judge response: {content[:500]}",
            "dimension_scores": {},
            "raw_response": content[:2000],
        }
    if "overall_score" not in parsed:
        scores = [
            v.get("score", 0.0) if isinstance(v, dict) else 0.0
            for v in (parsed.get("dimension_scores") or {}).values()
        ]
        parsed["overall_score"] = sum(scores) / len(scores) if scores else 0.0
    return parsed


def process_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    steps = trajectory.get("steps") or []
    names: list[str] = []
    for step in steps:
        for tc in step.get("tool_calls") or []:
            name = tc.get("name") or tc.get("tool") or ""
            if name and name not in {"text", "message", "assistant", "user"}:
                names.append(name)
    breakdown: dict[str, int] = {}
    for name in names:
        breakdown[name] = breakdown.get(name, 0) + 1
    return {
        "trajectory_available": True,
        "tool_call_count": sum(breakdown.values()),
        "mcp_tool_call_count": sum(
            v for k, v in breakdown.items() if k.startswith("mcp__") or k.startswith("tooluniverse_")
        ),
        "tool_call_breakdown": dict(sorted(breakdown.items())),
    }


def main() -> None:
    logs_dir = Path("/logs/verifier")
    logs_dir.mkdir(parents=True, exist_ok=True)

    rubric = read_text(Path("/tests/rubric.txt"))
    instruction = read_text(Path("/app/instruction.md"))
    if not instruction:
        instruction = read_text(Path("/tests/instruction.md"))

    trajectory_path = Path("/logs/agent/trajectory.json")
    trajectory: dict[str, Any] = {}
    if trajectory_path.exists():
        trajectory = read_json(trajectory_path)
    trajectory_text = format_trajectory(trajectory)

    answer = extract_answer_file_content()

    gold = read_json(Path("/tests/gold_metadata.json"))

    score = 0.0
    judge_result = "incorrect"
    judge_rationale = ""
    judge_error = None
    dimension_scores: dict[str, Any] = {}

    if not rubric:
        judge_rationale = "No rubric found at /tests/rubric.txt."
        judge_error = "missing_rubric"
    else:
        try:
            result = call_judge(rubric, instruction, trajectory_text, answer)
            score = float(result.get("overall_score", 0.0))
            score = max(0.0, min(1.0, score))
            judge_rationale = str(result.get("overall_rationale", ""))
            dimension_scores = result.get("dimension_scores", {})
        except Exception as exc:
            judge_error = str(exc)
            judge_rationale = f"Judge call failed: {exc}"
            traceback.print_exc()

    if score >= 0.8:
        judge_result = "correct"
    elif score >= 0.4:
        judge_result = "partial"
    else:
        judge_result = "incorrect"

    reward_data = {
        "schema_version": "biomnibench_da_reward.v1",
        "reward": score,
        "score": score,
        "judge_result": judge_result,
        "judge_rationale": judge_rationale,
        "judge_available": judge_error is None,
        "judge_model": os.environ.get("JUDGE_MODEL"),
        "judge_error": judge_error,
        "scoring_method": "llm_rubric_openai",
        "dimension_scores": dimension_scores,
        "answer": truncate(answer, 2000),
        "answer_source": "file" if answer else "none",
        "question_group_key": gold.get("question_group_key", ""),
        "partition": gold.get("partition", ""),
        "repeat_index": gold.get("repeat_index"),
        "n_repeats": gold.get("n_repeats"),
        "task_name": gold.get("task_name", ""),
        "source_task_id": gold.get("source_task_id", ""),
        "biomnibench_task_type": gold.get("biomnibench_task_type", ""),
        "biomnibench_category": gold.get("biomnibench_category", ""),
        "biomnibench_difficulty": gold.get("biomnibench_difficulty", ""),
        "process_metrics": process_metrics(trajectory) if trajectory else {"trajectory_available": False},
    }

    (logs_dir / "reward.json").write_text(
        json.dumps(reward_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (logs_dir / "reward.txt").write_text(f"{score}\n", encoding="utf-8")
    print(f"Reward: {score:.3f}")
    print(f"Judge result: {judge_result}")
    if judge_rationale:
        print(judge_rationale[:500])


if __name__ == "__main__":
    main()
'''


def patched_test_sh() -> str:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        "mkdir -p /logs/verifier\n"
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
    if not args.local_dir.is_dir():
        raise SystemExit(f"--local-dir does not exist: {args.local_dir}")
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
        "generator": "responses_api_agents/harbor_agent/scripts/materialize_biomnibench_da.py",
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
    }
    write_json(args.output_dir / "materialization_manifest.json", manifest)

    print(f"\nWrote {len(registry_tasks)} Harbor tasks to {args.output_dir}")
    print(f"Tasks selected: {len(selected_ids)}; repeats: {args.n_repeats}")
    print(f"Partition counts: {manifest['counts']['by_partition']}")
    print(f"Task type counts: {manifest['counts']['by_task_type']}")
    print("Judge: upstream logic, OpenAI-compatible endpoint only")
    if is_docker_bind(args):
        print(f"Data: bind mount -> {DEFAULT_CONTAINER_DATA_DIR} via docker-compose.yaml")
        print(
            f"Runtime image: {args.docker_image} (build with harbor_agent/docker/build_biomnibench_runtime_image.sh)"
        )
    else:
        print("Data: copied to environment/files/data with setup.sh staging for Singularity")
        print(f"Runtime image: {args.docker_image}")


if __name__ == "__main__":
    main()
