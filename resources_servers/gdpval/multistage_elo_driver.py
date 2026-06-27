# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Driver that wires the multi-stage ELO logic to the GDPVal comparison server.

This composes the pure staging logic in ``multistage_elo`` with the GDPVal
resources server's ``/verify`` (comparison mode). For each stage it:

1. asks the runner to select the stage's references (closest known ELO to the
   current estimate) and fix the stage's sampled tasks,
2. judges the evaluated model's cached deliverables against that reference
   subset, one ``/verify`` call per (task, repeat) with the per-request
   ``reference_ids`` filter,
3. pools the per-reference win/loss/tie votes and fits the stage ELO.

The evaluated model's deliverables are read from a directory laid out as
``<eval_deliverables_dir>/task_<id>/repeat_<n>/`` (the same layout the Stirrup
agent persists). Point ``eval_deliverables_dir`` at deliverables produced by an
earlier run to score them with **zero rollouts**. Tasks missing from the cache
are either produced on demand via an injected ``producer`` callback or reported,
controlled by ``produce_missing``.

The judging primitive ``verify_one`` is injected so the orchestration is
testable without a running server; ``make_http_verify_one`` provides the real
implementation that POSTs to the resources server.

CLI usage (run from the repo root, against a running comparison server)::

    python -m resources_servers.gdpval.multistage_elo_driver \\
        --server-url http://localhost:8000 \\
        --eval-deliverables-dir /path/to/eval/deliverables \\
        --reference-elos '@refs.json' \\
        --stage 5 --stage 88:4 \\
        --output elo_summary.json

where ``refs.json`` is ``{"<ref_id>": <elo>, ...}`` with ids matching the
server's configured ``reference_models``. Each stage has a set number of 
tasks and reference models set like ``--stage num_tasks:num_models``. 
See ``--help`` for all flags.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from resources_servers.gdpval.comparison import task_attempted
from resources_servers.gdpval.multistage_elo import (
    MultiStageEloConfig,
    MultiStageEloRunner,
    PerReferenceTotals,
    StageResult,
    StageSpec,
)


# verify_one(task_id, deliverables_dir, prompt, reference_ids) -> verify response dict
VerifyOneFn = Callable[[str, str, str, Sequence[str]], Dict[str, Any]]
# producer(task_ids) -> None: materialize eval deliverables for the given tasks.
ProducerFn = Callable[[Sequence[str]], None]


# ---------------------------------------------------------------------------
# Dataset / distribution loading
# ---------------------------------------------------------------------------


# Default location for distributions this driver builds on demand. Lives under
# the resources server's data dir so it is reachable from wherever the driver
# runs and is easy to inspect/reuse across runs.
DEFAULT_DISTRIBUTION_CACHE_DIR = Path(__file__).resolve().parent / "data" / "distributions"


def load_distribution(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a task-distribution JSON file produced by ``task_distribution.py``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Distribution file {path} must be a JSON object.")
    return data


def ensure_distribution(
    distribution_path: Optional[str | Path] = None,
    *,
    dataset_path: Optional[str | Path] = None,
    columns: Optional[Sequence[str]] = None,
    cache_dir: Optional[str | Path] = None,
) -> tuple[Dict[str, Dict[str, Any]], Path]:
    """Return ``(distribution, path)``, building the distribution if needed.

    If ``distribution_path`` exists it is loaded as-is. Otherwise a distribution
    is built from ``dataset_path`` (or the default GDPVal dataset) grouped by
    ``columns`` (default ``["occupation"]``) via ``task_distribution``, then saved
    so subsequent runs reuse it. It is written to ``distribution_path`` when
    given, else to ``<cache_dir>/<columns>_distribution.json`` (cache_dir
    defaults to ``DEFAULT_DISTRIBUTION_CACHE_DIR``).
    """
    column_list = list(columns) if columns else ["occupation"]

    if distribution_path is not None and Path(distribution_path).is_file():
        return load_distribution(distribution_path), Path(distribution_path)

    from responses_api_agents.stirrup_agent.task_distribution import (
        build_distribution_from_dataset,
        resolve_default_dataset,
    )

    resolved_dataset = Path(dataset_path) if dataset_path is not None else resolve_default_dataset()
    if resolved_dataset is None:
        raise FileNotFoundError(
            "No distribution file was provided and no default GDPVal dataset could be found to "
            "build one from. Provide distribution_path, pass dataset_path, or prepare the GDPVal "
            "dataset (gym eval prepare --benchmark gdpval)."
        )

    distribution = build_distribution_from_dataset(resolved_dataset, column_list)

    if distribution_path is not None:
        out_path = Path(distribution_path)
    else:
        base = Path(cache_dir) if cache_dir is not None else DEFAULT_DISTRIBUTION_CACHE_DIR
        out_path = base / f"{'_'.join(column_list)}_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(distribution, handle, indent=2, ensure_ascii=False)
    print(
        f"[multistage-elo] built task distribution over {column_list} from {resolved_dataset} -> {out_path}",
        flush=True,
    )
    return distribution, out_path


def load_task_prompts(jsonl_path: str | Path) -> Dict[str, str]:
    """Map ``task_id -> prompt`` from a benchmark JSONL.

    The prompt is needed when judging cached deliverables (the judge sees the
    task description). Looks for ``prompt`` and ``task_id`` at the top level and,
    failing that, under ``responses_create_params.metadata`` — covering both the
    prepared benchmark layout and the metadata-nested layout.
    """
    prompts: Dict[str, str] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = (row.get("responses_create_params") or {}).get("metadata") or {}
            task_id = row.get("task_id") or meta.get("task_id")
            prompt = row.get("prompt") or meta.get("prompt")
            if task_id is not None:
                prompts[str(task_id)] = prompt or ""

    return prompts


# ---------------------------------------------------------------------------
# Cached-deliverable discovery
# ---------------------------------------------------------------------------


def task_repeat_dirs(eval_deliverables_dir: str | Path, task_id: str) -> List[Path]:
    """Return attempted ``repeat_<n>`` dirs (or a flat task dir) for a task.

    Mirrors the resources server's reference-repeat resolution: prefers
    ``task_<id>/repeat_<n>/`` subdirs, falls back to a flat ``task_<id>/``, and
    only returns dirs that look like a completed run (``finish_params.json``).
    """
    task_root = Path(eval_deliverables_dir) / f"task_{task_id}"
    if not task_root.is_dir():
        return []
    repeats = sorted(p for p in task_root.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    candidates = repeats or [task_root]
    return [d for d in candidates if task_attempted(str(d))]


def cached_task_ids(eval_deliverables_dir: str | Path) -> set:
    """All task ids that have at least one attempted deliverable in the cache."""
    root = Path(eval_deliverables_dir)
    if not root.is_dir():
        return set()
    found = set()
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("task_"):
            task_id = child.name[len("task_") :]
            if task_repeat_dirs(eval_deliverables_dir, task_id):
                found.add(task_id)
    return found


def check_coverage(eval_deliverables_dir: str | Path, task_ids: Sequence[str]) -> tuple[List[str], List[str]]:
    """Split ``task_ids`` into ``(present, missing)`` against the cache."""
    present, missing = [], []
    for tid in task_ids:
        (present if task_repeat_dirs(eval_deliverables_dir, tid) else missing).append(tid)

    return present, missing


# ---------------------------------------------------------------------------
# Vote pooling
# ---------------------------------------------------------------------------


def pool_per_reference(verify_responses: Sequence[Mapping[str, Any]]) -> PerReferenceTotals:
    """Sum ``per_reference`` win/loss/tie counts across many verify responses."""
    totals: PerReferenceTotals = {}
    for vr in verify_responses:
        per_ref = vr.get("per_reference") or {}
        for ref_id, counts in per_ref.items():
            entry = totals.setdefault(ref_id, {"wins": 0, "losses": 0, "ties": 0, "reference_elo": None})
            entry["wins"] += int(counts.get("wins", 0) or 0)
            entry["losses"] += int(counts.get("losses", 0) or 0)
            entry["ties"] += int(counts.get("ties", 0) or 0)
            if entry["reference_elo"] is None:
                entry["reference_elo"] = counts.get("reference_elo")

    return totals


# ---------------------------------------------------------------------------
# judge_stage builder
# ---------------------------------------------------------------------------


def build_judge_stage(
    verify_one: VerifyOneFn,
    eval_deliverables_dir: str | Path,
    task_prompts: Mapping[str, str],
    *,
    produce_missing: bool = True,
    producer: Optional[ProducerFn] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
):
    """Build the ``judge_stage`` callable expected by ``MultiStageEloRunner``.

    For each stage's tasks, judges the cached eval deliverables against the
    selected references (one ``verify_one`` call per task-repeat) and pools the
    per-reference votes. Missing tasks are produced via ``producer`` when given;
    otherwise ``produce_missing=True`` raises an actionable error and
    ``produce_missing=False`` drops them with a warning.

    ``progress`` is an optional callback invoked as ``progress(done, total,
    task_id)`` after each ``verify_one`` completes, for live status reporting.
    """

    def judge_stage(task_ids: Sequence[str], reference_ids: Sequence[str]) -> PerReferenceTotals:
        present, missing = check_coverage(eval_deliverables_dir, task_ids)
        if missing:
            if producer is not None:
                producer(missing)
                present, missing = check_coverage(eval_deliverables_dir, task_ids)
            if missing and produce_missing and producer is None:
                raise FileNotFoundError(
                    f"{len(missing)} task(s) have no cached eval deliverable under "
                    f"{eval_deliverables_dir} (e.g. {missing[:3]}). Produce them first with an "
                    f"execute_only run, pass a producer, or set produce_missing=False to skip them."
                )
            if missing:
                print(
                    f"[multistage-elo] WARNING: skipping {len(missing)} task(s) with no cached "
                    f"deliverable (e.g. {missing[:3]})",
                    flush=True,
                )

        # Flatten to (task_id, repeat_dir) units up front so progress can report
        # an accurate done/total across all repeats in the stage.
        units = [(tid, repeat_dir) for tid in present for repeat_dir in task_repeat_dirs(eval_deliverables_dir, tid)]
        total = len(units)
        responses: List[Dict[str, Any]] = []
        for done, (task_id, repeat_dir) in enumerate(units, start=1):
            prompt = task_prompts.get(task_id, "")
            responses.append(verify_one(task_id, str(repeat_dir), prompt, list(reference_ids)))
            if progress is not None:
                progress(done, total, task_id)
        return pool_per_reference(responses)

    return judge_stage


# ---------------------------------------------------------------------------
# Real verify_one (HTTP)
# ---------------------------------------------------------------------------


def build_verify_request_body(
    task_id: str,
    deliverables_dir: str,
    prompt: str,
    reference_ids: Sequence[str],
    *,
    model: str = "eval",
) -> Dict[str, Any]:
    """Build a minimal comparison-mode ``/verify`` request body.

    In comparison mode the judge reads deliverable files from ``deliverables_dir``
    rather than the response payload, so a placeholder response is sufficient.
    """
    return {
        "responses_create_params": {"input": [], "model": model},
        "response": {
            "id": f"multistage-{task_id}",
            "created_at": 0,
            "model": model,
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        "task_id": task_id,
        "prompt": prompt,
        "deliverables_dir": deliverables_dir,
        "reference_ids": list(reference_ids),
    }


def make_http_verify_one(server_url: str, *, timeout: float = 1800.0, model: str = "eval") -> VerifyOneFn:
    """Return a blocking ``verify_one`` that POSTs to a running resources server.

    ``server_url`` is the resources server base URL (e.g. ``http://host:port``);
    ``/verify`` is appended. Uses stdlib ``urllib`` so the driver pulls in no
    async machinery — it is a standalone orchestration script, not part of the
    server hot path.
    """
    import urllib.request

    endpoint = server_url.rstrip("/") + "/verify"

    def verify_one(task_id: str, deliverables_dir: str, prompt: str, reference_ids: Sequence[str]) -> Dict[str, Any]:
        body = build_verify_request_body(task_id, deliverables_dir, prompt, reference_ids, model=model)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return verify_one


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_multistage_elo(
    config: MultiStageEloConfig,
    verify_one: VerifyOneFn,
    task_prompts: Mapping[str, str],
    *,
    rng=None,
    producer: Optional[ProducerFn] = None,
    on_event: Optional[Callable[[str, dict], None]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[StageResult]:
    """Run the full multi-stage ELO procedure and return per-stage results.

    ``config.eval_deliverables_dir`` must be set — it is the source of the eval
    model's (cached or produced) deliverables.

    ``on_event``/``progress`` are optional callbacks for live status reporting:
    ``on_event`` receives stage-level events (see ``MultiStageEloRunner``) and
    ``progress`` receives per-(task, repeat) judging progress.
    """
    if not config.eval_deliverables_dir:
        raise ValueError("config.eval_deliverables_dir must be set (source of eval deliverables).")

    distribution, _ = ensure_distribution(
        config.distribution_path,
        dataset_path=config.dataset_path,
        columns=config.column,
    )
    judge_stage = build_judge_stage(
        verify_one,
        config.eval_deliverables_dir,
        task_prompts,
        produce_missing=config.produce_missing,
        producer=producer,
        progress=progress,
    )
    runner = MultiStageEloRunner(config, distribution, judge_stage, rng=rng, on_event=on_event)
    return runner.run()


def stage_results_to_dict(results: Sequence[StageResult]) -> Dict[str, Any]:
    """Serialize stage results to a JSON-friendly summary dict."""
    final = results[-1] if results else None
    return {
        "final_eval_elo": final.eval_elo if final else None,
        "final_normalized_elo": final.normalized_elo if final else None,
        "num_stages": len(results),
        "stages": [
            {
                "stage_index": r.stage_index,
                "num_tasks": len(r.task_ids),
                "reference_ids": r.reference_ids,
                "eval_elo": r.eval_elo,
                "normalized_elo": r.normalized_elo,
                "num_references": r.num_references,
                "per_reference": r.per_reference,
                "task_ids": r.task_ids,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_TASK_PROMPTS = "benchmarks/gdpval/data/gdpval_benchmark.jsonl"


def _parse_stage(spec: str) -> StageSpec:
    """Parse a ``--stage`` value ``num_tasks[:num_models[:seed]]`` into a StageSpec.

    ``num_models`` may be ``all`` or empty for "all available references". Examples:
    ``5`` (5 tasks, all refs), ``88:4`` (88 tasks, 4 closest refs), ``5:all:7``
    (5 tasks, all refs, seed 7).
    """
    parts = spec.split(":")
    if not parts or not parts[0].strip():
        raise argparse.ArgumentTypeError(f"Invalid --stage {spec!r}: num_tasks is required.")
    try:
        num_tasks = int(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid --stage {spec!r}: num_tasks must be an integer.")

    num_models: Optional[int] = None
    if len(parts) >= 2 and parts[1].strip() and parts[1].strip().lower() != "all":
        try:
            num_models = int(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid --stage {spec!r}: num_models must be an integer or 'all'.")

    seed: Optional[int] = None
    if len(parts) >= 3 and parts[2].strip():
        try:
            seed = int(parts[2])
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid --stage {spec!r}: seed must be an integer.")

    return StageSpec(num_tasks=num_tasks, num_models=num_models, seed=seed)


def _load_reference_elos(value: str) -> Dict[str, float]:
    """Load reference ELOs from inline JSON or, if prefixed with ``@``, a JSON file.

    Accepts ``{"ref_id": elo, ...}``. The ids must match the running server's
    ``reference_models`` ids.
    """
    text = value
    if value.startswith("@"):
        text = Path(value[1:]).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict) or not data:
        raise argparse.ArgumentTypeError("--reference-elos must be a non-empty JSON object of {ref_id: elo}.")
    return {str(k): float(v) for k, v in data.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multistage_elo",
        description=(
            "Run multi-stage adaptive ELO estimation for a model's GDPVal deliverables "
            "against a running GDPVal comparison server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m resources_servers.gdpval.multistage_elo_driver \\\n"
            "    --server-url http://localhost:8000 \\\n"
            "    --eval-deliverables-dir /path/to/eval/deliverables \\\n"
            "    --reference-elos '@refs.json' \\\n"
            "    --stage 5 --stage 88:4 \\\n"
            "    --output elo_summary.json\n"
        ),
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="Base URL of the running GDPVal comparison-mode resources server (e.g. http://localhost:8000).",
    )
    parser.add_argument(
        "--eval-deliverables-dir",
        required=True,
        help="Directory of the evaluated model's deliverables (task_<id>/repeat_<n>/ layout).",
    )
    parser.add_argument(
        "--reference-elos",
        required=True,
        type=_load_reference_elos,
        metavar="JSON",
        help=(
            "Reference anchor ELOs as inline JSON ('{\"ref\": 1500, ...}') or '@path.json'. "
            "Keys must match the server's reference_models ids."
        ),
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        required=True,
        type=_parse_stage,
        metavar="N[:M[:SEED]]",
        help=(
            "A stage as num_tasks[:num_models[:seed]] (num_models 'all' or omitted = all references). "
            "Repeat for multiple stages, e.g. --stage 5 --stage 88:4."
        ),
    )
    parser.add_argument(
        "--task-prompts",
        default=DEFAULT_TASK_PROMPTS,
        help=f"Benchmark JSONL mapping task_id -> prompt (default: {DEFAULT_TASK_PROMPTS}).",
    )
    parser.add_argument(
        "--distribution",
        default=None,
        help="Existing task-distribution JSON to sample tasks from. If omitted, one is built and cached.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset JSONL to build the distribution from when --distribution is not given (default: GDPVal).",
    )
    parser.add_argument(
        "--column",
        dest="columns",
        action="append",
        default=None,
        metavar="COLUMN",
        help="Column(s) to group the distribution by when building one (default: occupation). Repeatable.",
    )
    parser.add_argument(
        "--nested-tasks",
        action="store_true",
        help="Make each stage's task set a superset of the previous (default: independent per-stage sampling).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop tasks with no cached eval deliverable instead of erroring (sets produce_missing=False).",
    )
    parser.add_argument(
        "--model",
        default="eval",
        help="Label for the evaluated model in verify requests (default: eval).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Per-request /verify timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Top-level RNG seed for reproducible task sampling and reference selection.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress live per-stage / per-task progress output on stderr.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to write the JSON ELO summary. Defaults to stdout.",
    )
    return parser


def _make_progress_printers():
    """Return ``(on_event, progress)`` callbacks that print human-readable status to stderr.

    ``on_event`` prints a banner at the start/end of each stage (selected
    references, task count, fitted ELO); ``progress`` prints a per-(task, repeat)
    counter as each ``/verify`` completes.
    """

    def on_event(name: str, data: dict) -> None:
        if name == "planned":
            counts = data.get("stage_task_counts", [])
            print(
                f"[multistage-elo] planned {data.get('total_stages')} stage(s); tasks per stage: {counts}",
                file=sys.stderr,
                flush=True,
            )
        elif name == "stage_start":
            idx = int(data["index"]) + 1
            total = data["total_stages"]
            refs = data.get("reference_ids", [])
            prior = data.get("prior_elo")
            prior_str = f"{prior:.1f}" if isinstance(prior, (int, float)) else "n/a"
            print(
                f"[multistage-elo] stage {idx}/{total}: {data.get('num_tasks')} task(s) "
                f"vs {len(refs)} ref(s) {refs} (prior ELO: {prior_str})",
                file=sys.stderr,
                flush=True,
            )
        elif name == "stage_end":
            idx = int(data["index"]) + 1
            total = data["total_stages"]
            elo = data.get("eval_elo")
            elo_str = f"{elo:.1f}" if isinstance(elo, (int, float)) else "unset (no games)"
            print(
                f"[multistage-elo] stage {idx}/{total} done: eval ELO = {elo_str} "
                f"(fit over {data.get('num_references')} ref(s))",
                file=sys.stderr,
                flush=True,
            )

    def progress(done: int, total: int, task_id: str) -> None:
        short = task_id[:18] + "…" if len(task_id) > 19 else task_id
        end = "\n" if done == total else "\r"
        print(f"[multistage-elo]   judged {done}/{total} (task {short})   ", end=end, file=sys.stderr, flush=True)

    return on_event, progress


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_deliverables_dir)
    if not eval_dir.is_dir():
        print(f"Eval deliverables dir not found: {eval_dir}", file=sys.stderr)
        return 2

    prompts_path = Path(args.task_prompts)
    if not prompts_path.is_file():
        print(f"Task prompts JSONL not found: {prompts_path}", file=sys.stderr)
        return 2

    config = MultiStageEloConfig(
        stages=list(args.stages),
        reference_elos=args.reference_elos,
        distribution_path=args.distribution,
        dataset_path=args.dataset,
        eval_deliverables_dir=str(eval_dir),
        produce_missing=not args.skip_missing,
        nested_tasks=args.nested_tasks,
        column=list(args.columns) if args.columns else ["occupation"],
    )

    verify_one = make_http_verify_one(args.server_url, timeout=args.timeout, model=args.model)
    task_prompts = load_task_prompts(prompts_path)
    rng = random.Random(args.seed) if args.seed is not None else None

    on_event, progress = (None, None) if args.quiet else _make_progress_printers()
    results = run_multistage_elo(config, verify_one, task_prompts, rng=rng, on_event=on_event, progress=progress)
    payload = json.dumps(stage_results_to_dict(results), indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        final = results[-1] if results else None
        final_elo = final.eval_elo if final else None
        print(f"Wrote ELO summary ({len(results)} stages, final_eval_elo={final_elo}) to {out_path}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
