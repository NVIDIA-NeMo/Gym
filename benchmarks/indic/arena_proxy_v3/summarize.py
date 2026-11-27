# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format the resources server's Indic Arena metrics from saved judgments."""

import argparse
import json
from pathlib import Path

from nemo_gym.rollouts import RolloutStore
from nemo_gym.utils import failures_path_for
from resources_servers.arena_judge.app import ArenaJudgeConfig, ArenaJudgeServer
from resources_servers.arena_judge.indic_metrics import INDIC_LANGUAGES, _read_jsonl, compute_indic_metrics


DEFAULT_INPUTS = Path(__file__).parent / "data/arena_proxy_v3_benchmark.jsonl"


def summarize(
    path: Path,
    *,
    inputs: Path = DEFAULT_INPUTS,
    languages: list[str] | None = None,
    reference_verdicts: list[Path] | None = None,
) -> dict:
    """Read selected Gym outcomes and format the same metrics as the resources server."""
    store = RolloutStore.read(path, import_legacy=False)
    if store is not None:
        results = store.selected("success")
        failures = store.selected("failure") + store.selected("omitted")
    else:
        results = _read_jsonl(path)
        sidecar = failures_path_for(path)
        failures = _read_jsonl(sidecar) if sidecar.exists() else []
        if any("_ng_run_id" in row for row in (*results, *failures)):
            raise ValueError("Run-tagged outcomes require their Gym manifest; restore it before summarizing")
    scorer = ArenaJudgeServer.model_construct(
        config=ArenaJudgeConfig.model_construct(arena_elo_bootstrap_rounds=100, arena_elo_bootstrap_seed=42)
    )
    return compute_indic_metrics(
        results,
        inputs=inputs,
        aggregate_scores=scorer._aggregate_arena_elo,
        failures=failures,
        reconcile_failures=store is None,
        languages=languages,
        reference_verdicts=reference_verdicts,
    )


def main() -> int:
    """Write strict JSON metrics and exit nonzero for incomplete selected coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument(
        "--inputs", type=Path, default=DEFAULT_INPUTS, help="Prepared benchmark JSONL defining coverage"
    )
    parser.add_argument("--languages", nargs="+", choices=("en", *INDIC_LANGUAGES))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-verdicts",
        type=Path,
        nargs="+",
        help="Original per-language judgment JSONLs; recompute reference CIs in prepared input order for comparison",
    )
    args = parser.parse_args()
    metrics = summarize(
        args.rollouts, inputs=args.inputs, languages=args.languages, reference_verdicts=args.reference_verdicts
    )
    report = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if metrics["_coverage"]["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
