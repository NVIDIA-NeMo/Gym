# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command line: inspect a suite, replay its oracles in-process, build datasets, emit a contribution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .dataset import build_rows, split_rows, write_jsonl
from .suite import Episode, Suite, VerifierError


def _cmd_inspect(args: argparse.Namespace) -> int:
    suite = Suite.load(Path(args.suite_dir))
    report = {
        "suite": suite.name,
        "root": str(suite.root),
        "tasks": len(suite),
        "tools": len(suite.tools),
        "renamed_tools": sum(1 for tool in suite.tools if tool.public_name != tool.name),
        "tasks_with_oracle": sum(1 for task in suite.tasks if task.oracle_steps),
        "profile": suite.tasks[0].profile,
        "sample_tools": [tool.public_name for tool in suite.tools[:8]],
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_replay_oracle(args: argparse.Namespace) -> int:
    suite = Suite.load(Path(args.suite_dir))
    tasks = suite.tasks[: args.limit] if args.limit else suite.tasks
    strict = 0
    fractions: list[float] = []
    failures: list[dict[str, object]] = []
    started = time.time()
    for task in tasks:
        if not task.oracle_steps:
            failures.append({"task_id": task.task_id, "error": "no oracle plan in package"})
            continue
        with Episode(task) as episode:
            results = episode.replay(task.oracle_steps)
            errors = [r for r in results if isinstance(r, dict) and "error" in r]
            try:
                verdict = episode.score()
            except VerifierError as exc:
                failures.append({"task_id": task.task_id, "error": str(exc)[:300]})
                continue
        fractions.append(verdict.reward_fraction)
        strict += int(verdict.strict_pass)
        if not verdict.strict_pass or errors:
            failures.append(
                {
                    "task_id": task.task_id,
                    "reward_fraction": verdict.reward_fraction,
                    "strict_pass": verdict.strict_pass,
                    "tool_errors": len(errors),
                    "profile": verdict.profile,
                }
            )
    report = {
        "suite": suite.name,
        "tasks_replayed": len(tasks),
        "strict_passes": strict,
        "mean_reward_fraction": round(sum(fractions) / len(fractions), 4) if fractions else None,
        "seconds": round(time.time() - started, 1),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if strict == len(tasks) else 1


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    suite = Suite.load(Path(args.suite_dir))
    rows = build_rows(suite, source_url=args.source_url)
    out = Path(args.out_dir)
    train, validation = split_rows(rows, validation_fraction=args.validation_fraction)
    written = {
        "example.jsonl": write_jsonl(rows[: args.example_count], out / "example.jsonl"),
        "train.jsonl": write_jsonl(train, out / "train.jsonl"),
        "validation.jsonl": write_jsonl(validation, out / "validation.jsonl"),
    }
    manifest = {
        "suite": suite.name,
        "tasks": len(rows),
        "tools": len(suite.tools),
        "files": written,
        "license": "CC-BY-4.0",
        "source": args.source_url,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_emit_contribution(args: argparse.Namespace) -> int:
    from .emit import emit_contribution

    written = emit_contribution(
        Path(args.out_dir),
        [Path(p) for p in args.suite_dir],
        example_count=args.example_count,
        rollouts_dir=Path(args.rollouts_dir) if args.rollouts_dir else None,
        calibration_readme=Path(args.calibration_readme) if args.calibration_readme else None,
    )
    for path in written:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blobfish-nemo-gym", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="summarise a downloaded suite")
    inspect.add_argument("suite_dir")
    inspect.set_defaults(func=_cmd_inspect)

    replay = sub.add_parser("replay-oracle", help="replay every task's oracle plan in-process and score it")
    replay.add_argument("suite_dir")
    replay.add_argument("--limit", type=int, default=0)
    replay.set_defaults(func=_cmd_replay_oracle)

    build = sub.add_parser("build-dataset", help="write example/train/validation JSONL for NeMo Gym")
    build.add_argument("suite_dir")
    build.add_argument("--out-dir", required=True)
    build.add_argument("--example-count", type=int, default=5)
    build.add_argument("--validation-fraction", type=float, default=0.1)
    build.add_argument("--source-url", default=None)
    build.set_defaults(func=_cmd_build_dataset)

    emit = sub.add_parser("emit-contribution", help="write a NeMo Gym resources_servers/ contribution directory")
    emit.add_argument("--out-dir", required=True)
    emit.add_argument("--suite-dir", action="append", required=True)
    emit.add_argument("--example-count", type=int, default=5)
    emit.add_argument("--rollouts-dir", default=None, help="NeMo Gym rollouts to draw example_rollouts.jsonl from")
    emit.add_argument(
        "--calibration-readme", default=None, help="results/calibration/README.md to embed as the reward profile"
    )
    emit.set_defaults(func=_cmd_emit_contribution)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
