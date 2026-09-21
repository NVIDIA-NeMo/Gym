# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Prepare VeriSoftBench for NeMo Gym.

Clones utopia-group/VeriSoftBench at the pinned commit (the harness *and* the dataset;
``data/verisoftbench.jsonl`` there is byte-identical to the Hugging Face mirror at its
pinned revision), verifies the dataset digest and row count, builds upstream's curated
context prompt for every task with upstream's own ``PromptBuilder``, and writes one
Gym row per task.

Preparation fails closed: the digest must match, exactly 500 rows must parse with ids
1..500, and nothing is written until every prompt has been built. ``--limit`` is the one
supported way to prepare a deliberate subset. No Lean container is needed for
preparation; prompts are built from the row alone.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upstream import (  # noqa: E402
    CONTEXT_MODE,
    DATASET_RELATIVE_PATH,
    DATASET_SHA256,
    EXPECTED_TASK_COUNT,
    UPSTREAM_COMMIT,
    Harness,
    ensure_checkout,
    load_upstream,
)


AGENT_REF = {"type": "responses_api_agents", "name": "verisoftbench_agent"}
DEFAULT_UPSTREAM_DIR = Path("cache/verisoftbench/VeriSoftBench-d066523")
DEFAULT_OUTPUT = Path("resources_servers/verisoftbench/data/verisoftbench_full.jsonl")


class CorpusError(RuntimeError):
    """The dataset does not match what the pinned commit is known to contain."""


def load_rows(upstream_dir: Path) -> list[dict[str, Any]]:
    path = upstream_dir / DATASET_RELATIVE_PATH
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != DATASET_SHA256:
        raise CorpusError(f"{path} has sha256 {digest}, expected {DATASET_SHA256}")
    rows = []
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise CorpusError(f"{path}:{line_no} is not JSON: {e}") from e
    if len(rows) != EXPECTED_TASK_COUNT:
        raise CorpusError(f"{len(rows)} rows parsed, expected {EXPECTED_TASK_COUNT}")
    ids = sorted(r.get("id") for r in rows)
    if ids != list(range(1, EXPECTED_TASK_COUNT + 1)):
        raise CorpusError("task ids are not exactly 1..500")
    return rows


class PromptSource:
    """Upstream's prompt builder without a Lean container (``Harness`` wants one for verification)."""

    def __init__(self, upstream_dir: Path):
        modules = load_upstream(upstream_dir)
        self._builder = modules.prompt_builder.PromptBuilder(modules.root / "prompts" / "templates", mode=CONTEXT_MODE)

    def system_prompt(self) -> str:
        return self._builder.retrive_sys_prompt()

    def user_prompt(self, row: dict[str, Any]) -> str:
        return self._builder.build_user_prompt(theorem_entry=row, mode=CONTEXT_MODE)


def format_row(row: dict[str, Any], system_prompt: str, user_prompt: str) -> dict[str, Any]:
    metadata = dict(row)
    metadata["context_mode"] = CONTEXT_MODE
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        },
        "verifier_metadata": metadata,
        "agent_ref": AGENT_REF,
    }


def prepare(rows: list[dict[str, Any]], prompts: PromptSource, limit: Optional[int]) -> list[dict[str, Any]]:
    system_prompt = prompts.system_prompt()
    if limit is not None:
        rows = rows[:limit]
    out = []
    for row in rows:
        user_prompt = prompts.user_prompt(row)
        if "<target_theorem>" not in user_prompt:
            raise CorpusError(f"task {row.get('id')}: prompt has no <target_theorem> section")
        out.append(format_row(row, system_prompt, user_prompt))
    return out


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="where the pinned checkout lives or is cloned to",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit", type=positive_int, default=None, help="prepare only the first N tasks (deliberate subset)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    ensure_checkout(args.upstream_dir)
    rows = load_rows(args.upstream_dir)
    prepared = prepare(rows, PromptSource(args.upstream_dir), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in prepared:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(prepared)} rows to {args.output} (upstream {UPSTREAM_COMMIT[:12]}, dataset sha256 {DATASET_SHA256[:12]})"
    )


__all__ = ["CorpusError", "Harness", "format_row", "load_rows", "prepare"]

if __name__ == "__main__":
    main()
