# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the SciCodePile runnable benchmark for the ``scicodepile`` resources server.

Notes on the upstream data, verified against all 200 released rows:

  * Every row is ``language: python``, ``runnable: true``, ``test_invalid: false``
    and ``primary_score_eligible: true``.
  * Every ``test`` defines ``check(candidate)``; the runner calls it with the
    function named by ``entry_point``.
  * The ``prompt`` field is **display text, not valid Python** — its docstring is
    not indented under the ``def`` line. It therefore cannot be prepended to the
    model's output the way BigCodeBench calibrates with ``code_prompt``. Every one
    of the 200 ``canonical_solution`` values is a complete function definition, and
    the verifier looks the function up by name, so the model has to produce one too.
  * ``setup_code`` is non-empty on 105 of 200 rows and must run before the
    model's code.

The upstream ``prompt`` is passed through untouched, HumanEval-style. Upstream does
not publish its own prompt, so any wrapper would be invention; measurements behind
that choice are in this benchmark's README.
"""

import argparse
from pathlib import Path

import orjson


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "scicodepile_benchmark.jsonl"

HF_DATASET = "SciCodePile/SciCode-Runnable-Benchmark-Reviewed"
HF_SPLIT = "train"
# Upstream ships a single 200-row split. Assert it so a silent upstream change
# surfaces here rather than as an unexplained score movement.
EXPECTED_ROWS = 200


def _build_question(prompt: str) -> str:
    """Return the user-visible question for one task.

    The upstream ``prompt`` — a function signature plus docstring — is passed
    through unmodified. Adding instructions changes what is being measured, and
    upstream publishes no prompt to match, so nothing is added here.
    """
    return prompt


def prepare(output_path: Path = OUTPUT_FPATH) -> Path:
    """Download and prepare the SciCodePile runnable benchmark. Returns the JSONL path."""
    from datasets import load_dataset

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    print(f"Downloading {HF_DATASET} from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT, token=hf_token)

    if len(dataset) != EXPECTED_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_ROWS} rows in {HF_DATASET}:{HF_SPLIT}, got {len(dataset)}; upstream may have drifted."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "wb") as f:
        for row in dataset:
            if not row["runnable"]:
                # Defensive: all 200 are runnable today, but never emit a task the
                # upstream authors marked unrunnable.
                continue

            meta = row.get("benchmark_meta") or {}
            if isinstance(meta, str):
                import ast

                try:
                    meta = ast.literal_eval(meta)
                except (ValueError, SyntaxError):
                    meta = {}

            out = {
                "question": _build_question(row["prompt"]),
                "verifier_metadata": {
                    "task_id": row["task_id"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "setup_code": row.get("setup_code") or "",
                    "audit_flags": meta.get("audit_flags"),
                    "primary_score_eligible": meta.get("primary_score_eligible"),
                },
            }
            f.write(orjson.dumps(out) + b"\n")
            n_written += 1

    print(f"Wrote {n_written} problems to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH, help="Output JSONL path.")
    args = parser.parse_args()
    prepare(args.output)
