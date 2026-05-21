# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compute-eval dataset prep.

Direct port of NeMo-Skills' ``nemo_skills/dataset/compute-eval/prepare.py``:
- downloads ``nvidia/compute-eval`` from HuggingFace (gated; requires HF_TOKEN)
- pre-formats the per-row ``context_files`` list into a single string block
  (``context_files_block``) using the same fence-language heuristic as Skills
- emits Gym-shaped JSONL with ``question`` + ``verifier_metadata``

The ``question`` field contains the three placeholder values the prompt
template substitutes in (``problem_prompt``, ``build_command``,
``context_files_block``). ``verifier_metadata`` carries the full original
problem record so the resources server can compile + run hidden tests via
``compute_eval.execution.evaluate_solutions``.
"""

import argparse
import os
from pathlib import Path

import datasets
import orjson


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_RELEASE = None  # None → compute-eval's default release

_CONTEXT_FILES_BLOCK_TEMPLATE = """
--- file: {path}
```{fence}
{content}
```
"""


def _fence_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith((".cu", ".cuh")):
        return "cuda"
    if p.endswith((".cc", ".cpp", ".cxx")):
        return "cpp"
    if p.endswith(".c"):
        return "c"
    if p.endswith(".h") or p.endswith(".hpp"):
        return "h"
    return ""


def _format_context_files_block(context_files: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for source in context_files:
        if "path" not in source or "content" not in source:
            continue
        fence = _fence_for_path(source["path"])
        blocks.append(
            _CONTEXT_FILES_BLOCK_TEMPLATE.format(path=source["path"], fence=fence, content=source["content"])
        )
    return "".join(blocks)


def prepare(
    output_path: Path = DATA_DIR / "compute_eval_benchmark.jsonl",
    release: str | None = DEFAULT_RELEASE,
) -> Path:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be set in the environment (nvidia/compute-eval is gated on HF).")

    dataset = datasets.load_dataset("nvidia/compute-eval", release, token=token)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "wb") as f:
        for item in dataset["eval"]:
            problem_prompt = item["prompt"]
            build_command = item["build_command"]
            context_files_block = _format_context_files_block(item["context_files"])

            out = {
                "problem_prompt": problem_prompt,
                "build_command": build_command,
                "context_files_block": context_files_block,
                "verifier_metadata": {
                    "task_id": item["task_id"],
                    "problem": item,
                },
            }
            f.write(orjson.dumps(out, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")
            n_written += 1

    print(f"Wrote {n_written} problems to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare nvidia/compute-eval for NeMo Gym")
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="HuggingFace release name (e.g., '2025-1', '2025-2'). Default: dataset default.",
    )
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "compute_eval_benchmark.jsonl")
    args = parser.parse_args()
    prepare(output_path=args.output_path, release=args.release)
