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

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


_OPEN_ENDED = {
    "molecule-name",
    "molecule-completion",
    "molecule-formula",
    "simple-formula",
    "reaction-prediction",
    "functional-group",
    "reaction-name",
    "retro-synthesis",
    "oracle-solubility",
}

_LABELS = ["A", "B", "C", "D", "E", "F"]

_MATH_TEMPLATE = (
    "Solve the following chemistry problem step by step. "
    "Put your answer inside \\boxed{{}}.\n\n{problem}\n\n"
    "Remember to put your answer inside \\boxed{{}}."
)

_MCQ_TEMPLATE = (
    "Answer the following multiple choice question. "
    "The last line of your response should be in the following format: "
    "'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n{problem}"
)


def _apply_aai_format(problem: str, problem_type: str, ideal: str | None) -> tuple[str, str | None, dict]:
    if problem_type in _OPEN_ENDED or problem_type.split("/")[0] in _OPEN_ENDED:
        return _MATH_TEMPLATE.format(problem=problem), ideal, {}

    lines = [line for line in problem.strip().split("\n") if line.strip()]
    q_lines = [line for line in lines if " " in line]
    c_lines = [line for line in lines if " " not in line]
    labels = _LABELS[: len(c_lines)]
    choices = {lbl: ch for lbl, ch in zip(labels, c_lines)}

    answer_label = next((lbl for lbl, ch in zip(labels, c_lines) if ch == ideal), None)
    labeled = "\n".join(f"{lbl}. {ch}" for lbl, ch in zip(labels, c_lines))
    prompt = _MCQ_TEMPLATE.format(problem=f"{' '.join(q_lines)}\n\n{labeled}")
    return prompt, answer_label, {"choices": choices}


def format_row(row: dict, aai_format: bool = False) -> dict:
    problem = row["problem"]
    problem_type = row.get("problem_type", "")
    ideal = row.get("ideal")
    extra_meta: dict = {}

    if aai_format:
        problem, ideal, extra_meta = _apply_aai_format(problem, problem_type, ideal)
        input_messages = [{"role": "user", "content": problem}]
    else:
        input_messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific reasoning agent. "
                    "Think step by step, then place your final answer inside <answer></answer> tags. "
                    "For example: <answer>CCO</answer>"
                ),
            },
            {"role": "user", "content": problem},
        ]

    return {
        "responses_create_params": {"input": input_messages},
        "verifier_metadata": {
            "solution": row["solution"],
            "problem_type": problem_type,
            "ideal": ideal,
            "id": row.get("id"),
            **extra_meta,
        },
        "agent_ref": {"type": "responses_api_agents", "name": "ether0_simple_agent"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare ether0-benchmark for NeMo Gym")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--problem-types", nargs="*", default=None, help="Problem type prefixes to include")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to output")
    parser.add_argument(
        "--aai-format",
        action="store_true",
        help="Use AAI prompt format: open-ended -> \\boxed{}, MCQ -> Answer: LETTER",
    )
    args = parser.parse_args()

    ds = load_dataset("futurehouse/ether0-benchmark", split="test")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as fout:
        for row in ds:
            if args.problem_types:
                pt = row.get("problem_type", "")
                if not any(pt.startswith(p) for p in args.problem_types):
                    continue

            fout.write(json.dumps(format_row(row, aai_format=args.aai_format), ensure_ascii=False) + "\n")
            count += 1

            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} rows to {output_path}", file=sys.stderr)


# python scripts/prepare_ether0.py --output data/val.jsonl
# python scripts/prepare_ether0.py --output data/example.jsonl --limit 5
# python scripts/prepare_ether0.py --output data/val_reactions.jsonl --problem-types reaction-prediction retro-synthesis
# python scripts/prepare_ether0.py --output data/val_aai.jsonl --aai-format
if __name__ == "__main__":
    main()
