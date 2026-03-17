#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


def _load_prompt_template(filename: str) -> str:
    with open(PROMPT_TEMPLATES_DIR / filename, encoding="utf-8") as fin:
        return yaml.safe_load(fin)["user"]


VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("verifier.yaml")

DEFAULT_AGENT_NAME = "proof_verification_simple_agent"


def convert_verification_rows(rows: list[dict[str, Any]], agent_name: str = DEFAULT_AGENT_NAME) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        problem = row["problem"]
        proof = row["proof"]
        user_content = VERIFIER_PROMPT_TEMPLATE.format(problem=problem, proof=proof)
        gym_example = {
            "agent_ref": {"name": agent_name},
            "responses_create_params": {
                "input": [{"role": "user", "content": user_content}],
            },
            "problem": problem,
            "proof": proof,
            "ground_truth_judgement": row["ground_truth_judgement"],
            "ground_truth_verify_score": row["ground_truth_verify_score"],
        }
        examples.append(gym_example)
    return examples


def convert_verification_jsonl(
    input_path: str,
    output_path: str,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> int:
    with open(input_path, encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    examples = convert_verification_rows(rows, agent_name=agent_name)

    with open(output_path, "w", encoding="utf-8") as fout:
        for example in examples:
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(examples)


def main():
    parser = argparse.ArgumentParser(description="Convert proof-verification JSONL to Gym-compatible format")
    parser.add_argument("--input", required=True, help="Path to proof-verification JSONL")
    parser.add_argument("--output", required=True, help="Path to Gym-compatible output JSONL")
    parser.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help=f"Agent name for agent_ref routing (default: '{DEFAULT_AGENT_NAME}')",
    )
    args = parser.parse_args()

    count = convert_verification_jsonl(args.input, args.output, agent_name=args.agent_name)
    print(f"Converted {count} examples: {args.input} -> {args.output}")
    print(f"Agent ref: {args.agent_name}")


if __name__ == "__main__":
    main()
