import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def format_row(row: dict) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific reasoning agent. "
                        "Think step by step, then place your final answer inside <answer></answer> tags. "
                        "For example: <answer>CCO</answer>"
                    ),
                },
                {"role": "user", "content": row["problem"]},
            ],
        },
        "verifier_metadata": {
            "solution": row["solution"],
            "problem_type": row.get("problem_type", ""),
            "ideal": row.get("ideal"),
            "id": row.get("id"),
        },
        "agent_ref": {"type": "responses_api_agents", "name": "ether0_simple_agent"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare ether0-benchmark for NeMo Gym")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--problem-types", nargs="*", default=None, help="Problem type prefixes to include")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to output")
    args = parser.parse_args()

    print("Downloading futurehouse/ether0-benchmark", file=sys.stderr)
    ds = load_dataset("futurehouse/ether0-benchmark", split="test")
    print(f"Loaded {len(ds)} rows", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as fout:
        for row in ds:
            if args.problem_types:
                pt = row.get("problem_type", "")
                if not any(pt.startswith(p) for p in args.problem_types):
                    continue

            fout.write(json.dumps(format_row(row), ensure_ascii=False) + "\n")
            count += 1

            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} rows to {output_path}", file=sys.stderr)


# python scripts/prepare_ether0.py --output data/val.jsonl
# python scripts/prepare_ether0.py --output data/example.jsonl --limit 5
# python scripts/prepare_ether0.py --output data/val_reactions.jsonl --problem-types reaction-prediction retro-synthesis
if __name__ == "__main__":
    main()
