"""Convert multi-step tool orchestration eval data to NeMo-Gym JSONL format.

Reads tools.json and evaluation_dataset.json from the evals source,
generates the multistep JSONL dataset plus example.jsonl.

Usage:
    python resources_servers/dassault_tool_call_multistep/preprocess_to_gym.py \
        --evals-dir /path/to/evals/multistep/test_data
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_json(filepath: str) -> Any:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(tools: Dict[str, Dict]) -> str:
    """Build system prompt with tool descriptions for multi-step planning."""
    tools_desc = "\n".join(
        [
            f"- {name}: {info['description']}\n"
            f"  Parameters: {info['parameters']}\n"
            f"  Returns: {info['returns']}"
            for name, info in tools.items()
        ]
    )

    return f"""You are a tool orchestration planner. Given a user query, determine the sequence of tool calls needed to fulfill the request.

Available tools:
{tools_desc}

IMPORTANT:
1. Identify ALL tools needed to complete the request
2. Determine the correct ORDER (some tools depend on outputs from others)
3. Use <from_previous> notation when a parameter depends on a previous step's output

Respond ONLY with a JSON array of tool calls in execution order:
[
  {{"function": "tool_name", "params": {{"param1": "value1"}}, "step": 1}},
  {{"function": "tool_name", "params": {{"param1": "<from_step_1:field>"}}, "step": 2}}
]

Rules:
- List tools in the order they should be executed
- If a parameter depends on a previous step, use "<from_step_N:field_path>" notation
- Include all necessary tools - don't skip steps
- The step number indicates execution order
- Output ONLY the JSON array, no explanations"""


def generate_dataset(tools: Dict[str, Dict], eval_dataset: List[Dict]) -> List[Dict]:
    system_prompt = build_system_prompt(tools)

    entries = []
    for query_data in eval_dataset:
        entry = {
            "responses_create_params": {
                "input": [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": query_data["query"]},
                ],
            },
            "verifier_metadata": {
                "query_id": query_data.get("id"),
                "expected_sequence": query_data["expected_sequence"],
                "difficulty": query_data.get("difficulty", "unknown"),
                "reasoning": query_data.get("reasoning", ""),
            },
        }
        entries.append(entry)
    return entries


def write_jsonl(entries: List[Dict], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(entries)} entries to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-step tool orchestration benchmark JSONL data")
    parser.add_argument(
        "--evals-dir",
        required=True,
        help="Path to evals/multistep/test_data/ directory containing tools.json and evaluation_dataset.json",
    )
    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    data_dir = Path(__file__).resolve().parent / "data"

    tools = load_json(evals_dir / "tools.json")
    eval_dataset = load_json(evals_dir / "evaluation_dataset.json")

    entries = generate_dataset(tools, eval_dataset)
    write_jsonl(entries, str(data_dir / "multistep.jsonl"))

    example_entries = entries[:5]
    write_jsonl(example_entries, str(data_dir / "example.jsonl"))

    print("Done!")


if __name__ == "__main__":
    main()
