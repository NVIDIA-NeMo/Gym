"""Convert scaled function-calling eval data to NeMo-Gym JSONL format.

Reads function_clusters.json and evaluation_dataset.json from the evals source,
generates a single combined JSONL dataset with all tool scales (20/40/60/80/100)
plus example.jsonl. Each row's verifier_metadata.function_count identifies the
tool scale it belongs to, enabling post-hoc aggregation by scale.

Usage:
    python resources_servers/dassault_tool_call_scaled/preprocess_to_gym.py \
        --evals-dir /path/to/evals/scaled/test_data
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You are a function calling assistant. Select the most appropriate function for the user query. "
    "Today's date is 2024-12-03. Always call a function when one is relevant."
)

TYPE_MAP = {
    "integer": "integer",
    "string": "string",
    "date": "string",
    "boolean": "boolean",
    "float": "number",
    "number": "number",
    "array": "array",
}


def load_json(filepath: str) -> Any:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def func_to_tool(func: Dict) -> Dict:
    """Convert a scaled function definition to OpenAI tool format."""
    properties = {}
    for param_name in func.get("parameters", []):
        raw_type = func.get("parameter_types", {}).get(param_name, "string")
        json_type = TYPE_MAP.get(raw_type, "string")
        properties[param_name] = {"type": json_type}

    return {
        "type": "function",
        "name": func["id"],
        "description": func.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(func.get("parameters", [])),
        },
        "strict": None,
    }


def get_functions_by_count(clusters: Dict[str, List[Dict]], count: int) -> List[Dict]:
    """Get a subset of functions balanced across clusters."""
    all_funcs = []
    for cluster_name, functions in clusters.items():
        for func in functions:
            func_copy = func.copy()
            func_copy["cluster"] = cluster_name
            all_funcs.append(func_copy)

    if count >= len(all_funcs):
        return all_funcs

    cluster_names = list(clusters.keys())
    per_cluster = count // len(cluster_names)
    remainder = count % len(cluster_names)

    selected = []
    for i, cluster in enumerate(cluster_names):
        take = per_cluster + (1 if i < remainder else 0)
        cluster_funcs = [f for f in all_funcs if f.get("cluster") == cluster]
        selected.extend(cluster_funcs[:take])

    return selected[:count]


def generate_dataset(
    clusters: Dict[str, List[Dict]],
    eval_dataset: List[Dict],
    func_count: int,
) -> List[Dict]:
    functions = get_functions_by_count(clusters, func_count)
    tools = [func_to_tool(f) for f in functions]
    available_ids = {f["id"] for f in functions}

    valid_queries = [q for q in eval_dataset if q["expected_function"] in available_ids]

    entries = []
    for query_data in valid_queries:
        entry = {
            "responses_create_params": {
                "input": [
                    {"role": "developer", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query_data["query"]},
                ],
                "tools": tools,
            },
            "verifier_metadata": {
                "query_id": query_data.get("id"),
                "expected_function": query_data["expected_function"],
                "expected_params": query_data.get("expected_params", {}),
                "confusion_candidates": query_data.get("confusion_candidates", []),
                "function_count": func_count,
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
    parser = argparse.ArgumentParser(description="Generate scaled tool-calling benchmark JSONL data")
    parser.add_argument(
        "--evals-dir",
        required=True,
        help="Path to evals/scaled/test_data/ directory containing function_clusters.json and evaluation_dataset.json",
    )
    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    data_dir = Path(__file__).resolve().parent / "data"

    clusters = load_json(evals_dir / "function_clusters.json")
    eval_dataset = load_json(evals_dir / "evaluation_dataset.json")

    all_entries = []
    for count in [20, 40, 60, 80, 100]:
        all_entries.extend(generate_dataset(clusters, eval_dataset, count))
    write_jsonl(all_entries, str(data_dir / "scaled.jsonl"))

    example_entries = generate_dataset(clusters, eval_dataset, 100)[:5]
    write_jsonl(example_entries, str(data_dir / "example.jsonl"))

    print("Done!")


if __name__ == "__main__":
    main()
