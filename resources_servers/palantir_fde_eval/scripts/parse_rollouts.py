#!/usr/bin/env python3
"""Parse rollout output JSONL into human-readable logs.

Produces two files:
  *_responses.log  — Raw LLM response output per row (LOG-01)
  *_summary.tsv    — Structured per-row summary with reward and verdicts (LOG-02)
"""
import argparse
import json
from pathlib import Path


def extract_response_text(response_output: list) -> list[str]:
    """Extract human-readable lines from response.output items."""
    lines = []
    for item in response_output:
        item_type = item.get("type", "")
        if item_type == "function_call":
            name = item.get("name", "unknown")
            args = item.get("arguments", "")
            lines.append(f"[function_call] {name}({args})")
        elif item_type == "message":
            for content_item in item.get("content", []):
                text = content_item.get("text", "")
                if text:
                    lines.append(f"[message] {text}")
    return lines


def parse_rollouts(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Parse rollout JSONL and write response log + summary TSV."""
    stem = input_path.stem
    responses_path = output_dir / f"{stem}_responses.log"
    summary_path = output_dir / f"{stem}_summary.tsv"

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"Warning: {input_path} is empty")
        responses_path.write_text("")
        summary_path.write_text(
            "task_index\trollout_index\treward\ttool_name_match\tstructure_valid\tjudge_score\tpredicted_tool_names\n"
        )
        print(f"Wrote 0 rows to {responses_path} and {summary_path}")
        return responses_path, summary_path

    # Write responses log
    with open(responses_path, "w") as f:
        for row in rows:
            task_idx = row.get("_ng_task_index", "?")
            rollout_idx = row.get("_ng_rollout_index", "?")
            f.write(f"=== Row {task_idx} (rollout {rollout_idx}) ===\n")

            response_output = row.get("response", {}).get("output", [])
            lines = extract_response_text(response_output)
            for line in lines:
                f.write(f"{line}\n")

            if not lines:
                f.write("[no extractable output]\n")
            f.write("---\n\n")

    # Write summary TSV
    with open(summary_path, "w") as f:
        f.write("task_index\trollout_index\treward\ttool_name_match\tstructure_valid\tjudge_score\tpredicted_tool_names\n")
        for row in rows:
            task_idx = row.get("_ng_task_index", "")
            rollout_idx = row.get("_ng_rollout_index", "")
            reward = row.get("reward", "")
            tool_match = row.get("tool_name_match", "")
            struct_valid = row.get("structure_valid", "")
            judge = row.get("judge_score", "")
            predicted = row.get("predicted_calls", [])
            tool_names = ",".join(c.get("name", "unknown") for c in predicted) if predicted else ""
            f.write(f"{task_idx}\t{rollout_idx}\t{reward}\t{tool_match}\t{struct_valid}\t{judge}\t{tool_names}\n")

    print(f"Wrote {len(rows)} rows to {responses_path} and {summary_path}")
    return responses_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Parse rollout output into logs")
    parser.add_argument("input", type=Path, help="Path to rollouts JSONL")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input.parent
    parse_rollouts(args.input, output_dir)


if __name__ == "__main__":
    main()
