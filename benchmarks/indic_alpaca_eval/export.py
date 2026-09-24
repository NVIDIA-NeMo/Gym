# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export one language of Gym rollouts for the upstream AlpacaEval package."""

import argparse
import json
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    # Literal Unicode line separators can occur inside instructions.
    with path.open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def _final_answer(row: dict) -> str:
    if row.get("_ng_failure_class") or row.get("failure_kind") or row.get("mask_sample"):
        raise ValueError(f"Cannot export failed or masked rollout {row['uid']}")
    response = row.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("output"), list):
        raise ValueError(f"Missing model response for {row['uid']}")
    if response.get("status") in {"failed", "cancelled", "queued", "in_progress"}:
        raise ValueError(f"Unfinished/failed model response for {row['uid']}")
    chunks = []
    for item in response["output"]:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                if not isinstance(content.get("text"), str):
                    raise ValueError(f"Non-text model answer for {row['uid']}")
                chunks.append(content["text"])
    # Empty or truncated candidate answers are still results, not dropped tasks.
    return "".join(chunks)


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_outputs(
    *,
    rollouts_file: str,
    dataset_file: str,
    language: str,
    generator: str,
    output_file: str,
    reference_output_file: str | None = None,
    reference_generator: str | None = None,
) -> Path:
    """Export a complete selected language, optionally with aligned references.

    Reject missing, extra, duplicate, or prompt-mismatched tasks before writing.
    Multiple repeats must be separated by the caller; never silently choose a
    favorable answer. References come from baseline_answer in the prepared data.
    """
    if not generator.strip():
        raise ValueError("Specify a nonempty candidate generator name.")
    if (reference_output_file is None) != (reference_generator is None):
        raise ValueError("Specify both reference_output_file and reference_generator.")
    if reference_generator is not None and not reference_generator.strip():
        raise ValueError("Specify a nonempty reference generator name.")
    output = Path(output_file)
    reference_output = Path(reference_output_file) if reference_output_file is not None else None
    paths = [Path(rollouts_file), Path(dataset_file), output]
    if reference_output is not None:
        paths.append(reference_output)
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Input and output paths must be distinct.")

    expected_rows = [r for r in _read_rows(Path(dataset_file)) if r["language"] == language]
    if not expected_rows:
        raise ValueError(f"No prepared instructions for language {language!r}")
    expected = {r["uid"]: r for r in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Duplicate prepared task IDs.")
    instructions = [r["instruction"] for r in expected_rows]
    if len(set(instructions)) != len(instructions):
        raise ValueError("Duplicate instruction text would make upstream instruction-based joins ambiguous.")
    actual = {}
    for row in _read_rows(Path(rollouts_file)):
        if row["language"] != language:
            continue
        uid = row["uid"]
        if uid in actual:
            raise ValueError(f"Duplicate rollout for {uid}; export one repeat at a time.")
        if uid not in expected:
            raise ValueError(f"Unexpected task {uid} for language {language}")
        if row.get("question") != expected[uid]["instruction"]:
            raise ValueError(f"Instruction mismatch for {uid}")
        actual[uid] = row
    if missing := set(expected) - set(actual):
        raise ValueError(f"Missing {len(missing)} candidate answers for {language}: {sorted(missing)[:5]}")

    candidates = []
    references = []
    for task in expected_rows:
        common = {"instruction": task["instruction"]}
        candidates.append({**common, "output": _final_answer(actual[task["uid"]]), "generator": generator})
        if reference_output is not None:
            answer = task.get("baseline_answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Missing reference answer for {task['uid']}; prepare with --reference-file.")
            references.append({**common, "output": answer, "generator": reference_generator})
    _write_json(output, candidates)
    if reference_output is not None:
        _write_json(reference_output, references)
    print(f"Exported {len(candidates)} {language} answers to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts-file", required=True)
    parser.add_argument("--dataset-file", required=True, help="Prepared Gym JSONL defining the expected tasks.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--generator", required=True, help="Name of the candidate model.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--reference-output-file")
    parser.add_argument("--reference-generator")
    export_outputs(**vars(parser.parse_args()))
