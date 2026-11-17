#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Converts CVDP local_export prompts to NeMo-Gym format.
# Optional --from-hf mode prepares pinned public non-agentic tasks without local_export.
#
# Input: prompts.jsonl produced by CVDP's local_export mode, with fields:
#   {id, prompt, system, user}
#
# Also requires the original CVDP dataset for verifier_metadata (harness_files,
# target_files), which the resource server (app.py) needs to run the Docker harness.
#
# Usage:
#   python convert_to_gym.py \
#       --prompts  prompts.jsonl \
#       --dataset  cvdp_dataset.jsonl \
#       --output   data/train.jsonl

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict


# Code comprehension categories use subjective scoring, not docker-compose harness.
# They have no target_files or harness_files — instead they carry a reference answer.
CODE_COMPREHENSION_CATEGORIES = [6, 8, 9, 10]


def _get_category_num(entry: dict) -> int | None:
    """Extract the numeric category from the categories list (e.g. 'cid010' -> 10)."""
    categories = entry.get("categories", [])
    if categories and isinstance(categories[0], str) and categories[0].startswith("cid"):
        return int(categories[0][3:])
    return None


def _get_target_files(entry: dict) -> list[str]:
    """Target paths from processed output.context or the published patch map."""
    output_context = (entry.get("output") or {}).get("context")
    if output_context is None:
        output_context = entry.get("patch") or {}
    return list(output_context.keys())


def _get_harness_files(entry: dict) -> dict[str, str | None]:
    """Docker-compose + test scripts — passed as-is, matching CVDP."""
    harness = entry.get("harness") or {}
    return harness.get("files", harness) or {}


def _get_context_files(entry: dict) -> dict[str, str]:
    """Keep every starting file, including buggy target RTL for repair tasks.

    Output/reference patches identify target paths but never supply starting content.
    """
    input_context = (entry.get("input") or {}).get("context", entry.get("context")) or {}
    return {k: v for k, v in input_context.items() if isinstance(v, str)}


def _get_subjective_reference(entry: dict) -> str | None:
    """Reference answer for code-comprehension categories — from output.response."""
    return (entry.get("output") or {}).get("response")


# Pinned public dataset revision and file integrity digests.
DATASET_REVISION = "5b807d945f6a99aa645f7e43a64a2115e281b4bf"  # pragma: allowlist secret
RAW_FILE = "cvdp_v1.1.0_nonagentic_code_generation_no_commercial.jsonl"
RAW_SHA256 = "cbcd81295561ebb16e4d857e096f4d9908d042c33aff3b58abf236e868411857"  # pragma: allowlist secret
OUTPUT_SHA256 = "3fa9800ad8696277abd7338be225a9388d35b9aa3e1b9e42b21989299735b36d"  # pragma: allowlist secret
TASK_COUNT = 302
_SUPPORTED_CATEGORIES = frozenset({"cid002", "cid003", "cid004", "cid007", "cid016"})
_LINT_RULE = re.compile(r"\s*lint_off\s+-rule\s+(\w+)")


class TaskInput(TypedDict):
    prompt: str
    context: NotRequired[dict[str, str | None] | None]


class TaskOutput(TypedDict):
    context: NotRequired[dict[str, str | None] | None]


class TaskHarness(TypedDict):
    files: NotRequired[dict[str, str | None] | None]


class RawTask(TypedDict):
    id: str
    input: TaskInput
    output: NotRequired[TaskOutput | None]
    harness: NotRequired[TaskHarness | None]
    categories: NotRequired[list[str] | None]
    difficulty: NotRequired[str]


def check_digest(path: Path, expected: str) -> None:
    """Reject changed public input or generated output instead of changing the protocol silently."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{path.name}: expected SHA256 {expected}, got {actual}")


def _harness_text(path: str, text: str | None) -> str | None:
    if not path.endswith(".vlt") or not text:
        return text
    # The reference simulator does not recognize this rule. Keep all other bytes,
    # including line endings, so the remaining lint checks retain their meaning.
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not ((match := _LINT_RULE.match(line)) and match.group(1) == "PROCASSINIT")
    )


def convert_public_task(task: RawTask) -> dict[str, object]:
    """Build one Gym row without changing the source task or exposing its reference answer."""
    from resources_servers.cvdp.cvdp_lib.model_helpers import ModelHelpers

    categories = task.get("categories") or []
    if not categories or categories[0] not in _SUPPORTED_CATEGORIES:
        raise ValueError(f"{task['id']}: unsupported non-commercial code-generation category {categories!r}")
    targets = list(((task.get("output") or {}).get("context") or {}).keys())
    if not targets:
        raise ValueError(f"{task['id']}: code-generation task has no target files")
    context = task["input"].get("context") or {}
    fragments = [
        f"\nConsider the following content for the file {name}:\n```\n{content}\n```"
        for name, content in context.items()
    ]
    fragments.append(f"\nProvide me one answer for this request: {task['input']['prompt']}")
    helpers = ModelHelpers()
    schema, no_schema = helpers.determine_schema(targets)
    if no_schema:
        fragments.append(
            "\nPlease provide your response as plain text without any JSON formatting. "
            f"Your response will be saved directly to: {targets[0]}.\n"
        )
    else:
        fragments.append(f"\nGenerate the following files: {', '.join(targets)}.\n")
    instructions = helpers.create_system_prompt(category=int(categories[0][3:]), schema=schema).rstrip()
    harness = (task.get("harness") or {}).get("files") or {}
    metadata: dict[str, object] = {
        "task_id": task["id"],
        "categories": categories,
        "difficulty": task.get("difficulty", ""),
        "target_files": targets,
        "harness_files": {name: _harness_text(name, text) for name, text in harness.items()},
    }
    # Generation targets may also contain input context required by the harness.
    supplied_files = {name: content for name, content in context.items() if content}
    if supplied_files:
        metadata["context_files"] = supplied_files
    return {
        "task_id": task["id"],
        "responses_create_params": {
            "input": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": "".join(fragments)},
            ]
        },
        "verifier_metadata": metadata,
    }


def prepare_public_dataset(
    *,
    output: Path,
    raw_path: Path | None = None,
    revision: str = DATASET_REVISION,
    filename: str = RAW_FILE,
    expected_raw_sha256: str = RAW_SHA256,
    expected_output_sha256: str = OUTPUT_SHA256,
    expected_count: int = TASK_COUNT,
) -> Path:
    """Prepare public non-commercial generation tasks, replacing output only after integrity checks.

    Defaults reproduce the pinned CVDP v1.1.0 non-agentic split from Gym PR #5900.
    Other compatible inputs require their own reviewed revision, digests and count.
    """
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if raw_path is None:
        from huggingface_hub import hf_hub_download

        raw_path = Path(
            hf_hub_download("nvidia/cvdp-benchmark-dataset", filename, repo_type="dataset", revision=revision)
        )
    check_digest(raw_path, expected_raw_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}-", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        count = 0
        with (
            raw_path.open(encoding="utf-8") as source,
            pending.open("w", encoding="utf-8", newline="\n") as destination,
        ):
            for line in source:
                if line.strip():
                    destination.write(json.dumps(convert_public_task(json.loads(line))) + "\n")
                    count += 1
        if count != expected_count:
            raise ValueError(f"Expected {expected_count} CVDP tasks, got {count}")
        check_digest(pending, expected_output_sha256)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CVDP export prompts to NeMo-Gym format")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompts", help="prompts.jsonl from CVDP local_export mode")
    mode.add_argument("--from-hf", action="store_true", help="Prepare pinned public non-agentic v1.1.0 tasks directly")
    parser.add_argument("--dataset", help="Original CVDP JSONL; optional offline raw input with --from-hf")
    public = parser.add_argument_group("public preparation overrides (used with --from-hf)")
    public.add_argument("--hf-revision", default=DATASET_REVISION, help="Hugging Face dataset commit")
    public.add_argument("--hf-file", default=RAW_FILE, help="File in nvidia/cvdp-benchmark-dataset")
    public.add_argument("--expected-sha256", default=RAW_SHA256, help="Expected raw input SHA-256")
    public.add_argument("--expected-output-sha256", default=OUTPUT_SHA256, help="Expected converted JSONL SHA-256")
    public.add_argument("--expected-count", type=int, default=TASK_COUNT, help="Expected task count")
    parser.add_argument("--output", required=True, help="Output NeMo-Gym JSONL")
    args = parser.parse_args()
    if args.from_hf:
        output = prepare_public_dataset(
            output=Path(args.output),
            raw_path=Path(args.dataset) if args.dataset else None,
            revision=args.hf_revision,
            filename=args.hf_file,
            expected_raw_sha256=args.expected_sha256,
            expected_output_sha256=args.expected_output_sha256,
            expected_count=args.expected_count,
        )
        print(f"Wrote verified public dataset to {output}")
        return
    if not args.dataset:
        parser.error("--dataset is required with --prompts")

    # Index original dataset by id for verifier_metadata lookup
    dataset: dict[str, dict] = {}
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                dataset[entry["id"]] = entry

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(args.prompts) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row["id"]

            if task_id not in dataset:
                skipped += 1
                continue

            raw = dataset[task_id]
            cat_num = _get_category_num(raw)
            is_comprehension = cat_num is not None and cat_num in CODE_COMPREHENSION_CATEGORIES

            target_files = _get_target_files(raw)
            # Code-generation categories require target_files; comprehension categories don't.
            if not target_files and not is_comprehension:
                skipped += 1
                continue

            context_files = _get_context_files(raw)
            verifier_metadata = {
                "task_id": task_id,
                "categories": raw.get("categories", []),
                "difficulty": raw.get("difficulty", ""),
                "target_files": target_files,
                "harness_files": _get_harness_files(raw),
            }
            if context_files:
                verifier_metadata["context_files"] = context_files

            # Code-comprehension categories carry the reference answer for BLEU/ROUGE scoring.
            if is_comprehension:
                ref = _get_subjective_reference(raw)
                if ref:
                    verifier_metadata["subjective_reference"] = ref
                else:
                    print(f"WARNING: no output.response for comprehension task {task_id}, skipping")
                    skipped += 1
                    continue

            gym_row = {
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["user"]},
                    ]
                },
                "verifier_metadata": verifier_metadata,
            }
            fout.write(json.dumps(gym_row) + "\n")
            written += 1

    print(f"Wrote {written} entries to {args.output}")
    if skipped:
        print(f"Skipped {skipped} entries (no dataset match, no target files, or missing reference)")


if __name__ == "__main__":
    main()
