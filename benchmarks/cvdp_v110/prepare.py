# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned public CVDP v1.1 dataset using Gym's public prompt helper."""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict

from resources_servers.cvdp.cvdp_lib.model_helpers import ModelHelpers


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


def is_prepared_data_current(fpath: Path) -> bool:
    """Let Gym reuse a cached dataset only when its complete contents match this protocol."""
    return fpath.is_file() and hashlib.sha256(fpath.read_bytes()).hexdigest() == OUTPUT_SHA256


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


def convert_task(task: RawTask) -> dict[str, object]:
    """Build one Gym row without changing the source task or exposing its reference answer."""
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


def prepare(*, raw_path: Path | None = None, output_dir: Path | None = None) -> Path:
    """Download or read the pinned public tasks and atomically publish checksum-verified Gym rows."""
    if raw_path is None:
        from huggingface_hub import hf_hub_download

        raw_path = Path(
            hf_hub_download("nvidia/cvdp-benchmark-dataset", RAW_FILE, repo_type="dataset", revision=DATASET_REVISION)
        )
    check_digest(raw_path, RAW_SHA256)
    directory = output_dir if output_dir is not None else Path(__file__).parent / "data"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "cvdp_v110.jsonl"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".cvdp_v110-", suffix=".jsonl", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        count = 0
        with (
            raw_path.open(encoding="utf-8") as source,
            pending.open("w", encoding="utf-8", newline="\n") as destination,
        ):
            for line in source:
                if line.strip():
                    destination.write(json.dumps(convert_task(json.loads(line))) + "\n")
                    count += 1
        if count != TASK_COUNT:
            raise ValueError(f"Expected {TASK_COUNT} CVDP tasks, got {count}")
        check_digest(pending, OUTPUT_SHA256)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Use an already downloaded input; the pinned checksum still applies")
    parser.add_argument("--output-dir", type=Path, help="Write the prepared dataset to this directory")
    args = parser.parse_args()
    print(prepare(raw_path=args.raw, output_dir=args.output_dir))
