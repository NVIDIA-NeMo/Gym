# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare original MBPP translations and a matching English baseline."""

import argparse
import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from textwrap import indent

from datasets import load_dataset
from huggingface_hub import get_token, hf_hub_download

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, maybe_get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "mbpp_benchmark.jsonl"
SOURCE_ID = "ai4bharat/indic-mbpp"
EXPECTED_ROWS = 500
LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
DEFAULT_LANGUAGES = tuple(LANGUAGE_NAMES)
SourceRow = Mapping[str, str | int | list[str]]


def _verifier_metadata(record: SourceRow) -> dict:
    task_id = record.get("task_id")
    tests = record.get("test_list")
    setup = record.get("test_setup_code")
    code = record.get("code")
    if type(task_id) is not int or task_id < 1:
        raise ValueError("task_id must be a positive integer")
    if not isinstance(tests, list) or len(tests) != 3 or any(not isinstance(t, str) or not t.strip() for t in tests):
        raise ValueError(f"Expected three nonempty original tests for task {task_id}")
    if not isinstance(setup, str) or not isinstance(code, str):
        raise ValueError(f"Missing setup or reference code for task {task_id}")
    functions = {node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(tests[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    entry_points = functions & calls
    if len(entry_points) != 1:
        raise ValueError(f"Cannot identify one tested function for task {task_id}")
    # The setup can depend on classes supplied by the candidate, so run it inside check().
    # Preserve a candidate function named check before the runner installs its callback.
    test = (
        'def check(candidate, _original_check=globals().get("check")):\n'
        "    if _original_check is not None:\n"
        '        globals()["check"] = _original_check\n' + indent(setup + "\n" + "\n".join(tests), "    ") + "\n"
    )
    ast.parse(test)
    return {"task_id": f"Mbpp/{task_id}", "entry_point": entry_points.pop(), "test": test}


def build_rows(
    records: Sequence[SourceRow],
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    task_ids: Sequence[int] | None = None,
) -> list[dict]:
    """Build translated prompts and unchanged original MBPP assertions."""
    if (
        isinstance(languages, str)
        or not languages
        or any(code not in (*DEFAULT_LANGUAGES, "en") for code in languages)
    ):
        raise ValueError(f"languages must be a nonempty sequence from {(*DEFAULT_LANGUAGES, 'en')}")
    if len(set(languages)) != len(languages):
        raise ValueError("languages must be unique")
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} MBPP tasks, got {len(records)}")
    metadata = {}
    for record in records:
        meta = _verifier_metadata(record)
        identity = record["task_id"]
        if identity in metadata:
            raise ValueError(f"Duplicate task_id: {identity}")
        metadata[identity] = meta
    if task_ids is not None:
        if not task_ids or any(type(value) is not int for value in task_ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must contain unique integer IDs")
        if set(task_ids) - metadata.keys():
            raise ValueError(f"Unknown task_ids: {sorted(set(task_ids) - metadata.keys())}")
    selected = set(metadata if task_ids is None else task_ids)
    rows = []
    for language in languages:
        column = "text" if language == "en" else f"text_{LANGUAGE_NAMES[language]}_translation"
        for record in records:
            identity = record["task_id"]
            text = record.get(column)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing problem text: {language}/{identity}")
            if identity not in selected:
                continue
            setup = record["test_setup_code"]
            example = (setup + "\n" if setup else "") + record["test_list"][0]
            question = ('"""\n' + text + "\n" + example + '\n"""\n').replace("    ", "\t")
            rows.append(
                {
                    "question": question,
                    "verifier_metadata": metadata[identity],
                    "language": language,
                    "uuid": f"{SOURCE_ID}/{language}/{identity}",
                }
            )
    return rows


def prepare(
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    task_ids: Sequence[int] | None = None,
    output_fpath: str | None = None,
) -> Path:
    """Download the translations and write native Gym tasks."""
    config = maybe_get_global_config_dict()
    token = config.get(HF_TOKEN_KEY_NAME) if config is not None else None
    source = hf_hub_download(
        repo_id=SOURCE_ID,
        filename="test.parquet",
        repo_type="dataset",
        token=token or get_token(),
    )
    records = load_dataset("parquet", data_files={"test": source}, split="test").to_list()
    rows = build_rows(records, languages=languages, task_ids=task_ids)
    output = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} problems to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
