# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rescore saved Gym or lm-eval MILU-1.5 final responses without model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import orjson

from benchmarks.indic.milu_1_5.prepare import COUNTS, LANGUAGES, OPTION_FIELDS, SOURCE_ID, SOURCE_REVISION
from nemo_gym.core import NeMoGymResponse
from resources_servers.mcqa.app import MILU_V2_ANSWER_PATTERN, extract_boxed_and_normalized_answers


@dataclass(frozen=True)
class SavedAnswer:
    """The final answer and its original question metadata, before extraction."""

    row_id: int
    gold: str
    options: list[dict[str, str]]
    text: str
    original_reward: float


def _jsonl(path, digest=None):
    with Path(path).open("rb") as handle:
        for line in handle:
            if digest is not None:
                digest.update(line)
            yield orjson.loads(line)


def _gym_answers(rollouts: Path, inputs: Path, language: str, digest) -> Iterator[SavedAnswer]:
    originals = {}
    for row in _jsonl(inputs):
        index = row["_ng_task_index"]
        if index in originals:
            raise ValueError(f"Duplicate materialized input index: {index}")
        metadata = row["metadata"]
        if metadata["language"] != language or metadata["source_revision"] != SOURCE_REVISION:
            raise ValueError(f"Wrong language or source revision in {inputs}")
        originals[index] = row
    for row in _jsonl(rollouts, digest):
        original = originals[row["_ng_task_index"]]
        if row.get("mask_sample") or row.get("failure_kind") or not row.get("response"):
            raise ValueError(f"Failed/masked response in {rollouts}; cannot report a complete score")
        if row["expected_answer"] != original["expected_answer"]:
            raise ValueError(f"Gold label changed in {rollouts}")
        response = NeMoGymResponse.model_validate(row["response"])
        yield SavedAnswer(
            original["metadata"]["source_row_index"],
            original["expected_answer"],
            original["options"],
            response.output_text.strip(),
            float(row["reward"]),
        )


def _harness_answers(rollouts: Path, language: str, digest) -> Iterator[SavedAnswer]:
    for row in _jsonl(rollouts, digest):
        doc = row["doc"]
        if doc["language"] != LANGUAGES[language]:
            raise ValueError(f"Wrong language in {rollouts}")
        gold = "ABCD"[OPTION_FIELDS.index(doc["target"])]
        if row["target"] != gold:
            raise ValueError(f"Gold label changed in {rollouts}")
        responses = row["resps"]
        if len(responses) != 1 or len(responses[0]) != 1 or not isinstance(responses[0][0], str):
            raise ValueError("Expected exactly one saved final response per question")
        yield SavedAnswer(
            row["doc_id"],
            gold,
            [{letter: doc[field]} for letter, field in zip("ABCD", OPTION_FIELDS, strict=True)],
            responses[0][0].strip(),
            float(row["exact_match"]),
        )


def rescore_shard(
    *,
    rollouts: Path,
    input_format: str,
    language: str,
    output: Path,
    inputs: Path | None = None,
    expected_count: int | None = None,
) -> dict:
    """Write audited per-question scores, rejecting incomplete/duplicate inputs.

    Saved rewards must match the legacy MILU v2 rule. Original responses and labels
    are never rewritten. Both new metrics have exactly the same denominator.
    """
    expected = COUNTS[language] if expected_count is None else expected_count
    if output.resolve() in {rollouts.resolve(), inputs.resolve() if inputs else None}:
        raise ValueError("Score output must not overwrite saved responses or inputs")
    digest = hashlib.sha256()
    if input_format == "gym" and inputs is not None:
        answers = _gym_answers(rollouts, inputs, language, digest)
    elif input_format == "lm_eval":
        answers = _harness_answers(rollouts, language, digest)
    else:
        raise ValueError("Select gym with materialized inputs, or lm_eval")
    seen = set()
    correct = {"strict_boxed": 0, "normalized": 0, "reference": 0}
    extracted = {"strict_boxed": 0, "normalized": 0}
    changed_examples = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for answer in answers:
                if type(answer.row_id) is not int or not 0 <= answer.row_id < expected or answer.row_id in seen:
                    raise ValueError(f"Duplicate or invalid source row index: {answer.row_id}")
                seen.add(answer.row_id)
                match = MILU_V2_ANSWER_PATTERN.search(answer.text)
                reference = next((group.upper() for group in match.groups() if group), None) if match else None
                if answer.original_reward != float(reference == answer.gold):
                    raise ValueError(f"Legacy reward mismatch in {rollouts}, question {answer.row_id}")
                strict, normalized = extract_boxed_and_normalized_answers(answer.text, options=answer.options)
                scores = {"strict_boxed": int(strict == answer.gold), "normalized": int(normalized == answer.gold)}
                correct["reference"] += int(answer.original_reward)
                for name, score in scores.items():
                    correct[name] += score
                extracted["strict_boxed"] += strict is not None
                extracted["normalized"] += normalized is not None
                record = {
                    "source_row_index": answer.row_id,
                    "expected_answer": answer.gold,
                    "strict_boxed_answer": strict,
                    "normalized_answer": normalized,
                    "strict_boxed_reward": scores["strict_boxed"],
                    "normalized_reward": scores["normalized"],
                    "reference_reward": answer.original_reward,
                    "final_text_sha256": hashlib.sha256(answer.text.encode()).hexdigest(),
                }
                handle.write(orjson.dumps(record) + b"\n")
                if strict != normalized and len(changed_examples) < 3:
                    changed_examples.append(record | {"final_text_tail": answer.text[-800:]})
            if len(seen) != expected:
                raise ValueError(f"Incomplete {language}: {len(seen)}/{expected}; no full-language score published")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "rows": len(seen),
        "correct": correct,
        "extracted": extracted,
        "accuracy": {name: 100 * count / len(seen) for name, count in correct.items()},
        "input_format": input_format,
        "rollouts": str(rollouts),
        "inputs": str(inputs) if inputs else None,
        "rollouts_sha256": digest.hexdigest(),
        "scores": str(output),
        "changed_examples": changed_examples,
    }


def main() -> None:
    """Build separate language-wise CSV tables from an explicit saved-run manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset"] != SOURCE_ID or manifest["dataset_revision"] != SOURCE_REVISION:
        raise ValueError("Manifest must identify the pinned MILU-1.5 dataset")
    records = []
    model_keys = set()
    for model in manifest["models"]:
        key = model["key"]
        if key in model_keys or Path(key).name != key or key in (".", ".."):
            raise ValueError(f"Duplicate or invalid model key: {key}")
        model_keys.add(key)
        scores = {}
        for language in LANGUAGES:
            if language not in model["languages"]:
                continue
            spec = model["languages"][language]
            scores[language] = rescore_shard(
                rollouts=Path(spec["rollouts"]),
                input_format=spec["format"],
                language=language,
                inputs=Path(spec["inputs"]) if spec.get("inputs") else None,
                output=args.output_dir / key / f"{language}.scores.jsonl",
            )
            print(f"{key}/{language}: {scores[language]['accuracy']}", flush=True)
        records.append(
            {
                "key": key,
                "model": model["model"],
                "settings": model["settings"],
                "completed_languages": len(scores),
                "languages": scores,
                "macro_average_completed": {
                    metric: sum(s["accuracy"][metric] for s in scores.values()) / len(scores) if scores else None
                    for metric in ("strict_boxed", "normalized", "reference")
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("strict_boxed", "normalized"):
        with (args.output_dir / f"{metric}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Model", "Settings", "Done", "Avg", "Avg completed (provisional if <11)", *LANGUAGES])
            for model in records:
                average = model["macro_average_completed"][metric]
                writer.writerow(
                    [
                        model["model"],
                        model["settings"],
                        model["completed_languages"],
                        average if model["completed_languages"] == len(LANGUAGES) else "",
                        average,
                        *(model["languages"].get(lang, {}).get("accuracy", {}).get(metric, "") for lang in LANGUAGES),
                    ]
                )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "dataset": SOURCE_ID,
        "dataset_revision": SOURCE_REVISION,
        "metric_units": "percent",
        "average": "unweighted mean of language accuracies; full Avg requires 11 languages",
        "strict_definition": "Gym strict_single_letter_boxed parser: last box, supported LaTeX wrappers/letter labels",
        "normalized_definition": "strict answer, otherwise normalized last box/last answer prefix/bare letter/unique exact option text",
        "models": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
