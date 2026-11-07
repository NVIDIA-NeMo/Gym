# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize and audit five-shot Indic GPQA Diamond prompts for base models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import shutil
from pathlib import Path
from typing import Any

import jinja2
import pyarrow.parquet as pq
import yaml
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
LANGUAGES = {
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
FIELDS = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
SOURCE_FILES = ("prepare.py", "evaluate.py", "launch.py", "job.sbatch", "README.md")


def sha(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def format_question(question: str, choices: list[str]) -> str:
    """Render one question in the reference harness format."""
    options = "\n".join(f"{letter}. {text.strip()}" for letter, text in zip("ABCD", choices, strict=True))
    return f"Question: {question.strip()}\n{options}\nAnswer:"


def load_config(path: Path) -> dict[str, Any]:
    """Load a campaign configuration and check its required shape."""
    config = json.loads(path.read_text())
    required = {"dataset_path", "canonical_gpqa_path", "reference_task_dir", "sampler_path", "models"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing campaign configuration keys: {sorted(missing)}")
    if not config["models"]:
        raise ValueError("Campaign configuration must contain at least one model")
    return config


def checkpoint_metadata(model: dict[str, Any]) -> dict[str, Any]:
    """Validate a local checkpoint and return immutable identity metadata."""
    path = Path(model["path"]).expanduser().resolve()
    index = path / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if not all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards):
        raise FileNotFoundError(f"Checkpoint {path} has a missing or empty shard")
    gpus = int(model["gpus"])
    if gpus < 1:
        raise ValueError(f"Invalid GPU count for {model['name']}: {gpus}")
    return {
        "name": model["name"],
        "path": str(path),
        "gpus": gpus,
        "memory_gb": int(model.get("memory_gb", 512 if gpus >= 8 else 192)),
        "tensor_parallel_size": int(model.get("tensor_parallel_size", gpus)),
        "config_sha256": sha(path / "config.json"),
        "index_sha256": sha(index),
        "tokenizer_sha256": sha(path / "tokenizer.json"),
        "checkpoint_shards": len(shards),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Campaign JSON; see campaign.example.json")
    parser.add_argument("--run", required=True, type=Path, help="New directory in which to freeze the campaign")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    run = args.run.expanduser().resolve()
    config = load_config(config_path)
    if (run / "jobs.json").exists():
        raise RuntimeError("Campaign already submitted; inputs are immutable. Use a new run directory.")

    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "tasks").mkdir(exist_ok=True)
    source = Path(config["dataset_path"]).expanduser().resolve()
    canonical_path = Path(config["canonical_gpqa_path"]).expanduser().resolve()
    reference = Path(config["reference_task_dir"]).expanduser().resolve()
    sampler_path = Path(config["sampler_path"]).expanduser().resolve()

    records = pq.read_table(source).to_pylist()
    with canonical_path.open() as handle:
        canonical = list(csv.DictReader(handle))
    if len(records) != len(canonical) or len(records) != 198:
        raise ValueError(
            f"Expected 198 aligned GPQA Diamond rows; translated={len(records)}, canonical={len(canonical)}"
        )
    if [row["Record ID"] for row in records] != [row["Record ID"] for row in canonical]:
        raise ValueError("Translated and canonical GPQA Record IDs are not aligned")

    sampler_spec = importlib.util.spec_from_file_location("reference_sampler", sampler_path)
    if sampler_spec is None or sampler_spec.loader is None:
        raise ImportError(f"Could not load reference sampler from {sampler_path}")
    sampler_module = importlib.util.module_from_spec(sampler_spec)
    sampler_spec.loader.exec_module(sampler_module)

    models = {key: checkpoint_metadata(model) for key, model in config["models"].items()}
    tokenizer_hashes = {model["tokenizer_sha256"] for model in models.values()}
    if len(tokenizer_hashes) != 1:
        raise ValueError("This campaign requires all checkpoints to use the same tokenizer")
    tokenizer = Tokenizer.from_file(next(iter(models.values()))["path"] + "/tokenizer.json")

    audits: dict[str, dict[str, Any]] = {}
    for language, name in LANGUAGES.items():
        reference_name = "odiya" if language == "or" else name.lower()
        reference_yaml = reference / f"gpqa_diamond_{reference_name}.yaml"
        task_config = yaml.safe_load(reference_yaml.read_text())
        if not (
            task_config["num_fewshot"] == 5
            and task_config["output_type"] == "multiple_choice"
            and task_config["doc_to_choice"] == list("ABCD")
            and task_config["test_split"] == "train"
        ):
            raise ValueError(f"Unexpected reference task semantics in {reference_yaml}")
        template = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(task_config["doc_to_text"])
        template_name = "Odiya" if language == "or" else name
        documents = []
        for index, record in enumerate(records):
            question = record[f"Question_{name}_translation"]
            options = [record[f"{field}_{name}_translation"] for field in FIELDS]
            if not all(isinstance(value, str) and value.strip() for value in [question, *options]):
                raise ValueError(f"Empty translation in {language} row {index}")
            order = list(range(4))
            seed = int(hashlib.md5(canonical[index]["Question"].encode()).hexdigest(), 16)
            random.Random(seed).shuffle(order)
            choices = [options[choice] for choice in order]
            text = format_question(question, choices)
            adapted = {
                f"question_{template_name}_translation": question,
                f"choices_{template_name}_translation": choices,
                "answer": order.index(0),
            }
            if text != template.render(**adapted):
                raise ValueError(f"Reference prompt mismatch in {language} row {index}")
            documents.append(
                {
                    "text": text,
                    "answer": "ABCD"[order.index(0)],
                    "index": index,
                    "record_id": record["Record ID"],
                }
            )

        sampler = sampler_module.ContextSampler(documents, rnd=42)
        rows = []
        lengths = []
        for document in documents:
            demonstrations = sampler.sample(5, eval_doc=document)
            indices = [example["index"] for example in demonstrations]
            if len(set(indices)) != 5 or document["index"] in indices:
                raise ValueError(f"Invalid few-shot sample for {language}/{document['record_id']}")
            prompt = (
                "".join(example["text"] + " " + example["answer"] + "\n\n" for example in demonstrations)
                + document["text"]
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            choice_ids = []
            for letter in "ABCD":
                combined = tokenizer.encode(prompt + " " + letter, add_special_tokens=False).ids
                if combined[:-1] != prompt_ids:
                    raise ValueError(f"Choice {letter} is not one token at {language} row {document['index']}")
                choice_ids.append(combined[-1])
            if len(set(choice_ids)) != 4:
                raise ValueError(f"Choice token collision at {language} row {document['index']}")
            rows.append(
                {
                    "id": f"{language}/{document['record_id']}",
                    "language": language,
                    "prompt": prompt,
                    "choices": [" A", " B", " C", " D"],
                    "answer": document["answer"],
                    "source_row_index": document["index"],
                    "fewshot_indices": indices,
                    "prompt_token_ids": prompt_ids,
                    "choice_token_ids": choice_ids,
                }
            )
            lengths.append(len(prompt_ids))

        output = run / "data" / f"{language}.jsonl"
        with output.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        task = {
            "task": f"indic_gpqa_diamond_5shot_{language}",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(output)}},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "{{prompt}}",
            "doc_to_choice": list("ABCD"),
            "doc_to_target": "{{ ['A', 'B', 'C', 'D'].index(answer) }}",
            "num_fewshot": 0,
            "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
            "metadata": {"num_fewshot_materialized": 5, "fewshot_seed": 42},
        }
        (run / "tasks" / f"{language}.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
        audits[language] = {
            "rows": len(rows),
            "data_sha256": sha(output),
            "reference_yaml_sha256": sha(reference_yaml),
            "max_prompt_tokens": max(lengths),
            "mean_prompt_tokens": sum(lengths) / len(lengths),
            "prompt_template_mismatches": 0,
            "self_example_leaks": 0,
            "single_token_continuations": True,
            "choice_token_ids": rows[0]["choice_token_ids"],
        }
        print(language, len(rows), "prompts audited; max tokens", max(lengths), flush=True)

    maximum = max(audit["max_prompt_tokens"] for audit in audits.values())
    slurm = {
        "partition": "batch",
        "constraint": "H100",
        "time": "02:00:00",
        **config.get("slurm", {}),
    }
    manifest = {
        "benchmark": "Indic GPQA Diamond pretrain",
        "dataset": "ai4bharat/indic-gpqa",
        "source_path": str(source),
        "source_sha256": sha(source),
        "canonical_path": str(canonical_path),
        "canonical_sha256": sha(canonical_path),
        "reference_path": str(reference),
        "reference_sampler_sha256": sha(sampler_path),
        "num_fewshot": 5,
        "fewshot_seed": 42,
        "fewshot_pool": "train; current question excluded",
        "option_order": "English-question MD5 shuffle, shared across languages",
        "render_chat_template": False,
        "add_special_tokens": False,
        "metric": "accuracy",
        "scoring": "argmax of raw log P(' A'/' B'/' C'/' D' | five-shot prompt)",
        "repeats": 1,
        "languages": audits,
        "models": models,
        "slurm": slurm,
        "max_model_len": max(8192, ((maximum + 2 + 1023) // 1024) * 1024),
        "jobs": len(audits) * len(models),
    }
    source_dir = run / "source"
    source_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for filename in SOURCE_FILES:
        shutil.copy2(ROOT / filename, source_dir / filename)
        source_hashes[filename] = sha(source_dir / filename)
    shutil.copy2(config_path, source_dir / "campaign.json")
    source_hashes["campaign.json"] = sha(source_dir / "campaign.json")
    manifest["source_hashes"] = source_hashes
    write_json(run / "manifest.json", manifest)
    write_json(run / "parity_report.json", audits)
    print("Prepared", len(audits) * len(records), "prompts per model; context", manifest["max_model_len"])


if __name__ == "__main__":
    main()
