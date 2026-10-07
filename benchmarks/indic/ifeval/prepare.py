# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the 12 supported Indic languages from ai4bharat/IndicIFEval Trans."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download
from pydantic import JsonValue

from resources_servers.instruction_following.indicifeval import IndicIFEvalMetadata
from resources_servers.instruction_following.setup_indicifeval import HARNESS_REVISION, LANGUAGES


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "ai4bharat/IndicIFEval"
SOURCE_REVISION = "4343c1b174322e32cc785ef89fe117038fdad7a9"
SOURCE_CONFIG = "indicifeval-trans"
LANGUAGE_ALIASES = {"ka": "kn", "mar": "mr", "mal": "ml"}
TranslationQuality = Literal["correct", "all", "parallel"]


def select_languages(languages: Sequence[str] | None) -> list[str]:
    """Normalize user aliases to the dataset's language split codes."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = [
        LANGUAGE_ALIASES.get(language, language) for language in (LANGUAGES if languages is None else languages)
    ]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {LANGUAGES}; aliases: {LANGUAGE_ALIASES}")
    return selected


def build_rows(
    records: Sequence[Mapping[str, JsonValue]],
    *,
    language: str,
    translation_quality: TranslationQuality = "correct",
) -> list[dict[str, JsonValue]]:
    """Preserve source prompts and checker arguments in English IFEval's schema."""
    language = select_languages([language])[0]
    if translation_quality not in ("correct", "all", "parallel"):
        raise ValueError("translation_quality must be correct, all, or parallel")
    rows = []
    seen = set()
    for record in records:
        key = record.get("key")
        if type(key) is not int or key in seen:
            raise ValueError(f"{language}: missing or duplicate source key {key}")
        seen.add(key)
        tags = record.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"{language}/{key}: tags must be a list of strings")
        if ("correct" in tags) == ("incorrect" in tags):
            raise ValueError(f"{language}/{key}: expected exactly one translation-quality tag")
        if translation_quality != "all" and "correct" not in tags:
            continue
        if translation_quality == "parallel" and "parallel" not in tags:
            continue
        metadata = IndicIFEvalMetadata.model_validate(
            {
                "language": language,
                "prompt": record.get("prompt"),
                "instruction_id_list": record.get("instruction_id_list"),
                "kwargs": record.get("kwargs"),
                "grading_mode": "binary",
            }
        )
        rows.append(
            {
                "id": key,
                "uuid": f"indic-ifeval-trans-{SOURCE_REVISION}-{language}-{key}",
                "subset_for_metrics": language,
                "responses_create_params": {
                    "input": [{"role": "user", "content": metadata.prompt}],
                    "tools": [],
                    "parallel_tool_calls": False,
                    "temperature": 0.0,
                    "max_output_tokens": 1280,
                },
                "verifier_metadata": metadata.model_dump(),
                "metadata": {
                    "source": SOURCE_ID,
                    "source_revision": SOURCE_REVISION,
                    "source_config": SOURCE_CONFIG,
                    "source_split": language,
                    "source_key": key,
                    "harness_revision": HARNESS_REVISION,
                    "tags": tags,
                    "resp_lang": record.get("resp_lang"),
                    "translation_quality": translation_quality,
                },
            }
        )
    if not rows:
        raise ValueError(f"{language}: no rows remain after translation_quality={translation_quality}")
    return rows


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    translation_quality: TranslationQuality = "correct",
) -> Path:
    """Download only pinned Trans parquet files and write Gym evaluation JSONL."""
    import pyarrow.parquet as pq

    selected = select_languages(languages)
    if translation_quality not in ("correct", "all", "parallel"):
        raise ValueError("translation_quality must be correct, all, or parallel")
    rows = []
    for language in selected:
        path = hf_hub_download(
            repo_id=SOURCE_ID,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            filename=f"{SOURCE_CONFIG}/{language}-00000-of-00001.parquet",
        )
        records = pq.read_table(path).to_pylist()
        if len(records) != 490:
            raise ValueError(f"{language}: expected 490 released Trans rows, found {len(records)}")
        language_rows = build_rows(records, language=language, translation_quality=translation_quality)
        print(f"{language}: {len(language_rows)} / {len(records)} Trans rows ({translation_quality})")
        rows.extend(language_rows)
    output_path = Path(output_fpath) if output_fpath else DIRECTORY / "data/ifeval_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    """Prepare all 12 languages or a requested subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=[*LANGUAGES, *LANGUAGE_ALIASES])
    parser.add_argument("--translation-quality", choices=["correct", "all", "parallel"], default="correct")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    print(prepare(args.output_path, languages=args.languages, translation_quality=args.translation_quality))


if __name__ == "__main__":
    main()
