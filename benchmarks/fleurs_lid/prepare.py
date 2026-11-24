# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the FLEURS spoken language identification benchmark for Gym.

The ``test`` split of every FLEURS locale (all 102 by default, or a subset such as the 36 language-ID locales) is read from the pinned ``google/fleurs`` revision. Each clip is written under ``data/audio/<locale>/``
and the prompt lists the selected locales as the allowed answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from benchmarks import audio_manifest, fleurs_common
from resources_servers.audio_eval._scoring.dataset.fleurs.languages import LID_LOCALES


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "fleurs_lid_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (
    Path(__file__),
    PROFILES,
    Path(audio_manifest.__file__),
    Path(fleurs_common.__file__),
)

# The 36 language-ID locales (28,033 clips), selected with ``languages=subset_36``; the prompt lists them in this order.
LOCALES_36 = LID_LOCALES
LANGUAGE_SETS = {"subset_36": LOCALES_36}


def language_prompt(locales: Sequence[str]) -> str:
    """The user prompt of the source rows: the selected locales are the allowed answers."""
    return (
        "Identify the language spoken in this audio. Respond with exactly one of the following locale labels and "
        "nothing else: " + ", ".join(locales) + "."
    )


def format_source_row(locale: str, row: dict, duration: float, audio_dir: str, prompt: str) -> dict:
    audio_path = f"{audio_dir}/{locale}/{row['wav_filename']}"
    name, group, _ = fleurs_common.LOCALES[locale]
    return {
        # Source sentence ids repeat across speakers: locale and WAV name identify a clip.
        "id": f"{locale}:{row['wav_filename']}",
        "expected_answer": locale,
        "audio_path": audio_path,
        "duration": duration,
        "messages": [{"role": "user", "content": prompt, "audio": {"path": audio_path, "duration": duration}}],
        "subset_for_metrics": locale,
        "task_type": "FLEURS-LID",
        "extra_fields": {
            "locale": locale,
            "locale_name": name,
            "lang_group": group,
            "source_id": row["id"],
            "source_wav_filename": row["wav_filename"],
            "source_revision": fleurs_common.HF_REVISION,
            "source_split": fleurs_common.SPLIT,
        },
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    languages: str | Sequence[str] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
    max_samples_per_locale: int | None = None,
) -> Path:
    """Download FLEURS and write the benchmark JSONL.

    Args:
        languages: FLEURS locales to keep, as a list, a comma-separated string or ``subset_36`` for the 36
            language-ID locales (default: all 102). The prompt lists exactly these locales, in this order.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N clips overall (smoke runs).
        max_samples_per_locale: Keep only the first N clips of each locale (smoke runs).
    """
    locales = fleurs_common.select_locales(languages, sets=LANGUAGE_SETS)
    cache_dir = DATA_DIR / "hf-fleurs"
    rows_by_locale = {locale: fleurs_common.load_locale(locale, cache_dir) for locale in locales}
    prompt = language_prompt(locales)

    clips = [(locale, row) for locale in locales for row in rows_by_locale[locale][:max_samples_per_locale]]
    clips = clips[:max_samples_per_profile]
    durations: dict[tuple[str, str], float] = {}
    for locale in locales:
        wanted = [row["wav_filename"] for clip_locale, row in clips if clip_locale == locale]
        if wanted:
            extracted = fleurs_common.extract_locale(
                locale, rows_by_locale[locale], wanted, DATA_DIR / "audio", cache_dir
            )
            durations.update(((locale, name), duration) for name, duration in extracted.items())

    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    prepared = audio_manifest.profile_rows(
        PROFILES,
        [
            format_source_row(locale, row, durations[locale, row["wav_filename"]], audio_dir, prompt)
            for locale, row in clips
        ],
    )
    options = {
        "languages": None if locales == fleurs_common.select_locales(None, sets=LANGUAGE_SETS) else list(locales),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
        "max_samples_per_locale": max_samples_per_locale,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
