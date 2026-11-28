# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the CoVoST 2 speech recognition and speech translation benchmark for Gym.

The ``test`` split of CoVoST 2 (Common Voice 4 clips with sentences and translations) is read from the pinned
``fixie-ai/covost2`` mirror, which stores the original MP3 clips next to the sentence and translation of each of the
36 language pairs. All 22 languages are prepared by default (ASR for each language, speech translation for every
pair whose two languages are selected); ``languages`` narrows them. Each clip is decoded, resampled to 16 kHz with
``soxr`` and written as a 16-bit WAV file under ``data/audio/<language>/test/``.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "covost2_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "fixie-ai/covost2"
HF_REVISION = "17c8c81e331e7a6929118121771a58c7ef7331d8"  # pragma: allowlist secret (a dataset commit, not a secret)
SPLIT = "test"
SAMPLE_RATE = 16000

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cy": "Welsh",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "lv": "Latvian",
    "mn": "Mongolian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "sl": "Slovenian",
    "sv-SE": "Swedish",
    "ta": "Tamil",
    "tr": "Turkish",
    "zh-CN": "Chinese",
}
ALL_LANGUAGES = tuple(LANGUAGE_NAMES)
# Languages scored by character error rate.
CER_LANGUAGES = frozenset({"zh-CN", "ja"})
# The 17 languages of the model-card protocol: 17 ASR subsets and 26 translation pairs (92,901 and 232,680 rows).
CARD_LANGUAGES = tuple("ar de en es et fr id it ja lv nl pt ru sl sv-SE tr zh-CN".split())
LANGUAGE_SETS = {"card": CARD_LANGUAGES}
# Test clips per language pair in the released data, keyed by the mirror's ``<source>_<target>`` configuration.
PAIR_TEST_ROWS = {
    "ar_en": 1695,
    "ca_en": 12730,
    "cy_en": 690,
    "de_en": 13511,
    "es_en": 13221,
    "et_en": 1571,
    "fa_en": 3445,
    "fr_en": 14760,
    "id_en": 844,
    "it_en": 8951,
    "ja_en": 684,
    "lv_en": 1629,
    "mn_en": 1759,
    "nl_en": 1699,
    "pt_en": 4023,
    "ru_en": 6300,
    "sl_en": 360,
    "sv-SE_en": 1595,
    "ta_en": 786,
    "tr_en": 1629,
    "zh-CN_en": 4898,
    **{f"en_{code}": 15531 for code in "ar ca cy de et fa id ja lv mn sl sv-SE ta tr zh-CN".split()},
}
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
ASR_PROMPT = "Transcribe the following audio."


def select_languages(languages: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize ``languages``: ``card``, a comma-separated string or a list of language codes (default: all 22)."""
    return audio_manifest.select(languages, ALL_LANGUAGES, "CoVoST 2 languages", LANGUAGE_SETS)


def translation_pairs(languages: Sequence[str]) -> list[str]:
    """Configurations of the available pairs whose source and target language are both selected."""
    return sorted(pair for pair in PAIR_TEST_ROWS if set(pair_languages(pair)) <= set(languages))


def pair_languages(pair: str) -> tuple[str, str]:
    source, _, target = pair.rpartition("_") if pair.endswith("_en") else pair.partition("_")
    return source, target


def audio_pair(language: str) -> str:
    """The configuration whose clips are the language's audio (English audio is the same in every ``en_*``)."""
    return "en_de" if language == "en" else f"{language}_en"


def open_shard(path: str):
    """Open a parquet shard of the pinned mirror for range reads."""
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(path, "rb")


def list_shards(pair: str) -> list[str]:
    from huggingface_hub import HfFileSystem

    prefix = f"datasets/{HF_REPO_ID}@{HF_REVISION}/{pair}"
    shards = sorted(HfFileSystem().glob(f"{prefix}/{SPLIT}-*.parquet"))
    if not shards:
        raise ValueError(f"No {SPLIT} shards for CoVoST 2 pair {pair} in {HF_REPO_ID}@{HF_REVISION}")
    return shards


def read_pair(pair: str) -> list[dict]:
    """``id``, ``sentence`` and ``translation`` of every test clip of a pair, ordered by clip file name."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for shard in list_shards(pair):
        with open_shard(shard) as stream:
            table = pq.ParquetFile(stream).read(columns=["id", "sentence", "translation"])
        rows.extend(table.to_pylist())
    if len(rows) != PAIR_TEST_ROWS[pair] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"CoVoST 2 {pair}: expected {PAIR_TEST_ROWS[pair]} unique clips, got {len(rows)}")
    return sorted(rows, key=lambda row: f"{row['id']}.wav")


def decode_clip(mp3: bytes) -> np.ndarray:
    """Decode one clip to mono float32 at 16 kHz."""
    import soundfile
    import soxr

    with soundfile.SoundFile(io.BytesIO(mp3)) as stream:
        rate, channels = int(stream.samplerate), int(stream.channels)
        samples = stream.read(dtype="float32", always_2d=False)
    if channels != 1 or len(samples) == 0:
        raise ValueError(f"CoVoST 2 clip must be non-empty mono; got channels={channels}, frames={len(samples)}")
    return samples if rate == SAMPLE_RATE else soxr.resample(samples, rate, SAMPLE_RATE)


def extract_language(language: str, wanted: set[str], audio_dir: Path) -> dict[str, float]:
    """Write the ``wanted`` clips of a language to ``audio_dir/<language>/test/`` and return their durations."""
    import pyarrow.parquet as pq
    import soundfile

    destination = audio_dir / language / SPLIT
    destination.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}
    for shard in list_shards(audio_pair(language)):
        with open_shard(shard) as stream:
            parquet = pq.ParquetFile(stream)
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(group, columns=["id"]).column("id").to_pylist()
                if not wanted.intersection(ids):
                    continue
                table = parquet.read_row_group(group, columns=["id", "audio"]).to_pylist()
                for item in table:
                    if item["id"] not in wanted:
                        continue
                    samples = decode_clip(item["audio"]["bytes"])
                    target = destination / f"{item['id']}.wav"
                    staged = target.with_name(f".{target.name}.tmp.wav")
                    soundfile.write(str(staged), samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                    os.replace(staged, target)
                    durations[item["id"]] = len(samples) / SAMPLE_RATE
    if missing := wanted - set(durations):
        raise ValueError(f"CoVoST 2 {language}: clips missing from the source: {sorted(missing)[:10]}")
    return durations


def _messages(instruction: str, audio_path: str, duration: float) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction, "audio": {"path": audio_path, "duration": duration}},
    ]


def format_asr_row(language: str, clip: dict, duration: float, audio_dir: str) -> dict:
    audio_path = f"{audio_dir}/{language}/{SPLIT}/{clip['id']}.wav"
    return {
        "expected_answer": clip["sentence"],
        "audio_path": audio_path,
        "duration": duration,
        "messages": _messages(ASR_PROMPT, audio_path, duration),
        "subset_for_metrics": language,
        "task_type": "Multilingual-ASR",
        "extra_fields": {
            "src_text": clip["sentence"],
            "src_lang_name": LANGUAGE_NAMES[language],
            "src_lang": language,
            "use_cer": language in CER_LANGUAGES,
        },
    }


def format_st_row(source: str, target: str, clip: dict, duration: float, audio_dir: str) -> dict:
    audio_path = f"{audio_dir}/{source}/{SPLIT}/{clip['id']}.wav"
    return {
        "expected_answer": clip["translation"],
        "audio_path": audio_path,
        "duration": duration,
        "messages": _messages(f"Please translate the given speech to {LANGUAGE_NAMES[target]}.", audio_path, duration),
        "subset_for_metrics": f"{source}->{target}",
        "task_type": "Multilingual-AST",
        "extra_fields": {
            "src_text": clip["sentence"],
            "tgt_text": clip["translation"],
            "src_lang_name": LANGUAGE_NAMES[source],
            "tgt_lang_name": LANGUAGE_NAMES[target],
            "src_lang": source,
            "tgt_lang": target,
        },
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def _capped(items: list, per_subset: int | None, overall: int | None, subset=lambda item: item[0]) -> list:
    """First ``per_subset`` items of every subset (items arrive grouped by subset), then the first ``overall``."""
    if per_subset is not None:
        counts: dict = {}
        kept = []
        for item in items:
            counts[subset(item)] = counts.get(subset(item), 0) + 1
            if counts[subset(item)] <= per_subset:
                kept.append(item)
        items = kept
    return items[:overall]


def prepare(
    *,
    languages: str | Sequence[str] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
    max_samples_per_subset: int | None = None,
) -> Path:
    """Download CoVoST 2 and write the combined benchmark JSONL (profiles ``covost2.asr`` and ``covost2.st``).

    Args:
        languages: Language codes to keep, as a list, a comma-separated string or ``card`` for the 17 languages of
            the model-card protocol (default: all 22). ASR covers the selected languages; speech translation covers
            the pairs with English whose two languages are both selected.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows of each profile (smoke runs).
        max_samples_per_subset: Keep only the first N rows of each language (ASR) or pair (speech translation).
    """
    for name, cap in (
        ("max_samples_per_profile", max_samples_per_profile),
        ("max_samples_per_subset", max_samples_per_subset),
    ):
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError(f"{name} must be a positive integer, got {cap!r}")
    selected = select_languages(languages)
    pairs = translation_pairs(selected)
    names = list(dict.fromkeys([*pairs, *(audio_pair(lang) for lang in selected)]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        clips_by_pair = dict(zip(names, pool.map(read_pair, names)))

    asr_specs = [(language, clip) for language in selected for clip in clips_by_pair[audio_pair(language)]]
    st_specs = [
        (pair_languages(pair)[0], pair_languages(pair)[1], clip) for pair in pairs for clip in clips_by_pair[pair]
    ]
    specs = {
        "covost2.asr": _capped(asr_specs, max_samples_per_subset, max_samples_per_profile),
        "covost2.st": _capped(st_specs, max_samples_per_subset, max_samples_per_profile, lambda spec: spec[:2]),
    }

    wanted: dict[str, set[str]] = {}
    for language, clip in specs["covost2.asr"]:
        wanted.setdefault(language, set()).add(clip["id"])
    for source, _, clip in specs["covost2.st"]:
        wanted.setdefault(source, set()).add(clip["id"])
    durations = {
        language: extract_language(language, ids, DATA_DIR / "audio")
        for language, ids in sorted(wanted.items())
        if ids
    }

    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"

    def sources(name: str) -> Iterator[dict]:
        if name == "covost2.asr":
            for language, clip in specs[name]:
                yield format_asr_row(language, clip, durations[language][clip["id"]], audio_dir)
        else:
            for source, target, clip in specs[name]:
                yield format_st_row(source, target, clip, durations[source][clip["id"]], audio_dir)

    prepared = audio_manifest.profile_rows(PROFILES, {name: sources(name) for name in specs})
    options = {
        "languages": None if selected == ALL_LANGUAGES else list(selected),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
        "max_samples_per_subset": max_samples_per_subset,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
