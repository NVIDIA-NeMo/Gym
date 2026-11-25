# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare CS-FLEURS (code-switched speech in 52 languages) for Gym.

The four test sets (``read``, ``mms``, ``xtts-test1``, ``xtts-test2``) come from the pinned ``byan/cs-fleurs``
revision. Each set's ``metadata.jsonl`` is authenticated (SHA-256 and row count), and its clips are downloaded
unchanged under ``data/raw``; one profile per test set shares the combined benchmark JSONL.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "cs_fleurs_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "byan/cs-fleurs"
HF_REVISION = "0cdbf166c5517ae4b6eb1c54248522eedec53017"  # pragma: allowlist secret (a dataset commit, not a secret)
# Profile -> (HF top-level directory, split directory, released row count, SHA-256 of metadata.jsonl).
SUBSETS = {
    "cs-fleurs.read": (
        "read",
        "test",
        5818,
        "1ed381025ed51c368e591e263eace85a6e4e59a215d3f8888ca07c1e188fd709",  # pragma: allowlist secret
    ),
    "cs-fleurs.mms": (
        "mms",
        "test",
        14932,
        "48c065a79260058951ba2f25e58c865bea313874beb15fc07db7e72b6bc63f37",  # pragma: allowlist secret
    ),
    "cs-fleurs.xtts-test1": (
        "xtts",
        "test1",
        11321,
        "677893d3db633645aedbb0a1d838e046afbad53c9817f66b3e5977c24fd0065d",  # pragma: allowlist secret
    ),
    "cs-fleurs.xtts-test2": (
        "xtts",
        "test2",
        17226,
        "bdee4723f79c3485e03b5476ec90215c0b24dd3dfc285f65a09491d8f705b9c8",  # pragma: allowlist secret
    ),
}
# The source rows carry the user prompt of profiles.json (its prompt_override applies the same text to the request).
USER_PROMPT = "Produce a verbatim transcript of the audio."

# ISO 639-3 code -> (display name, ISO 639-1 code). Metadata names languages as "<matrix>-<embedded>" in ISO 639-3;
# the ISO 639-1 code of the matrix language goes to the scorer's number and text normalization (None: no such code).
LANGUAGES = {
    "ara": ("Arabic", "ar"),
    "aze": ("Azerbaijani", "az"),
    "ben": ("Bengali", "bn"),
    "bul": ("Bulgarian", "bg"),
    "cat": ("Catalan", "ca"),
    "ceb": ("Cebuano", None),
    "ces": ("Czech", "cs"),
    "cmn": ("Mandarin Chinese", "zh"),
    "cym": ("Welsh", "cy"),
    "deu": ("German", "de"),
    "ell": ("Greek", "el"),
    "eng": ("English", "en"),
    "fin": ("Finnish", "fi"),
    "fra": ("French", "fr"),
    "guj": ("Gujarati", "gu"),
    "heb": ("Hebrew", "he"),
    "hin": ("Hindi", "hi"),
    "hun": ("Hungarian", "hu"),
    "ind": ("Indonesian", "id"),
    "isl": ("Icelandic", "is"),
    "ita": ("Italian", "it"),
    "jav": ("Javanese", "jv"),
    "jpn": ("Japanese", "ja"),
    "kan": ("Kannada", "kn"),
    "kaz": ("Kazakh", "kk"),
    "khm": ("Khmer", "km"),
    "kir": ("Kyrgyz", "ky"),
    "kor": ("Korean", "ko"),
    "lao": ("Lao", "lo"),
    "lav": ("Latvian", "lv"),
    "lug": ("Ganda", "lg"),
    "mal": ("Malayalam", "ml"),
    "mar": ("Marathi", "mr"),
    "mya": ("Burmese", "my"),
    "nld": ("Dutch", "nl"),
    "pan": ("Punjabi", "pa"),
    "pol": ("Polish", "pl"),
    "por": ("Portuguese", "pt"),
    "ron": ("Romanian", "ro"),
    "rus": ("Russian", "ru"),
    "slk": ("Slovak", "sk"),
    "spa": ("Spanish", "es"),
    "swe": ("Swedish", "sv"),
    "swh": ("Swahili", "sw"),
    "tam": ("Tamil", "ta"),
    "tel": ("Telugu", "te"),
    "tgk": ("Tajik", "tg"),
    "tgl": ("Tagalog", "tl"),
    "tha": ("Thai", "th"),
    "tur": ("Turkish", "tr"),
    "ukr": ("Ukrainian", "uk"),
    "urd": ("Urdu", "ur"),
    "uzb": ("Uzbek", "uz"),
    "vie": ("Vietnamese", "vi"),
    "yor": ("Yoruba", "yo"),
    "yue": ("Cantonese Chinese", None),
    "zlm": ("Malay", "ms"),
}
# Scripts without word boundaries are scored per character by the mixed error rate; those with combining marks (vowel
# signs, tone marks) are segmented by grapheme cluster instead of by code point, in the matrix or the embedded language.
GRAPHEME_LANGUAGES = frozenset({"tha", "lao", "mya", "khm"})


def split_pair(language: str) -> tuple[str, str]:
    """Split a metadata ``language`` (``"<matrix>-<embedded>"``) into its ISO 639-3 codes."""
    matrix, _, embedded = language.replace("_", "-").partition("-")
    return matrix, embedded.split("-")[0]


def load_metadata(profile: str) -> list[dict]:
    """Read one test set's pinned ``metadata.jsonl``, checking its hash, row count and id/path uniqueness."""
    hf_dir, split, expected_rows, expected_sha256 = SUBSETS[profile]
    path = Path(
        hf_hub_download(HF_REPO_ID, f"{hf_dir}/{split}/metadata.jsonl", repo_type="dataset", revision=HF_REVISION)
    )
    if (digest := audio_manifest.sha256_file(path)) != expected_sha256:
        raise ValueError(f"CS-FLEURS {profile} metadata hash drift: expected {expected_sha256}, found {digest}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"CS-FLEURS {profile} row-count drift: expected {expected_rows}, found {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"CS-FLEURS {profile} contains duplicate row ids")
    if len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError(f"CS-FLEURS {profile} contains duplicate audio paths")
    return rows


def clip_path(row: dict) -> str:
    """Path of a clip below its test set's directory (some metadata paths contain a doubled slash)."""
    return posixpath.normpath(row["file_name"])


def format_source_row(row: dict, profile: str, audio_root: str) -> dict:
    """Build the source row of one clip; ``audio_root`` is the directory that holds ``<hf_dir>/<split>/audio``."""
    hf_dir, split, expected_rows, sha256 = SUBSETS[profile]
    audio_path = f"{audio_root}/{hf_dir}/{split}/{clip_path(row)}"
    duration = float(row["duration"])
    language = row["language"]
    matrix, embedded = split_pair(language)
    extra_fields = {
        "src_text": row["text"],
        "src_lang": LANGUAGES[matrix][1],
        "matrix_lang": matrix,
        "matrix_lang_name": LANGUAGES[matrix][0],
        "embedded_lang": embedded,
        "embedded_lang_name": LANGUAGES[embedded][0] if embedded else "",
        "lang_pair": language,
        # Mixed error rate: characters for scripts without word boundaries, words for the rest, in one utterance.
        "use_mer": True,
        "mer_grapheme": matrix in GRAPHEME_LANGUAGES or embedded in GRAPHEME_LANGUAGES,
        "speaker": row.get("speaker"),
    }
    if "fluency" in row:
        extra_fields["fluency"] = row["fluency"]
    return {
        "id": row["id"],
        "expected_answer": row["text"],
        "audio_path": audio_path,
        "duration": duration,
        "messages": [{"role": "user", "content": USER_PROMPT, "audio": {"path": audio_path, "duration": duration}}],
        "subset_for_metrics": language,
        "task_type": "Multilingual-ASR",
        "extra_fields": extra_fields,
        "source_provenance": {
            "dataset": HF_REPO_ID,
            "dataset_revision": HF_REVISION,
            "metadata_sha256": sha256,
            "expected_rows": expected_rows,
            "subset": profile.removeprefix("cs-fleurs."),
        },
    }


def download_audio(profile: str, rows: list[dict], capped: bool) -> None:
    """Download the clips of ``rows`` (or the whole test set) unchanged into ``data/raw``; resumes a partial run."""
    hf_dir, split, _, _ = SUBSETS[profile]
    patterns = [f"{hf_dir}/{split}/{clip_path(row)}" for row in rows] if capped else [f"{hf_dir}/{split}/audio/**"]
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR / "raw"),
        allow_patterns=patterns,
        revision=HF_REVISION,
    )
    missing = [clip_path(row) for row in rows if not (DATA_DIR / "raw" / hf_dir / split / clip_path(row)).is_file()]
    if missing:
        raise FileNotFoundError(
            f"CS-FLEURS {profile}: {len(missing)} clips missing after download, e.g. {missing[:3]}"
        )


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Download CS-FLEURS and write the combined benchmark JSONL (one profile per test set).

    The full test sets are about 150 hours of 16 kHz audio (tens of GB).

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows of each test set and download only their clips
            (smoke runs).
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    row_audio_root = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/raw"
    profiles = audio_manifest.load_profiles(PROFILES)["profiles"]
    if set(profiles) != set(SUBSETS):
        raise ValueError(f"profiles.json and SUBSETS disagree: {sorted(profiles)} vs {sorted(SUBSETS)}")

    sources = {}
    for profile in profiles:
        rows = load_metadata(profile)[:max_samples_per_profile]
        download_audio(profile, rows, capped=max_samples_per_profile is not None)
        sources[profile] = [format_source_row(row, profile, row_audio_root) for row in rows]

    prepared = audio_manifest.profile_rows(PROFILES, sources)
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
