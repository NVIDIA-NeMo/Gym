# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the VoxLingua107 development set for Gym.

The public development archive (1,609 clips in 33 languages) is downloaded once and checked against its pinned
SHA-256 and per-language census. By default all 33 languages are prepared; ``languages`` narrows the set, for
example to ``LANGUAGES_24`` (a 24-language subset of 1,259 clips). The prompt lists the
selected languages as the allowed answers. WAV files go to ``data/audio/<language>/``.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path, PurePosixPath

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "voxlingua107_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

SOURCE_URL = "https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/dev.zip"
SOURCE_SHA256 = (
    "54c352d2ecbcbc03a4bb24151750a287573a5c2738636387a2240700bb89d2d2"  # pragma: allowlist secret (archive checksum)
)
# Clips per language in the 33-language archive.
ARCHIVE_LABEL_COUNTS = {
    "ar": 100,
    "az": 34,
    "da": 99,
    "de": 82,
    "el": 5,
    "en": 80,
    "es": 55,
    "et": 85,
    "fa": 100,
    "fi": 93,
    "fr": 100,
    "hr": 4,
    "hu": 4,
    "hy": 100,
    "is": 26,
    "it": 49,
    "ja": 42,
    "lt": 13,
    "lv": 95,
    "mk": 4,
    "nl": 100,
    "nn": 3,
    "no": 24,
    "pl": 2,
    "pt": 1,
    "ru": 29,
    "sl": 9,
    "sr": 7,
    "sv": 100,
    "tr": 72,
    "uk": 17,
    "ur": 52,
    "zh": 23,
}
EXPECTED_ROWS = 1609
# A 24-language subset (1,259 clips): the archive languages minus az fa hy is mk nn no sr ur.
LANGUAGES_24 = tuple("ar da de el en es et fi fr hr hu it ja lt lv nl pl pt ru sl sv tr uk zh".split())


def select_languages(languages: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize the ``languages`` argument (a list or a comma-separated string; default all) to sorted codes."""
    return audio_manifest.select(languages, sorted(ARCHIVE_LABEL_COUNTS), "VoxLingua107 languages")


def language_prompt(languages: tuple[str, ...]) -> str:
    """The user prompt of the source rows: the selected language codes are the allowed answers."""
    return (
        "Identify the language spoken in this audio. Respond with exactly one of the following language labels "
        "and nothing else: " + ", ".join(languages) + "."
    )


def list_clips(archive: zipfile.ZipFile, languages: tuple[str, ...]) -> list[tuple[zipfile.ZipInfo, str, str]]:
    """Check the archive against the released census; return ``(member, label, filename)`` of the selected clips."""
    counts = dict.fromkeys(ARCHIVE_LABEL_COUNTS, 0)
    clips = []
    for member in archive.infolist():
        if member.is_dir():
            continue
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or not path.name.lower().endswith(".wav"):
            raise ValueError(f"Unexpected VoxLingua107 archive member: {member.filename!r}")
        label, filename = path.parts
        if label not in counts:
            raise ValueError(f"Unexpected VoxLingua107 archive member: {member.filename!r}")
        counts[label] += 1
        clips.append((member, label, filename))
    if counts != ARCHIVE_LABEL_COUNTS:
        raise ValueError(f"VoxLingua107 archive census drift: expected {ARCHIVE_LABEL_COUNTS}, found {counts}")
    if len(clips) != EXPECTED_ROWS:
        raise ValueError(f"VoxLingua107 row-count drift: expected {EXPECTED_ROWS}, found {len(clips)}")
    clips = [clip for clip in clips if clip[1] in languages]
    return sorted(clips, key=lambda clip: (clip[1], clip[2]))


def extract_audio(archive: zipfile.ZipFile, clips: list[tuple[zipfile.ZipInfo, str, str]], audio_dir: Path) -> None:
    for member, label, filename in clips:
        destination = audio_dir / label / filename
        if destination.is_file() and destination.stat().st_size == member.file_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(destination.name + ".tmp")
        with archive.open(member) as source, staged.open("wb") as sink:
            while chunk := source.read(1 << 20):
                sink.write(chunk)
        os.replace(staged, destination)


def format_source_row(label: str, filename: str, duration: float, audio_dir: str, prompt: str) -> dict:
    audio_path = f"{audio_dir}/{label}/{filename}"
    return {
        "id": f"{label}-{Path(filename).stem}",
        "expected_answer": label,
        "messages": [{"role": "user", "content": prompt, "audio": {"path": audio_path, "duration": duration}}],
        "audio_path": audio_path,
        "duration": duration,
        "subset_for_metrics": label,
        "task_type": "VoxLingua107-LID",
        "extra_fields": {
            "language_label": label,
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "source_split": "dev",
        },
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    languages: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download VoxLingua107 dev and write the benchmark JSONL.

    Args:
        languages: Archive language codes to keep, as a list or a comma-separated string (default: all 33).
            The prompt lists exactly these codes.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N clips, in language and file-name order (smoke runs).
    """
    import soundfile

    selected = select_languages(languages)
    prompt = language_prompt(selected)
    archive_path = audio_manifest.download(SOURCE_URL, DATA_DIR / "source" / "dev.zip", SOURCE_SHA256)
    audio_dir = DATA_DIR / "audio"
    with zipfile.ZipFile(archive_path) as archive:
        clips = list_clips(archive, selected)[:max_samples_per_profile]
        extract_audio(archive, clips, audio_dir)

    prefix = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    durations = {
        (label, filename): float(soundfile.info(audio_dir / label / filename).duration) for _, label, filename in clips
    }
    prepared = audio_manifest.profile_rows(
        PROFILES,
        [
            format_source_row(label, filename, durations[label, filename], prefix, prompt)
            for _, label, filename in clips
        ],
    )
    options = {
        "languages": None if selected == select_languages(None) else list(selected),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
