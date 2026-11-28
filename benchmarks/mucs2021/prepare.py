# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the MUCS 2021 Subtask 2 blind test (Hindi-English and Bengali-English) for Gym.

The public OpenSLR SLR104 archive (about 1 GB) is downloaded once and checked against its pinned SHA-256, and every
Kaldi metadata file against its pinned digest and census. Each segment listed in ``segments`` is cut from its
recording without resampling and written as a 16 kHz mono 16-bit WAV to ``data/<language>/audio/``. The scorer's
word maps (CC BY-SA 4.0) are installed from the same archive into Gym's cache directory with their pinned SHA-256.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

from benchmarks import audio_manifest
from resources_servers.audio_eval import mucs_assets


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mucs2021_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__), Path(mucs_assets.__file__))

SOURCE_URL = mucs_assets.ARCHIVE_URL
SOURCE_SHA256 = mucs_assets.ARCHIVE_SHA256
ARCHIVE_ROOT = "subtask2_blindtest_wReadme"
SAMPLE_RATE = 16000
# The prompt of the source rows; each profile's prompt_override replaces it in the request.
PROMPT = "Transcribe the following audio."


@dataclass(frozen=True)
class Language:
    name: str  # profile suffix and ``subset_for_metrics``
    source_dir: str  # directory inside the archive
    rows: int
    recordings: int
    empty_references: int  # skipped by the official scorer
    word_map: str  # transliteration map of the release
    metadata_sha256: dict[str, str]


LANGUAGES = {
    "hindi-english": Language(
        name="hindi-english",
        source_dir="hindi",
        rows=4034,
        recordings=35,
        empty_references=8,
        word_map="hin2engwords.txt",
        metadata_sha256={
            "text": "83eab2b9420e252474964e2b0e916412a823b0754d04910d808a93a8832a69fe",
            "segments": "c1c7891dbe65af9013a28a19a59498921d0e31e8a0516b28f113724cb7189d3c",
            "wav.scp": "36a7be055cc3595ccc152d8529a6ee067aa69537b77ad0562a94e539dba7284d",
            "utt2spk": "861cdf50f2c884c4c35bcd775af90a5c288ff026da353e39e6b8480e584114ef",
        },
    ),
    "bengali-english": Language(
        name="bengali-english",
        source_dir="bengali",
        rows=3130,
        recordings=32,
        empty_references=4,
        word_map="ben2engwords.txt",
        metadata_sha256={
            "text": "dbb1308dd8f2a598cf9334a28dfea6ed498871e7ff4f60464989dd5989cd5326",
            "segments": "6ca79ae4155156b472bf2371e91583692480ba69f2c59211e6a204ff7fc64c50",
            "wav.scp": "4cb4a73e736ae5f7611d639fe10236efb1ee925e98cf0b7d61b6165342759bb7",
            "utt2spk": "95a09f27cfc8ced6c735b3d0a443cb2a17478c9c96dc2c00156fc2c318e37427",
        },
    ),
}
# The release has no cross-language aggregate: each language is scored on its own.


def select_languages(languages: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize the ``languages`` argument (a list or a comma-separated string; default all) to profile order."""
    return audio_manifest.select(languages, LANGUAGES, "MUCS languages")


def _member_path(member: tarfile.TarInfo) -> PurePosixPath | None:
    """The member's path below the archive root; None for directories and anything outside the expected tree."""
    path = PurePosixPath(member.name)
    if not member.isfile() or path.is_absolute() or ".." in path.parts or path.parts[:1] != (ARCHIVE_ROOT,):
        return None
    return PurePosixPath(*path.parts[1:])


def read_metadata(archive_path: Path) -> tuple[dict[str, dict[str, bytes]], dict[str, bytes]]:
    """One streaming pass: the Kaldi metadata per source directory and the scorer's ``inference/`` files."""
    metadata: dict[str, dict[str, bytes]] = {language.source_dir: {} for language in LANGUAGES.values()}
    inference: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = _member_path(member)
            if path is None:
                continue
            if len(path.parts) == 3 and path.parts[0] in metadata and path.parts[1] == "files":
                metadata[path.parts[0]][path.name] = archive.extractfile(member).read()
            elif len(path.parts) == 2 and path.parts[0] == "inference" and path.name in mucs_assets.ASSET_SHA256:
                inference[path.name] = archive.extractfile(member).read()
    return metadata, inference


def install_assets(inference: dict[str, bytes]) -> None:
    """Write the scorer's word maps from the archive to Gym's cache, verifying their pinned SHA-256."""
    directory = mucs_assets.assets_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in mucs_assets.ASSET_SHA256.items():
        if name not in inference or hashlib.sha256(inference[name]).hexdigest() != expected:
            raise ValueError(f"MUCS scoring asset {name} is missing from the archive or does not match its SHA-256")
        if not mucs_assets._is_valid(directory / name, name):
            staged = directory / f"{name}.{os.getpid()}.partial"
            staged.write_bytes(inference[name])
            os.replace(staged, directory / name)


def parse_language(language: Language, files: dict[str, bytes]) -> list[dict]:
    """Check the language's metadata against its pinned digests and census; return the segments in file order."""
    for name, expected in language.metadata_sha256.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError(f"MUCS {language.source_dir}/files/{name} is missing or differs from the pinned release")
    references: dict[str, str] = {}
    for number, line in enumerate(files["text"].decode("utf-8").splitlines(), 1):
        utterance_id, _, text = line.partition(" ")
        if not utterance_id or utterance_id in references:
            raise ValueError(f"Invalid or duplicate MUCS utterance ID at {language.source_dir}/text:{number}")
        references[utterance_id] = text
    speakers = dict(line.split() for line in files["utt2spk"].decode("utf-8").splitlines())
    recordings = dict(line.split() for line in files["wav.scp"].decode("utf-8").splitlines())
    segments = []
    for number, line in enumerate(files["segments"].decode("utf-8").splitlines(), 1):
        utterance_id, recording_id, start, end = line.split()
        if utterance_id not in references or recordings.get(recording_id) != f"{recording_id}.wav":
            raise ValueError(f"Unknown MUCS utterance or recording at {language.source_dir}/segments:{number}")
        if not utterance_id.replace("_", "").isalnum() or not Decimal(start) >= 0 or not Decimal(end) > Decimal(start):
            raise ValueError(f"Invalid MUCS segment {utterance_id!r} at {language.source_dir}/segments:{number}")
        segments.append(
            {
                "id": utterance_id,
                "recording": recording_id,
                "start": start,
                "end": end,
                "text": references[utterance_id],
                "speaker": speakers[utterance_id],
            }
        )
    empty = sum(not text.strip() for text in references.values())
    if (len(segments), len(references), len(speakers), len(recordings), empty) != (
        language.rows,
        language.rows,
        language.rows,
        language.recordings,
        language.empty_references,
    ):
        raise ValueError(f"MUCS {language.source_dir} census differs from the released one")
    return segments


def _frame(seconds: str) -> int:
    exact = Decimal(seconds) * SAMPLE_RATE
    if exact != exact.to_integral_value():
        raise ValueError(f"MUCS segment boundary {seconds}s is not sample-aligned")
    return int(exact)


def cut_segments(recording: bytes, segments: list[dict], audio_dir: Path) -> None:
    """Write each segment of one recording as a 16-bit mono WAV, without resampling."""
    import soundfile

    with soundfile.SoundFile(io.BytesIO(recording)) as source:
        if (source.samplerate, source.channels, source.format, source.subtype) != (
            SAMPLE_RATE,
            1,
            "WAV",
            "PCM_16",
        ):
            raise ValueError(f"Unexpected MUCS recording format: {source.samplerate} Hz, {source.subtype}")
        for segment in segments:
            start, end = _frame(segment["start"]), _frame(segment["end"])
            if end > source.frames:
                raise ValueError(f"MUCS segment {segment['id']} exceeds its recording")
            source.seek(start)
            samples = source.read(end - start, dtype="int16", always_2d=False)
            staged = audio_dir / f"{segment['id']}.wav.tmp"
            soundfile.write(staged, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            os.replace(staged, audio_dir / f"{segment['id']}.wav")


def write_audio(archive_path: Path, selected: dict[str, list[dict]]) -> None:
    """Cut the selected segments out of the archive's recordings in one streaming pass."""
    pending: dict[tuple[str, str], tuple[list[dict], Path]] = {}
    for language, segments in selected.items():
        audio_dir = DATA_DIR / language / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        by_recording: dict[str, list[dict]] = {}
        for segment in segments:
            by_recording.setdefault(segment["recording"], []).append(segment)
        for recording, group in by_recording.items():
            if not all((audio_dir / f"{segment['id']}.wav").is_file() for segment in group):
                pending[LANGUAGES[language].source_dir, recording] = (group, audio_dir)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = _member_path(member)
            if path is None or len(path.parts) != 2 or (path.parts[0], path.stem) not in pending:
                continue
            segments, audio_dir = pending.pop((path.parts[0], path.stem))
            cut_segments(archive.extractfile(member).read(), segments, audio_dir)
            if not pending:
                return
    if pending:
        raise ValueError(f"MUCS recordings missing from the archive: {sorted(pending)}")


def format_source_row(language: Language, segment: dict, audio_dir: str) -> dict:
    """The source row of one segment. ``audio_dir`` is the directory the row uses for the language's audio."""
    audio_path = f"{audio_dir}/{segment['id']}.wav"
    duration = float((_frame(segment["end"]) - _frame(segment["start"])) / SAMPLE_RATE)
    return {
        "audio_path": audio_path,
        "duration": duration,
        "expected_answer": segment["text"],
        "extra_fields": {
            "mucs_language": language.name,
            "official_word_mapping": language.word_map,
            "reference_source": f"{ARCHIVE_ROOT}/{language.source_dir}/files/text",
            "segment_end": segment["end"],
            "segment_start": segment["start"],
            "segments_source": f"{ARCHIVE_ROOT}/{language.source_dir}/files/segments",
            "source_archive_member": f"{ARCHIVE_ROOT}/{language.source_dir}/{segment['recording']}.wav",
            "source_recording_id": segment["recording"],
            "source_sha256": SOURCE_SHA256,
            "source_url": SOURCE_URL,
            "speaker_id": segment["speaker"],
        },
        "id": segment["id"],
        "messages": [
            {"audio": {"duration": duration, "path": audio_path}, "content": PROMPT, "role": "user"},
        ],
        "subset_for_metrics": language.name,
        "task_type": "Code-Switched-ASR",
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    languages: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download the MUCS 2021 blind test and write the benchmark JSONL.

    Args:
        languages: ``hindi-english`` and/or ``bengali-english`` (a list or a comma-separated string; default both,
            each is scored separately).
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N segments of each language (smoke runs).
    """
    if max_samples_per_profile is not None and max_samples_per_profile < 1:
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    selected = select_languages(languages)
    archive_path = audio_manifest.download(SOURCE_URL, DATA_DIR / "source" / Path(SOURCE_URL).name, SOURCE_SHA256)
    metadata, inference = read_metadata(archive_path)
    install_assets(inference)
    segments = {
        name: parse_language(LANGUAGES[name], metadata[LANGUAGES[name].source_dir])[:max_samples_per_profile]
        for name in selected
    }
    write_audio(archive_path, segments)

    prefix = audio_manifest.audio_prefix(DATA_DIR, audio_root)
    prepared = audio_manifest.profile_rows(
        PROFILES,
        {
            f"mucs2021.{name}": [
                format_source_row(LANGUAGES[name], segment, f"{prefix}/{name}/audio") for segment in segments[name]
            ]
            for name in selected
        },
    )
    options = {
        "languages": None if selected == select_languages(None) else list(selected),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
