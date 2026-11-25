# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare ContextASR-Bench for Gym.

The pinned ``MrSupW/ContextASR-Bench`` revision has four test sets: ``ContextASR-Speech`` (single-speaker, entity
rich monologues) and ``ContextASR-Dialogue`` (multi-speaker movie dialogues), each in English and Mandarin. All
four are prepared by default; ``subsets`` and ``languages`` narrow them (the English speech set is the benchmark's
original setting). Every set is used in three conditions (profiles):
``contextless`` (no context), ``coarse`` (the domain label or movie title) and ``fine`` (also the entity list);
``modes`` narrows them.

Audio is streamed from the dataset's tar archives and only the needed clips are written, as WAV files under
``data/audio/<set>/<language>`` (the dataset layout). The full default is about 100 GB of audio.
"""

from __future__ import annotations

import json
import os
import tarfile
from bisect import bisect_right
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "contextasr_bench_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "MrSupW/ContextASR-Bench"
HF_REVISION = "7906fd98d1d093bf244df80e726e07fc8afed6a6"  # pragma: allowlist secret

SUBSETS = ("speech", "dialogue")
LANGUAGES = ("English", "Mandarin")
MODES = ("contextless", "coarse", "fine")

# (subset, language) -> pinned metadata, audio archives and profile name prefix.
# The SHA-256 values are the digests of the metadata files at HF_REVISION.
SOURCES = {
    ("speech", "English"): {
        "jsonl_sha256": "b6bcb3de39aebe36328ab03d7d4d865899712c2dc375b6cd2b9ce156b6a952e4",
        "rows": 15326,
        "tars": 8,
        "profile": "contextasr-bench",
    },
    ("speech", "Mandarin"): {
        "jsonl_sha256": "229b2eeb82371fb2885f1a3e48cddebc9f8768399ad1149ce66b7212987209e9",
        "rows": 15498,
        "tars": 8,
        "profile": "contextasr-bench-mandarin",
    },
    ("dialogue", "English"): {
        "jsonl_sha256": "4bbf64387d1c581df2c7ab5db9af4461e1112ee489377b67084c9b40cb6d45e8",
        "rows": 5273,
        "tars": 6,
        "profile": "contextasr-bench-dialogue",
    },
    ("dialogue", "Mandarin"): {
        "jsonl_sha256": "10a1ff17678a151d45dc223ef24b106b4c0fbe92c7aaf418b073dab762046c98",
        "rows": 5232,
        "tars": 6,
        "profile": "contextasr-bench-dialogue-mandarin",
    },
}

TASK = "Transcribe the {language} audio into text, ensuring all punctuation marks are included."
COARSE_CONTEXT = {
    "speech": "This audio belongs to the {label} field",
    "dialogue": "This audio is a dialogue about the movie {label}",
}
ENTITIES = " and may contain the following words or phrases: {entity_list}"


def select_subsets(subsets) -> tuple[str, ...]:
    return audio_manifest.select(subsets, SUBSETS, "ContextASR-Bench subsets", ignore_case=True)


def select_languages(languages) -> tuple[str, ...]:
    return audio_manifest.select(languages, LANGUAGES, "ContextASR-Bench languages", ignore_case=True)


def select_modes(modes) -> tuple[str, ...]:
    return audio_manifest.select(modes, MODES, "ContextASR-Bench modes", ignore_case=True)


def metadata_name(subset: str, language: str) -> str:
    return f"ContextASR-{subset.capitalize()}_{language}.jsonl"


def load_metadata(subset: str, language: str, max_samples: int | None = None) -> list[dict]:
    """Read one pinned test set, check it against its hash, row count and identifiers, keep the first rows."""
    source = SOURCES[subset, language]
    path = Path(
        hf_hub_download(HF_REPO_ID, metadata_name(subset, language), repo_type="dataset", revision=HF_REVISION)
    )
    if (actual := audio_manifest.sha256_file(path)) != source["jsonl_sha256"]:
        raise ValueError(
            f"ContextASR-Bench {subset}/{language} metadata hash drift: expected {source['jsonl_sha256']}, found {actual}"
        )
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != source["rows"]:
        raise ValueError(
            f"ContextASR-Bench {subset}/{language} row-count drift: expected {source['rows']}, found {len(rows)}"
        )
    if len({row["uniq_id"] for row in rows}) != len(rows) or len({row["audio"] for row in rows}) != len(rows):
        raise ValueError(f"ContextASR-Bench {subset}/{language} has duplicate ids or audio paths")
    return rows[:max_samples]


def tar_paths(subset: str, language: str) -> list[str]:
    stem = f"ContextASR-{subset.capitalize()}_{language}"
    return [
        f"audio/ContextASR-{subset.capitalize()}/{language}/{stem}_{index}.tar"
        for index in range(1, SOURCES[subset, language]["tars"] + 1)
    ]


def _member_key(name: str) -> str:
    return name.removeprefix("./")


def _first_member(filesystem, path: str) -> str:
    with filesystem.open(path, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        return _member_key(archive.next().name)


def _stream_tar(filesystem, path: str, pending: dict[str, Path], stop: set[str]) -> None:
    """Extract the ``pending`` clips found in one archive; return early once all of ``stop`` are written."""
    with filesystem.open(path, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            key = _member_key(member.name)
            if key in pending and member.isfile():
                destination = pending.pop(key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged = destination.with_name(destination.name + ".tmp")
                with archive.extractfile(member) as source, staged.open("wb") as sink:
                    while chunk := source.read(1 << 20):
                        sink.write(chunk)
                os.replace(staged, destination)
                stop.discard(key)
                if not stop:
                    return


def extract_audio(subset: str, language: str, rows: list[dict], data_dir: Path) -> None:
    """Write the clips of ``rows`` that are not on disk yet, reading as little of the archives as possible.

    The archives hold name-sorted clips, so the first clip name of each archive tells which one holds a wanted
    clip; archives are read only up to their last wanted clip. Clips not found there are looked up in all archives.
    """
    pending = {
        Path(row["audio"]).name: data_dir / row["audio"] for row in rows if not (data_dir / row["audio"]).is_file()
    }
    if not pending:
        return
    filesystem = HfFileSystem()
    paths = [f"datasets/{HF_REPO_ID}@{HF_REVISION}/{path}" for path in tar_paths(subset, language)]
    firsts = [_first_member(filesystem, path) for path in paths]
    if firsts == sorted(firsts):
        assigned: dict[int, set[str]] = {}
        for key in pending:
            assigned.setdefault(max(bisect_right(firsts, key) - 1, 0), set()).add(key)
        for index, keys in sorted(assigned.items()):
            _stream_tar(filesystem, paths[index], pending, keys)
    for path in paths:
        if not pending:
            break
        _stream_tar(filesystem, path, pending, set(pending))
    if pending:
        raise FileNotFoundError(
            f"{len(pending)} ContextASR-Bench {subset}/{language} clips are missing from the archives; "
            f"first 10: {sorted(pending)[:10]}"
        )


def check_audio(rows: list[dict], data_dir: Path) -> None:
    import soundfile as sf

    for row in rows:
        path = data_dir / row["audio"]
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError) as error:
            raise ValueError(f"ContextASR-Bench audio is missing or unreadable: {path}") from error
        if info.frames <= 0:
            raise ValueError(f"ContextASR-Bench audio is empty: {path}")


def format_source_row(row: dict, mode: str, subset: str, language: str, audio_root: str) -> dict:
    """Skills-style source row of one clip in one condition; ``audio_root`` is the directory holding ``audio/``."""
    audio_path = f"{audio_root}/{row['audio']}"
    label = row["domain_label"] if subset == "speech" else row["movie_name"]
    task = TASK.format(language=language)
    if mode == "contextless":
        prompt = task
    elif mode == "coarse":
        prompt = f"{COARSE_CONTEXT[subset].format(label=label)}. {task}"
    elif mode == "fine":
        entities = ENTITIES.format(entity_list=", ".join(row["entity_list"]))
        prompt = f"{COARSE_CONTEXT[subset].format(label=label)}{entities}. {task}"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    source = {
        "messages": [
            {"role": "user", "content": prompt, "audio": {"path": audio_path, "duration": float(row["duration"])}}
        ],
        "expected_answer": row["text"],
        "entity_list": row["entity_list"],
        "uniq_id": row["uniq_id"],
        "duration": float(row["duration"]),
        "audio_filepath": audio_path,
        "source_provenance": {
            "dataset": HF_REPO_ID,
            "dataset_revision": HF_REVISION,
            "metadata_sha256": SOURCES[subset, language]["jsonl_sha256"],
            "expected_rows": SOURCES[subset, language]["rows"],
        },
    }
    if subset == "speech":
        source["domain_label"] = source["subset_for_metrics"] = row["domain_label"]
    else:
        source["movie_name"] = row["movie_name"]
    return source


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    subsets: str | list[str] | tuple[str, ...] | None = None,
    languages: str | list[str] | tuple[str, ...] | None = None,
    modes: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download ContextASR-Bench and write the benchmark JSONL.

    Args:
        subsets: ``speech`` and/or ``dialogue`` (default: both).
        languages: ``English`` and/or ``Mandarin`` (default: both).
        modes: ``contextless``, ``coarse`` and/or ``fine`` (default: all three).
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N clips of each test set (smoke runs); a positive integer.
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    expected = [f"{source['profile']}.{mode}" for source in SOURCES.values() for mode in MODES]
    if list(audio_manifest.load_profiles(PROFILES)["profiles"]) != expected:
        raise ValueError("profiles.json and SOURCES disagree on the profile names")
    chosen_subsets, chosen_languages, chosen_modes = (
        select_subsets(subsets),
        select_languages(languages),
        select_modes(modes),
    )
    sets = [(s, lang) for s in chosen_subsets for lang in chosen_languages]
    metadata = {key: load_metadata(*key, max_samples_per_profile) for key in sets}
    for (subset, language), rows in metadata.items():
        extract_audio(subset, language, rows, DATA_DIR)
        check_audio(rows, DATA_DIR)

    prefix = audio_manifest.audio_prefix(DATA_DIR, audio_root)
    prepared = audio_manifest.profile_rows(
        PROFILES,
        {
            f"{SOURCES[subset, language]['profile']}.{mode}": [
                format_source_row(row, mode, subset, language, prefix) for row in rows
            ]
            for (subset, language), rows in metadata.items()
            for mode in chosen_modes
        },
    )
    options = {
        "subsets": None if chosen_subsets == SUBSETS else list(chosen_subsets),
        "languages": None if chosen_languages == LANGUAGES else list(chosen_languages),
        "modes": None if chosen_modes == MODES else list(chosen_modes),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
