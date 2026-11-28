# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the public Preference-ASR benchmark for Gym.

The pinned ``nvidia/preference-asr-bench`` release has 2,822 instruction rows over 1,997 clips of four corpora
(AMI, Common Voice, LibriSpeech, VoxPopuli). Each row is used in two conditions (profiles): ``default`` (a plain
transcription prompt) and ``instructed`` (the row's preference instruction). All rows and both conditions are
prepared by default; ``corpora``, ``categories`` and ``variants`` narrow them. The release ships the audio of AMI,
LibriSpeech and VoxPopuli; Common Voice audio is read from the pinned Open ASR leaderboard parquet files. Clips are
converted to 16 kHz mono 16-bit WAV files under ``data/audio/<corpus>``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "preference_asr_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_RELEASE_REPO = "nvidia/preference-asr-bench"
HF_RELEASE_REVISION = "038bd44722b25a17abc9ad53265c6a96be8195ab"
MANIFEST_FILE = "data/preference_asr_testset.jsonl"
MANIFEST_SHA256 = "134214055ef2b6f4ff88b5526072036be17a59e6a238472fd1f71d9b0c81c2d9"
# The code companion repository publishes the same manifest (same SHA-256).
GITHUB_REPO = "nithinraok/preference-asr-bench"
GITHUB_REVISION = "efa5cae9aecb56ce28aeea6105425ded654acbdd"
ESB_REPO = "hf-audio/open-asr-leaderboard"
ESB_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
ANNOTATION_LICENSE = "CC-BY-NC-4.0"

CORPORA = ("ami", "common_voice", "librispeech", "voxpopuli")
CATEGORIES = ("case", "disfluencies", "entities", "normalization", "standard")
VARIANTS = ("default", "instructed")
PROFILE_PREFIX = "preference-asr-public."

# Census of the release: (rows, unique clips, total row duration in seconds, rows with an empty preference text).
EXPECTED_ROWS = 2822
EXPECTED_CLIPS = 1997
EXPECTED_DURATION_SECONDS = 21600.068
EXPECTED_EMPTY_PREFERENCE_TEXT = 5
EXPECTED_CORPUS_ROWS = {"ami": 632, "common_voice": 477, "librispeech": 749, "voxpopuli": 964}
EXPECTED_CATEGORY_ROWS = {"case": 1113, "disfluencies": 516, "entities": 906, "normalization": 228, "standard": 59}
EXPECTED_DIRECTION_ROWS = {None: 2078, "itn": 114, "keep": 258, "remove": 258, "tn": 114}

SAMPLE_RATE = 16000
MIN_SAMPLES = 320
DEFAULT_PROMPT = "Transcribe the English audio into text"
PROMPT_POLICY = "preference-asr-recording-prefix-v1"
SAFE_FILENAME = re.compile(r"[A-Za-z0-9._-]+")
CASE_TARGETS = {"lower_case", "upper_case", "proper_punctuated_capitalized_format"}
REQUIRED_FIELDS = {
    "audio_filepath",
    "duration",
    "text",
    "preference_text",
    "instruction",
    "preference_type",
    "sub_category",
    "dataset",
}


def rewrite_instruction(instruction: str) -> str:
    """Avoid two openings that made models stop immediately, without changing the task or any preference suffix."""
    if instruction == "Transcribe the speech in the input English audio":
        return "Transcribe this English recording."
    prefix = "Transcribe the speech in the input audio."
    if instruction.startswith(prefix):
        return "Transcribe this recording." + instruction[len(prefix) :]
    return instruction


def audio_parts(row: dict) -> tuple[str, str]:
    """The corpus and file name of a row's audio, checked to be ``data/<corpus>/<file>``."""
    path = PurePosixPath(str(row.get("audio_filepath", "")))
    if path.is_absolute() or len(path.parts) != 3 or path.parts[0] != "data":
        raise ValueError(f"Unsafe Preference-ASR audio path: {path}")
    corpus, filename = path.parts[1:]
    if corpus != row.get("dataset") or SAFE_FILENAME.fullmatch(filename) is None:
        raise ValueError(f"Preference-ASR audio path/corpus mismatch: {path}, dataset={row.get('dataset')!r}")
    return corpus, filename


def validate_manifest(rows: list[dict]) -> None:
    """Check every row and the whole manifest against the census of the pinned release."""
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Preference-ASR row-count drift: expected {EXPECTED_ROWS}, found {len(rows)}")
    clips: dict[str, float] = {}
    corpora, categories, directions = Counter(), Counter(), Counter()
    empty = 0
    total_duration = 0.0
    for number, row in enumerate(rows, 1):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Preference-ASR row {number} is missing fields: {sorted(missing)}")
        corpus, _ = audio_parts(row)
        duration = row["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise ValueError(f"Preference-ASR row {number} has invalid duration: {duration!r}")
        if duration <= 0:
            raise ValueError(f"Preference-ASR row {number} has invalid duration: {duration!r}")
        for field in ("text", "instruction"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Preference-ASR row {number} has an empty {field}")
        if not isinstance(row["preference_text"], str):
            raise TypeError(f"Preference-ASR row {number} has a non-string preference_text")
        if not row["preference_text"].strip():
            if row["preference_type"] != "disfluencies" or row.get("direction") != "remove":
                raise ValueError(f"Preference-ASR row {number} has an unexpected empty preference_text")
            empty += 1
        if not isinstance(row["sub_category"], list) or not all(isinstance(v, str) and v for v in row["sub_category"]):
            raise ValueError(f"Preference-ASR row {number} has invalid sub_category")
        if abs(clips.setdefault(row["audio_filepath"], float(duration)) - float(duration)) > 1e-9:
            raise ValueError(f"Preference-ASR reused audio has conflicting durations: {row['audio_filepath']}")
        total_duration += float(duration)
        corpora[corpus] += 1
        categories[row["preference_type"]] += 1
        directions[row.get("direction")] += 1
    observed = (len(clips), dict(corpora), dict(categories), dict(directions), empty)
    expected = (
        EXPECTED_CLIPS,
        EXPECTED_CORPUS_ROWS,
        EXPECTED_CATEGORY_ROWS,
        EXPECTED_DIRECTION_ROWS,
        EXPECTED_EMPTY_PREFERENCE_TEXT,
    )
    if observed != expected or abs(total_duration - EXPECTED_DURATION_SECONDS) > 1e-6:
        raise ValueError(f"Preference-ASR released-census drift: expected {expected}, found {observed}")


def load_manifest() -> list[dict]:
    """Read the pinned manifest of the release, check its hash and census, and return all rows."""
    path = Path(hf_hub_download(HF_RELEASE_REPO, MANIFEST_FILE, repo_type="dataset", revision=HF_RELEASE_REVISION))
    if (actual := audio_manifest.sha256_file(path)) != MANIFEST_SHA256:
        raise ValueError(f"Preference-ASR manifest hash drift: expected {MANIFEST_SHA256}, found {actual}")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    validate_manifest(rows)
    return rows


def canonical_samples(path_or_bytes) -> np.ndarray:
    """Decode one clip to mono float32 at 16 kHz (polyphase resampling when the source rate differs)."""
    import io

    import soundfile as sf
    from scipy.signal import resample_poly

    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    samples, rate = sf.read(source, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0 or rate <= 0 or not np.isfinite(samples).all():
        raise ValueError(f"Preference-ASR source audio is invalid: shape={samples.shape}, rate={rate}")
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, int(rate) // divisor)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < MIN_SAMPLES or not np.isfinite(samples).all():
        raise ValueError(f"Preference-ASR canonical audio is too short or not finite: {samples.size} samples")
    return samples


def write_wav(path: Path, samples: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.partial")
    sf.write(staged, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    os.replace(staged, path)


def target_path(data_dir: Path, corpus: str, filename: str) -> Path:
    return data_dir / "audio" / corpus / f"{Path(filename).stem}.wav"


def check_wav(path: Path, duration: float) -> None:
    import soundfile as sf

    try:
        with sf.SoundFile(str(path)) as stream:
            header = (stream.samplerate, stream.channels, stream.subtype)
            seconds = stream.frames / stream.samplerate
            frames = stream.frames
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Preference-ASR audio is missing or unreadable: {path}") from error
    if header != (SAMPLE_RATE, 1, "PCM_16") or frames < MIN_SAMPLES:
        raise ValueError(f"Preference-ASR audio contract failed for {path}: {header}, frames={frames}")
    if abs(seconds - duration) > 0.11:
        raise ValueError(f"Preference-ASR audio duration drift for {path}: metadata={duration}, WAV={seconds}")


def fetch_release_clip(filepath: str, destination: Path) -> None:
    """Convert one clip shipped in the release (AMI, LibriSpeech, VoxPopuli) into ``destination``."""
    cached = hf_hub_download(HF_RELEASE_REPO, filepath, repo_type="dataset", revision=HF_RELEASE_REVISION)
    write_wav(destination, canonical_samples(cached))


def esb_clip_name(clip_id: str) -> str:
    """File name of an Open ASR leaderboard clip id, as the release manifest spells it."""
    safe = clip_id.replace("/", "_").replace("\\", "_")
    for extension in (".wav", ".flac", ".mp3", ".ogg", ".opus"):
        if safe.lower().endswith(extension):
            safe = safe[: -len(extension)]
            break
    return f"{safe}.flac"


def extract_common_voice(wanted: dict[str, Path]) -> None:
    """Write the Common Voice clips (``{manifest file name: destination}``) from the Open ASR leaderboard parquet files.

    Only the row groups that hold a wanted clip are read; when a clip occurs more than once, the first one is used.
    """
    import pyarrow.parquet as pq

    pending = dict(wanted)
    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"datasets/{ESB_REPO}@{ESB_REVISION}/common_voice/test-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Common Voice test shards in {ESB_REPO}@{ESB_REVISION}")
    for shard in shards:
        if not pending:
            break
        with filesystem.open(shard, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            names = [esb_clip_name(str(clip_id)) for clip_id in parquet.read(columns=["id"]).to_pydict()["id"]]
            start = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                hits = {}
                for offset in range(size):
                    if names[start + offset] in pending and names[start + offset] not in hits.values():
                        hits[offset] = names[start + offset]
                if hits:
                    audio = parquet.read_row_group(group, columns=["audio"]).column("audio")
                    for offset, name in hits.items():
                        write_wav(pending.pop(name), canonical_samples(audio[offset].as_py()["bytes"]))
                start += size
    if pending:
        raise FileNotFoundError(
            f"{len(pending)} Common Voice clips are missing from {ESB_REPO}: {sorted(pending)[:10]}"
        )


def materialize_audio(rows: list[dict], data_dir: Path, workers: int = 8) -> None:
    """Write every clip the ``rows`` use that is not on disk yet, then check all of them."""
    clips: dict[str, tuple[str, str, float]] = {}
    for row in rows:
        corpus, filename = audio_parts(row)
        clips[row["audio_filepath"]] = (corpus, filename, float(row["duration"]))
    todo = {fp: c for fp, c in clips.items() if not target_path(data_dir, c[0], c[1]).is_file()}
    release = [(fp, target_path(data_dir, c[0], c[1])) for fp, c in todo.items() if c[0] != "common_voice"]
    if release:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda item: fetch_release_clip(*item), release))
    common_voice = {c[1]: target_path(data_dir, c[0], c[1]) for c in todo.values() if c[0] == "common_voice"}
    if common_voice:
        extract_common_voice(common_voice)
    for corpus, filename, duration in clips.values():
        check_wav(target_path(data_dir, corpus, filename), duration)


def format_source_row(row: dict, row_number: int, variant: str, audio_dir: str) -> dict:
    """Source row (messages plus scoring fields) of one manifest row; ``row_number`` is its 1-based position in the manifest."""
    corpus, filename = audio_parts(row)
    audio_path = f"{audio_dir}/{corpus}/{Path(filename).stem}.wav"
    prompt = DEFAULT_PROMPT if variant == "default" else rewrite_instruction(row["instruction"].strip())
    target_case = ""
    if row["preference_type"] == "case":
        matches = [value for value in row["sub_category"] if value in CASE_TARGETS]
        if len(matches) != 1:
            raise ValueError(f"Preference-ASR case row {row_number} has ambiguous target: {row['sub_category']!r}")
        target_case = matches[0]
    return {
        "id": f"preference-asr-public-{variant}-{row_number:04d}",
        "messages": [
            {"role": "user", "content": prompt, "audio": {"path": audio_path, "duration": float(row["duration"])}}
        ],
        "expected_answer": row["preference_text"].strip(),
        "standard_reference": row["text"].strip(),
        "audio_filepath": audio_path,
        "audio_duration": float(row["duration"]),
        "preference_type": row["preference_type"],
        "sub_category": list(row["sub_category"]),
        "direction": row.get("direction", ""),
        "target_case": target_case,
        "preference_source_dataset": row["dataset"],
        "preference_prompt_variant": variant,
        "source_instruction": row["instruction"].strip(),
        "prompt_policy": PROMPT_POLICY,
        "subset_for_metrics": row["preference_type"],
        "extra_fields": {
            "source_repo": GITHUB_REPO,
            "source_revision": GITHUB_REVISION,
            "source_manifest_sha256": MANIFEST_SHA256,
            "source_audio_filepath": row["audio_filepath"],
            "annotation_license": ANNOTATION_LICENSE,
            "release_census": EXPECTED_ROWS,
            "paper_census_note": "paper describes an earlier seven-corpus 3210 preference + 335 standard draft",
        },
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    corpora: str | list[str] | tuple[str, ...] | None = None,
    categories: str | list[str] | tuple[str, ...] | None = None,
    variants: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download Preference-ASR and write the benchmark JSONL.

    Args:
        corpora: Source corpora to keep (``ami``, ``common_voice``, ``librispeech``, ``voxpopuli``; default: all).
        categories: Preference types to keep (``case``, ``disfluencies``, ``entities``, ``normalization``,
            ``standard``; default: all).
        variants: Conditions to keep (``default``, ``instructed``; default: both). The default of every argument
            is the whole release in both conditions.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N selected rows of each condition (smoke runs).
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    chosen_corpora = audio_manifest.select(corpora, CORPORA, "Preference-ASR corpora")
    chosen_categories = audio_manifest.select(categories, CATEGORIES, "Preference-ASR categories")
    chosen_variants = audio_manifest.select(variants, VARIANTS, "Preference-ASR variants")
    numbered = [
        (number, row)
        for number, row in enumerate(load_manifest(), 1)
        if row["dataset"] in chosen_corpora and row["preference_type"] in chosen_categories
    ][:max_samples_per_profile]
    if not numbered:
        raise ValueError("No Preference-ASR rows match the selected corpora and categories")
    materialize_audio([row for _, row in numbered], DATA_DIR)

    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    prepared = audio_manifest.profile_rows(
        PROFILES,
        {
            PROFILE_PREFIX + variant: [format_source_row(row, number, variant, audio_dir) for number, row in numbered]
            for variant in chosen_variants
        },
    )
    options = {
        "corpora": None if chosen_corpora == CORPORA else list(chosen_corpora),
        "categories": None if chosen_categories == CATEGORIES else list(chosen_categories),
        "variants": None if chosen_variants == VARIANTS else list(chosen_variants),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
