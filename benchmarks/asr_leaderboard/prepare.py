# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Hugging Face Open ASR Leaderboard English benchmark for Gym.

Datasets are read from the repositories the leaderboard uses, with its own config names and session-level scoring
protocol, so the prepared data matches the leaderboard's current English protocol:

    - ami            -> config ``ami_cleaned``           (clips under 1 s removed upstream)
    - gigaspeech     -> config ``gigaspeech_cleaned``    (clips under 1 s removed upstream)
    - voxpopuli_aa   -> config ``voxpopuli_cleaned_aa``  (Artificial Analysis cleaned references)
    - earnings22_aa  -> ``ArtificialAnalysis/Earnings22-Cleaned-AA-chunked`` (341 VAD chunks of six calls)
    - urgent2024     -> config ``urgent2024``            (1,000 noisy rows, resampled to 16 kHz)

The Earnings22 chunks share the six parent-call references: every chunk row carries ``parent_id`` and
``chunk_index``, and the ``asr_with_pc`` server scores one concatenated transcript per call, as open_asr_leaderboard
PR #202 does. Audio is written as FLAC under ``data/audio/<dataset>/``.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
from huggingface_hub import hf_hub_download

from benchmarks import audio_manifest, hf_parquet
from resources_servers.asr_with_pc import open_asr_normalizer


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "asr_leaderboard_benchmark.jsonl"
CODE_FILES = (
    Path(__file__),
    Path(audio_manifest.__file__),
    Path(hf_parquet.__file__),
    Path(open_asr_normalizer.__file__),
)

AUDIO_SAMPLE_RATE = 16000
LEADERBOARD_REPO = "hf-audio/open-asr-leaderboard"
LEADERBOARD_REVISION = "de79e57adf9983993bcc72586d7313460825f5eb"  # pragma: allowlist secret
EARNINGS22_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
EARNINGS22_REVISION = "27f7d0056d593231974a752ce03c1e965da69b3f"  # pragma: allowlist secret
EARNINGS22_PARENT_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"
EARNINGS22_PARENT_REVISION = "5999936ee3d8d3cde160e3c65a8cc6ff5b7c5276"  # pragma: allowlist secret

# Request prompts; SPGISpeech references omit fillers, so its prompt does too.
VERBATIM_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)
NO_FILLERS_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining "
    "spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only."
)


@dataclass(frozen=True)
class DatasetSpec:
    """A leaderboard dataset: which Hugging Face config and split to read."""

    config: str | None
    split: str
    repo: str = LEADERBOARD_REPO
    revision: str = LEADERBOARD_REVISION
    chunked: bool = False  # Earnings22: rows are audio chunks of a parent call whose transcript is the reference
    expected_rows: int | None = None
    resample_audio: bool = False


DATASETS = {
    "librispeech_clean": DatasetSpec("librispeech", "test.clean"),
    "librispeech_other": DatasetSpec("librispeech", "test.other"),
    "voxpopuli_aa": DatasetSpec("voxpopuli_cleaned_aa", "test"),
    "gigaspeech": DatasetSpec("gigaspeech_cleaned", "test"),
    "spgispeech": DatasetSpec("spgispeech", "test"),
    "earnings22_aa": DatasetSpec(None, "test", repo=EARNINGS22_REPO, revision=EARNINGS22_REVISION, chunked=True),
    "ami": DatasetSpec("ami_cleaned", "test"),
    "urgent2024": DatasetSpec("urgent2024", "test", expected_rows=1000, resample_audio=True),
}

_NORMALIZER = open_asr_normalizer.OpenASREnglishTextNormalizer()


def is_target_text_in_range(text: str) -> bool:
    """Apply the Open ASR Leaderboard reference filter."""
    if text.strip() == "ignore time segment in scoring":
        return False
    return bool(_NORMALIZER(text).strip())


def extract_audio(audio_info: dict | Path | None) -> tuple[np.ndarray | None, int | None]:
    """Decode a Hugging Face audio value (undecoded ``bytes``, a file path, or a decoded array) to mono samples."""
    import soundfile as sf

    if audio_info is None:
        return None, None
    if isinstance(audio_info, Path):
        source = audio_info
    elif audio_info.get("bytes") is not None:
        source = io.BytesIO(audio_info["bytes"])
    elif audio_info.get("path"):
        source = audio_info["path"]
    else:
        try:
            return np.array(audio_info["array"]), int(audio_info["sampling_rate"])
        except (KeyError, TypeError, IndexError):
            return None, None
    audio_array, sampling_rate = sf.read(source, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    return np.asarray(audio_array), int(sampling_rate)


def format_entry(
    entry: dict,
    dataset_name: str,
    audio_dir: Path,
    audio_prefix: str,
    spec: DatasetSpec,
    *,
    reference_text: str | None = None,
    audio_info: dict | Path | None = None,
) -> dict | None:
    """Write the entry's audio and return its row, or ``None`` when the reference is filtered out."""
    import soundfile as sf

    text = (entry["text"] if reference_text is None else reference_text).strip()
    if not is_target_text_in_range(text):
        return None

    sample_id = str(entry["id"]).replace("/", "_").replace(":", "_")
    audio_filename = f"{Path(sample_id).stem}.flac"
    audio_array, sampling_rate = extract_audio(entry.get("audio") if audio_info is None else audio_info)
    duration = None
    if audio_array is not None and sampling_rate is not None:
        if sampling_rate != AUDIO_SAMPLE_RATE and spec.resample_audio:
            from scipy.signal import resample_poly

            divisor = gcd(sampling_rate, AUDIO_SAMPLE_RATE)
            audio_array = resample_poly(audio_array, AUDIO_SAMPLE_RATE // divisor, sampling_rate // divisor).astype(
                np.float32
            )
            sampling_rate = AUDIO_SAMPLE_RATE
        if sampling_rate != AUDIO_SAMPLE_RATE:
            raise ValueError(f"Expected {AUDIO_SAMPLE_RATE} Hz audio, got {sampling_rate} Hz for {sample_id}")
        duration = len(audio_array) / sampling_rate
        sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)
    elif entry.get("audio_length_s", entry.get("duration")) is not None:
        duration = float(entry.get("audio_length_s", entry.get("duration")))

    row = {
        "responses_create_params": {
            "input": [
                {"role": "user", "content": NO_FILLERS_PROMPT if dataset_name == "spgispeech" else VERBATIM_PROMPT}
            ],
            "metadata": {"audio_path": f"{audio_prefix}/{dataset_name}/{audio_filename}"},
        },
        "expected_answer": text,
        "subset_for_metrics": dataset_name,
        "benchmark_profile": "asr_leaderboard",
        "sample_id": sample_id,
    }
    if duration is not None:
        row["audio_duration"] = float(duration)
    if "speaker_id" in entry:
        row["speaker_id"] = entry["speaker_id"]
    if spec.chunked:
        row["parent_id"] = entry["parent_id"]
        row["chunk_index"] = int(entry["chunk_index"])
    return row


def _source_rows(spec: DatasetSpec, max_samples: int | None) -> Iterator[dict]:
    """The first ``max_samples`` rows of the pinned config split (Earnings22 chunks are JSON Lines, the rest parquet)."""
    files = hf_parquet.config_files(spec.repo, spec.revision, spec.config, spec.split)
    if spec.chunked:
        return itertools.islice(hf_parquet.iter_json_rows(spec.repo, spec.revision, files), max_samples)
    return hf_parquet.iter_rows(spec.repo, spec.revision, files, limit=max_samples)


def _rows(dataset_names: list[str], audio_prefix: str, max_samples: int | None) -> Iterator[dict]:
    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        print(
            f"Loading {dataset_name} from {spec.repo}@{spec.revision[:8]} (config={spec.config}, split={spec.split})"
        )
        parents = {}
        if spec.chunked:
            parent_files = hf_parquet.config_files(
                EARNINGS22_PARENT_REPO, EARNINGS22_PARENT_REVISION, None, spec.split
            )
            parent_rows = hf_parquet.iter_json_rows(EARNINGS22_PARENT_REPO, EARNINGS22_PARENT_REVISION, parent_files)
            parents = {str(parent["id"]): parent["transcript"] for parent in parent_rows}
        audio_dir = DATA_DIR / "audio" / dataset_name
        audio_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for count, entry in enumerate(_source_rows(spec, max_samples), 1):
            kwargs = {}
            if spec.chunked:
                if str(entry["parent_id"]) not in parents:
                    raise ValueError(f"No transcript in {EARNINGS22_PARENT_REPO} for parent id {entry['parent_id']!r}")
                kwargs["reference_text"] = parents[str(entry["parent_id"])]
                kwargs["audio_info"] = Path(
                    hf_hub_download(
                        spec.repo,
                        f"audio/{entry['file_name']}",
                        repo_type="dataset",
                        revision=spec.revision,
                    )
                )
            if (row := format_entry(entry, dataset_name, audio_dir, audio_prefix, spec, **kwargs)) is not None:
                yield row
        if max_samples is None and spec.expected_rows is not None and count != spec.expected_rows:
            raise ValueError(f"{dataset_name}: expected {spec.expected_rows} source rows, found {count}")


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    audio_root: str | None = None,
    datasets: list[str] | None = None,
    max_samples_per_subset: int | None = None,
) -> Path:
    """Download the leaderboard datasets and write the combined benchmark JSONL.

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        datasets: Subsets to prepare (default: all eight).
        max_samples_per_subset: Keep only the first N source rows of each subset (smoke runs).
    """
    if max_samples_per_subset is not None and (
        isinstance(max_samples_per_subset, bool)
        or not isinstance(max_samples_per_subset, int)
        or max_samples_per_subset < 1
    ):
        raise ValueError(f"max_samples_per_subset must be a positive integer, got {max_samples_per_subset!r}")
    datasets = list(datasets) if datasets else None  # a Hydra override arrives as a ListConfig
    dataset_names = datasets or list(DATASETS)
    if unknown := set(dataset_names) - set(DATASETS):
        raise ValueError(f"Unknown datasets {sorted(unknown)}. Available: {list(DATASETS)}")
    audio_prefix = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    options = {"audio_root": audio_root, "datasets": datasets, "max_samples_per_subset": max_samples_per_subset}
    return audio_manifest.write_rows(
        OUTPUT_FPATH, _rows(dataset_names, audio_prefix, max_samples_per_subset), code_files=CODE_FILES, **options
    )
