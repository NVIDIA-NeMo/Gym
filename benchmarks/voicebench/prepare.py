# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare VoiceBench for NeMo Gym.

VoiceBench (https://github.com/MatthewCYM/VoiceBench) is a spoken-assistant benchmark. Its public data, the
``hlt-lab/voicebench`` dataset at a pinned revision, is downloaded and written as one JSONL with the rows of every
subset: each row is a Responses request (the subset's system prompt, an empty user message and the audio path) plus
the ``verifier_metadata`` that ``resources_servers/voicebench`` scores against. MTBench is not included (the
upstream project publishes no scorer for it).
"""

import json
import re
import shutil
from pathlib import Path
from typing import Iterator

import yaml
from huggingface_hub import HfApi, HfFileSystem

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "voicebench_benchmark.jsonl"
SYSTEM_PROMPTS = BENCHMARK_DIR / "prompts" / "system_prompts.yaml"
CODE_FILES = (Path(__file__), SYSTEM_PROMPTS, Path(audio_manifest.__file__))

HF_REPO_ID = "hlt-lab/voicebench"
HF_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"  # pragma: allowlist secret

SPEAKER_SPLITS = (
    "en_AU_Wavenet_A_1.0_0.0_0.0",
    "en_AU_Wavenet_B_1.0_0.0_0.0",
    "en_IN_Wavenet_A_1.0_0.0_0.0",
    "en_IN_Wavenet_B_1.0_0.0_0.0",
    "en_GB_Wavenet_A_1.0_0.0_0.0",
    "en_GB_Wavenet_B_1.0_0.0_0.0",
    "en_US_Wavenet_A_1.0_0.0_0.0",
    "en_US_Wavenet_C_1.0_0.0_0.0",
    "en_US_Wavenet_A_1.5_0.0_0.0",
    "en_US_Wavenet_A_2.0_0.0_0.0",
    "en_US_Wavenet_A_0.5_0.0_0.0",
)
MMSU_ROWS = {
    "law": 51,
    "engineering": 107,
    "other": 546,
    "biology": 172,
    "business": 236,
    "economics": 280,
    "health": 406,
    "philosophy": 305,
    "psychology": 317,
    "history": 104,
    "chemistry": 167,
    "physics": 383,
}
SD_QA_SPLITS = ("aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf")

# Gym subset name -> (dataset config, {split: rows}) in the order of the rows.
SUBSETS = {
    "advbench": ("advbench", {"test": 520}),
    "alpacaeval": ("alpacaeval", {"test": 199}),
    "alpacaeval_full": ("alpacaeval_full", {"test": 636}),
    "alpacaeval_speaker": ("alpacaeval_speaker", {split: 636 for split in SPEAKER_SPLITS}),
    "bbh": ("bbh", {"test": 1000}),
    "commoneval": ("commoneval", {"test": 200}),
    "ifeval": ("ifeval", {"test": 345}),
    "mmsu": ("mmsu", MMSU_ROWS),
    "openbookqa": ("openbookqa", {"test": 455}),
    "sd-qa": ("sd-qa", {split: 553 for split in SD_QA_SPLITS}),
    "wildvoice": ("wildvoice", {"test": 1000}),
}
EXPECTED_ROWS = {subset: sum(splits.values()) for subset, (_, splits) in SUBSETS.items()}
# Columns of the dataset that the scorer reads or that identify the example, by subset.
SOURCE_FIELDS = {
    "bbh": ("reference", "id"),
    "ifeval": ("key", "instruction_id_list", "kwargs"),
    "mmsu": ("reference", "question_id", "cot_content", "category", "src"),
    "openbookqa": ("reference",),
    "sd-qa": ("reference",),
    "wildvoice": ("conversation_hash",),
}


def _shards(config: str) -> dict[str, list[str]]:
    """The parquet shards of a dataset configuration by split, in shard order."""
    shards: dict[str, list[str]] = {}
    for name in sorted(HfApi().list_repo_files(HF_REPO_ID, repo_type="dataset", revision=HF_REVISION)):
        match = re.fullmatch(rf"{re.escape(config)}/(.+)-\d{{5}}-of-\d{{5}}\.parquet", name)
        if match:
            shards.setdefault(match.group(1), []).append(name)
    return shards


def _source_rows(shards: list[str]) -> Iterator[dict]:
    """Rows of one split, streamed shard by shard; ``audio`` is ``{"bytes": ..., "path": ...}``."""
    import pyarrow.parquet as pq

    filesystem = HfFileSystem()
    for name in shards:
        with filesystem.open(f"datasets/{HF_REPO_ID}@{HF_REVISION}/{name}", "rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=16):
                yield from batch.to_pylist()


def _write_audio(audio: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    if audio.get("bytes") is not None:
        staged = destination.with_name(destination.name + ".tmp")
        staged.write_bytes(audio["bytes"])
        staged.replace(destination)
    elif audio.get("path"):
        shutil.copyfile(audio["path"], destination)
    else:
        raise ValueError("VoiceBench audio row has neither bytes nor a path")


def verifier_metadata(subset: str, split: str, index: int, source: dict) -> dict:
    """What the scorer needs: the transcript, the reference and the subset-specific fields."""
    metadata = {"subset": subset, "sample_id": f"{split}/{index:06d}", "split": split, "prompt": source["prompt"]}
    for field in SOURCE_FIELDS.get(subset, ()):
        metadata[field] = source[field]
    return metadata


def build_row(subset: str, metadata: dict, audio_path: str, system_prompt: str) -> dict:
    controls = audio_manifest.request_controls({"request": "reasoning"})
    return {
        "responses_create_params": {
            "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": ""}],
            "metadata": {
                "audio_path": audio_path,
                "extra_body": json.dumps(
                    {**controls["extra_body"], "seed": controls["seed"]}, sort_keys=True, separators=(",", ":")
                ),
            },
            "temperature": controls["temperature"],
            "top_p": controls["top_p"],
            "max_output_tokens": controls["max_output_tokens"],
        },
        "verifier_metadata": metadata,
    }


def iter_rows(audio_dir: str, audio_root: Path, max_samples: int | None) -> Iterator[dict]:
    system_prompts = yaml.safe_load(SYSTEM_PROMPTS.read_text(encoding="utf-8"))
    for subset, (config, splits) in SUBSETS.items():
        shards = _shards(config)
        for split, expected in splits.items():
            count = 0
            for index, source in enumerate(_source_rows(shards[split])):
                if max_samples is not None and index >= max_samples:
                    break
                audio = source["audio"]
                relative = f"{subset}/{re.sub(r'[/.]', '_', split)}/{index:06d}{Path(audio.get('path') or '').suffix or '.wav'}"
                _write_audio(audio, audio_root / relative)
                metadata = verifier_metadata(subset, split, index, source)
                yield build_row(subset, metadata, f"{audio_dir}/{relative}", system_prompts[subset])
                count += 1
            if max_samples is None and count != expected:
                raise ValueError(f"VoiceBench {config}/{split} has {count} rows; expected {expected}")


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Stream VoiceBench (11 GB of audio) from the Hub and write the combined benchmark JSONL and the audio files.

    Args:
        audio_root: Directory rows use for ``data/`` when the model server sees the data directory somewhere other
            than its local path (for example ``/data/voicebench`` in a container that mounts it there). Default: the
            absolute local path.
        max_samples_per_profile: Keep only the first N rows of every split (smoke runs).
    """
    if max_samples_per_profile is not None and max_samples_per_profile < 1:
        raise ValueError(f"max_samples_per_profile must be at least 1, got {max_samples_per_profile}")
    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(
        OUTPUT_FPATH,
        iter_rows(audio_dir, DATA_DIR / "audio", max_samples_per_profile),
        code_files=CODE_FILES,
        **options,
    )
