# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare VoiceCodeBench (transcription of code-like spoken entities) for Gym.

The rows are the 300 recordings of the pinned ``besimple-ai/voice-code-bench`` revision (8 workflow domains,
1,482 target entities). By default all of them are prepared; ``domains`` narrows the set. The source row keeps the
metadata the scorer needs (transcript layers and target entities); the model sees only the profile's transcription
prompt. The audio is written unchanged to ``data/voice-code-bench/data/audio/``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_DIR = DATA_DIR / "voice-code-bench"
OUTPUT_FPATH = DATA_DIR / "voice_code_bench.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "besimple-ai/voice-code-bench"
HF_REVISION = "8edfa398aa5575816d940793e4e2b8c6430184da"  # pragma: allowlist secret
METADATA_PATH = "data/metadata.jsonl"
METADATA_SHA256 = "a69c40387303ea70fe46160239266cdb64b6ad9a68ca160c7e12640b4afb74f5"  # pragma: allowlist secret
EXPECTED_ROWS = 300
EXPECTED_ENTITIES = 1482
EXPECTED_DOMAIN_COUNTS = {
    "technical_it_dev": 55,
    "contact_routing": 45,
    "retail_logistics_order": 45,
    "finance_billing": 40,
    "healthcare_admin": 35,
    "legal_insurance_government": 35,
    "education_workplace": 25,
    "dense_mixed_stress": 20,
}
DOMAINS = tuple(EXPECTED_DOMAIN_COUNTS)
# The placeholder instruction of the source rows; the profile's ``prompt_override`` replaces it in the request.
SOURCE_PROMPT = "Transcribe the following:"
SCORER_FIELDS = (
    "audio_id",
    "domain",
    "scenario",
    "difficulty",
    "transcripts",
    "entities",
    "entity_types",
    "entity_count",
)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Fail if the pinned metadata differs from the released one (rows, ids, transcript layers, entities, domains)."""
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} VoiceCodeBench rows, found {len(rows)}")
    audio_ids: set[str] = set()
    entity_total = 0
    for index, row in enumerate(rows, start=1):
        audio_id = row.get("audio_id")
        if not isinstance(audio_id, str) or not audio_id:
            raise ValueError(f"VoiceCodeBench row {index} is missing audio_id")
        if audio_id in audio_ids:
            raise ValueError(f"Duplicate VoiceCodeBench audio_id: {audio_id}")
        audio_ids.add(audio_id)
        transcripts = row.get("transcripts")
        if not isinstance(transcripts, dict) or any(
            not isinstance(transcripts.get(name), str) or not transcripts[name]
            for name in ("template", "acoustic", "canonical")
        ):
            raise ValueError(f"VoiceCodeBench row {audio_id} has invalid transcript layers")
        entities = row.get("entities")
        if not isinstance(entities, list) or len(entities) != row.get("entity_count"):
            raise ValueError(f"VoiceCodeBench row {audio_id} has inconsistent entities")
        entity_total += len(entities)
        if not isinstance(row.get("file_name"), str) or not row["file_name"].startswith("audio/"):
            raise ValueError(f"VoiceCodeBench row {audio_id} has an invalid file_name")
    if entity_total != EXPECTED_ENTITIES:
        raise ValueError(f"Expected {EXPECTED_ENTITIES} target entities, found {entity_total}")
    if (domains := Counter(row.get("domain") for row in rows)) != Counter(EXPECTED_DOMAIN_COUNTS):
        raise ValueError(f"VoiceCodeBench domain drift: {dict(sorted(domains.items(), key=str))}")


def select_domains(domains: str | list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Normalize a list or comma-separated string of domains; ``None`` selects all of them."""
    return audio_manifest.select(domains, DOMAINS, "VoiceCodeBench domains", keep_order=True)


def load_rows(
    domains: str | list[str] | tuple[str, ...] | None = None, max_samples: int | None = None
) -> list[dict[str, Any]]:
    """Read the pinned metadata, check it against the released census, then keep the selected rows (file order)."""
    selected = select_domains(domains)
    path = Path(hf_hub_download(HF_REPO_ID, METADATA_PATH, repo_type="dataset", revision=HF_REVISION))
    if (digest := audio_manifest.sha256_file(path)) != METADATA_SHA256:
        raise ValueError(f"VoiceCodeBench metadata hash drift: expected {METADATA_SHA256}, found {digest}")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    validate_rows(rows)
    return [row for row in rows if row["domain"] in selected][:max_samples]


def _audio_repo_path(row: dict[str, Any]) -> str:
    return f"data/{row['file_name']}"


def download_audio(rows: list[dict[str, Any]], *, full: bool) -> None:
    """Fetch the audio of ``rows`` into ``SOURCE_DIR``: the whole audio folder for the complete set, else file by file."""
    missing = [path for path in map(_audio_repo_path, rows) if not (SOURCE_DIR / path).is_file()]
    if missing and full:
        snapshot_download(
            HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=str(SOURCE_DIR),
            allow_patterns=["data/audio/*.wav"],
        )
    else:
        for filename in missing:
            hf_hub_download(HF_REPO_ID, filename, repo_type="dataset", revision=HF_REVISION, local_dir=str(SOURCE_DIR))
    paths = [SOURCE_DIR / _audio_repo_path(row) for row in rows]
    bad = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if bad:
        raise FileNotFoundError(f"Missing or empty VoiceCodeBench audio: {len(bad)} files (first: {bad[:3]})")


def format_source_row(row: dict[str, Any], audio_prefix: str) -> dict[str, Any]:
    audio_path = f"{audio_prefix}/{_audio_repo_path(row)}"
    return {
        "messages": [
            {
                "role": "user",
                "content": SOURCE_PROMPT,
                "audio": {"path": audio_path, "duration": float(row["duration"])},
            }
        ],
        "expected_answer": row["transcripts"]["canonical"],
        "audio_id": row["audio_id"],
        "audio_duration": float(row["duration"]),
        "audio_filepath": audio_path,
        "subset_for_metrics": row["domain"],
        "voice_code_bench_revision": HF_REVISION,
        "voice_code_bench": {key: row[key] for key in SCORER_FIELDS},
        "task_type": "VoiceCodeBench-ASR",
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    domains: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download VoiceCodeBench and write the benchmark JSONL.

    Args:
        domains: Workflow domains to keep (a list or a comma-separated string; default: all 8, i.e. all 300 recordings).
        audio_root: Directory the rows use for ``data/voice-code-bench/`` when the model server sees it somewhere
            other than its local path (for example ``/data/voice-code-bench`` when it is mounted there). Default:
            the absolute local path.
        max_samples_per_profile: Keep only the first N rows (smoke runs).
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    selected = select_domains(domains)
    rows = load_rows(selected, max_samples_per_profile)
    download_audio(rows, full=max_samples_per_profile is None and selected == DOMAINS)

    audio_prefix = audio_manifest.audio_prefix(SOURCE_DIR, audio_root)
    prepared = audio_manifest.profile_rows(PROFILES, [format_source_row(row, audio_prefix) for row in rows])
    options = {
        "domains": selected if selected != DOMAINS else None,
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
