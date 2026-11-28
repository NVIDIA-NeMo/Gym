# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare AIR-Bench Foundation (multiple-choice audio understanding) for Gym.

The rows are the questions of the pinned ``qyang1021/AIR-Bench-Dataset`` revision: 24,683 questions in 25
task/dataset leaves of three modalities (speech, sound, music). By default all of them are prepared; ``modalities``
and ``tasks`` narrow the set, for example ``modalities=speech`` selects the 10,973 speech questions in 11 leaves
(``SPEECH_LEAVES``). The prompt follows the instruction construction of the official
``Inference_Foundation.py`` (https://github.com/OFA-Sys/AIR-Bench). Misspellings in the public task names
(``Recontion``, ``Reconition``, ``Classfication``) are protocol identifiers that the official scorer matches
verbatim, so they are kept. The audio is written unchanged to ``data/Foundation/<leaf>/``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "air_bench_foundation_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "qyang1021/AIR-Bench-Dataset"
HF_REVISION = "8fa643aefd1d7248fd55fcf3e7ad22c81bbac1fa"
FOUNDATION_METADATA_PATH = "Foundation/Foundation_meta.json"
FOUNDATION_METADATA_SHA256 = "e59da5c972e30f03797517a009105428060ae34e3b6c112635db8f0bebc312fa"
EXPECTED_FOUNDATION_ROWS = 24683
# Revision of https://github.com/OFA-Sys/AIR-Bench whose protocol (prompt and scorer) this follows.
OFFICIAL_REPO_REVISION = "983247666e2a25519b5c122810c2bcd91f26fd8d"

OFFICIAL_QUESTION_PROMPT = (
    "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, "
    "you may only choose A or B or C or D."
)

# The AIR-Bench Foundation taxonomy: the modality of each of the 19 tasks (the release has no mixed-audio task; mixed
# audio belongs to the Chat benchmark). Vocal-sound classification is a sound task.
TASK_MODALITIES = {
    "Speech_Grounding": "speech",
    "Spoken_Language_Identification": "speech",
    "Speaker_Gender_Recognition": "speech",
    "Speaker_Emotion_Recontion": "speech",
    "Speaker_Age_Prediction": "speech",
    "Speech_Entity_Reconition": "speech",
    "Speaker_Intent_Classification": "speech",
    "Speaker_Number_Verification": "speech",
    "Synthesized_Voice_Detection": "speech",
    "Audio_Grounding": "sound",
    "Acoustic_Scene_Classification": "sound",
    "Sound_AQA": "sound",
    "vocal_sound_classification": "sound",
    "Music_AQA": "music",
    "Music_Genre_Recognition": "music",
    "Music_Instruments_Classfication": "music",
    "Music_Midi_Pitch_Analysis": "music",
    "Music_Midi_Velocity_Analysis": "music",
    "Music_Mood_Recognition": "music",
}
MODALITIES = ("speech", "sound", "music")
EXPECTED_LEAF_COUNTS = {
    "Acoustic_Scene_Classification_CochlScene": 1000,
    "Acoustic_Scene_Classification_TUT2017": 1000,
    "Audio_Grounding_AudioGrounding": 896,
    "Music_AQA_music_avqa": 814,
    "Music_Genre_Recognition_MTJ-Jamendo": 1000,
    "Music_Genre_Recognition_fma": 1000,
    "Music_Instruments_Classfication_MTJ-Jamendo": 1000,
    "Music_Instruments_Classfication_nsynth": 1000,
    "Music_Midi_Pitch_Analysis_nsynth": 1000,
    "Music_Midi_Velocity_Analysis_nsynth": 1000,
    "Music_Mood_Recognition_MTJ-Jamendo": 1000,
    "Sound_AQA_avqa": 1000,
    "Sound_AQA_clothoaqa": 1000,
    "Speaker_Age_Prediction_common_voice_13.0_en": 1000,
    "Speaker_Emotion_Recontion_iemocap": 1000,
    "Speaker_Emotion_Recontion_meld": 1000,
    "Speaker_Gender_Recognition_common_voice_13_en": 992,
    "Speaker_Gender_Recognition_meld": 1000,
    "Speaker_Intent_Classification_slurp": 1000,
    "Speaker_Number_Verification_voxceleb1": 1000,
    "Speech_Entity_Reconition_slurp": 1000,
    "Speech_Grounding_librispeech": 981,
    "Spoken_Language_Identification_covost2": 1000,
    "Synthesized_Voice_Detection_fake_or_real": 1000,
    "vocal_sound_classification_VocalSound": 1000,
}
EXPECTED_TOTAL_ROWS = 24683
EXPECTED_AUDIO_INVENTORY_SHA256 = "ced2bc8c9103744e63c9b7b4cd6e4ec01ed7a6ba7bc0f716dac0642d2c634d48"
# The 11 leaves (10,973 rows) of the speech tasks, and the digest of their audio inventory.
SPEECH_LEAVES = (
    "Speaker_Age_Prediction_common_voice_13.0_en",
    "Speaker_Emotion_Recontion_iemocap",
    "Speaker_Emotion_Recontion_meld",
    "Speaker_Gender_Recognition_common_voice_13_en",
    "Speaker_Gender_Recognition_meld",
    "Speaker_Intent_Classification_slurp",
    "Speaker_Number_Verification_voxceleb1",
    "Speech_Entity_Reconition_slurp",
    "Speech_Grounding_librispeech",
    "Spoken_Language_Identification_covost2",
    "Synthesized_Voice_Detection_fake_or_real",
)
EXPECTED_SPEECH_AUDIO_INVENTORY_SHA256 = "89571f90615c54d317a178c59691a24dd0223815499ab96c1e0df489d2eef873"


def _leaf_id(row: dict[str, Any]) -> str:
    return f"{row['task_name']}_{row['dataset_name']}"


def _audio_repo_path(row: dict[str, Any]) -> str:
    source_path = PurePosixPath(str(row["path"]))
    if source_path.is_absolute() or not source_path.parts or ".." in source_path.parts:
        raise ValueError(f"Unsafe AIR-Bench audio path: {row['path']!r}")
    if row["task_name"] == "Audio_Grounding":
        # The metadata names ``.wav`` files, the release stores ``.flac``; the official inference script swaps the suffix.
        source_path = source_path.with_suffix(".flac")
    return str(PurePosixPath("Foundation") / _leaf_id(row) / source_path)


def _validate_source_row(row: dict[str, Any], index: int) -> None:
    for key in ("path", "question", "choice_a", "choice_b", "answer_gt", "task_name", "dataset_name"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"AIR-Bench Foundation row {index} has invalid {key!r}")
    for key in ("choice_c", "choice_d"):
        if row.get(key) is not None and not isinstance(row[key], str):
            raise ValueError(f"AIR-Bench Foundation row {index} has invalid {key!r}")
    if not isinstance(row.get("uniq_id"), int):
        raise TypeError(f"AIR-Bench Foundation row {index} has invalid 'uniq_id'")
    if row["answer_gt"] not in [row.get(f"choice_{letter}") for letter in "abcd"]:
        raise ValueError(f"AIR-Bench Foundation row {index} answer_gt does not match a choice")
    _audio_repo_path(row)


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    inventory = "".join(f"{path}\n" for path in sorted(_audio_repo_path(row) for row in rows))
    return hashlib.sha256(inventory.encode("utf-8")).hexdigest()


def validate_census(rows: list[dict[str, Any]]) -> None:
    """Fail if the pinned metadata differs from the released one (rows, leaves, ids, audio inventories)."""
    for index, row in enumerate(rows):
        _validate_source_row(row, index)
    leaf_counts = Counter(_leaf_id(row) for row in rows)
    if leaf_counts != Counter(EXPECTED_LEAF_COUNTS):
        raise ValueError(
            f"AIR-Bench Foundation leaf-count drift: expected {EXPECTED_LEAF_COUNTS}, "
            f"found {dict(sorted(leaf_counts.items()))}"
        )
    unique_ids = {row["uniq_id"] for row in rows}
    if len(rows) != EXPECTED_TOTAL_ROWS or len(unique_ids) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"AIR-Bench Foundation row drift: expected {EXPECTED_TOTAL_ROWS}, "
            f"found {len(rows)} rows and {len(unique_ids)} unique ids"
        )
    if (digest := _inventory_digest(rows)) != EXPECTED_AUDIO_INVENTORY_SHA256:
        raise ValueError(f"AIR-Bench Foundation audio-inventory drift: found {digest}")
    speech_rows = [row for row in rows if _leaf_id(row) in SPEECH_LEAVES]
    if (digest := _inventory_digest(speech_rows)) != EXPECTED_SPEECH_AUDIO_INVENTORY_SHA256:
        raise ValueError(f"AIR-Bench Foundation speech audio-inventory drift: found {digest}")


def _names(value: str | list[str] | tuple[str, ...] | None, valid: dict | tuple, what: str) -> tuple[str, ...] | None:
    """Normalize a list or comma-separated string argument; ``None`` stays ``None`` (no restriction)."""
    return None if value is None else audio_manifest.select(value, valid, f"AIR-Bench {what}", keep_order=True)


def select_tasks(
    modalities: str | list[str] | tuple[str, ...] | None = None, tasks: str | list[str] | tuple[str, ...] | None = None
) -> set[str]:
    """The tasks of the given modalities (default: all), restricted to ``tasks`` when given."""
    chosen_modalities = _names(modalities, MODALITIES, "modalities")
    chosen_tasks = _names(tasks, TASK_MODALITIES, "tasks")
    selected = {
        task
        for task, modality in TASK_MODALITIES.items()
        if (chosen_modalities is None or modality in chosen_modalities)
        and (chosen_tasks is None or task in chosen_tasks)
    }
    if not selected:
        raise ValueError(f"No AIR-Bench task has modality {chosen_modalities} and name in {chosen_tasks}")
    return selected


def load_rows(
    modalities: str | list[str] | tuple[str, ...] | None = None,
    tasks: str | list[str] | tuple[str, ...] | None = None,
    max_samples: int | None = None,
    max_samples_per_subset: int | None = None,
) -> list[dict[str, Any]]:
    """Read the pinned metadata, check it against the released census, then keep the selected rows (file order).

    ``max_samples_per_subset`` keeps the first N rows of each task/dataset leaf, then ``max_samples`` the first N of
    what is left.
    """
    selected = select_tasks(modalities, tasks)
    path = Path(hf_hub_download(HF_REPO_ID, FOUNDATION_METADATA_PATH, repo_type="dataset", revision=HF_REVISION))
    if (digest := audio_manifest.sha256_file(path)) != FOUNDATION_METADATA_SHA256:
        raise ValueError(
            f"AIR-Bench Foundation metadata hash drift: expected {FOUNDATION_METADATA_SHA256}, found {digest}"
        )
    with path.open(encoding="utf-8") as stream:
        foundation_rows = json.load(stream)
    if not isinstance(foundation_rows, list) or len(foundation_rows) != EXPECTED_FOUNDATION_ROWS:
        raise ValueError(f"Expected {EXPECTED_FOUNDATION_ROWS} rows in the Foundation metadata")
    rows = [dict(row) for row in foundation_rows]
    validate_census(rows)
    chosen = [row for row in rows if row["task_name"] in selected]
    if max_samples_per_subset is not None:
        seen: Counter = Counter()
        kept = []
        for row in chosen:
            seen[_leaf_id(row)] += 1
            if seen[_leaf_id(row)] <= max_samples_per_subset:
                kept.append(row)
        chosen = kept
    return chosen[:max_samples]


def _check_cap(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")


def download_audio(rows: list[dict[str, Any]], data_dir: Path, *, full_leaves: bool) -> None:
    """Fetch the audio of ``rows`` into ``data_dir``: whole leaf folders for complete leaves, else file by file."""
    missing = [path for path in map(_audio_repo_path, rows) if not (data_dir / path).is_file()]
    if missing and full_leaves:
        snapshot_download(
            HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=str(data_dir),
            allow_patterns=[f"Foundation/{leaf}/**" for leaf in sorted({_leaf_id(row) for row in rows})],
        )
    else:
        for filename in missing:
            hf_hub_download(HF_REPO_ID, filename, repo_type="dataset", revision=HF_REVISION, local_dir=str(data_dir))
    check_audio(rows, data_dir)


def check_audio(rows: list[dict[str, Any]], data_dir: Path) -> None:
    paths = list(map(_audio_repo_path, rows))
    missing = [path for path in paths if not (data_dir / path).is_file()]
    empty = [path for path in paths if path not in missing and (data_dir / path).stat().st_size == 0]
    if missing or empty:
        raise FileNotFoundError(
            f"Invalid AIR-Bench Foundation audio: {len(missing)} missing (first: {missing[:3]}), "
            f"{len(empty)} empty (first: {empty[:3]})"
        )


def official_prompt(row: dict[str, Any]) -> str:
    """The instruction built by the official ``Inference_Foundation.py``."""
    choices = (
        f"A. {row['choice_a']}\nB. {row['choice_b']}\nC. {row.get('choice_c', None)}\nD. {row.get('choice_d', None)}"
    )
    return f"{OFFICIAL_QUESTION_PROMPT}\n{row['question']}\n{choices}"


def format_source_row(row: dict[str, Any], audio_prefix: str) -> dict[str, Any]:
    relative_path = _audio_repo_path(row)
    audio_path = f"{audio_prefix}/{relative_path}"
    return {
        "messages": [{"role": "user", "content": official_prompt(row), "audio": {"path": audio_path}}],
        "expected_answer": row["answer_gt"],
        "audio_path": audio_path,
        "audio_filepath": audio_path,
        "source_audio_path": relative_path,
        "question": row["question"],
        "choice_a": row["choice_a"],
        "choice_b": row["choice_b"],
        "choice_c": row.get("choice_c"),
        "choice_d": row.get("choice_d"),
        "answer_gt": row["answer_gt"],
        "task_name": row["task_name"],
        "dataset_name": row["dataset_name"],
        "uniq_id": row["uniq_id"],
        "subset_for_metrics": _leaf_id(row),
        "air_bench_foundation_leaf": _leaf_id(row),
        "air_bench_dataset_revision": HF_REVISION,
        "air_bench_official_repo_revision": OFFICIAL_REPO_REVISION,
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    modalities: str | list[str] | tuple[str, ...] | None = None,
    tasks: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
    max_samples_per_subset: int | None = None,
) -> Path:
    """Download AIR-Bench Foundation questions and audio and write the combined benchmark JSONL.

    Args:
        modalities: ``speech``, ``sound`` and/or ``music`` (a list or a comma-separated string; default: all).
            ``SPEECH_MODALITIES`` selects the 10,973 speech questions.
        tasks: Task names to keep, within the selected modalities (default: all tasks).
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows (smoke runs).
        max_samples_per_subset: Keep only the first N rows of each task/dataset leaf (stratified smoke runs).
    """
    _check_cap("max_samples_per_profile", max_samples_per_profile)
    _check_cap("max_samples_per_subset", max_samples_per_subset)
    selected = select_tasks(modalities, tasks)
    rows = load_rows(modalities, tasks, max_samples_per_profile, max_samples_per_subset)
    download_audio(rows, DATA_DIR, full_leaves=max_samples_per_profile is None and max_samples_per_subset is None)

    audio_prefix = audio_manifest.audio_prefix(DATA_DIR, audio_root)
    prepared = audio_manifest.profile_rows(PROFILES, [format_source_row(row, audio_prefix) for row in rows])
    narrowed = selected != set(TASK_MODALITIES)
    options = {
        "modalities": _names(modalities, MODALITIES, "modalities") if narrowed else None,
        "tasks": _names(tasks, TASK_MODALITIES, "tasks") if narrowed else None,
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
        "max_samples_per_subset": max_samples_per_subset,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
