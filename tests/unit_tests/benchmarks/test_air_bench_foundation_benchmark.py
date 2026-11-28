# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the AIR-Bench Foundation benchmark: declaration, census, row building, prepare()."""

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.air_bench_foundation import prepare as air
from nemo_gym.environment.manifest import load_manifest
from resources_servers.audio_eval.score import evaluator_settings, resolve
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "air_bench_foundation"
NAME = "air_bench_foundation"


def _meta_row(uniq_id: int, *, task: str = "Speech_Grounding", dataset: str = "librispeech", **overrides) -> dict:
    row = {
        "path": f"{uniq_id}.flac",
        "question": "Highlight when 'hate' is spoken.",
        "choice_a": "[7.67, 8.05]",
        "choice_b": "[1.03, 1.53]",
        "choice_c": "[3.07, 3.27]",
        "choice_d": None,
        "answer_gt": "[1.03, 1.53]",
        "task_name": task,
        "dataset_name": dataset,
        "uniq_id": uniq_id,
    }
    return row | overrides


def _synthetic_rows() -> list[dict]:
    """Speech rows in two leaves, a sound leaf, an Audio_Grounding clip and a music leaf."""
    return [
        _meta_row(0),
        _meta_row(1),
        _meta_row(2),
        _meta_row(3, task="Speaker_Gender_Recognition", dataset="meld", choice_c=None),
        _meta_row(4, task="vocal_sound_classification", dataset="VocalSound", path="4.wav"),
        _meta_row(5, task="Audio_Grounding", dataset="AudioGrounding", path="5.wav"),
        _meta_row(6, task="Music_AQA", dataset="music_avqa", path="6.mp3"),
    ]


def _install_census(monkeypatch, rows: list[dict]) -> None:
    """Make the synthetic rows the released census (counts, ids and audio inventories)."""
    speech = [air._leaf_id(row) for row in rows if air.TASK_MODALITIES[row["task_name"]] == "speech"]
    monkeypatch.setattr(air, "EXPECTED_TOTAL_ROWS", len(rows))
    monkeypatch.setattr(air, "EXPECTED_LEAF_COUNTS", dict(Counter(air._leaf_id(row) for row in rows)))
    monkeypatch.setattr(air, "EXPECTED_AUDIO_INVENTORY_SHA256", air._inventory_digest(rows))
    monkeypatch.setattr(air, "SPEECH_LEAVES", tuple(dict.fromkeys(speech)))
    speech_rows = [row for row in rows if air._leaf_id(row) in speech]
    monkeypatch.setattr(air, "EXPECTED_SPEECH_AUDIO_INVENTORY_SHA256", air._inventory_digest(speech_rows))


def test_declaration_specifics() -> None:
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "Apache-2.0"


def test_profile_contract_values() -> None:
    contract = audio_manifest.load_profiles(air.PROFILES)["profiles"]["air-bench-foundation"]
    assert contract["skills"]["METRICS_TYPE"] == "air_bench_foundation"
    controls = audio_manifest.request_controls(contract)
    assert controls["max_output_tokens"] == 20480 and controls["temperature"] == 0.6


def test_profile_contract_resolves_through_the_server() -> None:
    contract = audio_manifest.load_profiles(air.PROFILES)["profiles"]["air-bench-foundation"]
    assert evaluator_settings(contract) == ("air-bench-foundation", {})
    assert resolve("evaluators", "air-bench-foundation").__name__ == "AIRBenchFoundationEvaluator"
    assert resolve("metrics", contract["skills"]["METRICS_TYPE"]).__name__ == "AIRBenchFoundationMetrics"


def test_official_prompt_and_source_row_keep_the_official_strings() -> None:
    row = _meta_row(7, task="Speaker_Emotion_Recontion", dataset="meld")
    prompt = air.official_prompt(row)
    assert (
        prompt
        == f"{air.OFFICIAL_QUESTION_PROMPT}\n{row['question']}\nA. [7.67, 8.05]\nB. [1.03, 1.53]\nC. [3.07, 3.27]\nD. None"
    )

    source = air.format_source_row(row, "/data/air")
    leaf = "Speaker_Emotion_Recontion_meld"
    audio_path = f"/data/air/Foundation/{leaf}/7.flac"
    assert source["messages"] == [{"role": "user", "content": prompt, "audio": {"path": audio_path}}]
    assert source["audio_path"] == audio_path == source["audio_filepath"]
    assert source["source_audio_path"] == f"Foundation/{leaf}/7.flac"
    assert source["subset_for_metrics"] == leaf == source["air_bench_foundation_leaf"]
    assert source["expected_answer"] == "[1.03, 1.53]" == source["answer_gt"]
    assert source["choice_d"] is None and source["uniq_id"] == 7
    assert source["air_bench_dataset_revision"] == air.HF_REVISION
    assert source["air_bench_official_repo_revision"] == air.OFFICIAL_REPO_REVISION


def test_constants_describe_the_released_census() -> None:
    assert sum(air.EXPECTED_LEAF_COUNTS.values()) == air.EXPECTED_TOTAL_ROWS == air.EXPECTED_FOUNDATION_ROWS == 24683
    assert len(air.EXPECTED_LEAF_COUNTS) == 25 and len(air.TASK_MODALITIES) == 19
    assert set(air.MODALITIES) == set(air.TASK_MODALITIES.values())
    speech_tasks = [task for task, modality in air.TASK_MODALITIES.items() if modality == "speech"]
    speech_leaves = {leaf for leaf in air.EXPECTED_LEAF_COUNTS if any(leaf.startswith(f"{t}_") for t in speech_tasks)}
    assert speech_leaves == set(air.SPEECH_LEAVES) and len(air.SPEECH_LEAVES) == 11
    assert sum(air.EXPECTED_LEAF_COUNTS[leaf] for leaf in air.SPEECH_LEAVES) == 10973


@pytest.mark.parametrize(
    "override, error",
    [
        ({"path": "../escape.flac"}, "Unsafe"),
        ({"path": "/abs.flac"}, "Unsafe"),
        ({"question": ""}, "invalid 'question'"),
        ({"choice_c": 3}, "invalid 'choice_c'"),
        ({"answer_gt": "none of them"}, "does not match a choice"),
    ],
)
def test_source_row_validation_rejects_malformed_rows(override: dict, error: str) -> None:
    with pytest.raises((ValueError, TypeError), match=error):
        air.validate_census([_meta_row(0, **override)])


def test_source_row_validation_requires_an_integer_id() -> None:
    with pytest.raises(TypeError, match="uniq_id"):
        air.validate_census([_meta_row(0, uniq_id="0")])


def test_audio_grounding_clips_are_stored_as_flac() -> None:
    row = _meta_row(5, task="Audio_Grounding", dataset="AudioGrounding", path="Yy1saVTXsKwc.wav")
    assert air._audio_repo_path(row) == "Foundation/Audio_Grounding_AudioGrounding/Yy1saVTXsKwc.flac"
    other = _meta_row(6, task="Music_AQA", dataset="music_avqa", path="clip.wav")
    assert air._audio_repo_path(other) == "Foundation/Music_AQA_music_avqa/clip.wav"


def test_census_detects_count_id_and_inventory_drift(monkeypatch) -> None:
    rows = _synthetic_rows()
    _install_census(monkeypatch, rows)
    air.validate_census(rows)

    with pytest.raises(ValueError, match="leaf-count drift"):
        air.validate_census(rows[1:])
    duplicated_id = [*rows[:1], rows[1] | {"uniq_id": 0}, *rows[2:]]
    with pytest.raises(ValueError, match="row drift"):
        air.validate_census(duplicated_id)
    monkeypatch.setattr(air, "EXPECTED_AUDIO_INVENTORY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="audio-inventory drift"):
        air.validate_census(rows)
    monkeypatch.setattr(air, "EXPECTED_AUDIO_INVENTORY_SHA256", air._inventory_digest(rows))
    monkeypatch.setattr(air, "EXPECTED_SPEECH_AUDIO_INVENTORY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="speech audio-inventory drift"):
        air.validate_census(rows)


def test_select_tasks_defaults_to_all_and_intersects_modalities_and_tasks() -> None:
    assert air.select_tasks() == set(air.TASK_MODALITIES)
    assert air.select_tasks("speech") == {task for task, m in air.TASK_MODALITIES.items() if m == "speech"}
    assert air.select_tasks(["sound", "music"], None) == {t for t, m in air.TASK_MODALITIES.items() if m != "speech"}
    assert air.select_tasks(tasks="Music_AQA, Speech_Grounding") == {"Music_AQA", "Speech_Grounding"}
    assert air.select_tasks("music", ["Music_AQA", "Speech_Grounding"]) == {"Music_AQA"}
    for kwargs in ({"modalities": "voice"}, {"tasks": "Nope"}, {"modalities": ""}, {"modalities": []}):
        with pytest.raises(ValueError, match="Empty or unknown"):
            air.select_tasks(**kwargs)
    with pytest.raises(ValueError, match="No AIR-Bench task"):
        air.select_tasks("speech", "Music_AQA")


def test_load_rows_defaults_to_everything_narrows_checks_the_hash_and_caps(tmp_path: Path, monkeypatch) -> None:
    rows = _synthetic_rows()
    metadata = tmp_path / "Foundation_meta.json"
    metadata.write_text(json.dumps(rows))
    monkeypatch.setattr(air, "hf_hub_download", lambda *args, **kwargs: str(metadata))
    monkeypatch.setattr(air, "EXPECTED_FOUNDATION_ROWS", len(rows))
    monkeypatch.setattr(air, "FOUNDATION_METADATA_SHA256", audio_manifest.sha256_file(metadata))
    _install_census(monkeypatch, rows)

    assert [row["uniq_id"] for row in air.load_rows()] == [0, 1, 2, 3, 4, 5, 6]
    assert [row["uniq_id"] for row in air.load_rows("speech")] == [0, 1, 2, 3]
    assert [row["uniq_id"] for row in air.load_rows(modalities="sound")] == [4, 5]
    assert [row["uniq_id"] for row in air.load_rows(tasks=["Music_AQA"])] == [6]
    assert [row["uniq_id"] for row in air.load_rows(max_samples=2)] == [0, 1]
    # Per-leaf cap keeps the first row of each leaf (file order), then the global cap applies.
    assert [row["uniq_id"] for row in air.load_rows(max_samples_per_subset=1)] == [0, 3, 4, 5, 6]
    assert [row["uniq_id"] for row in air.load_rows(max_samples_per_subset=2, max_samples=3)] == [0, 1, 3]

    monkeypatch.setattr(air, "EXPECTED_FOUNDATION_ROWS", 8)
    with pytest.raises(ValueError, match="Expected 8 rows"):
        air.load_rows()
    monkeypatch.setattr(air, "FOUNDATION_METADATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash drift"):
        air.load_rows()


def test_check_audio_reports_missing_and_empty_files(tmp_path: Path) -> None:
    rows = [_meta_row(0), _meta_row(1), _meta_row(2)]
    leaf = tmp_path / "Foundation" / air._leaf_id(rows[0])
    leaf.mkdir(parents=True)
    (leaf / "0.flac").write_bytes(b"audio")
    (leaf / "1.flac").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="1 missing .*1 empty"):
        air.check_audio(rows, tmp_path)
    air.check_audio(rows[:1], tmp_path)


class _FakeHub:
    """Stands in for the Hugging Face downloads: writes a small file for every requested audio path."""

    def __init__(self, rows: list[dict], metadata: Path) -> None:
        self.rows = rows
        self.metadata = metadata
        self.single: list[str] = []
        self.snapshots: list[list[str]] = []

    def hf_hub_download(self, repo_id, filename, *, repo_type, revision, local_dir=None):
        assert repo_id == air.HF_REPO_ID and repo_type == "dataset" and revision == air.HF_REVISION
        if filename == air.FOUNDATION_METADATA_PATH:
            return str(self.metadata)
        self.single.append(filename)
        self._write(Path(local_dir) / filename)

    def snapshot_download(self, repo_id, *, repo_type, revision, local_dir, allow_patterns):
        assert repo_id == air.HF_REPO_ID and repo_type == "dataset" and revision == air.HF_REVISION
        self.snapshots.append(allow_patterns)
        for row in self.rows:
            if f"Foundation/{air._leaf_id(row)}/**" in allow_patterns:
                self._write(Path(local_dir) / air._audio_repo_path(row))

    @staticmethod
    def _write(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> tuple[Path, _FakeHub]:
    """Run prepare() against synthetic metadata and fake downloads, writing under tmp_path."""
    rows = _synthetic_rows()
    metadata = tmp_path / "Foundation_meta.json"
    metadata.write_text(json.dumps(rows))
    hub = _FakeHub(rows, metadata)
    monkeypatch.setattr(air, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(air, "snapshot_download", hub.snapshot_download)
    monkeypatch.setattr(air, "EXPECTED_FOUNDATION_ROWS", len(rows))
    monkeypatch.setattr(air, "FOUNDATION_METADATA_SHA256", audio_manifest.sha256_file(metadata))
    _install_census(monkeypatch, rows)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(air, "DATA_DIR", data_dir)
    monkeypatch.setattr(air, "OUTPUT_FPATH", data_dir / f"{NAME}_benchmark.jsonl")
    return data_dir, hub


def test_prepare_defaults_to_every_leaf_with_existing_audio(offline_prepare: tuple[Path, _FakeHub]) -> None:
    data_dir, hub = offline_prepare
    output = air.prepare()
    assert output == data_dir / f"{NAME}_benchmark.jsonl"
    assert len(hub.snapshots) == 1 and not hub.single, "the full leaves are fetched by folder"
    assert len(hub.snapshots[0]) == 5 and "Foundation/Audio_Grounding_AudioGrounding/**" in hub.snapshots[0]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [f"air-bench-foundation:{i}" for i in range(1, 8)]
    assert {row["benchmark_profile"] for row in rows} == {"air-bench-foundation"}
    assert {row["task_name"] for row in rows} >= {"Speech_Grounding", "vocal_sound_classification", "Music_AQA"}

    first = rows[0]
    request = first["responses_create_params"]
    audio_path = Path(request["metadata"]["audio_path"])
    assert audio_path == (data_dir / "Foundation" / "Speech_Grounding_librispeech" / "0.flac").resolve()
    assert audio_path.is_file()
    system, user = request["input"]
    assert system["role"] == "system" and "uppercase option letter" in system["content"]
    assert user["content"].startswith(air.OFFICIAL_QUESTION_PROMPT)
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["subset_for_metrics"] == "Speech_Grounding_librispeech" and first["uniq_id"] == 0

    grounding = rows[5]["responses_create_params"]["metadata"]["audio_path"]
    assert grounding.endswith("Foundation/Audio_Grounding_AudioGrounding/5.flac") and Path(grounding).is_file()

    assert air.is_prepared_data_current(output)
    assert not air.is_prepared_data_current(data_dir / "other.jsonl")

    hub.snapshots.clear()
    air.prepare()
    assert not hub.snapshots, "existing audio must not be fetched again"


def test_prepare_narrows_by_modality_and_task(offline_prepare: tuple[Path, _FakeHub]) -> None:
    data_dir, hub = offline_prepare
    output = air.prepare(modalities="speech")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["uniq_id"] for row in rows] == [0, 1, 2, 3]
    assert {row["air_bench_foundation_leaf"] for row in rows} == set(air.SPEECH_LEAVES)
    assert hub.snapshots == [[f"Foundation/{leaf}/**" for leaf in sorted(air.SPEECH_LEAVES)]]
    assert not (data_dir / "Foundation" / "Music_AQA_music_avqa").exists(), "other modalities are not fetched"
    assert not air.is_prepared_data_current(output), "narrowed data must be prepared again by default"

    output = air.prepare(modalities="music,sound", tasks=["Music_AQA"])
    assert [json.loads(line)["uniq_id"] for line in output.read_text().splitlines()] == [6]
    with pytest.raises(ValueError, match="Empty or unknown"):
        air.prepare(tasks="Nope")


def test_prepare_with_everything_listed_is_still_the_default(offline_prepare: tuple[Path, _FakeHub]) -> None:
    output = air.prepare(modalities="speech,sound,music", tasks=list(air.TASK_MODALITIES))
    assert air.is_prepared_data_current(output)


def test_prepare_audio_root_and_row_cap_override(offline_prepare: tuple[Path, _FakeHub]) -> None:
    data_dir, hub = offline_prepare
    output = air.prepare(audio_root="/data/air", max_samples_per_profile=2)
    leaf = "Speech_Grounding_librispeech"
    assert_capped_and_remapped_rows(air, output, rows=2, audio_path=f"/data/air/Foundation/{leaf}/0.flac")
    assert hub.single == [f"Foundation/{leaf}/0.flac", f"Foundation/{leaf}/1.flac"] and not hub.snapshots
    assert not (data_dir / "Foundation" / leaf / "2.flac").exists(), "capped runs fetch only the kept rows"


def test_prepare_per_subset_cap_fetches_file_by_file_and_validates_caps(
    offline_prepare: tuple[Path, _FakeHub],
) -> None:
    data_dir, hub = offline_prepare
    output = air.prepare(max_samples_per_subset=1)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["uniq_id"] for row in rows] == [0, 3, 4, 5, 6]
    assert len(hub.single) == 5 and not hub.snapshots
    assert not air.is_prepared_data_current(output)
    for name in ("max_samples_per_profile", "max_samples_per_subset"):
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
                air.prepare(**{name: bad})
