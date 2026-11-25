# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the VoiceCodeBench benchmark: declaration, census, row building, prepare()."""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from benchmarks import audio_manifest
from benchmarks.voice_code_bench import prepare as vcb
from nemo_gym.environment.manifest import load_manifest
from resources_servers.audio_eval.score import evaluator_settings, locate, resolve
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "voice_code_bench"
NAME = "voice_code_bench"


def _meta_row(index: int, domain: str = "contact_routing", entities: int = 2) -> dict:
    return {
        "audio_id": f"{domain}_{index:03d}",
        "file_name": f"audio/{index:03d}.wav",
        "domain": domain,
        "scenario": "callback_instructions",
        "difficulty": "light",
        "duration": 3.5 + index,
        "language": "en",
        "transcripts": {
            "template": "Call <e01> or <e02>",
            "acoustic": "Call five five five",
            "canonical": "Call 555 or 556",
        },
        "entities": [{"id": f"e{n:02d}", "type": "phone_number", "canonical": "555"} for n in range(entities)],
        "entity_count": entities,
        "entity_types": ["phone_number"],
        "speaker": {"id": "s1"},
    }


def _synthetic_rows() -> list[dict]:
    return [_meta_row(1), _meta_row(2), _meta_row(3, "finance_billing"), _meta_row(4, "technical_it_dev", 1)]


def _install_census(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr(vcb, "EXPECTED_ROWS", len(rows))
    monkeypatch.setattr(vcb, "EXPECTED_ENTITIES", sum(row["entity_count"] for row in rows))
    counts = Counter(row["domain"] for row in rows)
    monkeypatch.setattr(vcb, "EXPECTED_DOMAIN_COUNTS", dict(counts))
    monkeypatch.setattr(vcb, "DOMAINS", tuple(counts))


def test_declaration_specifics() -> None:
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "MIT AND LicenseRef-VoiceCodeBench-Audio" and manifest.domain == "asr"
    server = config[f"{NAME}_resources_server"]["resources_servers"]["audio_eval"]
    assert server["judge_model_server"] == {"type": "responses_api_models", "name": "judge_model"}
    assert "judge_model" in config


def test_profile_contract_resolves_through_the_server() -> None:
    contract = audio_manifest.load_profiles(vcb.PROFILES)["profiles"]["voice-code-bench"]
    assert evaluator_settings(contract) == (
        "voice_code_bench",
        {"verifier_id": "openai_gpt_5_5_v1", "cleanup_generation_loops": True},
    )
    module, evaluator = locate("evaluators", "voice_code_bench")
    assert evaluator.__name__ == "VoiceCodeBenchEvaluator" and hasattr(module, "JUDGE")
    assert resolve("metrics", contract["skills"]["METRICS_TYPE"]).__name__ == "VoiceCodeBenchMetrics"


def test_prepare_rejects_a_bad_row_cap_before_any_download(offline_prepare) -> None:
    _, hub = offline_prepare
    for bad in (0, -1, 2.5, True, "3"):
        with pytest.raises(ValueError, match="positive integer"):
            vcb.prepare(max_samples_per_profile=bad)
    assert not hub.single and not hub.snapshots


def test_profile_contract_resolves_to_the_greedy_transcription_request() -> None:
    contract = audio_manifest.load_profiles(vcb.PROFILES)["profiles"]["voice-code-bench"]
    assert contract["skills"]["METRICS_TYPE"] == "voice_code_bench"
    assert "++eval_config.cleanup_generation_loops=true" in contract["skills"]["EVAL_ARGS"]
    assert contract["prompt_override"]["system_prompt"] == ""
    controls = audio_manifest.request_controls(contract)
    assert controls["temperature"] == 0.0 and controls["max_output_tokens"] == 256
    assert controls["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_constants_describe_the_released_census() -> None:
    assert sum(vcb.EXPECTED_DOMAIN_COUNTS.values()) == vcb.EXPECTED_ROWS == 300
    assert len(vcb.DOMAINS) == 8 and vcb.EXPECTED_ENTITIES == 1482
    assert len(vcb.METADATA_SHA256) == 64 and len(vcb.HF_REVISION) == 40


@pytest.mark.parametrize(
    "override, error",
    [
        ({"audio_id": ""}, "missing audio_id"),
        ({"transcripts": {"template": "x", "acoustic": "y"}}, "invalid transcript layers"),
        ({"entity_count": 5}, "inconsistent entities"),
        ({"file_name": "wav/001.wav"}, "invalid file_name"),
    ],
)
def test_validation_rejects_malformed_rows(monkeypatch, override: dict, error: str) -> None:
    rows = _synthetic_rows()
    _install_census(monkeypatch, rows)
    vcb.validate_rows(rows)
    with pytest.raises(ValueError, match=error):
        vcb.validate_rows([rows[0] | override, *rows[1:]])


def test_validation_detects_count_duplicate_entity_and_domain_drift(monkeypatch) -> None:
    rows = _synthetic_rows()
    _install_census(monkeypatch, rows)
    with pytest.raises(ValueError, match="Expected 4"):
        vcb.validate_rows(rows[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        vcb.validate_rows([rows[0], rows[0] | {"file_name": "audio/009.wav"}, *rows[2:]])
    monkeypatch.setattr(vcb, "EXPECTED_ENTITIES", 99)
    with pytest.raises(ValueError, match="target entities"):
        vcb.validate_rows(rows)
    monkeypatch.setattr(vcb, "EXPECTED_ENTITIES", 7)
    monkeypatch.setattr(vcb, "EXPECTED_DOMAIN_COUNTS", {"contact_routing": 4})
    with pytest.raises(ValueError, match="domain drift"):
        vcb.validate_rows(rows)


def test_select_domains_defaults_to_all_and_rejects_unknown_names() -> None:
    assert vcb.select_domains() == vcb.DOMAINS
    assert vcb.select_domains("finance_billing, contact_routing,finance_billing") == (
        "finance_billing",
        "contact_routing",
    )
    assert vcb.select_domains(["dense_mixed_stress"]) == ("dense_mixed_stress",)
    for value in ("nope", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            vcb.select_domains(value)


def test_source_row_keeps_the_validated_message_and_scorer_metadata() -> None:
    row = _meta_row(7, "finance_billing")
    source = vcb.format_source_row(row, "/data/voice-code-bench")
    audio_path = "/data/voice-code-bench/data/audio/007.wav"
    assert source["messages"] == [
        {"role": "user", "content": "Transcribe the following:", "audio": {"path": audio_path, "duration": 10.5}}
    ]
    assert source["expected_answer"] == "Call 555 or 556" and source["audio_filepath"] == audio_path
    assert source["subset_for_metrics"] == "finance_billing" and source["audio_duration"] == 10.5
    assert source["task_type"] == "VoiceCodeBench-ASR" and source["voice_code_bench_revision"] == vcb.HF_REVISION
    assert set(source["voice_code_bench"]) == set(vcb.SCORER_FIELDS)
    assert "speaker" not in source["voice_code_bench"] and "language" not in source["voice_code_bench"]


class _FakeHub:
    """Stands in for the Hugging Face downloads: serves the metadata and writes a small file per audio path."""

    def __init__(self, metadata: Path) -> None:
        self.metadata = metadata
        self.single: list[str] = []
        self.snapshots: list[list[str]] = []

    def hf_hub_download(self, repo_id, filename, *, repo_type, revision, local_dir=None):
        assert repo_id == vcb.HF_REPO_ID and repo_type == "dataset" and revision == vcb.HF_REVISION
        if filename == vcb.METADATA_PATH:
            return str(self.metadata)
        self.single.append(filename)
        self._write(Path(local_dir) / filename)

    def snapshot_download(self, repo_id, *, repo_type, revision, local_dir, allow_patterns):
        assert repo_id == vcb.HF_REPO_ID and repo_type == "dataset" and revision == vcb.HF_REVISION
        self.snapshots.append(allow_patterns)
        for index in range(1, 5):
            self._write(Path(local_dir) / f"data/audio/{index:03d}.wav")

    @staticmethod
    def _write(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> tuple[Path, _FakeHub]:
    rows = _synthetic_rows()
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    hub = _FakeHub(metadata)
    monkeypatch.setattr(vcb, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(vcb, "snapshot_download", hub.snapshot_download)
    monkeypatch.setattr(vcb, "METADATA_SHA256", audio_manifest.sha256_file(metadata))
    _install_census(monkeypatch, rows)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(vcb, "DATA_DIR", data_dir)
    monkeypatch.setattr(vcb, "SOURCE_DIR", data_dir / "voice-code-bench")
    monkeypatch.setattr(vcb, "OUTPUT_FPATH", data_dir / f"{NAME}.jsonl")
    return data_dir, hub


def test_load_rows_checks_the_hash_and_narrows(offline_prepare: tuple[Path, _FakeHub], monkeypatch) -> None:
    assert [row["audio_id"] for row in vcb.load_rows()] == [r["audio_id"] for r in _synthetic_rows()]
    assert [row["domain"] for row in vcb.load_rows("finance_billing,technical_it_dev")] == [
        "finance_billing",
        "technical_it_dev",
    ]
    assert len(vcb.load_rows(max_samples=1)) == 1
    monkeypatch.setattr(vcb, "METADATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash drift"):
        vcb.load_rows()


def test_prepare_defaults_to_every_recording(offline_prepare: tuple[Path, _FakeHub]) -> None:
    data_dir, hub = offline_prepare
    output = vcb.prepare()
    assert output == data_dir / f"{NAME}.jsonl"
    assert hub.snapshots == [["data/audio/*.wav"]] and not hub.single
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [f"voice-code-bench:{i}" for i in range(1, 5)]
    assert {row["benchmark_profile"] for row in rows} == {"voice-code-bench"}

    first = rows[0]
    request = first["responses_create_params"]
    audio_path = Path(request["metadata"]["audio_path"])
    assert audio_path == (data_dir / "voice-code-bench" / "data" / "audio" / "001.wav").resolve()
    assert audio_path.is_file()
    (user,) = request["input"]
    assert user["role"] == "user" and user["content"].startswith("Transcribe the audio as a clean written transcript.")
    assert first["ng_source_row"]["messages"][0]["content"] == "Transcribe the following:"
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["subset_for_metrics"] == "contact_routing"

    assert vcb.is_prepared_data_current(output)
    assert not vcb.is_prepared_data_current(data_dir / "other.jsonl")
    hub.snapshots.clear()
    vcb.prepare()
    assert not hub.snapshots, "existing audio must not be fetched again"


def test_prepare_narrows_by_domain_and_fetches_file_by_file(offline_prepare: tuple[Path, _FakeHub]) -> None:
    data_dir, hub = offline_prepare
    output = vcb.prepare(domains="finance_billing")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["audio_id"] for row in rows] == ["finance_billing_003"]
    assert hub.single == ["data/audio/003.wav"] and not hub.snapshots
    assert not (data_dir / "voice-code-bench" / "data" / "audio" / "001.wav").exists()
    assert not vcb.is_prepared_data_current(output), "narrowed data must be prepared again by default"
    with pytest.raises(ValueError, match="Empty or unknown"):
        vcb.prepare(domains="nope")


def test_prepare_with_every_domain_listed_is_still_the_default(offline_prepare: tuple[Path, _FakeHub]) -> None:
    output = vcb.prepare(domains=list(vcb.DOMAINS))
    assert vcb.is_prepared_data_current(output)


def test_prepare_audio_root_and_row_cap_override(offline_prepare: tuple[Path, _FakeHub]) -> None:
    _, hub = offline_prepare
    output = vcb.prepare(audio_root="/data/voice-code-bench/", max_samples_per_profile=2)
    assert_capped_and_remapped_rows(vcb, output, rows=2, audio_path="/data/voice-code-bench/data/audio/001.wav")
    assert hub.single == ["data/audio/001.wav", "data/audio/002.wav"] and not hub.snapshots


def test_download_rejects_missing_or_empty_audio(offline_prepare: tuple[Path, _FakeHub], monkeypatch) -> None:
    data_dir, hub = offline_prepare
    rows = vcb.load_rows()[:1]
    path = data_dir / "voice-code-bench" / "data" / "audio" / "001.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Missing or empty"):
        vcb.download_audio(rows, full=False)
