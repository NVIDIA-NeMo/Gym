# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the VoiceBench benchmark: declaration, prompts, row building, prepare()."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from benchmarks import audio_manifest
from benchmarks.voicebench import prepare as voicebench
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from resources_servers.voicebench.task_data import SUBSET_SCORERS, VoiceBenchMetadata
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows, read_rows


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "voicebench"
PROMPTS = BENCHMARK / "prompts"

# SHA-256 of the two judge prompt files, headers included: any edit must be deliberate.
JUDGE_PROMPT_SHA256 = {
    "judge_open.yaml": "b831ec07d10895812d690618d1683476c616588eb661d0502f9ad4957962b1c1",  # pragma: allowlist secret
    "judge_reference_qa.yaml": "684b614d44675bce03cc9d9f51bb268157f38dc4eac51f76c6289df949909140",  # pragma: allowlist secret
}


def _config() -> dict:
    return yaml.safe_load((BENCHMARK / "config.yaml").read_text())


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert dataset.prepare_dependencies == ["pyarrow"]
    assert manifest.resources_server == "voicebench" and manifest.licensing == "Apache-2.0"
    assert manifest.canonical_split == "benchmark" and manifest.prompt_source == "prepared" and manifest.experimental


def test_subset_row_counts_match_the_server_config() -> None:
    server = _config()["voicebench_benchmark_resources_server"]["resources_servers"]["voicebench"]
    assert server["expected_rows"] == voicebench.EXPECTED_ROWS
    assert set(voicebench.SUBSETS) == set(SUBSET_SCORERS)
    # `all` takes neither: the inherited server defaults (null) must not be overridden here.
    assert "scoring_type" not in server and "expected_num_rows" not in server


def test_every_subset_has_a_system_prompt() -> None:
    prompts = yaml.safe_load(voicebench.SYSTEM_PROMPTS.read_text())
    assert set(prompts) == set(voicebench.SUBSETS)
    assert all(isinstance(text, str) and text for text in prompts.values())


@pytest.mark.parametrize("name", ["judge_open.yaml", "judge_reference_qa.yaml"])
def test_judge_prompts_keep_the_placeholders_the_server_fills(name: str) -> None:
    prompt = yaml.safe_load((PROMPTS / name).read_text())
    assert set(prompt) == {"system", "user"}
    placeholders = {"{prompt}", "{response}"} | ({"{reference}"} if name == "judge_reference_qa.yaml" else set())
    assert all(prompt["user"].count(token) == 1 for token in placeholders)
    assert "{reference}" not in prompt["user"] or name == "judge_reference_qa.yaml"


@pytest.mark.parametrize("name", sorted(JUDGE_PROMPT_SHA256))
def test_judge_prompt_files_are_pinned(name: str) -> None:
    assert hashlib.sha256((PROMPTS / name).read_bytes()).hexdigest() == JUDGE_PROMPT_SHA256[name]


def test_judge_prompts_are_the_upstream_text() -> None:
    """The prompts are VoiceBench's own (``api_judge.py``): pin the exact text, not just its shape."""
    open_prompt = yaml.safe_load((PROMPTS / "judge_open.yaml").read_text())
    assert open_prompt["system"] == "You are a helpful assistant who tries to help answer the user's question."
    assert open_prompt["user"].startswith("\nI need your help to evaluate the performance of several models")
    assert "### [Instruction]: {prompt}\n### [Response]: {response}\n" in open_prompt["user"]
    assert open_prompt["user"].endswith("You don’t need to provide any explanations.\n")
    qa_prompt = yaml.safe_load((PROMPTS / "judge_reference_qa.yaml").read_text())
    assert qa_prompt["user"] == (
        "### Question\n{prompt}\n\n### Reference answer\n{reference}\n\n### Candidate answer\n{response}\n\n"
        'Is the candidate answer correct based on the question and reference answer? \nPlease only output a single "Yes" '
        'or "No". Do not output anything else.'
    )


def _fake_sources(monkeypatch, rows_per_split: int) -> None:
    """Replace the Hugging Face reads: every split yields ``rows_per_split`` rows with dummy audio bytes."""
    subset_of = {config: subset for subset, (config, _) in voicebench.SUBSETS.items()}

    def shards(config: str) -> dict[str, list[str]]:
        return {split: [f"{config}/{split}"] for split in voicebench.SUBSETS[subset_of[config]][1]}

    def source_rows(shard_names: list[str]):
        config, split = shard_names[0].split("/")
        subset = subset_of[config]
        for index in range(rows_per_split):
            row = {"prompt": f"{subset} {split} {index}", "audio": {"bytes": b"RIFF", "path": f"{index}.wav"}}
            row |= {field: 0 if field == "question_id" else "A" for field in voicebench.SOURCE_FIELDS.get(subset, ())}
            if subset == "ifeval":
                row |= {"key": index, "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
            yield row

    monkeypatch.setattr(voicebench, "_shards", shards)
    monkeypatch.setattr(voicebench, "_source_rows", source_rows)


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(voicebench, "DATA_DIR", data_dir)
    monkeypatch.setattr(voicebench, "OUTPUT_FPATH", data_dir / "voicebench_benchmark.jsonl")
    _fake_sources(monkeypatch, rows_per_split=3)
    return data_dir


def test_verifier_metadata_keeps_what_each_scorer_reads() -> None:
    source = {"prompt": "q", "reference": "B", "id": "x1", "extra": "dropped"}
    metadata = voicebench.verifier_metadata("bbh", "test", 7, source)
    assert metadata == {
        "subset": "bbh",
        "sample_id": "test/000007",
        "split": "test",
        "prompt": "q",
        "reference": "B",
        "id": "x1",
    }
    assert VoiceBenchMetadata.model_validate(metadata).subset == "bbh"
    with pytest.raises(KeyError):
        voicebench.verifier_metadata("sd-qa", "usa", 0, {"prompt": "q"})


def test_build_row_carries_audio_and_sampling_but_no_answers() -> None:
    row = voicebench.build_row("openbookqa", {"subset": "openbookqa"}, "/data/a.wav", "SYSTEM")
    params = row["responses_create_params"]
    assert params["input"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": ""}]
    assert params["metadata"]["audio_path"] == "/data/a.wav"
    assert (params["temperature"], params["top_p"], params["max_output_tokens"]) == (0.6, 0.95, 20480)
    extra = json.loads(params["metadata"]["extra_body"])
    assert extra["seed"] == 0 and extra["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 16384}
    assert "reference" not in json.dumps(params)


def test_prepare_with_a_cap_writes_the_first_rows_of_every_split(offline_prepare: Path) -> None:
    output = voicebench.prepare(max_samples_per_profile=2)
    rows = read_rows(output)
    splits = sum(len(splits) for _, splits in voicebench.SUBSETS.values())
    assert splits == 42 and len(rows) == 2 * splits
    assert {row["verifier_metadata"]["subset"] for row in rows} == set(voicebench.SUBSETS)
    first = rows[0]
    assert first["verifier_metadata"]["sample_id"] == "test/000000"
    audio = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio == (offline_prepare / "audio" / "advbench" / "test" / "000000.wav").resolve() and audio.is_file()
    # Every written row is accepted by the server's metadata model, including the subset-specific fields.
    for row in rows:
        VoiceBenchMetadata.model_validate(row["verifier_metadata"])
    mmsu = [row["verifier_metadata"] for row in rows if row["verifier_metadata"]["subset"] == "mmsu"]
    assert {m["split"] for m in mmsu} == set(voicebench.MMSU_ROWS) and all(m["category"] for m in mmsu)


def test_a_capped_or_remapped_run_is_not_current_and_audio_root_prefixes_rows(offline_prepare: Path) -> None:
    output = voicebench.prepare(audio_root="/data/voicebench", max_samples_per_profile=1)
    assert_capped_and_remapped_rows(
        voicebench, output, rows=42, audio_path="/data/voicebench/audio/advbench/test/000000.wav"
    )
    assert (offline_prepare / "audio" / "advbench" / "test" / "000000.wav").is_file(), "audio is written locally"


def test_a_split_with_the_wrong_row_count_fails_without_a_cap(offline_prepare: Path) -> None:
    with pytest.raises(ValueError, match="has 3 rows; expected 520"):
        voicebench.prepare()


@pytest.mark.parametrize("cap", [0, -1])
def test_prepare_rejects_a_row_cap_below_one_before_reading_anything(cap: int, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("nothing may be read")

    monkeypatch.setattr(voicebench, "_shards", fail)
    with pytest.raises(ValueError, match="at least 1"):
        voicebench.prepare(max_samples_per_profile=cap)


def test_audio_without_bytes_or_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither bytes nor a path"):
        voicebench._write_audio({"bytes": None, "path": None}, tmp_path / "x.wav")


def test_the_code_stamp_covers_the_prompts_and_the_row_builder() -> None:
    assert voicebench.SYSTEM_PROMPTS in voicebench.CODE_FILES
    assert Path(audio_manifest.__file__) in voicebench.CODE_FILES
