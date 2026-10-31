# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from benchmarks.indic.gsm8k import prepare as prepare_module


def source_row() -> dict:
    return {
        "id": "one",
        "question": "English question",
        "answer": "Original work\n#### 18",
        "answer_Hindi_translation": "bad translation #### 99",
        **{
            f"question_{name}_translation": f"{name} question\u2028line two"
            for name in prepare_module.LANGUAGES.values()
        },
    }


def test_translations_and_original_answers_without_demos() -> None:
    rows = prepare_module.build_rows([source_row()], languages=["hi", "ta"], source_sha256="abc")
    assert [r["question"] for r in rows] == ["Hindi question\u2028line two", "Tamil question\u2028line two"]
    assert all(r["expected_answer"] == "Original work\n#### 18" for r in rows)
    assert all(r["responses_create_params"] == {"max_output_tokens": 120000} for r in rows)
    assert rows[0]["uuid"] != rows[1]["uuid"]
    assert "prompt" not in rows[0]


def test_default_languages_and_stable_source_identity() -> None:
    rows = prepare_module.build_rows([source_row()], source_sha256="abc")
    assert [r["language_code"] for r in rows] == [
        "bn",
        "gu",
        "hi",
        "kn",
        "mr",
        "ml",
        "ne",
        "or",
        "pa",
        "ta",
        "te",
        "ur",
    ]
    assert rows == prepare_module.build_rows([source_row()], source_sha256="abc")
    changed = prepare_module.build_rows([source_row()], source_sha256="different")
    assert rows[0]["uuid"] != changed[0]["uuid"]
    assert all(r["source_repository"] == "ai4bharat/indic-gsm8k" for r in rows)


@pytest.mark.parametrize("change", [{"question_Hindi_translation": ""}, {"answer": "no final"}, {"id": None}])
def test_invalid_source_data(change: dict) -> None:
    with pytest.raises(ValueError):
        prepare_module.build_rows([{**source_row(), **change}], languages=["hi"], source_sha256="abc")


@pytest.mark.parametrize("rows", [[], [source_row(), source_row()]])
def test_empty_or_duplicate_source(rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        prepare_module.build_rows(rows, languages=["hi"], source_sha256="abc")


@pytest.mark.parametrize("languages", [[], ["hi", "hi"], "hi", ["unknown"]])
def test_invalid_language_selection(languages: object) -> None:
    with pytest.raises(ValueError):
        prepare_module.build_rows([source_row()], languages=languages, source_sha256="abc")


@pytest.fixture
def parquet_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist([source_row()]), source)
    monkeypatch.setattr(prepare_module, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    return source


def test_downloads_pinned_hf_source(parquet_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(parquet_source)

    monkeypatch.setattr(prepare_module, "hf_hub_download", download)
    output = prepare_module.prepare(output=tmp_path / "out.jsonl", languages=["hi"])
    assert calls == [
        {
            "repo_id": "ai4bharat/indic-gsm8k",
            "filename": "test.parquet",
            "repo_type": "dataset",
            "revision": "e21f3ff7e9fe703a7f4faeefdba27801be278eb5",
        }
    ]
    # Read physical JSONL records; U+2028 inside question text is not a record boundary.
    with output.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 1
    assert rows[0]["question"] == "Hindi question\u2028line two"
    assert prepare_module.is_prepared_data_current(output)


def test_local_copy_does_not_download(parquet_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_download(**kwargs: object) -> str:
        pytest.fail("Local preparation must not access the Hub")

    monkeypatch.setattr(prepare_module, "hf_hub_download", unexpected_download)
    output = prepare_module.prepare(
        source=str(parquet_source), output=str(tmp_path / "out.jsonl"), languages=["as", "sa"]
    )
    with output.open() as stream:
        assert [json.loads(line)["language_code"] for line in stream] == ["as", "sa"]


def test_changed_source_preserves_output(parquet_source: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    output.write_text("previous complete data\n")
    parquet_source.write_bytes(b"different dataset")
    with pytest.raises(ValueError, match="checksum differs"):
        prepare_module.prepare(source=parquet_source, output=output)
    assert output.read_text() == "previous complete data\n"


@pytest.mark.parametrize("content", ["", "invalid json\n", "null\n", "{}\n"])
def test_rejects_stale_cached_data(tmp_path: Path, content: str) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(content)
    assert not prepare_module.is_prepared_data_current(path)


def test_missing_cached_data(tmp_path: Path) -> None:
    assert not prepare_module.is_prepared_data_current(tmp_path / "missing.jsonl")


def test_chat_config_routes_environment_agent_and_verifier() -> None:
    root = Path(__file__).parents[4]
    config = OmegaConf.load(root / "benchmarks/indic/gsm8k/config.yaml")
    resources = OmegaConf.load(root / "resources_servers/gsm8k_reference/configs/gsm8k_reference.yaml")
    model = config.policy_model.responses_api_models.vllm_model
    assert model.use_completions_api is False
    assert model.chat_template_kwargs.enable_thinking is True
    assert not model.get("sampling_overrides")
    environment = config.indic_gsm8k_environment_server.environment_servers.legacy_agent
    assert environment.agent_server.name == "indic_gsm8k_simple_agent"
    assert (
        config.indic_gsm8k_simple_agent.responses_api_agents.simple_agent.resources_server.name
        == "indic_gsm8k_resources_server"
    )
    assert resources.gsm8k_reference_simple_agent.responses_api_agents.simple_agent.max_steps == 1
