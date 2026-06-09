# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/bunsenbench_chemistry_mcq/prepare.py`` and materialization helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.bunsenbench_chemistry_mcq import prepare as prepare_module
from benchmarks.bunsenbench_chemistry_mcq import upstream
from benchmarks.bunsenbench_chemistry_mcq.materialize import (
    PROMPT_VERSION,
    materialize_dataset,
    materialize_row,
    validate_reconstituted_rows,
)


def _row(label: dict[str, str] | None = None) -> dict:
    label = label or {"bct_field": "general", "bct_subfield": "bonding"}
    return {
        **upstream.UPSTREAM_CONFIG_METADATA,
        "bunsen_id": "bunsen:example:1",
        "source": "chembench_general_chemistry",
        "source_dataset": "jablonkagroup/ChemBench",
        "source_config": "general_chemistry",
        "source_split": "train",
        "source_revision": "rev",
        "source_record_id": "1",
        "source_row_index": 0,
        "source_record_sha256": "source-hash",
        "canonical_problem_sha256": "problem-hash",
        "bct_field": label["bct_field"],
        "bct_subfield": label["bct_subfield"],
        "question": "Which formula is water?",
        "choices": ["H2O", "CO2", "NaCl"],
        "answer": "H2O",
        "answer_index": 0,
        "source_meta": {"subfield": "general_chemistry"},
    }


def test_upstream_config_metadata_matches_expected_versions() -> None:
    builder = _builder()

    assert upstream.config_metadata(builder) == upstream.UPSTREAM_CONFIG_METADATA
    assert upstream.validate_config_metadata(builder) == upstream.UPSTREAM_CONFIG_METADATA


def test_upstream_config_metadata_rejects_unexpected_versions() -> None:
    builder = _builder(version="0.1.1")

    with pytest.raises(ValueError, match="bunsen_bench_config_version"):
        upstream.validate_config_metadata(builder)


def test_upstream_config_metadata_rejects_unexpected_split() -> None:
    builder = _builder(splits={"manifest": object()})

    with pytest.raises(ValueError, match="Unexpected Bunsen Bench splits"):
        upstream.validate_config_metadata(builder)


def test_reconstitute_upstream_dataset_uses_hf_builder_and_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    calls = []

    class FakeTool:
        @staticmethod
        def reconstitute(*args, **kwargs):
            calls.append((args, kwargs))
            return [_row()]

    monkeypatch.setattr(upstream, "get_hf_token", lambda token=None: "hf-token")
    monkeypatch.setattr(upstream, "load_manifest_builder", lambda *, token: builder)
    monkeypatch.setattr(upstream, "load_reconstitute_tool", lambda *, token: FakeTool)

    dataset = upstream.reconstitute_upstream_dataset(limit=7, verify_hashes=True, verbose=False)

    assert dataset == [_row()]
    assert calls == [
        (
            (builder,),
            {
                "token": "hf-token",
                "limit": 7,
                "verify_hashes": True,
                "include_raw_row": False,
                "verbose": False,
            },
        )
    ]


def test_reconstitute_upstream_dataset_rejects_metadata_collisions(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()

    class FakeTool:
        @staticmethod
        def reconstitute(*args, **kwargs):
            row = _row()
            row["bunsen_bench_revision"] = "unexpected"
            return [row]

    monkeypatch.setattr(upstream, "get_hf_token", lambda token=None: "hf-token")
    monkeypatch.setattr(upstream, "load_manifest_builder", lambda *, token: builder)
    monkeypatch.setattr(upstream, "load_reconstitute_tool", lambda *, token: FakeTool)

    with pytest.raises(ValueError, match="bunsen_bench_revision"):
        upstream.reconstitute_upstream_dataset()


def test_load_manifest_builder_uses_pinned_upstream_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    builder = SimpleNamespace()

    def fake_load_dataset_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return builder

    monkeypatch.setattr("datasets.load_dataset_builder", fake_load_dataset_builder)

    assert upstream.load_manifest_builder(token="hf-token") is builder
    assert calls == [
        (
            ("nvidia/bunsen-bench", "chemistry_mcq"),
            {"revision": upstream.BUNSEN_BENCH_REVISION, "token": "hf-token"},
        )
    ]


def test_load_reconstitute_tool_downloads_from_upstream_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_path = tmp_path / "reconstitute.py"
    tool_path.write_text("VALUE = 1\n", encoding="utf-8")
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tool_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    module = upstream.load_reconstitute_tool(token="hf-token")

    assert module.VALUE == 1
    assert calls == [
        {
            "repo_id": "nvidia/bunsen-bench",
            "repo_type": "dataset",
            "filename": "tools/reconstitute.py",
            "revision": upstream.BUNSEN_BENCH_REVISION,
            "token": "hf-token",
        }
    ]


def test_prepare_materializes_reconstituted_upstream_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_module, "reconstitute_upstream_dataset", lambda limit=None: [_row()])

    output_path = tmp_path / "bunsen_chem.jsonl"
    assert prepare_module.prepare(output_path, limit=1) == output_path

    prepared = json.loads(output_path.read_text(encoding="utf-8"))
    assert prepared["uuid"] == "bunsen:example:1"
    assert prepared["metadata"]["bct_field"] == "general"


def test_reconstituted_row_validation_rejects_payload_drift() -> None:
    row = _row()
    row.pop("canonical_problem_sha256")

    with pytest.raises(ValueError, match="missing fields"):
        validate_reconstituted_rows([row])


def test_reconstituted_row_validation_rejects_inconsistent_answer_index() -> None:
    row = _row()
    row["answer_index"] = 2

    with pytest.raises(ValueError, match="answer_index"):
        validate_reconstituted_rows([row])


def test_reconstituted_row_validation_rejects_unexpected_fields() -> None:
    row = _row()
    row["legacy_taxonomy"] = "general_chemistry"

    with pytest.raises(ValueError, match="unexpected fields"):
        validate_reconstituted_rows([row])


def test_empty_dataset_materialization_writes_empty_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "bunsen_chem.jsonl"

    assert materialize_dataset([], output_path) == output_path
    assert output_path.read_text(encoding="utf-8") == ""


def test_materialize_dataset_rejects_duplicate_source_locators(tmp_path: Path) -> None:
    first = _row()
    second = _row()
    second["bunsen_id"] = "bunsen:example:2"

    with pytest.raises(ValueError, match="Duplicate source locator"):
        materialize_dataset([first, second], tmp_path / "bunsen_chem.jsonl")


def test_materialize_row_is_deterministic_and_letter_grades() -> None:
    row = _row()
    first = materialize_row(row)
    second = materialize_row(row)

    assert first == second
    assert first["expected_answer"] in {"A", "B", "C"}
    option_by_letter = {letter: text for option in first["options"] for letter, text in option.items()}
    assert option_by_letter[first["expected_answer"]] == row["answer"]
    assert first["options_text"].startswith("<choices>\n<choice>")
    assert first["options_text"].endswith("</choice>\n</choices>")
    assert "A:" not in first["options_text"]
    assert first["metadata"]["source_row_index"] == 0
    assert first["metadata"]["bunsen_bench_revision"] == upstream.BUNSEN_BENCH_REVISION
    assert first["metadata"]["bunsen_bench_config"] == upstream.BUNSEN_BENCH_CONFIG_NAME
    assert first["metadata"]["bunsen_bench_config_version"] == upstream.BUNSEN_BENCH_VERSION
    assert first["metadata"]["prompt_version"] == PROMPT_VERSION
    assert "filter_flags" not in first["metadata"]
    assert "release" not in first["metadata"]
    assert "taxonomy_version" not in first["metadata"]
    assert "answer" not in first["metadata"]
    assert "choices" not in first["metadata"]
    assert "question" not in first["metadata"]
    assert "source_meta" not in first["metadata"]
    assert "responses_create_params" not in first
    assert "grading_mode" not in first


def _builder(
    *,
    name: str = "chemistry_mcq",
    version: str = upstream.BUNSEN_BENCH_VERSION,
    description: str = "Chemistry MCQ evaluation manifest",
    splits: dict[str, object] | None = None,
) -> SimpleNamespace:
    splits = splits or {upstream.BUNSEN_BENCH_SPLIT_NAME: object()}
    return SimpleNamespace(
        config=SimpleNamespace(name=name, version=version, description=description),
        info=SimpleNamespace(splits=splits),
    )
