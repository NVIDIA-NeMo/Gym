# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/bunsen_chem/prepare.py`` and materialization helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.bunsen_chem import prepare as prepare_module
from benchmarks.bunsen_chem import upstream
from benchmarks.bunsen_chem.materialize import (
    PROMPT_VERSION,
    materialize_dataset,
    materialize_row,
    validate_reconstituted_rows,
)
from benchmarks.bunsen_chem.taxonomy import BCT_SUBFIELDS, normalize_taxonomy_label


def _row(label: dict[str, str] | None = None) -> dict:
    label = label or {"bct_field": "general", "bct_subfield": "bonding"}
    return {
        **upstream.EXPECTED_CONFIG_METADATA,
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
        "filter_flags": ["chemistry", "mcq", "valid_answer", "public_source", "chembench"],
        "bct_field": label["bct_field"],
        "bct_subfield": label["bct_subfield"],
        "question": "Which formula is water?",
        "choices": ["H2O", "CO2", "NaCl"],
        "answer": "H2O",
        "answer_index": 0,
        "source_meta": {"subfield": "general_chemistry"},
    }


def test_upstream_config_metadata_matches_expected_versions() -> None:
    builder = SimpleNamespace(
        config=SimpleNamespace(
            description=(
                "Chemistry MCQ; release=bunsen_chem_public_v0.1.0; "
                "transform_version=bunsen_chem_sources_v2; filter_version=mcq_public_v1; "
                "taxonomy_version=bct_gpt55_low_v1"
            )
        )
    )

    assert upstream.config_metadata(builder) == upstream.EXPECTED_CONFIG_METADATA
    assert upstream.validate_config_metadata(builder) == upstream.EXPECTED_CONFIG_METADATA


def test_upstream_config_metadata_rejects_unexpected_versions() -> None:
    builder = SimpleNamespace(
        config=SimpleNamespace(
            description=(
                "Chemistry MCQ; release=bunsen_chem_public_v0.1.0; "
                "transform_version=bunsen_chem_sources_v2; filter_version=mcq_public_v1; "
                "taxonomy_version=unexpected"
            )
        )
    )

    with pytest.raises(ValueError, match="taxonomy_version"):
        upstream.validate_config_metadata(builder)


def test_reconstitute_upstream_dataset_uses_hf_builder_and_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = SimpleNamespace(config=SimpleNamespace(description=_metadata_description()))
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
    builder = SimpleNamespace(config=SimpleNamespace(description=_metadata_description()))

    class FakeTool:
        @staticmethod
        def reconstitute(*args, **kwargs):
            row = _row()
            row["taxonomy_version"] = "unexpected"
            return [row]

    monkeypatch.setattr(upstream, "get_hf_token", lambda token=None: "hf-token")
    monkeypatch.setattr(upstream, "load_manifest_builder", lambda *, token: builder)
    monkeypatch.setattr(upstream, "load_reconstitute_tool", lambda *, token: FakeTool)

    with pytest.raises(ValueError, match="taxonomy_version"):
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
    verify_calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tool_path)

    def fake_verify_file_sha256(path: Path, expected_sha256: str) -> None:
        verify_calls.append((path, expected_sha256))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(upstream, "verify_file_sha256", fake_verify_file_sha256)

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
    assert verify_calls == [(tool_path, upstream.RECONSTITUTE_TOOL_SHA256)]


def test_verify_file_sha256_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("payload", encoding="utf-8")

    upstream.verify_file_sha256(path, "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5")

    with pytest.raises(ValueError, match="Unexpected sha256"):
        upstream.verify_file_sha256(path, "0" * 64)


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


def test_taxonomy_subfields_are_unique_across_fields() -> None:
    owners: dict[str, list[str]] = {}
    for field, subfields in BCT_SUBFIELDS.items():
        for subfield in subfields:
            owners.setdefault(subfield, []).append(field)

    duplicates = {subfield: fields for subfield, fields in owners.items() if len(fields) > 1}

    assert duplicates == {}


def test_preference_metabolism_is_replaced_by_metabolic_stability() -> None:
    with pytest.raises(ValueError, match="Unknown bct_subfield"):
        normalize_taxonomy_label({"bct_field": "preference", "bct_subfield": "metabolism"})

    assert normalize_taxonomy_label({"bct_field": "preference", "bct_subfield": "metabolic_stability"}) == {
        "bct_field": "preference",
        "bct_subfield": "metabolic_stability",
    }
    assert normalize_taxonomy_label({"bct_field": "biochemistry", "bct_subfield": "metabolism"}) == {
        "bct_field": "biochemistry",
        "bct_subfield": "metabolism",
    }


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
    assert first["metadata"]["release"] == "bunsen_chem_public_v0.1.0"
    assert first["metadata"]["taxonomy_version"] == "bct_gpt55_low_v1"
    assert first["metadata"]["prompt_version"] == PROMPT_VERSION
    assert "answer" not in first["metadata"]
    assert "choices" not in first["metadata"]
    assert "question" not in first["metadata"]
    assert "source_meta" not in first["metadata"]
    assert "responses_create_params" not in first
    assert "grading_mode" not in first


def _metadata_description() -> str:
    return (
        "Chemistry MCQ; release=bunsen_chem_public_v0.1.0; "
        "transform_version=bunsen_chem_sources_v2; filter_version=mcq_public_v1; "
        "taxonomy_version=bct_gpt55_low_v1"
    )
