# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Gym's dataset preparation entrypoint and generated answers."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from benchmarks.long_transduction import prepare as generator
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


def test_prepare_returns_cached_path_without_dependencies(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    output = tmp_path / "existing.jsonl"
    original = '{"existing": true}\n'
    output.write_text(original)
    monkeypatch.setattr(generator, "OUTPUT_FPATH", output)

    def unexpected_load() -> None:
        pytest.fail("Cached preparation must not load generation dependencies")

    monkeypatch.setattr(generator, "_get_encoder", unexpected_load)
    assert generator.prepare() == output
    assert output.read_text() == original


def test_prepare_declares_dependencies() -> None:
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "benchmarks/long_transduction/config.yaml")
    dataset = config.long_transduction_agent.responses_api_agents.simple_agent.datasets[0]
    assert {"tiktoken", "wonderwords"}.issubset(dataset.prepare_dependencies)


def test_prepare_generates_all_types(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("tiktoken")
    pytest.importorskip("wonderwords")
    output = tmp_path / "generated.jsonl"
    monkeypatch.setattr(generator, "DATA_DIR", tmp_path)
    monkeypatch.setattr(generator, "OUTPUT_FPATH", output)
    monkeypatch.setattr(generator, "TARGET_TOKENS_LIST", [2048, 4096])
    monkeypatch.setattr(generator, "N_SAMPLES", 1)
    monkeypatch.setattr(generator, "MAX_OPERANDS_RANGE", [2])
    monkeypatch.setattr(generator, "N_VARIABLES_RANGE", [8])
    monkeypatch.setattr(generator, "PERM_FRACTIONS", [1.0])
    monkeypatch.setattr(generator, "VOCAB_FRACTIONS", [1.0])
    output.write_text("old contents\n")
    assert generator.prepare(force=True) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 24
    assert {row["type"] for row in rows} == set(generator.PROMPT_TEMPLATES)
    assert all(row["question"] and row["expected_output"] for row in rows)
    assert {row["target_tokens"] for row in rows} == {2048, 4096}
    prompt_config = load_prompt_config("benchmarks/prompts/generic/default.yaml")
    for row in rows:
        materialized = apply_prompt_to_row(row, prompt_config)
        assert materialized["responses_create_params"]["max_output_tokens"] == row["target_tokens"] * 3 // 2
        assert materialized["responses_create_params"]["input"][-1]["content"] == row["question"]
        assert materialized["target_tokens"] == row["target_tokens"]

    # Check against the scoring contract as well as the file shape.
    from resources_servers.long_transduction.parse import (
        score_csv_permutation,
        score_response,
        score_response_numbered,
        score_unnumbered_uuid_sort,
        score_uuid_sort,
        score_var_expand_numbered,
        score_var_expand_unnumbered,
    )

    scorers = {
        "unnumbered_streaming_sum": (score_response, "expressions"),
        "streaming_sum": (score_response_numbered, "expressions"),
        "shuffled_streaming_sum": (score_response_numbered, "expressions"),
        "unnumbered_uuid_sort": (score_unnumbered_uuid_sort, "uuid_lines"),
        "streaming_uuid_sort": (score_uuid_sort, "uuid_lines"),
        "shuffled_streaming_uuid_sort": (score_uuid_sort, "uuid_lines"),
        "unnumbered_var_expand": (score_var_expand_unnumbered, "expressions"),
        "streaming_var_expand": (score_var_expand_numbered, "expressions"),
        "shuffled_streaming_var_expand": (score_var_expand_numbered, "expressions"),
    }
    for row in rows:
        if row["type"].startswith("csv"):
            scores, _, _ = score_csv_permutation(
                row["expected_output"], row["expected_output"], row["n_rows"], row["n_cols"]
            )
        else:
            scorer, field = scorers[row["type"]]
            scores = scorer(row["expected_output"], row[field])
        assert scores and all(all(item) for item in scores), row["type"]
