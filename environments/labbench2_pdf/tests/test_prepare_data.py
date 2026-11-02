# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from environments.labbench2_pdf import prepare_data as prepare_data_module
from environments.labbench2_pdf.prepare_data import PreparationOptions, prepare_data


def _write_question(questions_dir: Path, benchmark: str = "litqa3") -> None:
    questions_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "11111111-item",
        "tag": benchmark,
        "version": "2.0",
        "question": "What is the answer?",
        "ideal": "answer",
        "sources": ["https://doi.org/10.1000/example"],
    }
    (questions_dir / f"{benchmark}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_provided_paper_folder_bypasses_downloaders(monkeypatch, tmp_path: Path) -> None:
    questions_dir = tmp_path / "provided-questions"
    papers_dir = tmp_path / "provided-papers"
    output_dir = tmp_path / "generated" / "tasks"
    _write_question(questions_dir)
    papers_dir.mkdir()
    (papers_dir / "10.1000_example.pdf").write_bytes(b"local snapshot")

    def unexpected_download(*args, **kwargs):
        del args, kwargs
        raise AssertionError("provided inputs must bypass downloaders")

    monkeypatch.setattr(prepare_data_module, "download_questions", unexpected_download)
    monkeypatch.setattr(prepare_data_module, "download_corpus", unexpected_download)

    result = prepare_data(
        PreparationOptions(
            questions_dir=questions_dir,
            papers_dir=papers_dir,
            output_dir=output_dir,
            benchmarks=("litqa3",),
        )
    )

    assert result["preparation"]["questions"]["mode"] == "provided"
    assert result["preparation"]["papers"]["mode"] == "provided"
    assert result["materialization"]["materialized_tasks"] == 1
    assert (output_dir / "litqa3-0001-11111111" / "task.toml").is_file()
    assert (output_dir / "preparation_manifest.json").is_file()


def test_omitted_inputs_download_then_filter_and_materialize(monkeypatch, tmp_path: Path) -> None:
    questions_cache = tmp_path / "cache" / "questions"
    papers_cache = tmp_path / "cache" / "papers"
    output_dir = tmp_path / "generated" / "tasks"
    calls: list[str] = []

    def fake_download_questions(options):
        calls.append("questions")
        _write_question(options.output_dir)
        return {
            "repo_id": options.repo_id,
            "revision": options.revision,
            "configs": {"litqa3": 1},
        }

    def fake_download_papers(options):
        calls.append("papers")
        assert options.questions_dir == questions_cache.resolve()
        options.papers_dir.mkdir(parents=True)
        (options.papers_dir / "10.1000_example.pdf").write_bytes(b"downloaded paper")
        return {
            "doi_count": 1,
            "downloaded_count": 1,
            "cached_count": 0,
            "failed_count": 0,
        }

    monkeypatch.setattr(prepare_data_module, "download_questions", fake_download_questions)
    monkeypatch.setattr(prepare_data_module, "download_corpus", fake_download_papers)

    result = prepare_data(
        PreparationOptions(
            questions_cache_dir=questions_cache,
            papers_cache_dir=papers_cache,
            output_dir=output_dir,
            benchmarks=("litqa3",),
        )
    )

    assert calls == ["questions", "papers"]
    assert result["preparation"]["questions"]["mode"] == "downloaded"
    assert result["preparation"]["papers"]["mode"] == "downloaded"
    assert result["materialization"]["materialized_tasks"] == 1


def test_provided_papers_can_be_combined_with_downloaded_questions(monkeypatch, tmp_path: Path) -> None:
    questions_cache = tmp_path / "cache" / "questions"
    papers_dir = tmp_path / "provided-papers"
    output_dir = tmp_path / "generated" / "tasks"
    papers_dir.mkdir()
    (papers_dir / "10.1000_example.pdf").write_bytes(b"local snapshot")

    def fake_download_questions(options):
        _write_question(options.output_dir)
        return {
            "repo_id": options.repo_id,
            "revision": options.revision,
            "configs": {"litqa3": 1},
        }

    def unexpected_paper_download(*args, **kwargs):
        del args, kwargs
        raise AssertionError("--papers-dir must bypass public paper downloaders")

    monkeypatch.setattr(prepare_data_module, "download_questions", fake_download_questions)
    monkeypatch.setattr(prepare_data_module, "download_corpus", unexpected_paper_download)

    result = prepare_data(
        PreparationOptions(
            papers_dir=papers_dir,
            questions_cache_dir=questions_cache,
            output_dir=output_dir,
            benchmarks=("litqa3",),
        )
    )

    assert result["preparation"]["questions"]["mode"] == "downloaded"
    assert result["preparation"]["papers"]["mode"] == "provided"
    assert result["materialization"]["materialized_tasks"] == 1
