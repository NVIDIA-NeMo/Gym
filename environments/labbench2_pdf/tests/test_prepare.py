# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from environments.labbench2_pdf.prepare import MaterializationOptions, build_parser, materialize


def _write_questions(path: Path, benchmark: str, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{benchmark}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(item_id: str, source: str, *, benchmark: str = "litqa3", ideal: str = "answer") -> dict:
    return {
        "id": item_id,
        "tag": benchmark,
        "version": "2.0",
        "question": f"Question for {item_id}?",
        "ideal": ideal,
        "sources": [f"https://doi.org/{source}"],
    }


def _options(
    tmp_path: Path,
    *,
    environment_type: str = "docker",
    missing_paper: str = "skip",
    corpus_view_mode: str = "copy",
    source_policy: str = "all",
    limit_per_benchmark: int | None = None,
) -> MaterializationOptions:
    return MaterializationOptions(
        questions_dir=tmp_path / "questions",
        papers_dir=tmp_path / "papers",
        output_dir=tmp_path / "generated" / environment_type,
        benchmarks=("litqa3",),
        environment_type=environment_type,
        image="labbench2-pdf-runtime:test",
        corpus_view_mode=corpus_view_mode,
        missing_paper=missing_paper,
        source_policy=source_policy,
        limit_per_benchmark=limit_per_benchmark,
    )


def test_docker_materialization_mounts_only_benchmark_relevant_union(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    rows = [
        _row("11111111-first", "10.1000/first", ideal="gold one"),
        _row("22222222-second", "10.1000/second", ideal="gold two"),
    ]
    _write_questions(questions, "litqa3", rows)
    for name in ("10.1000_first.pdf", "10.1000_second.pdf", "unrelated.pdf"):
        (papers / name).write_bytes(name.encode())

    manifest = materialize(_options(tmp_path))

    assert manifest["materialized_tasks"] == 2
    assert manifest["pdf_access_mode"] == "direct"
    assert manifest["corpus_scope"] == "relevant"
    view = Path(manifest["corpus_views"]["litqa3"]["path"])
    assert sorted(path.name for path in view.glob("*.pdf")) == [
        "10.1000_first.pdf",
        "10.1000_second.pdf",
    ]

    task = Path(manifest["output_dir"]) / "litqa3-0001-11111111"
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    assert rows[0]["question"] in instruction
    assert rows[0]["ideal"] not in instruction
    assert rows[0]["sources"][0] not in instruction
    assert "10.1000_first.pdf" not in instruction
    assert "skill" not in instruction.casefold()
    assert "/papers" in instruction
    assert "do not use a\npatch or file-edit tool" in instruction
    assert "cat /app/answer.txt" in instruction

    task_toml_text = (task / "task.toml").read_text(encoding="utf-8")
    task_toml = tomllib.loads(task_toml_text)
    assert task_toml["metadata"]["corpus_scope"] == "relevant"
    assert task_toml["metadata"]["pdf_access_mode"] == "direct"
    assert "sources" not in task_toml["metadata"]
    assert "skill" not in task_toml_text.casefold()
    assert task_toml["verifier"]["env"] == {
        "OPENAI_API_KEY": "${JUDGE_API_KEY}",
        "OPENAI_BASE_URL": "${JUDGE_BASE_URL}",
        "JUDGE_MODEL": "${JUDGE_MODEL}",
        "JUDGE_MAX_TOKENS": "2048",
    }

    compose = (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    assert str(view) in compose
    assert str(papers.resolve()) not in compose
    assert "unrelated.pdf" not in compose
    assert "HOST_VERIFIER_LOGS_PATH" in compose
    assert "HOST_AGENT_LOGS_PATH" in compose
    assert 'target: "/papers"' in compose

    gold = json.loads((task / "tests" / "gold_metadata.json").read_text(encoding="utf-8"))
    assert gold["ideal"] == "gold one"
    assert "sources" not in gold
    rollout_rows = [
        json.loads(line) for line in (Path(manifest["output_dir"]) / "litqa3_input.jsonl").read_text().splitlines()
    ]
    assert [row["task_name"] for row in rollout_rows] == [
        "litqa3-0001-11111111",
        "litqa3-0002-22222222",
    ]
    assert all(row["agent_ref"] == {"name": "harbor_agent_general"} for row in rollout_rows)
    example_rows = [
        json.loads(line) for line in (Path(manifest["output_dir"]) / "example_input.jsonl").read_text().splitlines()
    ]
    assert example_rows == rollout_rows


def test_missing_paper_rows_are_skipped_and_audited(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    _write_questions(
        questions,
        "litqa3",
        [
            _row("11111111-present", "10.1000/present"),
            _row("22222222-missing", "10.1000/missing"),
        ],
    )
    (papers / "10.1000_present.pdf").touch()

    manifest = materialize(_options(tmp_path))

    assert manifest["selected_rows"] == 2
    assert manifest["materialized_tasks"] == 1
    assert [row["item_id"] for row in manifest["skipped_rows"]] == ["22222222-missing"]
    audit = json.loads((Path(manifest["output_dir"]) / "coverage_audit.json").read_text())
    assert audit["missing_count"] == 1
    assert audit["answerable_count"] == 1


def test_missing_paper_error_preserves_existing_output(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    _write_questions(questions, "litqa3", [_row("11111111-missing", "10.1000/missing")])
    options = _options(tmp_path, missing_paper="error")
    options.output_dir.mkdir(parents=True)
    marker = options.output_dir / "keep-me"
    marker.write_text("existing", encoding="utf-8")
    options = MaterializationOptions(
        questions_dir=options.questions_dir,
        papers_dir=options.papers_dir,
        output_dir=options.output_dir,
        benchmarks=options.benchmarks,
        environment_type=options.environment_type,
        image=options.image,
        missing_paper="error",
        overwrite=True,
    )

    with pytest.raises(RuntimeError, match="rows do not satisfy the PDF source policy"):
        materialize(options)

    assert marker.read_text(encoding="utf-8") == "existing"


def test_all_source_policy_skips_partially_resolved_questions(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    partial = _row("11111111-partial", "10.1000/present")
    partial["sources"].append("https://doi.org/10.1000/missing")
    complete = _row("22222222-complete", "10.1000/complete")
    _write_questions(questions, "litqa3", [partial, complete])
    (papers / "10.1000_present.pdf").touch()
    (papers / "10.1000_complete.pdf").touch()

    manifest = materialize(_options(tmp_path))

    assert manifest["materialized_tasks"] == 1
    assert manifest["source_policy"] == "all"
    assert manifest["skipped_rows"][0]["item_id"] == "11111111-partial"
    assert manifest["skipped_rows"][0]["partial"] is True
    assert manifest["skipped_rows"][0]["missing_sources"] == ["https://doi.org/10.1000/missing"]


def test_any_source_policy_can_keep_partially_resolved_questions(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    row = _row("11111111-partial", "10.1000/present")
    row["sources"].append("https://doi.org/10.1000/missing")
    _write_questions(questions, "litqa3", [row])
    (papers / "10.1000_present.pdf").touch()

    manifest = materialize(_options(tmp_path, source_policy="any"))

    assert manifest["materialized_tasks"] == 1
    assert manifest["skipped_rows"] == []
    assert manifest["coverage"]["partial_count"] == 1


def test_limit_counts_answerable_rows_and_preserves_source_position(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    _write_questions(
        questions,
        "litqa3",
        [
            _row("11111111-missing", "10.1000/missing"),
            _row("22222222-present", "10.1000/present"),
            _row("33333333-unused", "10.1000/unused"),
        ],
    )
    (papers / "10.1000_present.pdf").touch()

    manifest = materialize(_options(tmp_path, limit_per_benchmark=1))

    assert manifest["materialized_tasks"] == 1
    assert manifest["tasks"][0]["name"] == "litqa3-0002-22222222"
    assert [row["item_id"] for row in manifest["skipped_rows"]] == ["11111111-missing"]


def test_benchmark_with_no_papers_does_not_block_other_benchmarks(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    _write_questions(questions, "litqa3", [_row("11111111-present", "10.1000/present")])
    _write_questions(
        questions,
        "figqa2",
        [_row("22222222-missing", "10.1000/missing", benchmark="figqa2")],
    )
    (papers / "10.1000_present.pdf").touch()
    base_options = _options(tmp_path)
    options = MaterializationOptions(
        questions_dir=base_options.questions_dir,
        papers_dir=base_options.papers_dir,
        output_dir=base_options.output_dir,
        benchmarks=("litqa3", "figqa2"),
    )

    manifest = materialize(options)

    assert manifest["materialized_tasks"] == 1
    assert manifest["coverage"]["by_benchmark"]["figqa2"]["answerable"] == 0
    assert (Path(manifest["output_dir"]) / "figqa2_input.jsonl").read_text() == ""
    assert Path(manifest["corpus_views"]["figqa2"]["path"]).is_dir()


def test_singularity_stages_one_copied_store_without_per_task_copies(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    row = _row("11111111-first", "10.1000/first")
    _write_questions(questions, "litqa3", [row])
    source_pdf = papers / "10.1000_first.pdf"
    source_pdf.write_bytes(b"pdf")

    manifest = materialize(_options(tmp_path, environment_type="singularity"))

    task = Path(manifest["output_dir"]) / "litqa3-0001-11111111"
    staged_pdf = task / "environment" / "files" / "papers" / source_pdf.name
    view_pdf = Path(manifest["corpus_views"]["litqa3"]["path"]) / source_pdf.name
    assert source_pdf.stat().st_ino != view_pdf.stat().st_ino
    assert view_pdf.stat().st_ino == staged_pdf.stat().st_ino
    assert not (task / "environment" / "docker-compose.yaml").exists()
    setup = (task / "environment" / "files" / "setup.sh").read_text(encoding="utf-8")
    assert 'ln -s "${HARBOR_STAGING}/papers" /papers' in setup
    assert "cp -r" not in setup


def test_hardlink_mode_is_an_explicit_zero_copy_option(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    papers = tmp_path / "papers"
    papers.mkdir()
    _write_questions(questions, "litqa3", [_row("11111111-first", "10.1000/first")])
    source_pdf = papers / "10.1000_first.pdf"
    source_pdf.write_bytes(b"pdf contents")

    manifest = materialize(_options(tmp_path, corpus_view_mode="hardlink"))

    view_pdf = Path(manifest["corpus_views"]["litqa3"]["path"]) / source_pdf.name
    assert view_pdf.read_bytes() == source_pdf.read_bytes()
    assert view_pdf.stat().st_ino == source_pdf.stat().st_ino


def test_cli_modes_are_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--questions-dir",
            "questions",
            "--papers-dir",
            "papers",
            "--environment-type",
            "singularity",
            "--benchmarks",
            "figqa2",
            "tableqa2",
        ]
    )

    assert args.environment_type == "singularity"
    assert args.benchmarks == ["figqa2", "tableqa2"]
    assert args.corpus_view_mode == "copy"


def test_default_runtime_image_does_not_replace_the_standalone_image() -> None:
    parser = build_parser()
    args = parser.parse_args(["--questions-dir", "questions", "--papers-dir", "papers"])

    assert args.image == "nemo-gym-labbench2-pdf:2.0"
