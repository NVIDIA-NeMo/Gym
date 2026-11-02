# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from pathlib import Path

import pytest

from benchmarks.imo_answerbench import prepare as prepare_module


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_FPATH = REPO_ROOT / "resources_servers" / "math_with_autograder" / "data" / "example.jsonl"
CATEGORIES = {"Algebra", "Combinatorics", "Geometry", "Number theory"}

# Verbatim from the pinned answerbench_v2.csv (lines 1 and 148-158). imo-bench-algebra-036's
# Problem field has no closing quote (google-deepmind/superhuman#13); its neighbours are well formed.
PINNED_CSV_EXCERPT = "\n".join(
    [
        "Problem ID,Problem,Short Answer,Category,Subcategory,Source",
        r'imo-bench-algebra-035,"Find all real-coefficient polynomials $f$ such that $2f(0.5f(n))=f(f(n))-f(n)^2$ holds for all real numbers $n$.',
        r'","$f(x)=-1, f(x)=2x^{2}+b x, f(x)=0$",Algebra,Polynomial,Iran 2002',
        r'imo-bench-algebra-036,"Find all functions $Y: \mathbb{R} \backslash\{0\} \rightarrow \mathbb{R}$ such that for any non-zero real numbers $a, b$ with $ab \neq -1$, the following equation holds:',
        r"\[",
        r"a Y\left(a+\frac{1}{b}\right)+b Y(b)+\frac{a}{b}=b Y\left(b+\frac{1}{a}\right)+a Y(a)+\frac{b}{a}",
        r"\]",
        r',"$Y(x)=A+\frac{B}{x}-x$",Algebra,Functional Equation,Iran 2002',
        r'imo-bench-algebra-037,"Find all functions $X: \mathbb{C} \rightarrow \mathbb{C}$ such that the equation',
        r"$$X(X(a)+b X(b)-b-1)=1+a+|b|^{2}$$",
        r"holds for all complex numbers $a,b\in \mathbb{C}$ and that $X(1)=u$ for some $u\in \mathbb{C}$ such that $|u-1|=1$.",
        r'",$X(y)=1+(u-1) \bar{y}$,Algebra,Functional Equation,Iran 2024',
        "",
    ]
)


def _prepare_from(csv_text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    output_fpath = tmp_path / "imo_answerbench_benchmark.jsonl"
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_fpath)
    monkeypatch.setattr(
        prepare_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(csv_text.encode("utf-8"))
    )

    assert prepare_module.prepare() == output_fpath
    return [json.loads(line) for line in output_fpath.read_text(encoding="utf-8").splitlines()]


def test_prepare_realigns_the_row_with_an_unterminated_problem_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = {row["problem_id"]: row for row in _prepare_from(PINNED_CSV_EXCERPT, tmp_path, monkeypatch)}

    assert list(rows) == ["imo-bench-algebra-035", "imo-bench-algebra-036", "imo-bench-algebra-037"]
    shifted = rows["imo-bench-algebra-036"]
    assert shifted["expected_answer"] == r"$Y(x)=A+\frac{B}{x}-x$"
    assert (shifted["category"], shifted["subcategory"], shifted["source"]) == (
        "Algebra",
        "Functional Equation",
        "Iran 2002",
    )
    # The answer cell no longer leaks into the prompt; the problem keeps its trailing newline like its neighbours.
    assert shifted["question"].endswith("+\\frac{b}{a}\n\\]\n")
    assert "Y(x)=A+" not in shifted["question"]
    assert rows["imo-bench-algebra-035"]["expected_answer"] == "$f(x)=-1, f(x)=2x^{2}+b x, f(x)=0$"
    assert rows["imo-bench-algebra-037"]["source"] == "Iran 2024"


def test_prepare_rejects_a_misaligned_row_instead_of_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row one cell short that is not the unterminated-quote shape must fail loudly, not ship a null column.
    malformed = "Problem ID,Problem,Short Answer,Category,Subcategory,Source\nimo-bench-x-001,What is 1+1?,2,Algebra\n"

    with pytest.raises(ValueError, match="imo-bench-x-001"):
        _prepare_from(malformed, tmp_path, monkeypatch)
    assert not (tmp_path / "imo_answerbench_benchmark.jsonl").exists()


def test_shipped_example_rows_match_the_realigned_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = {row["problem_id"]: row for row in _prepare_from(PINNED_CSV_EXCERPT, tmp_path, monkeypatch)}
    examples = [json.loads(line) for line in EXAMPLE_FPATH.read_text(encoding="utf-8").splitlines()]

    for example in examples:
        assert example["category"] in CATEGORIES, example["problem_id"]
        assert example["expected_answer"] not in CATEGORIES, example["problem_id"]
        assert example["source"], example["problem_id"]
        assert example["responses_create_params"]["input"][0]["content"].endswith(example["question"])

    example = next(row for row in examples if row["problem_id"] == "imo-bench-algebra-036")
    for key in ("question", "expected_answer", "category", "subcategory", "source"):
        assert example[key] == prepared["imo-bench-algebra-036"][key], key
