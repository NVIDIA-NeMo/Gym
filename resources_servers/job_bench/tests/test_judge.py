# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from resources_servers.job_bench.judge import (
    build_failed_rubric_result,
    build_rubric_prompt,
    build_rubric_result,
    build_scorecard,
    build_user_content,
    collect_image_attachments,
    convert_file_to_text,
    extract_all_file_contents,
    normalize_criteria,
    parse_judge_json,
    rubric_needs_vision,
)


# A 1x1 PNG; the smallest input that exercises the real image code paths.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
OTHER_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

RUBRIC = {
    "rubric": "Does the report state the population count?",
    "weight": 10,
    "criterion": ["A count is given", "The count is 1,108"],
}


def test_convert_reads_plain_text_formats(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    path.write_text("id,age\n1,39\n", encoding="utf-8")

    assert convert_file_to_text(path) == "id,age\n1,39\n"


def test_convert_reports_unreadable_text_without_raising(tmp_path: Path) -> None:
    assert "ERROR: Failed to read text file" in convert_file_to_text(tmp_path / "absent.csv")


def test_convert_renders_a_notebook_source_and_stream_output(tmp_path: Path) -> None:
    path = tmp_path / "analysis.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": ["print(1108)\n"], "outputs": [{"text": ["1108\n"]}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    rendered = convert_file_to_text(path)
    assert "=== code ===" in rendered
    assert "print(1108)" in rendered
    assert "1108" in rendered


def test_convert_reports_a_malformed_notebook(tmp_path: Path) -> None:
    path = tmp_path / "broken.ipynb"
    path.write_text("{not json", encoding="utf-8")

    assert "ERROR: Failed to read notebook" in convert_file_to_text(path)


def test_convert_renders_sqlite_schema_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE subjects (id INTEGER, name TEXT)")
    connection.execute("INSERT INTO subjects VALUES (1, 'a'), (2, NULL)")
    connection.commit()
    connection.close()

    rendered = convert_file_to_text(path)
    assert "=== Schema ===" in rendered
    assert "=== Table: subjects ===" in rendered
    assert "total_rows: 2" in rendered
    assert "id,name" in rendered
    # NULLs render as empty cells rather than the string "None".
    assert "2," in rendered


def test_convert_reports_a_corrupt_sqlite_file(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    path.write_bytes(b"not a database")

    assert "ERROR" in convert_file_to_text(path)


def test_convert_labels_images_and_unknown_binaries(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(PNG_BYTES)
    binary = tmp_path / "archive.bin"
    binary.write_bytes(b"\x00\x01")

    assert "cannot extract text content" in convert_file_to_text(image)
    assert "Binary or unsupported file type" in convert_file_to_text(binary)


def test_extract_all_concatenates_files_with_headers(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")

    contents = extract_all_file_contents(tmp_path)
    assert "=== FILE: a.txt ===" in contents
    assert "=== FILE: b.txt ===" in contents
    assert "alpha" in contents and "beta" in contents


def test_extract_all_truncates_an_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")

    contents = extract_all_file_contents(tmp_path, max_chars_per_file=10)
    assert "Content truncated at 10 characters" in contents
    assert "x" * 11 not in contents


def test_extract_all_exempts_sqlite_from_the_char_cap(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (value TEXT)")
    connection.execute("INSERT INTO t VALUES (?)", ("y" * 200,))
    connection.commit()
    connection.close()

    contents = extract_all_file_contents(tmp_path, max_chars_per_file=10)
    assert "truncated" not in contents
    assert "y" * 200 in contents


def test_extract_all_returns_empty_for_a_missing_directory(tmp_path: Path) -> None:
    assert extract_all_file_contents(tmp_path / "absent") == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Is there a plot of ages?", True),
        ("Does it include a Q-Q diagram?", True),
        ("Is the total correct?", False),
    ],
)
def test_rubric_needs_vision_detects_visual_language(text: str, expected: bool) -> None:
    assert rubric_needs_vision({"rubric": text, "criterion": []}) is expected


def test_rubric_needs_vision_reads_criteria_in_both_shapes() -> None:
    assert rubric_needs_vision({"rubric": "", "criterion": ["a histogram is shown"]}) is True
    assert rubric_needs_vision({"rubric": "", "criterion": "a heatmap is shown"}) is True
    assert rubric_needs_vision({"rubric": "", "criterion": "a table is shown"}) is False


def test_collect_attachments_deduplicates_identical_images(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(PNG_BYTES)
    (tmp_path / "b.png").write_bytes(PNG_BYTES)
    (tmp_path / "c.png").write_bytes(OTHER_PNG_BYTES)

    attachments = collect_image_attachments(tmp_path)
    assert len(attachments) == 2
    assert all(url.startswith("data:image/png;base64,") for _, url in attachments)


def test_collect_attachments_honours_the_cap(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(PNG_BYTES)
    (tmp_path / "b.png").write_bytes(OTHER_PNG_BYTES)

    assert len(collect_image_attachments(tmp_path, cap=1)) == 1
    assert collect_image_attachments(tmp_path, cap=0) == []
    assert collect_image_attachments(tmp_path / "absent") == []


def test_collect_attachments_reads_images_embedded_in_a_docx(tmp_path: Path) -> None:
    docx_path = tmp_path / "report.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("word/media/image1.png", PNG_BYTES)

    attachments = collect_image_attachments(tmp_path)
    assert len(attachments) == 1
    assert attachments[0][0] == "report.docx:word/media/image1.png"


def test_collect_attachments_skips_a_corrupt_docx(tmp_path: Path) -> None:
    (tmp_path / "report.docx").write_bytes(b"not a zip")

    assert collect_image_attachments(tmp_path) == []


def test_collect_attachments_reads_images_embedded_in_a_notebook(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [],
                "outputs": [{"data": {"image/png": base64.b64encode(PNG_BYTES).decode("ascii")}}],
            }
        ]
    }
    (tmp_path / "plots.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    attachments = collect_image_attachments(tmp_path)
    assert len(attachments) == 1
    assert "cell-0-output-0:image/png" in attachments[0][0]


def test_collect_attachments_ignores_undecodable_notebook_images(tmp_path: Path) -> None:
    notebook = {"cells": [{"cell_type": "code", "source": [], "outputs": [{"data": {"image/png": "!!!not base64"}}]}]}
    (tmp_path / "plots.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    assert collect_image_attachments(tmp_path) == []


def test_normalize_criteria_accepts_a_bare_string() -> None:
    assert normalize_criteria({"criterion": "only one"}) == ["only one"]
    assert normalize_criteria({}) == []


def test_prompt_lists_every_criterion_and_the_expected_count() -> None:
    prompt = build_rubric_prompt(RUBRIC, "=== FILE: r.txt ===\n1,108", vision_used=False)

    assert "Criterion 0: A count is given" in prompt
    assert "Criterion 1: The count is 1,108" in prompt
    assert "must have exactly 2 items" in prompt
    assert "1,108" in prompt
    assert "and the attached images" not in prompt


def test_prompt_mentions_images_only_when_they_are_attached() -> None:
    assert "and the attached images" in build_rubric_prompt(RUBRIC, "", vision_used=True)


def test_user_content_is_a_plain_string_without_attachments() -> None:
    assert build_user_content("prompt", []) == "prompt"


def test_user_content_interleaves_image_parts_with_their_names() -> None:
    content = build_user_content("prompt", [("chart.png", "data:image/png;base64,AAA")])

    assert content[0] == {"type": "text", "text": "prompt"}
    assert content[1]["text"] == "\n## Attached Images (1 file)"
    assert content[2] == {"type": "text", "text": "Image 1: chart.png"}
    assert content[3] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


def test_user_content_pluralizes_multiple_images() -> None:
    content = build_user_content("p", [("a.png", "data:1"), ("b.png", "data:2")])

    assert content[1]["text"] == "\n## Attached Images (2 files)"


@pytest.mark.parametrize(
    "content, expected_status",
    [
        ('{"rubric_passed": true}', "direct_json"),
        ('```json\n{"rubric_passed": true}\n```', "markdown_fence"),
        ('Sure! {"rubric_passed": true} Hope that helps.', "first_last_brace"),
    ],
)
def test_parse_judge_json_recovers_common_wrappings(content: str, expected_status: str) -> None:
    parsed, status = parse_judge_json(content)

    assert parsed["rubric_passed"] is True
    assert status == expected_status


def test_parse_judge_json_falls_back_to_a_regex_scan() -> None:
    content = 'prelude { "criteria_results": [{"index": 0}], "rubric_passed": false } trailing }'

    parsed, status = parse_judge_json(content)
    assert status == "regex_extract"
    assert parsed["rubric_passed"] is False


def test_parse_judge_json_raises_when_nothing_is_recoverable() -> None:
    with pytest.raises(ValueError, match="Could not extract JSON"):
        parse_judge_json("no json here at all")


def test_rubric_scores_its_full_weight_only_when_it_passes() -> None:
    parsed = {
        "criteria_results": [
            {"index": 0, "passed": True, "reasoning": "r0", "evidence": "e0"},
            {"index": 1, "passed": True, "reasoning": "r1", "evidence": "e1"},
        ],
        "rubric_passed": True,
        "overall_reasoning": "all good",
    }

    result = build_rubric_result(0, RUBRIC, parsed)
    assert result["result"]["passed"] is True
    assert result["result"]["score"] == 10
    assert result["result"]["criteria_passed"] == 2
    assert result["result"]["criteria_results"][1]["evidence"] == "e1"


def test_rubric_scores_zero_when_the_judge_fails_it() -> None:
    parsed = {
        "criteria_results": [{"index": 0, "passed": True}],
        "rubric_passed": False,
        "overall_reasoning": "second criterion missing",
    }

    result = build_rubric_result(0, RUBRIC, parsed)
    assert result["result"]["score"] == 0
    assert result["result"]["criteria_passed"] == 1
    # A criterion the judge omitted defaults to failed rather than raising.
    assert result["result"]["criteria_results"][1]["passed"] is False


def test_rubric_result_survives_malformed_criteria_entries() -> None:
    parsed = {"criteria_results": ["nonsense", {"passed": True}], "rubric_passed": True}

    result = build_rubric_result(0, RUBRIC, parsed)
    assert result["result"]["criteria_results"][0]["passed"] is False
    assert result["result"]["criteria_results"][1]["passed"] is True


def test_failed_rubric_result_zeroes_every_criterion() -> None:
    result = build_failed_rubric_result(1, RUBRIC, "judge unavailable")

    assert result["index"] == 1
    assert result["result"]["passed"] is False
    assert result["result"]["score"] == 0
    assert result["result"]["criteria_count"] == 2
    assert all(item["reasoning"] == "judge unavailable" for item in result["result"]["criteria_results"])


def test_scorecard_normalizes_by_total_weight_not_rubric_count() -> None:
    results = [
        {"index": 0, "weight": 10, "result": {"passed": True, "score": 10}},
        {"index": 1, "weight": 5, "result": {"passed": False, "score": 0}},
        {"index": 2, "weight": 5, "result": {"passed": False, "score": 0}},
    ]

    scorecard = build_scorecard(results)
    assert scorecard["total_score"] == 10
    assert scorecard["max_score"] == 20
    assert scorecard["normalized_score"] == 0.5
    # pass_rate counts rubrics, so it differs from the weighted score.
    assert scorecard["pass_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert scorecard["passed_count"] == 1
    assert scorecard["total_count"] == 3


def test_scorecard_of_no_rubrics_is_zero_not_a_division_error() -> None:
    scorecard = build_scorecard([])

    assert scorecard["normalized_score"] == 0.0
    assert scorecard["pass_rate"] == 0.0


# --------------------------------------------------- office-format converters
#
# The optional readers (openpyxl, mammoth, pdfplumber, python-pptx) are declared
# in requirements.txt but are not guaranteed in every environment, so these
# exercise our adapter logic against stub modules: what we own is the call shape
# and the rendered layout, not the third-party parsing itself.


class SimpleNamespaceLike:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_stub(monkeypatch: pytest.MonkeyPatch, name: str, module: object) -> None:
    import sys

    monkeypatch.setitem(sys.modules, name, module)


def _hide_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import name`` raise ImportError inside the converter."""
    import builtins

    real_import = builtins.__import__

    def fake_import(module_name, *args, **kwargs):
        if module_name == name or module_name.startswith(f"{name}."):
            raise ImportError(f"No module named {module_name!r}")
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_convert_renders_every_excel_sheet_as_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pandas = pytest.importorskip("pandas")
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"")

    frames = {
        "Summary": pandas.DataFrame({"count": [1108]}),
        "Detail": pandas.DataFrame({"age": [39, 46]}),
    }
    monkeypatch.setattr(pandas, "ExcelFile", lambda _p: SimpleNamespaceLike(sheet_names=list(frames)))
    monkeypatch.setattr(pandas, "read_excel", lambda _xl, sheet_name: frames[sheet_name])

    rendered = convert_file_to_text(path)
    assert "=== Sheet: Summary ===" in rendered
    assert "1108" in rendered
    assert "=== Sheet: Detail ===" in rendered
    # to_csv(index=False): no unnamed index column.
    assert "count\n1108" in rendered.replace("\r\n", "\n")


def test_convert_reports_an_unreadable_workbook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pandas = pytest.importorskip("pandas")
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"")

    def boom(_p):
        raise ValueError("not a zip file")

    monkeypatch.setattr(pandas, "ExcelFile", boom)

    assert "ERROR: Failed to read Excel" in convert_file_to_text(path)


def test_convert_reports_a_missing_excel_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"")
    _hide_module(monkeypatch, "pandas")

    assert "pandas/openpyxl not available" in convert_file_to_text(path)


def test_convert_renders_docx_as_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "report.docx"
    path.write_bytes(b"")

    captured = {}

    def convert_to_markdown(handle, convert_image=None):
        captured["convert_image"] = convert_image
        return SimpleNamespaceLike(value="# Findings\n\n1,108 subjects")

    stub = SimpleNamespaceLike(
        convert_to_markdown=convert_to_markdown,
        images=SimpleNamespaceLike(img_element=lambda fn: ("img_element", fn)),
    )
    _install_stub(monkeypatch, "mammoth", stub)

    rendered = convert_file_to_text(path)
    assert rendered == "# Findings\n\n1,108 subjects"
    # Embedded images are replaced by a placeholder rather than inlined.
    placeholder = captured["convert_image"][1](SimpleNamespaceLike(content_type="image/png"))
    assert placeholder == {"src": "embedded-image", "alt": "Embedded image: image/png"}


def test_convert_reports_a_missing_docx_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "report.docx"
    path.write_bytes(b"")
    _hide_module(monkeypatch, "mammoth")

    assert "mammoth not available" in convert_file_to_text(path)


def test_convert_reports_an_unreadable_docx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "report.docx"
    path.write_bytes(b"")

    def boom(*_args, **_kwargs):
        raise ValueError("corrupt")

    _install_stub(
        monkeypatch,
        "mammoth",
        SimpleNamespaceLike(convert_to_markdown=boom, images=SimpleNamespaceLike(img_element=lambda fn: fn)),
    )

    assert "ERROR: Failed to read DOCX" in convert_file_to_text(path)


def test_convert_renders_pdf_pages_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "guidance.pdf"
    path.write_bytes(b"")

    class Pdf:
        pages = [
            SimpleNamespaceLike(extract_text=lambda layout=True: "Section 5.7"),
            # A page with no extractable text must still produce a header.
            SimpleNamespaceLike(extract_text=lambda layout=True: None),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    _install_stub(monkeypatch, "pdfplumber", SimpleNamespaceLike(open=lambda _p: Pdf()))

    rendered = convert_file_to_text(path)
    assert "=== Page 1 ===\nSection 5.7" in rendered
    assert "=== Page 2 ===\n" in rendered


def test_convert_reports_a_missing_pdf_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "guidance.pdf"
    path.write_bytes(b"")
    _hide_module(monkeypatch, "pdfplumber")

    assert "pdfplumber not available" in convert_file_to_text(path)


def test_convert_reports_an_unreadable_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "guidance.pdf"
    path.write_bytes(b"")

    def boom(_p):
        raise ValueError("damaged")

    _install_stub(monkeypatch, "pdfplumber", SimpleNamespaceLike(open=boom))

    assert "ERROR: Failed to read PDF" in convert_file_to_text(path)


def test_convert_renders_pptx_slide_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "deck.pptx"
    path.write_bytes(b"")

    slide = SimpleNamespaceLike(
        shapes=[SimpleNamespaceLike(text="Title"), SimpleNamespaceLike(text=""), SimpleNamespaceLike()]
    )
    _install_stub(
        monkeypatch, "pptx", SimpleNamespaceLike(Presentation=lambda _p: SimpleNamespaceLike(slides=[slide]))
    )

    rendered = convert_file_to_text(path)
    assert "=== Slide 1 ===" in rendered
    assert "Title" in rendered


def test_convert_reports_a_missing_pptx_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "deck.pptx"
    path.write_bytes(b"")
    _hide_module(monkeypatch, "pptx")

    assert "python-pptx not available" in convert_file_to_text(path)


def test_convert_reports_an_unreadable_pptx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "deck.pptx"
    path.write_bytes(b"")

    def boom(_p):
        raise ValueError("corrupt")

    _install_stub(monkeypatch, "pptx", SimpleNamespaceLike(Presentation=boom))

    assert "ERROR: Failed to read PowerPoint" in convert_file_to_text(path)


def test_sqlite_render_reports_a_table_it_cannot_query(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE ok (id INTEGER)")
    # A embedded double quote defeats the renderer's simple quoting, so this
    # table fails to read. It must be reported inline, not abort the whole file.
    connection.execute('CREATE TABLE "od""d" (id INTEGER)')
    connection.commit()
    connection.close()

    rendered = convert_file_to_text(path)
    assert "=== Table: ok ===" in rendered
    assert "[ERROR reading table" in rendered


def test_collect_attachments_skips_files_it_cannot_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "chart.png").write_bytes(PNG_BYTES)
    real_read_bytes = Path.read_bytes

    def failing_read_bytes(self):
        if self.suffix == ".png":
            raise OSError("permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)

    assert collect_image_attachments(tmp_path) == []


def test_collect_attachments_ignores_unsupported_image_extensions(tmp_path: Path) -> None:
    (tmp_path / "diagram.svg").write_bytes(b"<svg/>")

    assert collect_image_attachments(tmp_path) == []


def test_collect_attachments_skips_an_unreadable_notebook(tmp_path: Path) -> None:
    (tmp_path / "plots.ipynb").write_text("{not json", encoding="utf-8")

    assert collect_image_attachments(tmp_path) == []


def test_collect_attachments_accepts_a_chunked_notebook_image(tmp_path: Path) -> None:
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [],
                # nbformat splits long base64 payloads across a list of lines.
                "outputs": [{"data": {"image/png": [encoded[:10], encoded[10:]]}}, {"data": "not-a-dict"}],
            }
        ]
    }
    (tmp_path / "plots.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    assert len(collect_image_attachments(tmp_path)) == 1
