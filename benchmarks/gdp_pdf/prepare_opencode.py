# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the same GDP.pdf questions, rubrics and PDFs without pre-rendering/OCR."""

from pathlib import Path

from benchmarks.gdp_pdf.prepare import DATA_DIR, DOCUMENTS_DIR, MEDIA_DIR, _prepare_rows, _write_jsonl


def prepare() -> Path:
    output = DATA_DIR / "gdp_pdf_opencode.jsonl"
    rows = _prepare_rows(output, MEDIA_DIR, DOCUMENTS_DIR, render_documents=False)
    for row in rows:
        row["responses_create_params"]["input"][0]["content"][0]["text"] = (
            f"The document is available at `/workspace/document.pdf`.\n\n{row['verifier_metadata']['prompt']}"
        )
    _write_jsonl(output, rows)
    return output


if __name__ == "__main__":
    prepare()
