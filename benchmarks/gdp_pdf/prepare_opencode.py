# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the same GDP.pdf questions, rubrics and PDFs without pre-rendering/OCR."""

from pathlib import Path

from benchmarks.gdp_pdf.prepare import DATA_DIR, DOCUMENTS_DIR, MEDIA_DIR, _prepare_rows


def prepare() -> Path:
    output = DATA_DIR / "gdp_pdf_opencode.jsonl"
    _prepare_rows(output, MEDIA_DIR, DOCUMENTS_DIR, render_documents=False)
    return output


if __name__ == "__main__":
    prepare()
