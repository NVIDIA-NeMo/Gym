# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared direct-PDF runtime for LitQA3, FigQA2, and TableQA2 Harbor tasks.
FROM python:3.13.14-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        asciinema \
        bash \
        ca-certificates \
        curl \
        file \
        ghostscript \
        git \
        imagemagick \
        jq \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        ripgrep \
        tesseract-ocr \
        tk \
        tmux \
    && rm -rf /var/lib/apt/lists/*

COPY docker/runtime-requirements.txt /tmp/runtime-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt

RUN python -c "from PIL import Image; from wand.image import Image as WandImage" \
    && python -c "import PyPDF2, bs4, cv2, fastapi, fitz, imageio, matplotlib, numpy" \
    && python -c "import pandas, pdf2image, pdfminer, pdfplumber, pymupdf, pypdf" \
    && python -c "import pytesseract, tkinter, uvicorn" \
    && python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot" \
    && pdfinfo -v 2>&1 | grep -q '^pdfinfo version' \
    && tesseract --version | grep -q '^tesseract' \
    && mkdir -p /app /app/pdf_pages

WORKDIR /app
CMD ["bash"]
