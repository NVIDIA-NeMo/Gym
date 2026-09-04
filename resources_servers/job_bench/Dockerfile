# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sandbox image for the JobBench agent.
#
# JobBench tasks are office data work: reconciling spreadsheets, tracing PDF
# citations, querying SQLite, producing xlsx/docx/pptx deliverables and plots.
# Upstream runs on the operator's workstation with those libraries already
# installed; the sandbox has no such luxury, so the stack is baked in here and
# the agent never needs to pip install (which also keeps hermetic runs working).
#
#   docker build -t job-bench-sandbox:latest resources_servers/job_bench
#   docker push <your-registry>/job-bench-sandbox:latest
#   export JOB_BENCH_SANDBOX_IMAGE=<your-registry>/job-bench-sandbox:latest

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# ca-certificates for TLS when the agent searches online; git/curl/unzip because
# tasks and agents routinely reach for them; the poppler and libreoffice bits
# cover PDF and Office conversions that pure-Python readers cannot handle.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        unzip \
        zip \
        jq \
        sqlite3 \
        poppler-utils \
        libreoffice-calc \
        libreoffice-writer \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        pandas==2.2.3 \
        numpy==2.1.3 \
        scipy==1.14.1 \
        openpyxl==3.1.5 \
        xlrd==2.0.1 \
        XlsxWriter==3.2.0 \
        python-docx==1.1.2 \
        python-pptx==1.0.2 \
        mammoth==1.8.0 \
        pdfplumber==0.11.4 \
        pypdf==5.1.0 \
        reportlab==4.2.5 \
        matplotlib==3.9.2 \
        statsmodels==0.14.4 \
        beautifulsoup4==4.12.3 \
        lxml==5.3.0 \
        tabulate==0.9.0 \
        Pillow==11.0.0

WORKDIR /workspace
