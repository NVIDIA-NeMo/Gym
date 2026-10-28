# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for the GDP.pdf python tool: an IPython kernel plus common PDF and data libraries.
#   docker build -t gdp-pdf-sandbox resources_servers/gdp_pdf/containers
FROM python:3.13-slim
RUN pip install --no-cache-dir \
    ipykernel==6.30.1 jupyter-client==8.6.3 \
    pymupdf==1.26.4 pdfplumber==0.11.7 \
    pandas==2.3.2 numpy==2.3.3 pillow==11.3.0 matplotlib==3.10.6
