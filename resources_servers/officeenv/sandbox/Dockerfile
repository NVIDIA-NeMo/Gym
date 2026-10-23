# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The workspace an OfficeEnv episode lends a sandboxed harness: a shell, Python with data libraries,
# and curl for the episode's file API (OFFICEENV.md). Pi's installer also needs bash, tar, gzip and
# sha256sum, and network access to nodejs.org and npm when the session starts.
#
#   docker build -t officeenv-sandbox:pilot resources_servers/officeenv/sandbox
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates coreutils curl gzip tar \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas==2.2.3 openpyxl==3.1.5

WORKDIR /workspace
