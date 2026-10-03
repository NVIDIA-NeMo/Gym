# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN uv pip install --system "mcp>=1.28.1,<2"

COPY tools/server.py /opt/hello-tools/server.py

WORKDIR /workspace
