# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runtime for code_gen's sandboxed unit tests. The runner next to this file and the grading
# code are uploaded with every verification, not built in (see ../sandbox_execution.py), so
# the image pins what that code imports plus the libraries candidates can import on the Ray
# path (scipy, pandas). Keep Python, numpy, scipy and pandas in step with the code_gen server
# environment (requirements.txt) so both paths grade alike.
#
# The interpreter is uv's managed CPython, the same build the server venv uses. Time limits
# make interpreter speed part of the grade: the official python:3.13-slim build ran a
# CPU-bound benchmark ~40% slower than this one, enough to turn borderline passes into
# time-limit failures.
#
# Base images are digest-pinned, as in opencode_sandboxed_agent/offline_science_image.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

COPY --from=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:$PATH

RUN uv venv --python 3.13.14 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache numpy==2.2.6 scipy==1.18.1 pandas==3.0.6
