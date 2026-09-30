# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the base with NVIDIA/SOL-ExecBench's docker/Dockerfile at this revision.
# The verifier independently checks the installed native Python source hashes.
ARG SOL_EXECBENCH_BASE_IMAGE
FROM ${SOL_EXECBENCH_BASE_IMAGE}
USER root
RUN test -x /venv/bin/sol-execbench \
    && printf '%s\n' a9fa0804c793d438e70850c33fe34426e66d53dd > /opt/sol-execbench-revision \
    && mkdir -p /sol-eval/assets
ENV FLASHINFER_TRACE_DIR=/sol-eval/assets
