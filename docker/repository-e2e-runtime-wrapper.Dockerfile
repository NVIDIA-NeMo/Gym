# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN ln -s ../dev-venv/bin/python /opt/repository-e2e-gym/bin/python3 && \
    python3 --version | grep -E '^Python 3[.]12[.]' && \
    test "$(uv --version | awk '{print $2}')" = 0.11.19
