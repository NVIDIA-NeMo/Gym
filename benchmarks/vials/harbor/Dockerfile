# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13-slim
RUN useradd -m -u 10001 -s /bin/sh agent && mkdir -p /task && chown agent:agent /task
WORKDIR /task
USER agent
