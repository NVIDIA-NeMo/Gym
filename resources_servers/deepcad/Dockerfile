# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libxrender1 libxext6 libgomp1 libosmesa6 libegl1 libxcursor1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cadquery==2.8.0
RUN mkdir /sandbox && useradd --uid 10001 --create-home cad
RUN chown cad:cad /sandbox
USER cad
WORKDIR /sandbox
