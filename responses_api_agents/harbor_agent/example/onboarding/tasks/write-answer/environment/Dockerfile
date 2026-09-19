# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Official multi-architecture manifest verified on 2026-09-10.
FROM ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254

# Terminus2 uses both tools. Install at build time; the task runs without network access.
# Apt packages are not version-locked: this pins the base image, not the complete build.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux asciinema \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
