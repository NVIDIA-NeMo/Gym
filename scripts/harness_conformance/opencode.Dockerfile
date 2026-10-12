# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM node:22-bookworm-slim
ARG OPENCODE_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g opencode-ai@${OPENCODE_VERSION}

ENV OPENCODE_DISABLE_AUTOUPDATE=true OPENCODE_DISABLE_MODELS_FETCH=true OPENCODE_DISABLE_PROJECT_CONFIG=true
