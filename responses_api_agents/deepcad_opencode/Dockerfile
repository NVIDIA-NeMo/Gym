# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM node:22-bookworm-slim AS node
FROM deepcad-cadquery:2.8.0
USER root
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN node /usr/local/lib/node_modules/npm/bin/npm-cli.js install -g opencode-ai@1.17.11 \
    && pip install --no-cache-dir 'numpy>=2.2' 'pillow>=11'
RUN mkdir /workspace && chown cad:cad /workspace
USER cad
WORKDIR /workspace
