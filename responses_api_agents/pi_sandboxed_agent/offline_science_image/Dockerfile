# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG SCIENCE_IMAGE
FROM ${SCIENCE_IMAGE}
RUN curl -fL --retry 3 https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.xz -o /tmp/node.tar.xz && \
    echo 'd60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307  /tmp/node.tar.xz' | sha256sum -c - && \
    mkdir /opt/node && tar -xf /tmp/node.tar.xz -C /opt/node --strip-components=1 && rm /tmp/node.tar.xz
ENV PATH=/opt/node/bin:/opt/pi/node_modules/.bin:$PATH
COPY package.json package-lock.json /opt/pi/
RUN cd /opt/pi && npm ci --ignore-scripts --omit=dev --no-audit --no-fund && pi --version
WORKDIR /workspace
