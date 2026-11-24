# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Overlay mode: the base image is pulled and the RUN line runs at every seed; nothing is built or pushed.
# The verifier imports math-verify and runs in a sandbox of this image (a fresh one with configs/harbor_arm.yaml),
# so it is installed at the versions resources_servers/math_with_judge/uv.lock pins. Every seed downloads them from
# PyPI, so the sandbox needs that egress; a prebuilt image with them installed is the way out once a registry
# OpenSandbox pulls from exists.
# The base is pinned by digest (linux/amd64, the platform OpenSandbox runs) so two prepares resolve the same image.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

WORKDIR /workspace

RUN uv pip install --system --no-cache math-verify==0.8.0 latex2sympy2_extended==1.10.2 sympy==1.14.0 mpmath==1.3.0 antlr4-python3-runtime==4.9.3
