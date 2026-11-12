# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runtime for competitive_coding_challenges' sandboxed grading. The runner next to this file
# and ../ccc_eval.py are uploaded with every verification, not built in (see
# ../sandbox_evaluator.py); problem data is mounted or added at
# sandbox_config.problem_data_dir (see ../README.md).
#
# The toolchain matches the NeMo Skills sandbox the baselines were scored with
# (nemo-skills-sandbox-dc43f3e: Debian 13 "trixie", g++ 14.2.0-19). The compiler decides
# what compiles and how fast it runs, and IOI time limits are wall-clock.
#
# Base images are digest-pinned, as in opencode_sandboxed_agent/offline_science_image.
#
# Some problems' run.sh call bare `python` (the NeMo Skills sandbox has it); python-is-python3 provides it.
FROM debian:trixie-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++-14=14.2.0-19 g++=4:14.2.0-1 python3 python-is-python3 \
    && rm -rf /var/lib/apt/lists/*
