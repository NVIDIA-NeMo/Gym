# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Preserve the 2024/2025 grading environments used by EFB, without reinstalling packages.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY test_data_cleaned.h5 /data/test_data_cleaned.h5
RUN echo '2b41a7df40ddc23ce651ec05b8ecb6f8  /data/test_data_cleaned.h5' | md5sum --check
