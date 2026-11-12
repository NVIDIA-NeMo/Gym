#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n "${NEMO_GYM_TEST_DATA_ROOT:-}" ]]; then
    export SACREBLEU="${NEMO_GYM_TEST_DATA_ROOT%/}/sacrebleu"
fi
