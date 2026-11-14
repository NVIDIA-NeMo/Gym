#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SQL Server aborts with "Invalid mapping of address ... reserved address space"
# when the soft stack limit is unlimited (legacy VA layout). Cap it, then hand
# off to the image's all-in-one entrypoint.
ulimit -s 8192 || true
exec /entrypoint-aio.sh
