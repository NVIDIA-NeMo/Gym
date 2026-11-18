#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu
runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -e /lib/ld-musl-x86_64.so.1 ]; then
    exec "$runtime_dir/musl/bin/python3" "$@"
fi
exec "$runtime_dir/bin/python3" "$@"
