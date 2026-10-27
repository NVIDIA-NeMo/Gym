#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Publishes a Python runtime that sandbox_python.py built on the Agent Server host. Runs in the task sandbox and
# needs only sh, tar with gzip, mv and rm: no network, no uv and no Python of the task image.
#
# usage: install_python_runtime.sh RUNTIME_DIR SETUP_DIR IMPORT_CHECK
#
# SETUP_DIR, a private directory inside RUNTIME_DIR, holds python.tar.gz and site-packages.tar.gz and is removed
# on exit. The runtime is unpacked and checked there, then renamed to RUNTIME_DIR/python. Exits 0 once
# RUNTIME_DIR/python passes IMPORT_CHECK, whichever seed published it.
set -eu
runtime_dir=$1
setup_dir=$2
import_check=$3
trap 'rm -rf "$setup_dir"' EXIT
# Some harnesses write their state under HOME when imported; keep the checks out of the task's HOME.
export HOME="$runtime_dir/home"
mkdir -p "$HOME"
tar -xzf "$setup_dir/python.tar.gz" -C "$setup_dir"
tar -xzf "$setup_dir/site-packages.tar.gz" -C "$setup_dir"
fresh=$setup_dir/python
"$fresh/bin/python3" -I -c "$import_check"
runtime=$runtime_dir/python
attempt=0
# Ends as soon as a runtime that passes the check is in place, so one that a concurrent seed published is kept.
until "$runtime/bin/python3" -I -c "$import_check" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 3 ]; then
        "$runtime/bin/python3" -I -c "$import_check"
        exit 1
    fi
    # A runtime that fails the check, for example one baked into the image for another platform, is set aside.
    # A concurrent seed may move it first.
    if [ -e "$runtime" ]; then
        mv "$runtime" "$setup_dir/replaced-$attempt" 2>/dev/null || true
    fi
    # rename(2) publishes atomically and fails when a concurrent seed published first.
    if [ -e "$fresh" ]; then
        "$fresh/bin/python3" -I -c 'import os, sys; os.rename(*sys.argv[1:])' "$fresh" "$runtime" 2>/dev/null || true
    fi
done
