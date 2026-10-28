#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Publishes a Python runtime that sandbox_python.py built on the Agent Server host. Runs in the task sandbox and
# needs only bash, tar with gzip and rm: no network, no uv and no Python of the task image.
#
# usage: install_python_runtime.sh RUNTIME_DIR SETUP_DIR IMPORT_CHECK
#
# SETUP_DIR, a private directory inside RUNTIME_DIR, holds python.tar.gz and site-packages.tar.gz and is removed
# on exit. The runtime is unpacked and checked there, then renamed to RUNTIME_DIR/python. Exits 0 once
# RUNTIME_DIR/python passes IMPORT_CHECK, whichever seed published it.

# Set explicitly, so an inherited SHELLOPTS cannot turn a failed step into a published runtime.
set -o errexit -o nounset -o pipefail
runtime_dir=$1
setup_dir=$2
import_check=$3
trap 'rm -rf "$setup_dir"' EXIT
# Some harnesses write their state under HOME when imported; keep the checks out of the task's HOME.
export HOME=$runtime_dir/home
mkdir -p "$HOME"
tar -xzf "$setup_dir/python.tar.gz" -C "$setup_dir"
tar -xzf "$setup_dir/site-packages.tar.gz" -C "$setup_dir"
fresh=$setup_dir/python
"$fresh/bin/python3" -I -c "$import_check"
# Concurrent seeds take turns on a lock over RUNTIME_DIR, held by the fresh runtime's own Python, so a runtime is
# only set aside by a seed that saw it fail its check and never one that another seed has just published. The
# lock is released when the process exits, however it exits.
publish='import fcntl, os, subprocess, sys
runtime_dir, setup_dir, check = sys.argv[1:]
runtime = f"{runtime_dir}/python"
fcntl.flock(os.open(runtime_dir, os.O_RDONLY), fcntl.LOCK_EX)
try:
    usable = subprocess.run([f"{runtime}/bin/python3", "-I", "-c", check], capture_output=True).returncode == 0
except OSError:
    usable = False
if not usable:
    # A runtime that fails the check, for example one baked into the image for another platform, is set aside.
    if os.path.lexists(runtime):
        os.rename(runtime, f"{setup_dir}/replaced")
    os.rename(f"{setup_dir}/python", runtime)'
"$fresh/bin/python3" -I -c "$publish" "$runtime_dir" "$setup_dir" "$import_check"
# Reports why the runtime in place fails, if it does.
"$runtime_dir/python/bin/python3" -I -c "$import_check"
