#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Grades the agent's patch with the Scale-SWE server's own verification code (tests/nemo_gym/grade.py), in a
# separate sandbox of the row's image. Exit 75 without a reward masks the sample (no verdict), as the server does.
mkdir -p /logs/verifier
for candidate in python3 python /usr/local/bin/python3 /usr/bin/python3 /opt/conda/bin/python3; do
  if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -I -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    exec "$candidate" -I /tests/nemo_gym/grade.py
  fi
done
echo "[nemo_gym] no Python 3.9+ in this image; the task cannot be graded here (masked)" >&2
exit 75
