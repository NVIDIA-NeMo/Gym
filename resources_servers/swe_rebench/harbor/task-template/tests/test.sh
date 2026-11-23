#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Grades the agent's patch with the SWE-rebench server's own verification code (tests/nemo_gym/grade.py), in a
# separate sandbox of the row's image. Exit 75 without a reward masks the sample (no verdict), as the server does.
# Python: the image's 3.10+, else the pinned standalone release (python.sh); this runs before any agent code.
mkdir -p /logs/verifier
. /tests/nemo_gym/python.sh
if py=$(nemo_gym_python /opt/nemo_gym_verifier); then
  exec "$py" -I /tests/nemo_gym/grade.py
fi
echo "[nemo_gym] no Python 3.10+ in this image and the pinned one could not be installed; cannot grade (masked)" >&2
exit 75
