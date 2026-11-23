#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Seed of resources_servers/swe_rebench/app.py (_start_task_sandbox), run by the overlay Dockerfile's second RUN line
# in the agent's sandbox before the agent starts. Finds or installs a Python 3.10+ for the server's vendored seed and
# capture code, records it in python_bin, and runs seed.py (anti-cheat scrub, git identity, pristine untracked
# files). A non-zero exit fails the seed, as the server fails a seed whose required scrub or untracked listing failed.
S=/var/lib/nemo_gym
. "$S/python.sh"
py=$(nemo_gym_python "$S") || { echo "[nemo_gym setup] no Python 3.10+ for the seed" >&2; exit 1; }
printf '%s\n' "$py" > "$S/python_bin"
exec "$py" -I "$S/seed.py"
