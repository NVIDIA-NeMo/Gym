# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity between a native-path run and a Harbor-path run of the same tasks.

After an environment is converted to Harbor task folders, both paths are run on the same tasks with the same harness
and model, N repeats each. This package matches the two runs task by task, computes per-task and overall mean reward
with a bootstrap interval over tasks, counts masked, failed and missing rollouts, and gives a PASS, FAIL or
INCONCLUSIVE verdict. A strict mode checks model-free runs (reference solution, negative control) for exact equality.

`reading` streams one side's rollout files into per-task counts; `parity` compares the two sides, renders the report
and holds the command line. Run it with `python -m scripts.harbor_parity --help`.
"""
