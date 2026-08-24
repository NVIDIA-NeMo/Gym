# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import shim exposing Gym's AutomationBench taskset to Verifiers V1."""

from benchmarks.automationbench.taskset import AutomationBenchTaskset


__all__ = ["AutomationBenchTaskset"]
