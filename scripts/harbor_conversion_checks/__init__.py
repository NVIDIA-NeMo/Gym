# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks and probes for converting a native Gym environment to Harbor task folders.

`python -m scripts.harbor_conversion_checks <command>`: `check` (content checks of the converted folders), `controls`
(negative-control and reward-forgery datasets), `fake-model` (a scripted OpenAI chat endpoint), `first-request`
(compare the first model request of two runs) and `subset` (a reproducible task id list). Standard library only.
"""
