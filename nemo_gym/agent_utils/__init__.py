# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared agent execution and supervision helpers.

Sandbox sessions and supervisor clients run on the agent server. The standalone
process supervisor is uploaded into task sandboxes and uses only the standard
library. Core sandbox APIs and providers live in :mod:`nemo_gym.sandbox`.
"""
