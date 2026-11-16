# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking CI checks that steer new Gym components toward the Environment Server architecture.

Selection runs under bare `python3` before Gym is installed, so `ci` and `rules` import only the standard library
at module level. Each component is inspected in its own venv, installed the way Gym installs it for its tests.
"""
