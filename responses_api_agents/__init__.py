# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility namespace for the former agent-harness package location.

New code should import built-in harnesses from :mod:`harnesses`. The
``responses_api_agents`` name remains the config/server protocol key, and this
namespace keeps historical Python imports working without duplicating the
implementations in the wheel.
"""

import importlib
import sys


# Importing the old package first still returns the canonical package object. ``harnesses`` also
# installs a small alias finder so every nested legacy import shares canonical module identity.
sys.modules[__name__] = importlib.import_module("harnesses")
