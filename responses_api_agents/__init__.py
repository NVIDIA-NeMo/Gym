# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility namespace for the former agent-harness package location.

New code should import built-in harnesses from :mod:`harnesses`. The
``responses_api_agents`` name remains the config/server protocol key, and this
namespace keeps historical Python imports working without duplicating the
implementations in the wheel.
"""

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parent.parent / "harnesses")]
