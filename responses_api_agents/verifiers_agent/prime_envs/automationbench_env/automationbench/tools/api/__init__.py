# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""API interfaces to AutomationBench's simulated services."""

from automationbench.tools.api.contract import APIContract
from automationbench.tools.api.encode import base64_encode

__all__ = ["APIContract", "base64_encode"]
