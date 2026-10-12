"""SWE-bench wrapper agent for NeMo-Gym.

This module provides integration between NeMo-Skills' SWE-bench evaluation
capabilities and NeMo-Gym's agent framework.
"""

from .app import (
    SWEBenchRunRequest,
    SWEBenchVerifyRequest,
    SWEBenchVerifyResponse,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
)


__all__ = [
    "SWEBenchWrapper",
    "SWEBenchWrapperConfig",
    "SWEBenchRunRequest",
    "SWEBenchVerifyRequest",
    "SWEBenchVerifyResponse",
]
