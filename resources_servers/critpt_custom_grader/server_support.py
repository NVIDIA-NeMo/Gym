# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config models and helpers this grader owns, so it runs against unmodified core.

These types were once shared symbols in ``nemo_gym.server_utils``. They live here now, scoped
to this one server. Two of them declare operator config for this server. One builds the fixed
error body this server answers with. None of them touches shared code.

Scope note on ``ServerRequestPolicy``. ``no_resubmission`` and ``deadline_seconds`` state operator
intent. They do not change how the stock ``ServerClient`` sends a request. The single-dispatch and
the request deadline that a shared client change once added are not present here. So the stock
client may retry a ``/verify``. This server is safe under a retry: it grades one job at a time,
refuses a second concurrent job past its queue, and re-grades an idempotent request without any
double-scoring. See the server's own ``verify`` path. The one cost of a client retry is one more
graded run; scoring is unaffected.
"""

from typing import Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerRequestPolicy(BaseModel):
    """Operator intent for how a client should send a job to this server.

    ``no_resubmission``: send the request once. ``deadline_seconds``: bound the whole exchange.
    Both are declarative here. The stock client does not read them, so this server tolerates a
    retried request rather than relying on single dispatch (see the module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    no_resubmission: bool = False
    deadline_seconds: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _require_deadline_with_no_resubmission(self) -> "ServerRequestPolicy":
        if self.no_resubmission and self.deadline_seconds is None:
            raise ValueError("`no_resubmission: true` requires a finite `deadline_seconds` > 0")
        return self


class ServerRequestPrivacy(BaseModel):
    """Operator intent that a request's content, and a callee's answer, leave only through returns.

    ``private_requests``: this server withholds request bodies, validation detail, provider
    exceptions and task sources from its logs and its error bodies. This server enforces that at
    its own HTTP boundary (its private route class, its private ingress, its overridden setup
    hooks), not through shared code. The field records the operator's declaration.
    """

    model_config = ConfigDict(extra="forbid")

    private_requests: bool = False


def private_error_response(category: str, status_code: int) -> JSONResponse:
    """The fixed category-only error body this server answers with: ``{"error": {"category": <category>}}``."""
    return JSONResponse(content={"error": {"category": category}}, status_code=status_code)
