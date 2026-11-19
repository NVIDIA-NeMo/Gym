# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the episode budget before routing, on both agent and model servers."""

from dataclasses import replace

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from nemo_gym.core.correlation import encode, parse_path
from nemo_gym.core.generation_budget import EpisodeGenerationBudget, generation_budget_context
from nemo_gym.server_runtime.request_handling.rollout_context import scope_with_routed_path


class GenerationBudgetMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        budget = None
        try:
            parsed = parse_path(scope["path"])
            if parsed is not None and parsed[0].generation_budget is not None:
                context, rest = parsed
                budget = EpisodeGenerationBudget.from_segment(context.generation_budget)
                scope = scope_with_routed_path(scope, encode(replace(context, generation_budget=None)) + rest)
        except ValueError as error:
            await JSONResponse({"detail": str(error)}, status_code=422)(scope, receive, send)
            return
        with generation_budget_context(budget):
            await self.app(scope, receive, send)
