# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep GDPval's own answers out of what ``web_search`` and ``fetch_web_page`` return.

GDPval's gold deliverables and other models' submissions are public (the ``openai/gdpval`` dataset
and many copies on Hugging Face and GitHub). An agent that searches for the benchmark or the task id
finds them and copies the answer: Kimi K2.6 did so on task 11dcc268. The guard refuses queries and
URLs that name the benchmark or the current task, and drops search results that do.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import httpx


# Case-insensitive patterns matched against queries, URLs and search results.
DEFAULT_BLOCKED_WEB_PATTERNS: tuple[str, ...] = (
    # The benchmark's name in any spelling: gdpval, gdp-val, GDP_Val, "GDP val".
    r"gdp[\s_-]?val",
    # The directory of gold answers in the dataset and in the copies of it.
    r"deliverable_files",
)

BLOCKED_MESSAGE = "blocked: the GDPval benchmark and its tasks cannot be looked up during the evaluation"


class BlockedURLError(httpx.HTTPError):
    """A fetch, or a redirect it followed, went to a blocked URL."""


class WebGuard:
    def __init__(self, patterns: Iterable[str], task_id: Optional[str] = None):
        parts = list(patterns)
        if task_id:
            parts.append(re.escape(task_id))
        self._pattern = re.compile("|".join(f"(?:{part})" for part in parts), re.IGNORECASE) if parts else None

    def blocks(self, text: str) -> bool:
        return self._pattern is not None and self._pattern.search(text) is not None

    def filter_search_reply(self, data: dict) -> dict:
        """The Tavily reply without the results that mention a blocked pattern.

        Tavily writes its answer from all the results, so the answer goes too when any result went.
        """
        results = data.get("results", [])
        kept = [
            result
            for result in results
            if not any(self.blocks(str(result.get(field, ""))) for field in ("title", "url", "content"))
        ]
        answer = data.get("answer")
        if len(kept) < len(results) or (answer and self.blocks(answer)):
            answer = None
        return {**data, "answer": answer, "results": kept}

    async def check_request(self, request: httpx.Request) -> None:
        """httpx request hook: refuse the page and every redirect it follows to a blocked URL."""
        if self.blocks(str(request.url)):
            raise BlockedURLError(BLOCKED_MESSAGE)
