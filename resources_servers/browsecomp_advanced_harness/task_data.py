# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the browsecomp_advanced_harness server.

This server reuses the tavily_search lineage: its app.py redeclares ``TavilySearchRunRequest``
with the identical two required fields (question, ground_truth), so the task schema is imported
from tavily_search rather than redefined. Rows additionally carry tools inside
``responses_create_params`` (framework-owned, not typed here). verify() branches on server
config.use_judge — LLM judge against ground_truth, or (when use_judge=false) exact string
equality between ground_truth and the span extracted from the last assistant message by a fixed
"Answer: ... Confidence:" pattern; ground_truth is never interpreted as a regex.
"""

from resources_servers.tavily_search.task_data import TaskData as TavilySearchTaskData


class TaskData(TavilySearchTaskData):
    """BrowseComp rows validate against the shared tavily_search task schema (no extra fields)."""
