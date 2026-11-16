# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the bird_sql_agentic server.

This server uses the same task rows as bird_sql, so the schema is imported from bird_sql
rather than redefined.
"""

from resources_servers.bird_sql.task_data import TaskData as BirdSqlTaskData


class TaskData(BirdSqlTaskData):
    """bird_sql_agentic rows validate against the shared bird_sql task schema (no extra fields)."""