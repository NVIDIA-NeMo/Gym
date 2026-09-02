# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gpqa_diamond server.

gpqa_diamond reuses mcqa's wire model verbatim (its app.py imports ``MCQAVerifyRequest`` from
``resources_servers.mcqa.app``), so its task data is the mcqa schema unchanged. Committed rows
carry uuid, 4-key options ([{A: ...}, {B: ...}, {C: ...}, {D: ...}]), a single-letter
expected_answer, grading_mode='strict_single_letter_boxed' (present in every row but unread by
the overridden verify(), which uses GPQA-specific letter extraction), and a provenance
``metadata`` dict {explanation, subset_for_metrics, difficulty} that grading explicitly ignores.
"""

from resources_servers.mcqa.task_data import TaskData as MCQATaskData


class TaskData(MCQATaskData):
    """GPQA-Diamond rows validate against the shared MCQA task schema (no extra fields)."""
