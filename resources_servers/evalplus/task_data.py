# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the evalplus server.

Same pointer shape as code_fim (imported, not redefined): rows carry only
``verifier_metadata.task_id``. Here task_id is an EvalPlus key such as 'HumanEval/0' or 'Mbpp/2';
prompts, tests, and ground truth live OUT of the row in the ``evalplus`` package, selected by
``config.dataset`` (humaneval | mbpp) at server startup. Both dataset flows share this row shape
— only the task_id namespace differs.
"""

from resources_servers.code_fim.task_data import TaskData as CodeFIMTaskData


class TaskData(CodeFIMTaskData):
    """EvalPlus pointer row; ``task_id`` is namespaced 'HumanEval/<n>' or 'Mbpp/<n>'."""
