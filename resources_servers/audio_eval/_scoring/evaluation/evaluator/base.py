# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/evaluator/base.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills
# Modifications: the batch path (`eval_full`, with its `tqdm` dependency) and `supports_single_eval` are removed,
# and `nested_dataclass` from `nemo_skills.utils` is replaced by `dataclass`; the rest is unchanged.

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(kw_only=True)
class BaseEvaluatorConfig:
    # TODO: should we pass input_file separately everywhere?
    input_file: str | None = None  # could be None for interleaved evals
    data_dir: str | None = None
    split: str = "test"


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, config: Dict[str, Any], num_parallel_requests=10):
        """Initialize evaluator with configuration."""
        self.config = config
        self.num_parallel_requests = num_parallel_requests

    async def eval_single(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single data point during generation (optional).

        Args:
            data_point: Single data point with generation results

        Returns:
            Dict with evaluation results to merge into data_point

        Raises:
            NotImplementedError: If single evaluation is not supported
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support single evaluation during generation")
