# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract base for pluggable task strategies used by StirrupAgentWrapper.

Implement a subclass of ``TaskStrategy`` to support a new benchmark or
evaluation task under the shared Stirrup agent harness.  The wrapper
calls these hooks at well-defined points during ``responses()`` and
``run()`` so that all Stirrup mechanics (agent creation, history
conversion, Ray execution) stay in one place.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse


class TaskSampleSkipError(Exception):
    """Raised by a task strategy to skip a sample without crashing the run."""


class TaskStrategy(ABC):
    """Pluggable task logic for a Stirrup-based agent wrapper.

    Subclasses define *what* the agent works on; the wrapper defines
    *how* the Stirrup agent is launched and its history is converted.
    """

    # ------------------------------------------------------------------
    # Prompt / input preparation
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_task_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse task-specific fields out of the request metadata dict.

        The returned dict is passed to every other hook as ``task_info``.
        """
        ...

    @abstractmethod
    def build_system_prompt(self, task_info: Dict[str, Any], config: Any) -> str:
        """Return the system prompt for the Stirrup agent."""
        ...

    @abstractmethod
    def build_user_prompt(self, task_info: Dict[str, Any], config: Any) -> str:
        """Return the user/task prompt for the Stirrup agent."""
        ...

    def prepare_input_files(
        self, task_info: Dict[str, Any]
    ) -> Optional[str]:
        """Download / stage any input files the agent needs.

        Return the path to a directory containing the files, or ``None``
        if the task has no reference files.  The caller is responsible for
        cleaning up the directory after the agent run.
        """
        return None

    # ------------------------------------------------------------------
    # Execution provider
    # ------------------------------------------------------------------

    def get_exec_provider(
        self, task_info: Dict[str, Any], config: Any
    ) -> Any:
        """Return a custom ``CodeExecToolProvider``, or ``None`` for the default local one.

        Override this in tasks that need a specialised execution
        environment (e.g. Apptainer containers for SWE-bench).
        The returned object must implement Stirrup's
        ``CodeExecToolProvider`` interface (async context manager).
        """
        return None

    # ------------------------------------------------------------------
    # Response metadata
    # ------------------------------------------------------------------

    @abstractmethod
    def build_response_metadata(
        self,
        task_info: Dict[str, Any],
        deliverable_text: str,
        elapsed_seconds: float,
    ) -> Dict[str, str]:
        """Build the ``metadata`` dict attached to the NeMoGymResponse."""
        ...

    @abstractmethod
    def response_id(self, task_info: Dict[str, Any]) -> str:
        """Return a stable ID string for the NeMoGymResponse."""
        ...

    def fallback_message_id(self, task_info: Dict[str, Any]) -> str:
        """ID used for the empty-output fallback message."""
        return f"msg-{self.response_id(task_info)}-empty"

    # ------------------------------------------------------------------
    # Reward / scoring
    # ------------------------------------------------------------------

    @abstractmethod
    async def compute_reward(
        self,
        deliverable_text: str,
        task_info: Dict[str, Any],
        config: Any,
        model_base_url: str,
        model_name: str,
        api_key: str = "dummy",
        deliverable_content_blocks: list | None = None,
    ) -> float:
        """Score the agent's deliverable.  Return a float in [0, 1]."""
        ...

    # ------------------------------------------------------------------
    # Verify response construction
    # ------------------------------------------------------------------

    @abstractmethod
    def build_verify_response(
        self,
        *,
        responses_create_params: Any,
        response: NeMoGymResponse,
        reward: float,
        task_info: Dict[str, Any],
        deliverable_text: str,
        elapsed_seconds: float,
        judge_response: dict | None = None,
    ) -> BaseVerifyResponse:
        """Construct the task-specific ``BaseVerifyResponse`` subclass."""
        ...

    def build_skipped_verify_response(
        self,
        *,
        responses_create_params: Any,
        task_info: Dict[str, Any],
        reason: str,
    ) -> BaseVerifyResponse:
        """Construct a verify response for a skipped sample.

        Override in subclasses to add task-specific fields.
        Raises ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support skipping samples"
        )
