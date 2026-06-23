# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""SWE environment verifier resources server.

A ``SimpleResourcesServer`` whose ``verify()`` extracts the agent's patch from
the response, builds a ``SweTask`` from the per-task metadata, grades it in its
own fresh, stateless sandbox via the ``verify_task`` orchestrator, and returns
the eval-side fields plus reward.

For the apptainer provider it must co-locate with the ``.sif``/Lustre storage
(and Docker for nested families). The reward is a non-nullable float; masking is
carried as ``reward=0.0`` plus ``mask_sample``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweTask


_FENCED_DIFF = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def _as_list(value: Any) -> list[str]:
    """Coerce a metadata value into a list of strings.

    Accepts ``None`` (yields an empty list), an existing list (each element
    stringified), or a string. A string that looks like a JSON array is parsed
    into its elements; any other non-empty string becomes a single-element list.

    Args:
        value: The raw metadata value to normalize.

    Returns:
        list[str]: The value expressed as a list of strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return [str(v) for v in json.loads(stripped)]
            except json.JSONDecodeError:
                pass
        return [stripped] if stripped else []
    return [str(value)]


class SweEnvVerifierConfig(BaseResourcesServerConfig):
    """Configuration for the SWE environment verifier.

    Attributes:
        sandbox_provider: Single-key provider mapping selecting the sandbox
            backend (e.g. ``{"docker": {}}``).
        model_patch_field: Metadata key under which the agent's patch is stored.
        opensandbox_service_url: Optional URL of an opensandbox service.
    """

    sandbox_provider: dict[str, Any] = {"docker": {}}
    model_patch_field: str = "model_patch"
    opensandbox_service_url: str | None = None


class SweEnvVerifyResponse(BaseVerifyResponse):
    """Verify response carrying the SWE eval outcome alongside the reward.

    Attributes:
        resolved: Whether the patch resolved the task.
        patch_exists: Whether a non-empty patch was supplied.
        patch_applied: Whether the patch applied cleanly in the sandbox.
        eval_error: Whether evaluation failed and the sample is masked.
        error_kind: Typed error category when evaluation failed, else ``None``.
        mask_sample: Whether the sample should be excluded from training.
        instance_id: Identifier of the graded instance.
    """

    resolved: bool = False
    patch_exists: bool = False
    patch_applied: bool = False
    eval_error: bool = False
    error_kind: str | None = None
    mask_sample: bool = False
    instance_id: str = ""


class SweEnvVerifier(SimpleResourcesServer):
    """Resources server that grades an agent's patch in a fresh sandbox."""

    config: SweEnvVerifierConfig

    async def verify(self, body: BaseVerifyRequest) -> SweEnvVerifyResponse:
        """Grade the agent's patch for one task and build the verify response.

        Builds a ``SweTask`` from the request, evaluates it in a fresh sandbox,
        converts the eval report into a reward, and masks the sample when the
        report carries an error.

        Args:
            body: The verify request with per-task metadata and agent response.

        Returns:
            SweEnvVerifyResponse: The reward together with the eval-side fields.
        """
        task = build_task(body, self.config.model_patch_field)
        report = await verify_task(self.config.sandbox_provider, task)
        reward = reward_from_report(report)
        masked = report.error_kind is not None
        return SweEnvVerifyResponse(
            **body.model_dump(),
            reward=reward,
            resolved=report.resolved,
            patch_exists=report.patch_exists,
            patch_applied=report.patch_applied,
            eval_error=masked,
            error_kind=report.error_kind,
            mask_sample=masked,
            instance_id=report.instance_id,
        )


def build_task(body: BaseVerifyRequest, patch_field: str) -> SweTask:
    """Map a verify request onto a SweTask.

    Reads the per-task metadata and the agent response and assembles the task
    fields the eval harness needs. Module-level (not a method) so it is
    unit-testable without instantiating the Pydantic server.

    Args:
        body: The verify request with per-task metadata and agent response.
        patch_field: Metadata key under which the agent's patch is stored.

    Returns:
        SweTask: The task to grade, including the extracted model patch.
    """
    metadata: dict[str, Any] = dict(body.responses_create_params.metadata or {})
    # Some rows nest task fields inside a stringified ``instance_dict`` (e.g.
    # fail_to_pass_select / base_dockerfile). Surface those keys at top level so the
    # ``*_select`` + dockerfile-ENV handling and the other harnesses find them;
    # explicit top-level metadata keys take precedence.
    instance_dict = metadata.get("instance_dict")
    if isinstance(instance_dict, str):
        try:
            instance_dict = json.loads(instance_dict)
        except json.JSONDecodeError:
            instance_dict = None
    if isinstance(instance_dict, dict):
        metadata = {**instance_dict, **metadata}
    patch = extract_patch(body.response, metadata, patch_field)
    return SweTask(
        instance_id=str(metadata.get("instance_id", "unknown")),
        image=metadata.get("image"),
        base_commit=metadata.get("base_commit"),
        repo_workdir=str(metadata.get("repo_workdir", "/testbed")),
        test_command=str(metadata.get("test_command", "")),
        test_framework=str(metadata.get("test_framework", "")),
        model_patch=patch,
        test_patch=str(metadata.get("test_patch", "")),
        # Some rows carry the required-test lists under UPPERCASE FAIL_TO_PASS /
        # PASS_TO_PASS keys. Reading lowercase only would leave those rows with empty
        # required-test lists, which the rebench/ext resolution rule (empty subset <= any
        # set) scores resolved=True for EVERY sample -> reward inflation. Fall back to the
        # UPPERCASE keys.
        fail_to_pass=_as_list(metadata.get("fail_to_pass") or metadata.get("FAIL_TO_PASS")),
        pass_to_pass=_as_list(metadata.get("pass_to_pass") or metadata.get("PASS_TO_PASS")),
        benchmark=str(metadata.get("benchmark", "swe-bench-ext")),
        split=str(metadata.get("split", "test")),
        metadata=metadata,
    )


def extract_patch(response: NeMoGymResponse, metadata: dict[str, Any], patch_field: str) -> str:
    """Read the patch from the response.

    Prefers the normalized patch stored on the response metadata; if absent,
    falls back to the first fenced ``diff``/``patch`` block found in the output
    text.

    Args:
        response: The agent response to read the patch from.
        metadata: Per-task metadata (unused for lookup but kept for parity).
        patch_field: Metadata key under which the patch is stored.

    Returns:
        str: The extracted patch, or an empty string if none is found.
    """
    response_metadata = getattr(response, "metadata", None) or {}
    patch = response_metadata.get(patch_field)
    if patch:
        return str(patch)
    for item in getattr(response, "output", []) or []:
        text = _item_text(item)
        if text:
            match = _FENCED_DIFF.search(text)
            if match:
                return match.group(1)
    return ""


def _item_text(item: Any) -> str:
    """Extract the text content from a response output item.

    Args:
        item: A response output item whose ``content`` may be a string or a list
            of content chunks each carrying a ``text`` attribute.

    Returns:
        str: The concatenated text, or an empty string if there is none.
    """
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


if __name__ == "__main__":
    SweEnvVerifier.run_webserver()
