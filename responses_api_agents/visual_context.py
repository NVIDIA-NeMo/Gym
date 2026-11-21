# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared admission checks for the visual agents' capture-backed history client."""

from collections.abc import Mapping

from nemo_gym.agents import SimpleResponsesAPIAgent
from nemo_gym.agents.context_management import ContextHistoryConfig, ContextManagedResponsesClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


def visual_context_client(
    agent: SimpleResponsesAPIAgent,
    *,
    config: ContextHistoryConfig | None,
    logical_rollout_id: str | None,
    initial_request: NeMoGymResponseCreateParamsNonStreaming,
    cookies: Mapping[str, str] | None = None,
) -> ContextManagedResponsesClient | None:
    """Opt in explicitly; a disabled history policy is a capture-backed identity control.

    Seed observations must already be in initial_request, so their images receive
    the same protection as instructions. Never allocate an independent capture ID.
    """
    if config is None:
        return None
    if not agent._token_id_capture_enabled() or not logical_rollout_id:
        raise ValueError("Visual context management requires enabled token capture and a dispatched rollout ID")
    if config.max_response_retries:
        raise ValueError("Visual context management does not add model resampling retries")
    if getattr(agent.config, "return_transitions", False):
        raise ValueError("Visual context management requires return_transitions=false")
    if (
        getattr(agent.config, "max_no_boxed_truncation_retries", 0)
        and getattr(agent.config, "no_boxed_truncation_retry_factor", 1) != 1
    ):
        raise ValueError("Visual context management requires a fixed per-call output budget")
    budget = initial_request.max_output_tokens
    if not budget or config.guards.reserved_generation_tokens < budget:
        raise ValueError("Context guards must reserve the complete explicit max_output_tokens budget")
    limit = config.guards.max_total_tokens
    if limit is None:
        raise ValueError("Visual context management requires an explicit max_total_tokens guard")
    legacy_limit = getattr(agent.config, "max_total_sequence_length", None)
    if legacy_limit is not None and legacy_limit != limit:
        raise ValueError("Agent max_total_sequence_length and context max_total_tokens must agree")
    image_limit = getattr(agent.config, "max_images", None)
    if image_limit is not None and image_limit != config.guards.max_active_images:
        raise ValueError("Agent max_images and context max_active_images must agree")
    return ContextManagedResponsesClient(
        server_client=agent.server_client,
        model_server=agent.config.model_server,
        logical_rollout_id=logical_rollout_id,
        config=config,
        initial_request=initial_request,
        cookies=cookies,
    )
