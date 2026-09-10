# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit OSWorld agent behavior contracts.

The contract separates the model-facing protocol from history selection and
gives training/evaluation profiles a semantic identity.  Sampling parameters
remain request-owned and are recorded elsewhere; this module covers Gym-owned
agent behavior only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from responses_api_agents.osworld_agent.adapter_agents import (
    DEFAULT_NEMOTRON_PROTOCOL_ID,
    normalize_nemotron_agent_options,
    resolve_nemotron_protocol,
)
from responses_api_agents.osworld_agent.history_policy import HistoryPolicySpec
from responses_api_agents.osworld_agent.runner_registry import RunnerSpec
from responses_api_agents.osworld_agent.trajectory import stable_id


RolloutPurpose = Literal["training", "evaluation"]
ParityMode = Literal["strict", "declared"]

_LEGACY_HISTORY_KEYS = frozenset({"max_image_history_length", "max_live_images"})
_EXPLICIT_CONTRACT_KEYS = frozenset({"history_policy", "model_protocol_id"})


@dataclass(frozen=True)
class ResolvedAgentContract:
    """One resolved, serializable Gym-owned behavior profile."""

    contract: dict[str, Any]
    agent_kwargs: dict[str, Any]
    history_policy: HistoryPolicySpec | None
    model_protocol_id: str | None

    @property
    def contract_id(self) -> str:
        return str(self.contract["agent_contract_id"])


def _nemotron_options(agent_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize every constructor option that changes adapter behavior."""

    return normalize_nemotron_agent_options(agent_kwargs)


def resolve_agent_contract(
    *,
    runner_spec: RunnerSpec,
    max_steps: int,
    max_trajectory_length: int,
    screen_size: tuple[int, int] = (1920, 1080),
    platform: str = "ubuntu",
    explicit_history_policy: Mapping[str, Any] | None,
    model_protocol_id: str | None,
    rollout_purpose: RolloutPurpose | None,
    parity_mode: ParityMode,
) -> ResolvedAgentContract:
    """Normalize legacy settings and return the exact child-agent contract."""

    agent_kwargs = dict(runner_spec.agent_kwargs)
    misplaced_contract_fields = sorted(_EXPLICIT_CONTRACT_KEYS.intersection(agent_kwargs))
    if misplaced_contract_fields:
        raise ValueError(
            "Configure OSWorld contract fields at agent config level, not inside agent_kwargs: "
            + ", ".join(misplaced_contract_fields)
        )
    history_policy: HistoryPolicySpec | None = None
    protocol_contract: dict[str, Any] | None = None

    if runner_spec.kind == "nemotron_v3_nano_omni_agent":
        legacy_history_fields = sorted(_LEGACY_HISTORY_KEYS.intersection(agent_kwargs))
        if explicit_history_policy is not None and legacy_history_fields:
            raise ValueError(
                "Explicit history_policy cannot be combined with legacy agent kwargs: "
                + ", ".join(legacy_history_fields)
            )
        if explicit_history_policy is None:
            keep_images = agent_kwargs.pop("max_image_history_length", max_trajectory_length)
            max_live_images = agent_kwargs.pop("max_live_images", None)
            history_policy = HistoryPolicySpec.from_legacy(
                keep_images=keep_images,
                max_live_images=max_live_images,
            )
        else:
            history_policy = HistoryPolicySpec.from_mapping(explicit_history_policy)

        protocol = resolve_nemotron_protocol(model_protocol_id or DEFAULT_NEMOTRON_PROTOCOL_ID)
        thinking = bool(_nemotron_options(agent_kwargs)["thinking"])
        protocol_contract = protocol.to_contract(thinking=thinking)
        resolved_protocol_id: str | None = protocol.protocol_id
        agent_options = _nemotron_options(agent_kwargs)
    else:
        if explicit_history_policy is not None:
            raise ValueError(f"runner {runner_spec.name!r} does not support an explicit history_policy")
        if model_protocol_id is not None:
            raise ValueError(f"runner {runner_spec.name!r} does not support model_protocol_id")
        resolved_protocol_id = None
        # Other runners keep their existing behavior and parity semantics.
        # Do not echo arbitrary native-agent kwargs into evidence because a
        # custom runner may carry credentials there.
        agent_options = None

    semantic_contract: dict[str, Any] = {
        "schema_version": 1,
        "runner": {
            "name": runner_spec.name,
            "kind": runner_spec.kind,
            "action_space": runner_spec.action_space,
            "observation_type": runner_spec.observation_type,
            "agent_class_path": runner_spec.agent_class_path,
            "platform": platform,
            "screen_size": list(screen_size),
        },
        "max_steps": max_steps,
        "model_protocol": protocol_contract,
        "history_policy": history_policy.to_contract() if history_policy is not None else None,
        "agent_options": agent_options,
    }
    contract_id = stable_id("osworld-agent-contract", semantic_contract)
    contract = {
        **semantic_contract,
        "agent_contract_id": contract_id,
        # Purpose and parity describe selection/enforcement, not behavior, so
        # they intentionally do not participate in agent_contract_id.
        "rollout_purpose": rollout_purpose,
        "parity_mode": parity_mode,
    }
    return ResolvedAgentContract(
        contract=contract,
        agent_kwargs=agent_kwargs,
        history_policy=history_policy,
        model_protocol_id=resolved_protocol_id,
    )


def assert_train_eval_parity(
    training: ResolvedAgentContract,
    evaluation: ResolvedAgentContract,
    *,
    parity_mode: ParityMode,
) -> None:
    """Fail closed for the Nemotron adapter without changing other runners."""

    training_kind = (training.contract.get("runner") or {}).get("kind")
    evaluation_kind = (evaluation.contract.get("runner") or {}).get("kind")
    parity_is_in_scope = (
        training_kind == "nemotron_v3_nano_omni_agent" and evaluation_kind == "nemotron_v3_nano_omni_agent"
    )
    if parity_is_in_scope and parity_mode == "strict" and training.contract_id != evaluation.contract_id:
        raise ValueError(
            "Training/evaluation OSWorld agent contracts differ while "
            "agent_contract_parity_mode='strict': "
            f"training={training.contract_id}, evaluation={evaluation.contract_id}. "
            "Make the protocol/history/agent options equal, or set "
            "agent_contract_parity_mode='declared' for an intentional difference."
        )
