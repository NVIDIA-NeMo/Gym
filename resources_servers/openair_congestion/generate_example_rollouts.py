# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the five committed replay rollouts without a model service.

These examples exercise the resource server's real ``reset``/``step``/``close``
contract with the deterministic scripted-relief policy used by ``client.py``.
They prove contribution wiring and reward provenance; they are explicitly not
evidence of SFT or GRPO policy quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.openair_congestion.app import (
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
)
from resources_servers.openair_congestion.client import _tool_response, choose_action


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_INPUT = DATA_DIR / "example.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "example_rollouts.jsonl"
EVIDENCE_MAX_STEPS = 2
NEUTRAL_USER_PROMPT = (
    "Inspect the observed telemetry and choose exactly one safe congestion-control tool call or noop."
)
_STEP_INFO_KEYS = (
    "guardrail_accepted",
    "rejection_reason",
    "step_idx",
    "kpi_source",
    "dynamics_mode",
    "reward_version",
    "reward_measurements",
    "reward_terms",
    "causal_action_effects",
    "training_usable",
    "diagnostic_only",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_model_inputs(rows: list[dict[str, Any]]) -> None:
    """Reject example prompts that reveal evaluator-only scenario labels."""

    for index, row in enumerate(rows):
        scenario_id = str(row.get("scenario_id") or "")
        regime_names = {str(name) for name in (row.get("regime_mix") or {})}
        hidden_names = {scenario_id, *regime_names} - {""}
        content = " ".join(str(message.get("content", "")) for message in row["responses_create_params"]["input"])
        normalized = content.lower().replace("-", "_").replace(" ", "_")
        leaked = sorted(name for name in hidden_names if name.lower() in normalized)
        if leaked:
            raise ValueError(f"example row {index} leaks evaluator metadata: {leaked}")


async def _generate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _validate_model_inputs(rows)
    config = OpenAirCongestionResourcesServerConfig(
        host="",
        port=0,
        entrypoint="app.py",
        name="openair_congestion_example_generator",
        pool_size=1,
        agent_max_steps=EVIDENCE_MAX_STEPS,
    )
    env = OpenAirCongestionEnv(config=config, server_client=MagicMock(spec=ServerClient))
    generated: list[dict[str, Any]] = []

    for task_index, source_row in enumerate(rows):
        session_id = f"example-generator-{task_index}"
        # Keep the checked-in receipt compact while still showing one
        # intervention followed by persistent synthetic state transitions.
        row = {
            **source_row,
            "max_steps": min(
                int(source_row.get("max_steps", EVIDENCE_MAX_STEPS)),
                EVIDENCE_MAX_STEPS,
            ),
        }
        observation, _ = await env.reset(row, session_id=session_id)
        trajectory: list[dict[str, Any]] = []
        episode_return = 0.0
        terminated = truncated = False
        last_response: dict[str, Any] | None = None
        last_info: dict[str, Any] = {}

        try:
            for step_idx in range(int(row.get("max_steps", 16))):
                action = choose_action(observation, step_idx)
                last_response = _tool_response(action["name"], action["arguments"], step_idx)
                step_tuple = await env.step(
                    NeMoGymResponse.model_validate(last_response),
                    {"_ng_step_request_id": f"example-{task_index}-{step_idx}"},
                    session_id=session_id,
                )
                next_observation, raw_reward, raw_terminated, raw_truncated, raw_info = step_tuple
                reward = float(raw_reward)
                assert math.isfinite(reward)
                episode_return += reward
                last_info = {key: value for key, value in raw_info.items() if key != "_ng_lifecycle_generation"}
                trajectory.append(
                    {
                        "step": step_idx,
                        "observation": observation,
                        "action": action,
                        "reward": reward,
                        "next_observation": next_observation,
                        "terminated": bool(raw_terminated),
                        "truncated": bool(raw_truncated),
                        **{key: last_info.get(key) for key in _STEP_INFO_KEYS},
                    }
                )
                observation = next_observation
                terminated = bool(raw_terminated)
                truncated = bool(raw_truncated)
                if terminated or truncated:
                    break
        finally:
            await env.explicit_close(session_id)

        if last_response is None or not (terminated or truncated):
            raise RuntimeError(f"example row {task_index} did not complete a bounded episode")

        final_info = {key: last_info.get(key) for key in _STEP_INFO_KEYS}
        generated.append(
            {
                **row,
                "response": last_response,
                "reward": episode_return,
                "terminated": terminated,
                "truncated": truncated,
                "info": final_info,
                "trajectory": trajectory,
                "example_policy": "deterministic_scripted_relief_v1",
                "evidence_scope": "resource_server_wiring_not_model_quality",
                "_ng_task_index": task_index,
                "_ng_rollout_index": 0,
            }
        )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    if len(rows) != 5:
        raise ValueError(f"expected exactly five example rows, got {len(rows)}")
    rollouts = asyncio.run(_generate(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rollouts),
        encoding="utf-8",
    )
    print(f"wrote {len(rollouts)} deterministic rollouts to {args.output}")


if __name__ == "__main__":
    main()
