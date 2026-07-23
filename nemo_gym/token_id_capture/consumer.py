# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""The consumer that turns a rollout's captured tokens into trajectories.

This is the single primitive both consumers call after a rollout finishes: Gym's
rollout collection (co-located, reading the token store's files) and a trainer's
finalizer (which passes a ``TokenSource``, e.g. HTTP or TransferQueue-backed).
The only difference between them is where the ``TokenEntry`` records come from;
the build and projection are identical.

It is deliberately free of any rollout-record or model-server imports, so it
does not couple to those layers. The caller supplies the ``rollout_id`` (Gym's
rollout collection derives it from the record's task/rollout/attempt indices)
and the reward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from nemo_gym.token_id_capture.builder import (
    assert_nemo_rl_contiguity,
    build_trajectories,
    per_request,
    prefix_merging,
    project_main_chain_response,
)
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.source import TokenSource
from nemo_gym.token_id_capture.store import TokenCaptureStore


def token_id_capture_dirs_from_config(global_config_dict) -> list[Path]:
    """Resolve the token store directory when training-token capture is enabled, else []."""
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    config = TokenIdCaptureConfig.model_validate(global_config_dict)
    directory = config.resolved_dir()
    return [directory] if (config.token_id_capture_enabled and directory is not None) else []


def _assemble(
    rollout_id: str,
    entries: list[TokenEntry],
    builder: str,
    reward: float,
    reward_components: Optional[dict[str, float]],
    model: str,
) -> dict:
    out = prefix_merging(entries) if builder == "prefix_merging" else per_request(entries)
    response = project_main_chain_response(rollout_id, out, model=model)
    assert_nemo_rl_contiguity(response)
    trajectories = build_trajectories(
        rollout_id, entries, builder=builder, reward=reward, reward_components=reward_components
    )
    return {
        "rollout_id": rollout_id,
        "builder": builder,
        "trajectories": [t.model_dump() for t in trajectories],
        "nemo_rl_response": response,
    }


def trajectories_for_rollout(
    rollout_id: str,
    token_capture_dirs: list[Path],
    *,
    builder: str = "prefix_merging",
    reward: float = 0.0,
    reward_components: Optional[dict[str, float]] = None,
    model: str = "",
) -> Optional[dict]:
    """Co-located path: read the rollout's tokens from the store files and build its trajectories.

    ``reward`` (scalar, for GRPO) and ``reward_components`` (named per-objective scores, for GDPO)
    come from the verifier result and ride the trajectory; they are not read from the token store.
    Returns ``None`` when no tokens were captured for the rollout (capture off, or a dialect the
    engine returned no ids for). Mirrors how evaluation capture is merged into a rollout record.
    """
    for directory in token_capture_dirs:
        entries = TokenCaptureStore(directory).read_entries(rollout_id)
        if entries:
            return _assemble(rollout_id, entries, builder, reward, reward_components, model)
    return None


async def trajectories_from_source(
    rollout_id: str,
    source: TokenSource,
    *,
    builder: str = "prefix_merging",
    reward: float = 0.0,
    reward_components: Optional[dict[str, float]] = None,
    model: str = "",
) -> Optional[dict]:
    """Non-co-located path: read the rollout's tokens through a ``TokenSource`` (HTTP, or a
    trainer's own transport) and build its trajectories. Returns ``None`` when none were captured."""
    entries = await source.tokens_for(rollout_id)
    if not entries:
        return None
    return _assemble(rollout_id, entries, builder, reward, reward_components, model)
