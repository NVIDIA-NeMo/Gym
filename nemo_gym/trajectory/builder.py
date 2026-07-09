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

"""The trajectory builder and the NeMo-RL delivery projection.

The builder is a library called in the agent-server process after verify().
It reads token ids through a TokenSource and the ModelCallRecords used to
classify calls, drops the auxiliary calls, runs the chosen builder, applies the
loss mask, joins the reward, and returns a list of Trajectory objects.

The projection turns a chain into the shape NeMo-RL consumes. NeMo-RL's
`_postprocess_nemo_gym_to_nemo_rl_result` walks a response's output items that
carry prompt_token_ids / generation_token_ids / generation_log_probs and
asserts that each item's prompt extends the tokens seen so far. So the
projection emits, per chain, an ordered output list where item k's prompt is
the chain's cumulative tokens before generation k — contiguous by
construction — and the existing NeMo-RL consumer ingests these rollouts with no
code changes. Only the main chain is delivered today; branches are counted in
provenance but not yet emitted for training.
"""

from __future__ import annotations

import logging
from typing import Optional

from nemo_gym.observability.records import ModelCallRecord
from nemo_gym.trajectory.registry import get_builder
from nemo_gym.trajectory.sources import ScopedReward, TokenSource, Trajectory, classify_auxiliary
from nemo_gym.trajectory.strategies import Chain


logger = logging.getLogger(__name__)


def build_trajectories(
    rollout_id: str,
    token_source: TokenSource,
    model_call_records: Optional[list[ModelCallRecord]] = None,
    builder: str = "per_request",
    reward: float = 0.0,
    rewards: Optional[list[ScopedReward]] = None,
    is_resolved: Optional[bool] = None,
    policy_model: str = "",
    aux_fingerprints: Optional[tuple[str, ...]] = None,
) -> list[Trajectory]:
    entries = token_source.entries(rollout_id)
    aux: dict[str, str] = {}
    if model_call_records:
        kwargs = {"policy_model": policy_model}
        if aux_fingerprints is not None:
            kwargs["fingerprints"] = aux_fingerprints
        aux = classify_auxiliary(model_call_records, **kwargs)
        entries = [e for e in entries if e.request_id not in aux]
    if not entries:
        return []

    out = get_builder(builder)(entries)
    resp_by_reqid = {s.request_id: s.response for s in (model_call_records or [])}

    trajectories: list[Trajectory] = []
    for chain in out.chains:
        ids, mask, lps, _ = chain.flatten()
        messages = [resp_by_reqid.get(link.entry.request_id, {}) for link in chain.links]
        trajectories.append(
            Trajectory(
                rollout_id=rollout_id,
                chain_id=chain.chain_id,
                token_ids=ids,
                loss_mask=mask,
                logprobs=lps,
                messages=messages,
                reward=reward,
                rewards=rewards or [ScopedReward(value=reward)],
                is_resolved=is_resolved,
                provenance={
                    "builder": out.notes.get("builder", builder),
                    "n_entries": len(chain.links),
                    "n_aux_excluded": len(aux),
                    "quarantined_branches": len(out.quarantined),
                    "notes": out.notes,
                    "token_source": type(token_source).__name__,
                },
            )
        )
    return trajectories


def main_chain(trajectories: list[Trajectory]) -> Optional[Trajectory]:
    for t in trajectories:
        if t.chain_id == "main":
            return t
    return max(trajectories, key=lambda t: len(t.token_ids), default=None)


# ---------------------------------------------------------------------------
# NeMo-RL inline projection
# ---------------------------------------------------------------------------


def project_chain_to_response_output(chain: Chain, resp_by_reqid: dict[str, dict]) -> list[dict]:
    """Turn a chain into an ordered list of output items carrying token ids.
    Item k's prompt_token_ids is the cumulative tokens before generation k, so
    each item's prompt extends the previous one. The readable content is taken
    from the captured response for that call so decoding and logging stay
    meaningful."""
    items: list[dict] = []
    cumulative: list[int] = list(chain.root_prompt)
    for link in chain.links:
        cumulative = cumulative + link.interstitial
        base = {}
        src = resp_by_reqid.get(link.entry.request_id) or {}
        src_out = src.get("output") or []
        if src_out and isinstance(src_out[-1], dict):
            base = dict(src_out[-1])
        base.setdefault("type", "message")
        base["prompt_token_ids"] = list(cumulative)
        base["generation_token_ids"] = list(link.entry.generation_token_ids)
        base["generation_log_probs"] = list(link.entry.generation_log_probs)
        if link.entry.routed_experts is not None:
            base["routed_experts"] = link.entry.routed_experts
        items.append(base)
        cumulative = cumulative + list(link.entry.generation_token_ids)
    return items


def project_main_chain_response(
    rollout_id: str, chains: list[Chain], resp_by_reqid: dict[str, dict], model: str = ""
) -> dict:
    explicit_main = [c for c in chains if c.chain_id == "main"]
    mains = explicit_main or chains[:1]
    # NeMo-RL consumes one trajectory per rollout. When there is no single merged
    # main chain but several (e.g. per_request made one chain per call, or there
    # are sub-agent branches), only the first is delivered — the rest would be
    # dropped. Surface that loudly rather than silently losing training signal:
    # for a multi-turn append-only harness use prefix_merging so all turns land
    # in one contiguous main chain.
    if not explicit_main and len(chains) > 1:
        dropped = sum(len(link.entry.generation_token_ids) for c in chains[1:] for link in c.links)
        logger.warning(
            f"project_main_chain_response delivering only the first of {len(chains)} chains for "
            f"rollout {rollout_id}; {dropped} generated tokens in the other chains are not projected. "
            "Use builder=prefix_merging for multi-turn append-only rollouts to merge all turns."
        )
    if not mains:
        return {"id": f"proj-{rollout_id}", "model": model, "output": [], "usage": {}}
    output = project_chain_to_response_output(mains[0], resp_by_reqid)
    n_in = len(output[0]["prompt_token_ids"]) if output else 0
    n_out = sum(len(i["generation_token_ids"]) for i in output)
    return {
        "id": f"proj-{rollout_id}",
        "model": model,
        "object": "response",
        "output": output,
        "usage": {"input_tokens": n_in, "output_tokens": n_out},
        "ng_projection": {"rollout_id": rollout_id, "n_chains": len(chains), "n_branches": max(len(chains) - 1, 0)},
    }


def assert_nemo_rl_contiguity(response: dict) -> None:
    """Vendored replica of the contiguity assertion in NeMo-RL's
    _postprocess_nemo_gym_to_nemo_rl_result — run in Gym CI on every
    projected rollout so the consumer contract is enforced at the producer."""
    seen: list[int] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("generation_token_ids") is None:
            continue
        prompt = item.get("prompt_token_ids") or []
        if prompt[: len(seen)] != seen:
            raise AssertionError(
                "projection violates NeMo-RL prefix contiguity: an output "
                "item's prompt_token_ids does not extend the tokens seen so "
                "far (truncated/rewritten history or a branch leaked in)"
            )
        seen = list(prompt) + list(item["generation_token_ids"])
