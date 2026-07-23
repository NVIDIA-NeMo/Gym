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

"""Turn a rollout's captured token records into trainable trajectories.

The builder is a pure function over a list of ``TokenEntry`` records (whatever a
``TokenSource`` returns). It has two strategies:

  per_request     assumes nothing about how the calls relate; every call becomes
                  its own training sequence. Always valid.
  prefix_merging  chains calls by the token-prefix relationship: each call is
                  parented to the earlier call whose full token sequence (prompt
                  plus generation) is the longest prefix of this call's prompt.
                  This rebuilds a multi-turn, append-only rollout into one chain.
                  A prompt that no longer extends any earlier call starts a new
                  root (a compacted or rewritten context). Two candidate parents
                  with identical sequences are ambiguous, so that subtree is
                  quarantined rather than guessed.

Both are order-independent: they do not depend on arrival order or any sequence
number. ``prefix_merging`` processes entries by increasing prompt length, which
is derived from the tokens themselves (a parent's prompt is shorter than its
child's), so concurrent or out-of-order capture yields the same result.

Loss masks follow provenance: tokens the policy generated are marked 1 (with
their captured log probabilities), and everything re-fed into a prompt (history,
tool output, tokens added between calls) is marked 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, Field

from nemo_gym.token_id_capture.records import TokenEntry


@dataclass
class ChainLink:
    entry: TokenEntry
    interstitial: list[int]  # prompt tokens added since the parent (tool output, new user turn); mask 0


@dataclass
class Chain:
    chain_id: str
    links: list[ChainLink] = field(default_factory=list)
    root_prompt: list[int] = field(default_factory=list)

    def flatten(self) -> tuple[list[int], list[int], list[Optional[float]], list[tuple[int, int, str]]]:
        """Expand the chain into (token_ids, loss_mask, log_probs, spans).

        Generated tokens get mask 1 and their log probabilities; prompt/interstitial
        tokens get mask 0 and no log probability. ``spans`` records, per generated
        segment, ``(start, end, model_call_id)`` so a trainer can attach per-call
        metadata (e.g. a weight version under async) to the tokens of each call."""
        ids: list[int] = list(self.root_prompt)
        mask: list[int] = [0] * len(ids)
        lps: list[Optional[float]] = [None] * len(ids)
        spans: list[tuple[int, int, str]] = []
        for link in self.links:
            ids += link.interstitial
            mask += [0] * len(link.interstitial)
            lps += [None] * len(link.interstitial)
            gen = link.entry.generation_token_ids
            glp = link.entry.generation_log_probs
            if len(glp) != len(gen):
                raise ValueError(
                    f"log-prob/token length mismatch on {link.entry.model_call_id}: {len(glp)} vs {len(gen)}"
                )
            start = len(ids)
            ids += gen
            mask += [1] * len(gen)
            lps += list(glp)
            spans.append((start, len(ids), link.entry.model_call_id))
        return ids, mask, lps, spans


@dataclass
class BuildOutput:
    chains: list[Chain]
    quarantined: list[str] = field(default_factory=list)  # model_call_ids
    notes: dict = field(default_factory=dict)


class Trajectory(BaseModel):
    """One trainable sequence: a flat token stream with a per-token loss mask and
    the behavior-policy log probabilities at the generated positions."""

    rollout_id: str
    chain_id: str
    token_ids: list[int]
    loss_mask: list[int]
    log_probs: list[Optional[float]]
    # Per-call provenance: (start, end, model_call_id) for each generated span, so async training can
    # attach a per-call weight version to the right tokens.
    spans: list[tuple[int, int, str]] = Field(default_factory=list)
    # The scalar reward drives single-objective training (GRPO). reward_components carries the
    # named per-objective scores for multi-objective training (GDPO); it is None for single-reward
    # environments. Both are per-response (per-rollout), copied from the verifier result; neither is
    # an engine fact, so neither appears on TokenEntry.
    reward: float = 0.0
    reward_components: Optional[dict[str, float]] = None
    provenance: dict = Field(default_factory=dict)


def per_request(entries: list[TokenEntry]) -> BuildOutput:
    ordered = sorted(entries, key=lambda e: (len(e.prompt_token_ids), e.model_call_id))
    chains = [
        Chain(chain_id=f"req-{i}", root_prompt=list(e.prompt_token_ids), links=[ChainLink(entry=e, interstitial=[])])
        for i, e in enumerate(ordered)
    ]
    return BuildOutput(chains=chains, notes={"builder": "per_request"})


def _is_prefix(a: list[int], b: list[int]) -> bool:
    return len(a) <= len(b) and b[: len(a)] == a


@dataclass(eq=False)  # identity-based, so nodes are hashable for set membership
class _Node:
    entry: TokenEntry
    cumulative: list[int]  # prompt + generation for this call
    parent: Optional["_Node"] = None
    children: list["_Node"] = field(default_factory=list)
    quarantined: bool = False


def prefix_merging(entries: list[TokenEntry]) -> BuildOutput:
    # Increasing prompt length is an order derived from the tokens: a parent's
    # cumulative sequence is a prefix of its child's prompt, so the parent's
    # prompt is shorter. This makes the pass order-independent.
    ordered = sorted(entries, key=lambda e: (len(e.prompt_token_ids), e.model_call_id))
    nodes: list[_Node] = []
    roots: list[_Node] = []
    quarantined: list[str] = []

    for entry in ordered:
        prompt = list(entry.prompt_token_ids)
        candidates = [n for n in nodes if _is_prefix(n.cumulative, prompt)]
        node = _Node(entry=entry, cumulative=prompt + list(entry.generation_token_ids))
        if candidates:
            best_len = max(len(n.cumulative) for n in candidates)
            best = [n for n in candidates if len(n.cumulative) == best_len]
            node.parent = best[0]
            if len(best) > 1:
                # Two or more parents with identical sequences: ambiguous, quarantine.
                node.quarantined = True
                quarantined.append(entry.model_call_id)
            best[0].children.append(node)
        else:
            roots.append(node)
        nodes.append(node)

    # Resolve retry siblings. A harness (Claude Code) retries on timeout / 5xx / dropped SSE, and the
    # capture point records a call even if the client never received it, so a retry yields two nodes
    # with identical prompt ids under the same parent and divergent generations. Rule: the sibling a
    # later call extended (has children) wins; unextended siblings are quarantined. When none was
    # extended (a retry of the final call), keep one deterministically so the main chain is stable.
    siblings_by_parent: dict[int, list[_Node]] = {}
    for node in nodes:
        siblings_by_parent.setdefault(id(node.parent), []).append(node)
    for group in siblings_by_parent.values():
        by_prompt: dict[tuple, list[_Node]] = {}
        for node in group:
            by_prompt.setdefault(tuple(node.entry.prompt_token_ids), []).append(node)
        for retry_group in by_prompt.values():
            if len(retry_group) < 2:
                continue
            extended = [n for n in retry_group if n.children]
            keep = set(extended) if extended else {min(retry_group, key=lambda n: n.entry.model_call_id)}
            for node in retry_group:
                if node not in keep and not node.quarantined:
                    node.quarantined = True
                    quarantined.append(node.entry.model_call_id)

    chains: list[Chain] = []

    def walk(node: _Node, path: list[_Node]) -> None:
        path = path + [node]
        if not node.children:
            if any(p.quarantined for p in path):
                return
            root = path[0]
            chain = Chain(chain_id="", root_prompt=list(root.entry.prompt_token_ids))
            prev_cumulative = list(root.entry.prompt_token_ids)
            for step, p in enumerate(path):
                interstitial = [] if step == 0 else list(p.entry.prompt_token_ids[len(prev_cumulative) :])
                chain.links.append(ChainLink(entry=p.entry, interstitial=interstitial))
                prev_cumulative = list(p.entry.prompt_token_ids) + list(p.entry.generation_token_ids)
            chains.append(chain)
            return
        for child in node.children:
            walk(child, path)

    for root in roots:
        walk(root, [])

    # The main chain is the longest root-to-leaf path from the first root; the rest are branches.
    def chain_length(c: Chain) -> int:
        return len(c.root_prompt) + sum(len(link.entry.generation_token_ids) for link in c.links)

    if chains:
        first_root_prompt = list(roots[0].entry.prompt_token_ids) if roots else []
        mains = [c for c in chains if c.root_prompt == first_root_prompt] or chains
        main = max(mains, key=chain_length)
        main.chain_id = "main"
        branch = 0
        for c in chains:
            if c is not main:
                c.chain_id = f"branch-{branch}"
                branch += 1

    return BuildOutput(
        chains=chains,
        quarantined=quarantined,
        notes={"builder": "prefix_merging", "roots": len(roots)},
    )


_BUILDERS: dict[str, Callable[[list[TokenEntry]], BuildOutput]] = {
    "per_request": per_request,
    "prefix_merging": prefix_merging,
}


def build_trajectories(
    rollout_id: str,
    entries: list[TokenEntry],
    builder: str = "prefix_merging",
    reward: float = 0.0,
    reward_components: Optional[dict[str, float]] = None,
) -> list[Trajectory]:
    """Build the trainable trajectories for one rollout from its token records."""
    if builder not in _BUILDERS:
        raise ValueError(f"unknown builder {builder!r}; known: {sorted(_BUILDERS)}")
    if not entries:
        return []
    out = _BUILDERS[builder](entries)
    total_calls = len(entries)
    quarantined_fraction = (len(out.quarantined) / total_calls) if total_calls else 0.0
    trajectories: list[Trajectory] = []
    for chain in out.chains:
        ids, mask, lps, spans = chain.flatten()
        trained = sum(mask)
        trajectories.append(
            Trajectory(
                rollout_id=rollout_id,
                chain_id=chain.chain_id,
                token_ids=ids,
                loss_mask=mask,
                log_probs=lps,
                spans=spans,
                reward=reward,
                reward_components=reward_components,
                provenance={
                    "builder": out.notes.get("builder", builder),
                    "n_calls": len(chain.links),
                    # Metrics that make silent training loss visible: how much of the rollout was
                    # dropped, and how much of this chain is actually trained on. A reasoning model in
                    # a multi-call rollout can quietly collapse to turn-1-only; these surface it.
                    "quarantined_calls": len(out.quarantined),
                    "quarantined_fraction": round(quarantined_fraction, 4),
                    "trained_token_fraction": round(trained / len(ids), 4) if ids else 0.0,
                    "notes": out.notes,
                },
            )
        )
    return trajectories


# --- Projection to the shape NeMo-RL consumes ---


def project_chain_to_output_items(chain: Chain) -> list[dict]:
    """Project the chain into content-bearing Responses output items whose prompts are
    contiguous. For each call, emit its captured output items (assistant text, tool
    calls preserved) and set the contiguous prompt on the item that carries the
    generation, so each generated item's prompt extends the previous one — the shape
    NeMo-RL ingests, with the text its penalties read (section 7.2). Falls back to a
    synthesized token-only item only when a call captured no content items."""
    items: list[dict] = []
    cumulative = list(chain.root_prompt)
    for step, link in enumerate(chain.links):
        cumulative = cumulative + (link.interstitial if step > 0 else [])
        entry = link.entry
        content_items = [dict(item) for item in (entry.output_items or [])]
        generated = [item for item in content_items if item.get("generation_token_ids") is not None]
        if not generated and content_items:
            # No item carried token fields (unexpected); attach to the last so tokens are not lost.
            generated = content_items[-1:]
        if content_items:
            for item in generated:
                item["prompt_token_ids"] = list(cumulative)
                item["generation_token_ids"] = list(entry.generation_token_ids)
                item["generation_log_probs"] = list(entry.generation_log_probs)
                if entry.routed_experts is not None:
                    item["routed_experts"] = entry.routed_experts
            items.extend(content_items)
        else:
            item = {
                "type": "message",
                "prompt_token_ids": list(cumulative),
                "generation_token_ids": list(entry.generation_token_ids),
                "generation_log_probs": list(entry.generation_log_probs),
            }
            if entry.routed_experts is not None:
                item["routed_experts"] = entry.routed_experts
            items.append(item)
        cumulative = cumulative + list(entry.generation_token_ids)
    return items


def project_main_chain_response(rollout_id: str, out: BuildOutput, model: str = "") -> dict:
    """Project the main chain into a Responses-shaped object with contiguous output items."""
    mains = [c for c in out.chains if c.chain_id == "main"] or out.chains[:1]
    output = project_chain_to_output_items(mains[0]) if mains else []
    # Token fields ride only on generated items; a content-only leading item (e.g. assistant
    # text emitted before a tool call) carries none. Read the usage counts from the items that
    # actually have token fields so a leading content item does not KeyError or skew the totals.
    generated = [item for item in output if item.get("generation_token_ids") is not None]
    n_in = len(generated[0]["prompt_token_ids"]) if generated else 0
    n_out = sum(len(item["generation_token_ids"]) for item in generated)
    return {
        "id": f"proj-{rollout_id}",
        "model": model,
        "object": "response",
        "output": output,
        "usage": {"input_tokens": n_in, "output_tokens": n_out},
    }


def assert_nemo_rl_contiguity(response: dict) -> None:
    """Enforce the invariant NeMo-RL's ingestion relies on: each output item's
    prompt_token_ids must extend the tokens seen so far (prompt plus generation
    of all prior items). Raises AssertionError otherwise."""
    seen: list[int] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("generation_token_ids") is None:
            continue
        prompt = item.get("prompt_token_ids") or []
        if prompt[: len(seen)] != seen:
            raise AssertionError(
                "projection violates NeMo-RL prefix contiguity: an output item's prompt_token_ids "
                "does not extend the tokens seen so far"
            )
        seen = list(prompt) + list(item["generation_token_ids"])
