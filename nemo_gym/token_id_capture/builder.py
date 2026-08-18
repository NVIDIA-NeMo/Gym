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

"""Build trainable trajectories from a frozen token capture.

The builder consumes ``TokenEntry`` records from a ``TokenCaptureSnapshot``.
The snapshot is frozen before the consumer passes its entries to the builder.

``per_request`` creates one training sequence per call.
It does not infer relationships between calls.
It can return multiple trajectories.

``prefix_merging`` chains calls by token-prefix relationships.
Each call uses the earlier call with the longest matching cumulative sequence as its parent.
The cumulative sequence contains the prompt and generation.
This strategy rebuilds an append-only rollout as one chain.
A rewritten or compacted prompt starts a new root.
Identical candidate parents are ambiguous.
The builder quarantines the ambiguous subtree.

Both strategies are independent of capture order.
``prefix_merging`` processes entries by increasing prompt length.
This order comes from the tokens.
A parent's prompt is shorter than its child's prompt.

Loss masks follow token provenance.
Policy-generated tokens have a mask value of 1 and retain their captured log probabilities.
Prompt tokens have a mask value of 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from nemo_gym.token_id_capture.records import TokenEntry, compute_digest


@dataclass
class ChainLink:
    entry: TokenEntry
    interstitial: list[int]  # Prompt tokens added after the parent have a mask value of 0.


@dataclass
class Chain:
    chain_id: str
    links: list[ChainLink] = field(default_factory=list)
    root_prompt: list[int] = field(default_factory=list)

    def validate(self) -> None:
        """Require one log probability for each generated token.

        A trainer cannot use a chain with mismatched token and log-probability counts.
        """
        for link in self.links:
            generated = link.entry.generation_token_ids
            log_probs = link.entry.generation_log_probs
            if len(log_probs) != len(generated):
                raise ValueError(
                    f"log-prob/token length mismatch on {link.entry.model_call_id}: "
                    f"{len(log_probs)} vs {len(generated)}"
                )


@dataclass
class BuildNotes:
    """Describe what the build kept, dropped, or could not resolve.

    The consumer converts these fields into run metrics.
    Typed fields prevent renamed keys from appearing as zero values on dashboards.
    """

    builder: str
    roots: int = 0
    chains: int = 0
    generated_tokens_captured: int = 0
    generated_tokens_delivered: int = 0
    # Only one chain is delivered per rollout.
    # Sub-agent branches and post-compaction chains are dropped.
    # The delivered fraction exposes this limitation.
    delivered_fraction: float = 0.0
    # These calls have a retry sibling that the harness may not have kept.
    # A final-call retry is unresolved because no later call identifies the survivor.
    unresolved_retries: list[str] = field(default_factory=list)
    # Calls without generated tokens are excluded from the chain.
    empty_generation_calls: list[str] = field(default_factory=list)
    # Count why recorded parent links were not used.
    parent_link_failures: dict[str, int] = field(default_factory=dict)


@dataclass
class BuildOutput:
    chains: list[Chain]
    quarantined: list[str] = field(default_factory=list)  # Model call IDs.
    notes: BuildNotes = field(default_factory=lambda: BuildNotes(builder=""))


def per_request(entries: list[TokenEntry]) -> BuildOutput:
    ordered = sorted(entries, key=lambda e: (len(e.prompt_token_ids), e.model_call_id))
    chains = [
        Chain(chain_id=f"req-{i}", root_prompt=list(e.prompt_token_ids), links=[ChainLink(entry=e, interstitial=[])])
        for i, e in enumerate(ordered)
    ]
    return BuildOutput(chains=chains, notes=BuildNotes(builder="per_request", chains=len(chains)))


@dataclass(eq=False)  # Identity-based equality keeps nodes hashable.
class _Node:
    entry: TokenEntry
    cumulative: list[int]  # Prompt and generation tokens for this call.
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    quarantined: bool = False


@dataclass
class _PrefixTrieNode:
    children: dict[int, "_PrefixTrieNode"] = field(default_factory=dict)
    candidates: list[_Node] = field(default_factory=list)


class _PrefixIndex:
    """Index cumulative token sequences for linear-time parent lookup."""

    def __init__(self) -> None:
        self._root = _PrefixTrieNode()

    def add(self, candidate: _Node) -> None:
        current = self._root
        for token_id in candidate.cumulative:
            current = current.children.setdefault(token_id, _PrefixTrieNode())
        current.candidates.append(candidate)

    def infer_parent(self, prompt: list[int]) -> tuple[_Node | None, bool]:
        best: list[_Node] = []
        current = self._root
        for token_id in prompt:
            next_node = current.children.get(token_id)
            if next_node is None:
                break
            current = next_node
            if current.candidates:
                best = current.candidates
        return (best[0], len(best) > 1) if best else (None, False)


def _resolve_parent(
    node: "_Node",
    by_call_id: dict[str, "_Node"],
    prefix_index: _PrefixIndex,
) -> tuple["_Node | None", bool, str | None]:
    """Find this call's parent.

    Prefer a verified recorded link.
    A missing parent target falls back to strict longest token-prefix matching.
    A digest mismatch returns no parent and quarantines the call.
    ``note`` reports why the recorded link was not used.
    """
    prompt = list(node.entry.prompt_token_ids)
    claimed = node.entry.parent_call_id
    if claimed is not None:
        parent = by_call_id.get(claimed)
        if parent is None:
            inferred, ambiguous = _infer_parent(prompt, candidates)
            return inferred, ambiguous, "parent_call_id_missing"
        cum_len = parent.entry.cum_len
        if cum_len is None:
            cum_len = len(parent.cumulative)
        if cum_len <= len(prompt) and compute_digest(prompt[:cum_len]) == (
            parent.entry.digest or compute_digest(parent.cumulative)
        ):
            return parent, False, None
        return None, False, "parent_digest_mismatch"
    return _infer_parent(prompt, prefix_index) + (None,)


def _infer_parent(prompt: list[int], index: _PrefixIndex) -> tuple["_Node | None", bool]:
    """Infer the parent from the longest cumulative prefix.

    This is the fallback when no verified parent link exists.
    Identical cumulative sequences are ambiguous.
    The caller quarantines an ambiguous subtree.
    """
    return index.infer_parent(prompt)


def prefix_merging(entries: list[TokenEntry]) -> BuildOutput:
    # A call without generated tokens has no training signal.
    # Its cumulative sequence equals its prompt.
    # Keeping it would make it the parent of another call with the same prompt.
    # A filtered call and its retry would then look like a two-turn chain.
    # Retry resolution compares only siblings and would miss that pair.
    empty_generation = [e.model_call_id for e in entries if not e.generation_token_ids]
    entries = [e for e in entries if e.generation_token_ids]
    if not entries:
        return BuildOutput(
            chains=[], notes=BuildNotes(builder="prefix_merging", empty_generation_calls=empty_generation)
        )

    # Increasing prompt length defines an order from the tokens.
    # A parent's cumulative sequence is a prefix of its child's prompt.
    # The parent's prompt is therefore shorter.
    # This makes the pass independent of capture order.
    ordered = sorted(entries, key=lambda e: (len(e.prompt_token_ids), e.model_call_id))
    nodes: list[_Node] = []
    roots: list[_Node] = []
    quarantined: list[str] = []
    prefix_index = _PrefixIndex()

    nodes_by_call_id: dict[str, _Node] = {}
    parent_link_failures: dict[str, int] = {}

    for entry in ordered:
        prompt = list(entry.prompt_token_ids)
        node = _Node(entry=entry, cumulative=prompt + list(entry.generation_token_ids))
        parent, ambiguous, note = _resolve_parent(node, nodes_by_call_id, prefix_index)
        if note:
            parent_link_failures[note] = parent_link_failures.get(note, 0) + 1
            if note == "parent_digest_mismatch":
                node.quarantined = True
                quarantined.append(entry.model_call_id)
        if parent is not None:
            node.parent = parent
            if ambiguous:
                # Quarantine calls with identical candidate parents.
                node.quarantined = True
                quarantined.append(entry.model_call_id)
            parent.children.append(node)
        else:
            roots.append(node)
        nodes.append(node)
        prefix_index.add(node)
        nodes_by_call_id[entry.model_call_id] = node

    # Resolve retry siblings.
    # A harness can retry after a timeout, server error, or dropped stream.
    # The capture point can record a call that the client never received.
    # A retry then creates siblings with identical prompts and different generations.
    #
    # A recorded parent link identifies the sibling retained by the harness.
    # Without a link, retain the sibling extended by a later call.
    # Neither method can resolve a retry of the final call.
    # Mark that retry as unresolved.
    # The consumer masks the rollout instead of training on an unconfirmed generation.
    unresolved_retries: list[str] = []
    # Group calls by parent identity.
    # All roots share the explicit ROOTS key.
    # A retry of the first call creates roots with the same prompt.
    # Treat those roots as siblings.
    ROOTS = "roots"
    siblings_by_parent: dict[object, list[_Node]] = {}
    for node in nodes:
        siblings_by_parent.setdefault(ROOTS if node.parent is None else id(node.parent), []).append(node)
    for group in siblings_by_parent.values():
        by_prompt: dict[tuple, list[_Node]] = {}
        for node in group:
            by_prompt.setdefault(tuple(node.entry.prompt_token_ids), []).append(node)
        for retry_group in by_prompt.values():
            if len(retry_group) < 2:
                continue
            extended = [n for n in retry_group if any(not child.quarantined for child in n.children)]
            if extended:
                keep = set(extended)
            else:
                keep = {min(retry_group, key=lambda n: n.entry.model_call_id)}
                unresolved_retries.extend(n.entry.model_call_id for n in retry_group)
            for node in retry_group:
                if node not in keep and not node.quarantined:
                    node.quarantined = True
                    quarantined.append(node.entry.model_call_id)

    chains: list[Chain] = []

    # Materialize root-to-leaf chains without recursion.
    # Agent rollouts can exceed Python's recursion limit.
    for leaf in (node for node in nodes if not node.children):
        reverse_path: list[_Node] = []
        current: _Node | None = leaf
        while current is not None:
            reverse_path.append(current)
            current = current.parent
        path = list(reversed(reverse_path))
        if any(node.quarantined for node in path):
            continue
        root = path[0]
        chain = Chain(chain_id="", root_prompt=list(root.entry.prompt_token_ids))
        prev_cumulative = list(root.entry.prompt_token_ids)
        for step, node in enumerate(path):
            interstitial = [] if step == 0 else list(node.entry.prompt_token_ids[len(prev_cumulative) :])
            chain.links.append(ChainLink(entry=node.entry, interstitial=interstitial))
            prev_cumulative = list(node.entry.prompt_token_ids) + list(node.entry.generation_token_ids)
        chains.append(chain)

    # Deliver only one chain per rollout.
    # Choose the chain whose root call completed first.
    #
    # Completion order matches dispatch order for a sequential harness.
    # The model server awaits token capture before returning the response.
    # The next call therefore starts after the previous record is durable.
    # The capture point sets ``created_at``.
    # Use this timestamp instead of file order.
    # Multiple server workers append in lock order rather than completion order.
    #
    # Two unsupported shapes involve a second agent.
    #
    # A short auxiliary call can finish before the agent's first task call.
    # Selection then treats the auxiliary call as the main root.
    #
    # Parallel sub-agents make completion order differ from dispatch order.
    # Records do not identify the calling agent.
    # Ordering cannot recover that identity.
    # Some harnesses can route sub-agent calls to a different model server.
    #
    # These shapes require future work.
    # A split rollout reports multiple chains and a delivered fraction below 1.
    def selection_key(c: Chain) -> tuple:
        # Sort by the earliest root.
        # Break timestamp ties by call ID.
        # Sort an empty chain last.
        if not c.links:
            return (float("inf"), "")
        root = c.links[0].entry
        return (root.created_at, root.model_call_id)

    if chains:
        main = min(chains, key=selection_key)
        main.chain_id = "main"
        branch = 0
        for c in chains:
            if c is not main:
                c.chain_id = f"branch-{branch}"
                branch += 1

    delivered = sum(len(link.entry.generation_token_ids) for link in main.links) if chains else 0
    captured = sum(len(e.generation_token_ids) for e in entries)
    notes = BuildNotes(
        builder="prefix_merging",
        roots=len(roots),
        chains=len(chains),
        generated_tokens_captured=captured,
        generated_tokens_delivered=delivered,
        delivered_fraction=round(delivered / captured, 4) if captured else 0.0,
        unresolved_retries=unresolved_retries,
        empty_generation_calls=empty_generation,
        parent_link_failures=parent_link_failures,
    )
    return BuildOutput(chains=chains, quarantined=quarantined, notes=notes)


_BUILDERS: dict[str, Callable[[list[TokenEntry]], BuildOutput]] = {
    "per_request": per_request,
    "prefix_merging": prefix_merging,
}


def run_builder(entries: list[TokenEntry], builder: str = "prefix_merging") -> BuildOutput:
    """Chain frozen snapshot entries with the named strategy."""
    if builder not in _BUILDERS:
        raise ValueError(f"unknown builder {builder!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[builder](entries)


# --- Projection to a contiguous, token-bearing response ---


def project_chain_to_output_items(chain: Chain) -> list[dict]:
    """Project a chain into Responses output items with contiguous prompts.

    Preserve captured assistant text and tool calls.
    Put the contiguous prompt on the item that carries each generation.
    Each generated item's prompt extends the previous generated item.
    Preserve text for downstream scoring.
    Create a token-only item only when a call has no captured content items.
    """
    items: list[dict] = []
    cumulative = list(chain.root_prompt)
    for step, link in enumerate(chain.links):
        cumulative = cumulative + (link.interstitial if step > 0 else [])
        entry = link.entry
        content_items = [dict(item) for item in (entry.output_items or [])]
        index = entry.token_item_index
        if index is not None and 0 <= index < len(content_items):
            # Capture records the item that originally carried the token arrays.
            generated = [content_items[index]]
        else:
            # Older records keep token arrays inline.
            generated = [item for item in content_items if item.get("generation_token_ids") is not None]
        if not generated and content_items:
            # Attach tokens to the last item when no item carries token fields.
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
    """Rebuild the main chain as a Responses object whose output items are contiguous.

    The result is a Gym-native Responses payload.
    It contains ``object: "response"``, ``output`` items, and ``usage``.
    Token fields describe one unbroken sequence across the rollout.
    The sequence combines items from multiple model calls.
    """
    if not out.chains:
        raise ValueError("capture produced no safe trainable chain")
    if out.notes.builder == "per_request" and len(out.chains) != 1:
        raise ValueError("per_request produced multiple trajectories for a single-response delivery")
    mains = [c for c in out.chains if c.chain_id == "main"] or out.chains[:1]
    output = project_chain_to_output_items(mains[0])
    if not any(item.get("generation_token_ids") for item in output):
        raise ValueError("capture produced no trainable generated tokens")
    # Only generated items carry token fields.
    # A leading content-only item carries no token fields.
    # Read usage counts from token-bearing items.
    # This avoids missing keys and incorrect totals.
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


def assert_prefix_contiguity(response: dict) -> None:
    """Require each generated item to extend all preceding tokens.

    The preceding tokens include the prompt and all prior generations.
    Raise ``AssertionError`` when the response is not contiguous.
    """
    seen: list[int] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("generation_token_ids") is None:
            continue
        prompt = item.get("prompt_token_ids") or []
        if prompt[: len(seen)] != seen:
            raise AssertionError(
                "projection is not prefix-contiguous: an output item's prompt_token_ids "
                "does not extend the tokens seen so far"
            )
        seen = list(prompt) + list(item["generation_token_ids"])
