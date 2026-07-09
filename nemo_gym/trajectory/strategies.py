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

"""The built-in trajectory builders.

Two builders turn a rollout's model calls into token chains (both registered in
the plugin registry in ``registry.py``; users add their own the same way):

  per_request     — assumes nothing about how the calls relate; every call is
                    its own sample. Correct for rewritten contexts, sliding
                    windows, anything. The simple, universal fallback.
  prefix_merging  — parents each call to the earlier call whose full token
                    sequence is the longest prefix of this call's prompt. This
                    rebuilds multi-turn chains and sub-agent branches. A prompt
                    that no longer extends any earlier call starts a new root
                    (a context that was compacted or rewritten). Two candidate
                    parents with identical sequences are ambiguous, so that
                    subtree is quarantined and excluded rather than guessed.

Loss masks are based on where each token came from: tokens the policy sampled
are marked 1, everything re-fed into the prompt (history, tool output, tokens
added between calls) is marked 0. Logprobs are attached only at the marked-1
positions. A unit test checks that both builders produce the same set of
marked-1 tokens on an append-only rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from nemo_gym.observability.records import TokenEntry
from nemo_gym.trajectory.registry import register


@dataclass
class ChainLink:
    entry: TokenEntry
    interstitial: list[int]  # prompt tokens added since parent (tools/user), mask 0


@dataclass
class Chain:
    chain_id: str
    links: list[ChainLink] = field(default_factory=list)
    root_prompt: list[int] = field(default_factory=list)

    def flatten(self) -> tuple[list[int], list[int], list[Optional[float]], list[int]]:
        ids: list[int] = list(self.root_prompt)
        mask: list[int] = [0] * len(ids)
        lps: list[Optional[float]] = [None] * len(ids)
        boundaries: list[int] = []
        for link in self.links:
            ids += link.interstitial
            mask += [0] * len(link.interstitial)
            lps += [None] * len(link.interstitial)
            boundaries.append(len(ids))
            gen = link.entry.generation_token_ids
            glp = link.entry.generation_log_probs
            if len(glp) != len(gen):
                raise ValueError(f"logprob/token length mismatch on {link.entry.request_id}: {len(glp)} vs {len(gen)}")
            ids += gen
            mask += [1] * len(gen)
            lps += list(glp)
        return ids, mask, lps, boundaries


@dataclass
class BuildOutput:
    chains: list[Chain]
    quarantined: list[str] = field(default_factory=list)  # request_ids
    notes: dict = field(default_factory=dict)


@register("per_request")
def per_request(entries: list[TokenEntry]) -> BuildOutput:
    chains = []
    for i, e in enumerate(sorted(entries, key=lambda x: x.seq)):
        chains.append(
            Chain(
                chain_id=f"req-{i}", root_prompt=list(e.prompt_token_ids), links=[ChainLink(entry=e, interstitial=[])]
            )
        )
    return BuildOutput(chains=chains, notes={"builder": "per_request"})


def _is_prefix(a: list[int], b: list[int]) -> bool:
    return len(a) <= len(b) and b[: len(a)] == a


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _Node:
    entry: TokenEntry
    cumulative: list[int]  # root prompt + all tokens through this gen
    children: list["_Node"] = field(default_factory=list)
    parent: Optional["_Node"] = None
    quarantined: bool = False


@register("prefix_merging")
def prefix_merging(entries: list[TokenEntry], compaction_min_shared_frac: float = 0.5) -> BuildOutput:
    ordered = sorted(entries, key=lambda x: x.seq)
    roots: list[_Node] = []
    nodes: list[_Node] = []
    quarantined: list[str] = []
    n_compaction_roots = 0

    for e in ordered:
        prompt = list(e.prompt_token_ids)
        # Parent = node whose cumulative sequence is the LONGEST full prefix
        # of this prompt. Note: exact-match at PoC scale; production uses
        # rolling-hash prefix fingerprints for O(total tokens).
        candidates = [n for n in nodes if _is_prefix(n.cumulative, prompt)]
        if candidates:
            best_len = max(len(n.cumulative) for n in candidates)
            best = [n for n in candidates if len(n.cumulative) == best_len]
            parent = best[0]
            node = _Node(entry=e, cumulative=prompt + list(e.generation_token_ids), parent=parent)
            if len(best) > 1:
                # Ambiguous: >=2 candidate parents with identical cumulative
                # sequences (concurrent identical prefixes). Quarantine.
                node.quarantined = True
                quarantined.append(e.request_id)
            parent.children.append(node)
            nodes.append(node)
            continue
        # No full-prefix parent: either the first call, or a compaction /
        # rewritten-context split -> new root.
        if nodes:
            best_shared = max(_common_prefix_len(n.cumulative, prompt) for n in nodes)
            if prompt and best_shared / max(len(prompt), 1) >= compaction_min_shared_frac:
                n_compaction_roots += 1  # partial overlap: compaction signature
        node = _Node(entry=e, cumulative=prompt + list(e.generation_token_ids))
        roots.append(node)
        nodes.append(node)

    # Emit one chain per root-to-leaf path (a FOREST, not one line).
    chains: list[Chain] = []

    def walk(node: _Node, path: list[_Node]):
        path = path + [node]
        if not node.children:
            if any(p.quarantined for p in path):
                return
            root = path[0]
            chain = Chain(chain_id="", root_prompt=list(root.entry.prompt_token_ids))
            prev_cum = list(root.entry.prompt_token_ids)
            for p in path:
                inter = list(p.entry.prompt_token_ids[len(prev_cum) :]) if p is not path[0] else []
                chain.links.append(ChainLink(entry=p.entry, interstitial=inter))
                prev_cum = list(p.entry.prompt_token_ids) + list(p.entry.generation_token_ids)
            chains.append(chain)
            return
        for c in node.children:
            walk(c, path)

    for r in roots:
        walk(r, [])

    # Main chain = the longest chain starting from the first root; everything
    # else is a branch.
    def chain_len(c: Chain) -> int:
        return sum(len(link.entry.generation_token_ids) for link in c.links) + len(c.root_prompt)

    if chains:
        first_root_prompt = roots[0].entry.prompt_token_ids if roots else []
        mains = [c for c in chains if c.root_prompt == list(first_root_prompt)]
        main = max(mains or chains, key=chain_len)
        main.chain_id = "main"
        b = 0
        for c in chains:
            if c is not main:
                c.chain_id = f"branch-{b}"
                b += 1

    return BuildOutput(
        chains=chains,
        quarantined=quarantined,
        notes={"builder": "prefix_merging", "roots": len(roots), "compaction_roots": n_compaction_roots},
    )
