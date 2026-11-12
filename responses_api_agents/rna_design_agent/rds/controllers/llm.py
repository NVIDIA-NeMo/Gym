# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The agent arm: a model picks the next tool from the campaign state."""

from __future__ import annotations

from .. import registry
from ..llm import Provider, Reply
from ..runner import Action
from ..state import StateView


SYSTEM_PROMPT = """You are running an RNA design campaign.

Each round you are shown the campaign state and must choose exactly one tool
call. Call the `stop` tool when further spending will not improve the result.

The state's `objective` block is the thing you are being judged on: it names
the metric candidates are ranked by and the gates they must pass. Generating is
only half the job — a candidate with no score counts for nothing, so whenever
`objective.unscored_candidates` is large, score before you generate again.

What to keep in mind:
- The budget is hard. When any dimension is spent the campaign ends, whatever
  state it is in, so spend expensive tools on candidates worth the cost.
- Leave enough budget to score. Ending with many candidates and no scores is
  the worst outcome available to you; it reports as zero.
- To score candidates that already exist, pass one of the strings in
  `sequence_selectors` as a tool's `sequences` argument — "@unscored" and
  "@top:50" are expanded for you. Never invent a filename, and never retype a
  sequence you were shown truncated.
- Omit an optional argument you have no opinion about rather than passing an
  empty string for it. An empty value is not the same as a default.
- Tool descriptions say what a score does and does not mean. Read them before
  treating a number as evidence.
- A tool that failed is information. Do not repeat a call that just failed with
  the same arguments."""

#: The prompt when the campaign withholds its scoring, so it never mentions a ranked metric.
BENCHMARK_PROMPT = """You are designing RNA.

Each round you are shown the campaign state and must choose exactly one tool
call. The state's `objective` block states the design task and the hard
constraints the molecule must satisfy. You are not shown how your work will be
scored: design for the structure you were given, not for any number you happen
to measure.

How the loop works:
- One tool call per round. Its result comes back in the next round's state,
  along with what you have spent and what is left.
- `budget.limits` is what you may spend and `budget.spend` is what you have.
  Unused budget is not saved — a run that stops at a third of its rounds has
  thrown the rest away. Candidates are a budget line too: the limit is how many
  designs you may hold, and it is usually far larger than one generator call
  returns.
- To act on designs you already have, pass a selector string — the ones in
  `sequence_selectors`, such as "@all" or "@top:50" — as a tool's `sequences`
  argument. Never invent a filename, and never retype a sequence you were shown
  truncated.
- Sequences you write yourself count as designs: pass them to a scorer and they
  enter the population like anything a generator made. They must be exactly the
  target's length and use only A, C, G and U. There is no separate tool for
  handing over a sequence — scoring it is how it arrives.
- Omit an optional argument you have no opinion about rather than passing an
  empty string for it.

What to keep in mind:
- You are judged on the designs you produce. Producing a sequence that folds
  into the target is the whole task.
- If the tools do not offer the method you want and `code.run` is available,
  write it. Saving a script with `save_as` keeps it for the targets that
  follow, and `code.list` shows what you have kept.
- Measuring your own candidates is allowed and sensible — folding engines and
  structure predictors are on your tool list — but the numbers you get are
  yours, not the ones you will be judged by.
- A tool that failed is information. Read the error: it names the argument or
  the value it refused. Do not repeat a call that just failed with the same
  arguments.
- Say why before you call. One sentence, in the message beside the call: what
  you believe about this target and what this call is meant to settle. It is
  recorded with the call, and it is read — a design nobody can explain is worth
  less than one that can be."""

STOP_TOOL = {
    "name": Action.STOP,
    "description": "End the campaign. Use when more spending will not help.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    },
}


class LLMController:
    """Model-agnostic by construction: it talks only to a `Provider`."""

    def __init__(
        self,
        provider: Provider,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 4096,
        allowed: list[str] | None = None,
    ) -> None:
        self.provider = provider
        #: Tool names this campaign may use. None means the whole registry.
        self.allowed = None if allowed is None else set(allowed)
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.usage_total = None
        self.last_reply: Reply | None = None
        #: Model calls made this round (charged), and extra tool calls not run (recorded). Read by the runner.
        self.calls_this_round = 0
        self.dropped_this_round = 0

    @property
    def name(self) -> str:
        return f"llm:{getattr(self.provider, 'model', self.provider.name)}"

    def tools(self) -> list[dict]:
        # Registry order with stop last: a stable tool list keeps the prompt cache valid.
        offered = [
            t for t in registry.llm_tools() if self.allowed is None or registry.get(t["name"]).name in self.allowed
        ]
        return [*offered, STOP_TOOL]

    def next_action(self, view: StateView) -> Action:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": view.to_prompt()},
        ]
        self.calls_this_round = 0
        self.dropped_this_round = 0

        for attempt in range(2):
            reply = self.provider.chat(messages, tools=self.tools(), max_tokens=self.max_tokens)
            self.calls_this_round += 1
            self.last_reply = reply
            self.usage_total = reply.usage if self.usage_total is None else self.usage_total + reply.usage
            if reply.tool_calls:
                break
            if attempt == 0:
                # A prose reply is not a decision to stop. Ask once more and charge the extra call.
                messages = [
                    *messages,
                    {"role": "assistant", "content": reply.text or "(no answer)"},
                    {
                        "role": "user",
                        "content": "That round needs a tool call. Choose one tool and "
                        "call it, or call stop if nothing further will help.",
                    },
                ]
        else:
            return Action(Action.STOP, {}, rationale=(reply.text[:500] or "no tool call, twice"))

        # One call per round: the first is run and the rest are counted.
        self.dropped_this_round = max(0, len(reply.tool_calls) - 1)
        call = reply.tool_calls[0]
        return Action(call.name, call.arguments, rationale=(reply.reasoning or reply.text)[:500], error=call.error)
