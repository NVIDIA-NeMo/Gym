# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPVal context compaction for the OpenHands runner (uploaded into the sandbox).

Stirrup semantics for OpenHands: when the last agent reply reports ``prompt + completion``
tokens at or above ``cutoff * max_context``, the whole history after the task prompt is
summarized by one policy call and the agent continues from ``system prompt + task + summary``
(nothing else is kept, so no stale assistant tokens are ever re-sent into the new context).

The summarization call is a policy call like any other: it goes through the runner's
``DirectModelClient`` (same rollout-prefixed URL, so Gym token-capture / lineage correlation
sees it; same retries, timing log and ``/trace/completions`` completion log with the echoed
token fields). Its log entry carries ``gdpval_call_kind = "compaction_summary"`` so the agent
server (``compaction.py``) can split the rollout into context segments for NeMo RL's
multi-trace training.

Everything that needs OpenHands is created inside ``install``; the module-level helpers are
pure so the agent's offline tests can exercise the trigger and prompt logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

CALL_KIND_KEY = "gdpval_call_kind"
SUMMARY_CALL_KIND = "compaction_summary"

# Verbatim copy of the OpenHands 0.62 LLMSummarizingCondenser prompt (the GDPVal condenser
# only changes when it fires and what it keeps, not what it asks the model).
SUMMARY_PROMPT = """You are maintaining a context-aware state summary for an interactive agent.
You will be given a list of events corresponding to actions taken by the agent, and the most recent previous summary if one exists.
If the events being summarized contain ANY task-tracking, you MUST include a TASK_TRACKING section to maintain continuity.
When referencing tasks make sure to preserve exact task IDs and statuses.

Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

TASK_TRACKING: {Active tasks, their IDs and statuses - PRESERVE TASK IDs}

COMPLETED: (Tasks completed so far, with brief results)
PENDING: (Tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant state)

For code-specific tasks, also include:
CODE_STATE: {File paths, function signatures, data structures}
TESTS: {Failing cases, error messages, outputs}
CHANGES: {Code edits, variable updates}
DEPS: {Dependencies, imports, external calls}
VERSION_CONTROL_STATUS: {Repository state, current branch, PR status, commit history}

PRIORITIZE:
1. Adapt tracking format to match the actual task type
2. Capture key user requirements and goals
3. Distinguish between completed and pending tasks
4. Keep all sections concise and relevant

SKIP: Tracking irrelevant details for the current task type

Example formats:

For code tasks:
USER_CONTEXT: Fix FITS card float representation issue
COMPLETED: Modified mod_float() in card.py, all tests passing
PENDING: Create PR, update documentation
CODE_STATE: mod_float() in card.py updated
TESTS: test_format() passed
CHANGES: str(val) replaces f"{val:.16G}"
DEPS: None modified
VERSION_CONTROL_STATUS: Branch: fix-float-precision, Latest commit: a1b2c3d

For other tasks:
USER_CONTEXT: Write 20 haikus based on coin flip results
COMPLETED: 15 haikus written for results [T,H,T,H,T,H,T,T,H,T,H,T,H,T,H]
PENDING: 5 more haikus needed
CURRENT_STATE: Last flip: Heads, Haiku count: 15/20"""


def truncate_middle(content: str, max_chars: int) -> str:
    """OpenHands' ``truncate_content``: keep both ends of an over-long event rendering."""
    if max_chars < 0 or len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n[... Observation truncated due to length ...]\n" + content[-half:]


def build_summary_prompt(previous_summary: str, forgotten: list[tuple[int, str]], *, max_event_chars: int) -> str:
    """The upstream condenser prompt: instructions, previous summary, the forgotten events."""
    prompt = SUMMARY_PROMPT + "\n\n"
    prompt += f"<PREVIOUS SUMMARY>\n{truncate_middle(previous_summary or '', max_event_chars)}\n</PREVIOUS SUMMARY>\n"
    prompt += "\n\n"
    for event_id, text in forgotten:
        prompt += f"<EVENT id={event_id}>\n{truncate_middle(text, max_event_chars)}\n</EVENT>\n"
    prompt += "Now summarize the events using the rules above."
    return prompt


def head_length(events: list[Any]) -> int:
    """Leading events kept verbatim: through the first user message (system prompt + task).

    OpenHands puts the ``SystemMessageAction`` first and the task ``MessageAction`` (source
    ``user``) right after it. Falls back to one event when no user message is present.
    """
    for index, event in enumerate(events):
        if type(event).__name__ == "MessageAction" and str(getattr(event, "source", "") or "") in ("user", "EventSource.USER"):
            return index + 1
    return 1


def should_compact(
    *, last_usage: Any, max_context: int, cutoff: float, usage_count: int, compacted_at: int
) -> bool:
    """Stirrup's trigger: the last agent reply's ``prompt + completion`` reached ``cutoff`` of the window.

    ``usage_count`` / ``compacted_at`` (number of agent calls now / at the last compaction)
    make sure a compaction is never re-triggered by the same reply: the next check needs a
    reply produced inside the new, summarized context.
    """
    if last_usage is None or usage_count <= compacted_at or max_context <= 0:
        return False
    total = int(getattr(last_usage, "prompt_tokens", 0) or 0) + int(getattr(last_usage, "completion_tokens", 0) or 0)
    return total / max_context >= cutoff


def install(
    *,
    cutoff: float,
    max_context: int,
    max_event_chars: int,
    log_path: str | None = None,
):
    """Patch OpenHands so ``LLMSummarizingCondenserConfig`` builds the GDPVal condenser.

    Put a plain ``LLMSummarizingCondenserConfig`` on ``AgentConfig.condenser`` (its
    ``keep_first`` / ``max_size`` are ignored) and call this after ``use_direct_model_client``
    is active: the summarization client subclasses the patched ``NemoGymClient`` (the runner's
    direct transport) and extends its completion log with the ``gdpval_call_kind`` marker.
    Returns the condenser class (for diagnostics).
    """
    from openhands.agenthub import nemo_gym_client
    from openhands.core.message import Message, TextContent
    from openhands.events.action.agent import CondensationAction
    from openhands.events.observation.agent import AgentCondensationObservation
    from openhands.llm.metrics import Metrics
    from openhands.memory.condenser.condenser import Condensation, RollingCondenser, View
    from openhands.memory.condenser.impl.llm_summarizing_condenser import LLMSummarizingCondenser

    if not 0 < cutoff <= 1:
        raise ValueError("context summarization cutoff must be in (0, 1]")

    def _note(record: dict[str, Any]) -> None:
        if not log_path:
            return
        try:
            with open(log_path, "a") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            pass

    class _CondenserLLM:
        """What ``NemoGymClient`` reads off an LLM: config, request kwargs, its own metrics."""

        def __init__(self, agent_llm):
            self.config = agent_llm.config
            self.metrics = Metrics(model_name=agent_llm.config.model)
            self._nemo_gym_llm_kwargs = dict(agent_llm._nemo_gym_llm_kwargs)

    class SummaryModelClient(nemo_gym_client.NemoGymClient):
        """The runner's direct policy client; its completion log names the call kind."""

        def _log_completion(self, messages, model_response_json, provider_specific_fields, params):
            log_file = os.path.join(
                self.llm.config.log_completions_folder,
                f"{self.llm.config.model.replace('/', '__')}-{time.time()}.json",
            )
            record = {
                "messages": [m.model_dump() for m in messages],
                "response": model_response_json,
                "provider_specific_fields": provider_specific_fields,
                "kwargs": {k: v for k, v in params.items() if k not in ("messages", "client")},
                "timestamp": time.time(),
                CALL_KIND_KEY: SUMMARY_CALL_KIND,
            }
            fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(log_file))
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(record))
            os.replace(temp_path, log_file)

    class GDPValSummarizingCondenser(RollingCondenser):
        def __init__(self, agent_llm):
            self._agent_llm = agent_llm
            self._client = SummaryModelClient(_CondenserLLM(agent_llm))
            self._compacted_at = 0
            self.compactions = 0
            super().__init__()

        # -- trigger -------------------------------------------------------------
        def should_condense(self, view: View) -> bool:
            events = list(view.events)
            if len(events) <= head_length(events):
                return False  # nothing after the task prompt to summarize
            usages = self._agent_llm.metrics.token_usages
            if view.unhandled_condensation_request:
                return True
            return should_compact(
                last_usage=usages[-1] if usages else None,
                max_context=max_context,
                cutoff=cutoff,
                usage_count=len(usages),
                compacted_at=self._compacted_at,
            )

        # -- summary call --------------------------------------------------------
        def _summarize(self, prompt: str):
            messages = [Message(role="user", content=[TextContent(text=prompt)])]

            def run():
                return asyncio.run(self._client.model_call(messages, None))

            # ``get_condensation`` is synchronous but runs inside the agent's async step; the
            # direct transport opens its own session per call, so drive it on a helper loop.
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(run).result()

        def get_condensation(self, view: View) -> Condensation:
            events = list(view.events)
            head = head_length(events)
            previous = ""
            forgotten = []
            for event in events[head:]:
                if isinstance(event, AgentCondensationObservation):
                    previous = event.message or ""
                    continue
                forgotten.append(event)
            if not forgotten:
                raise ValueError("compaction requested with no events to forget")
            prompt = build_summary_prompt(
                previous, [(event.id, str(event)) for event in forgotten], max_event_chars=max_event_chars
            )
            usages = self._agent_llm.metrics.token_usages
            started = time.time()
            response = self._summarize(prompt)
            summary = response.choices[0].message.content or ""
            self._compacted_at = len(usages)
            self.compactions += 1
            self.add_metadata("response", response.model_dump())
            self.add_metadata("metrics", self._client.llm.metrics.get())
            _note(
                {
                    "compaction": self.compactions,
                    "forgotten_events": len(forgotten),
                    "forgotten_range": [forgotten[0].id, forgotten[-1].id],
                    "trigger_usage": usages[-1].model_dump() if usages else None,
                    "summary_chars": len(summary),
                    "seconds": round(time.time() - started, 2),
                    "response_id": getattr(response, "id", None),
                }
            )
            return Condensation(
                action=CondensationAction(
                    forgotten_events_start_id=min(event.id for event in forgotten),
                    forgotten_events_end_id=max(event.id for event in forgotten),
                    summary=summary,
                    summary_offset=head,
                )
            )

    def from_config(cls, config, llm_registry):
        agent_llm = llm_registry.service_to_llm.get("agent")
        if agent_llm is None:
            raise RuntimeError("GDPVal condenser needs the agent LLM registered before the condenser is built")
        return GDPValSummarizingCondenser(agent_llm)

    LLMSummarizingCondenser.from_config = classmethod(from_config)
    return GDPValSummarizingCondenser
