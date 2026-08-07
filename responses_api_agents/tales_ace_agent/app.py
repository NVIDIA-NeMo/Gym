# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""
TalesAceAgent — ACE (Agentic Context Engineering) agent for TALES text-adventure
environments, with full NeMo Relay instrumentation.

Integrates ace_relay_trace.py logic into the NeMo Gym agent harness (Option B).
The agent owns the full DSPy + Haystack memory loop and evolving ACE playbook,
calling the TALES resources server via ServerClient for env interaction.

Tracing mirrors ace_relay_trace.py v0.4.0:
  [1] episode_goal + agent_step events  (goal/obs/action/reasoning per step)
  [2] reflection_summary event          (game state + steps + next best action)
  [3] llm_rate_limit_hit on 429/RPM     (wait + retry on reasoner and reflector)
  [+] cache_lookup, playbook_injected, episode_outcome, playbook_updated,
      reasoner_fallback, ATIF trajectory per episode, ATOF event stream.

Requires:
  - TALES resources server running with expose_admissible_commands: true
  - NVIDIA_API_KEY env var (or .env file in this directory)
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
import threading
import time
import uuid
from typing import Any

import dspy
import nemo_relay
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv
from fastapi import Body, HTTPException, Request, Response
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import Tool
from haystack.utils import Secret
from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder, NvidiaTextEmbedder
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.gymnasium import EnvResetResponse, EnvStepResponse


colorama_init(autoreset=True)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Model / API constants
# ---------------------------------------------------------------------------
_NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
_REASONER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
_SUMMARIZER_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

# ---------------------------------------------------------------------------
# Rate limiter: sliding-window, 28 RPM cap, shared across all NVIDIA API calls
# ---------------------------------------------------------------------------
_RPM_CAP = 28
_MIN_GAP = 60.0 / _RPM_CAP
_rate_lock = threading.Lock()
_call_times: collections.deque = collections.deque()


def _throttle() -> None:
    while True:
        with _rate_lock:
            now = time.monotonic()
            while _call_times and now - _call_times[0] >= 60.0:
                _call_times.popleft()
            if len(_call_times) < _RPM_CAP:
                gap = (now - _call_times[-1]) if _call_times else _MIN_GAP
                if gap >= _MIN_GAP:
                    _call_times.append(now)
                    return
                wait = _MIN_GAP - gap
            else:
                wait = 60.0 - (now - _call_times[0]) + 0.1
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Haystack singletons (lazy, one set per process)
# ---------------------------------------------------------------------------
_HAYSTACK: dict = {}


def _reasoner() -> NvidiaChatGenerator:
    if "gen" not in _HAYSTACK:
        _HAYSTACK["gen"] = NvidiaChatGenerator(
            model=_REASONER_MODEL,
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            api_base_url=_NVIDIA_API_BASE,
            max_retries=8,
        )
    return _HAYSTACK["gen"]


def _doc_embedder() -> NvidiaDocumentEmbedder:
    if "doc" not in _HAYSTACK:
        e = NvidiaDocumentEmbedder(
            model=_EMBED_MODEL, api_key=Secret.from_env_var("NVIDIA_API_KEY"), api_url=_NVIDIA_API_BASE
        )
        e.warm_up()
        _HAYSTACK["doc"] = e
    return _HAYSTACK["doc"]


def _text_embedder() -> NvidiaTextEmbedder:
    if "txt" not in _HAYSTACK:
        e = NvidiaTextEmbedder(
            model=_EMBED_MODEL, api_key=Secret.from_env_var("NVIDIA_API_KEY"), api_url=_NVIDIA_API_BASE
        )
        e.warm_up()
        _HAYSTACK["txt"] = e
    return _HAYSTACK["txt"]


# ---------------------------------------------------------------------------
# Memory helpers (ported from haystack_memory)
# ---------------------------------------------------------------------------
_NOOP_MARKERS = ["nothing happens", "nothing special", "already"]
_INFO_ACTIONS = ("examine", "look", "inventory")
_TOP_K_MEMORY = 10

_UNLIKELY_LOCATIONS: dict[str, list[str]] = {
    "cd": ["vase", "plant", "bathtub", "toilet", "sink", "fridge", "microwave", "garbagecan", "pot", "pan"],
    "book": ["vase", "plant", "bathtub", "toilet", "fridge", "microwave"],
    "laptop": ["vase", "plant", "bathtub", "toilet", "sink", "fridge", "microwave", "garbagecan"],
    "cellphone": ["vase", "plant", "bathtub", "toilet", "fridge", "microwave"],
    "phone": ["vase", "plant", "bathtub", "toilet", "fridge", "microwave"],
    "pen": ["bathtub", "toilet", "fridge", "microwave"],
    "pencil": ["bathtub", "toilet", "fridge", "microwave"],
    "keychain": ["bathtub", "toilet", "fridge", "microwave", "vase", "plant"],
    "creditcard": ["bathtub", "toilet", "fridge", "microwave", "vase"],
    "remotecontrol": ["bathtub", "toilet", "fridge", "microwave", "vase", "plant"],
    "newspaper": ["bathtub", "toilet", "fridge", "sink", "vase", "plant", "microwave"],
    "magazine": ["bathtub", "toilet", "fridge", "sink", "vase", "microwave"],
    "watch": ["bathtub", "toilet", "fridge", "microwave", "vase", "plant"],
    "candle": ["bathtub", "toilet", "fridge", "vase", "plant"],
    "statue": ["bathtub", "toilet", "fridge", "microwave"],
    "spraybottle": ["bathtub", "toilet", "fridge", "vase", "plant"],
    "soapbottle": ["bathtub", "toilet", "fridge", "vase", "plant"],
    "knife": ["vase", "plant"],
    "fork": ["vase", "plant"],
    "spoon": ["vase", "plant"],
    "toiletpaper": ["fridge", "microwave", "vase", "plant"],
    "handtowel": ["fridge", "microwave", "vase"],
    "cloth": ["fridge", "microwave"],
}


def _common_sense_hint(goal: str) -> str:
    g = goal.lower()
    for obj, unlikely in _UNLIKELY_LOCATIONS.items():
        if obj in g:
            return (
                f"Common sense: {obj}s are NEVER found in {', '.join(unlikely)}. Do NOT navigate to those locations."
            )
    return ""


def parse_goal(task: str) -> str:
    return task.split("Your task is to:")[-1].strip() if "Your task is to:" in task else task.strip()


def ground_fact(fact: str, observation: str) -> str | None:
    if not fact or fact.strip().lower() == "none":
        return None
    norm = re.sub(r"\s+", "", observation.lower())
    objs = re.findall(r"[a-z]+ \d+", fact.lower())
    if objs and not all(re.sub(r"\s+", "", o) in norm for o in objs):
        return None
    return fact.strip()


def is_noop(action: str, before: str, after: str) -> bool:
    if any(m in after.lower() for m in _NOOP_MARKERS):
        return True
    return after.strip() == before.strip() and not action.startswith("go to")


def make_action_tool(admissible: list[str]) -> Tool:
    return Tool(
        name="take_action",
        description="Choose the single best next action toward completing the task.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(admissible),
                    "description": "next action, copied verbatim from the list",
                }
            },
            "required": ["action"],
        },
        function=lambda action: action,
    )


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------
class MemoryUpdate(dspy.Signature):
    """Maintain a COMPACT running memory for a text household task (ALFWorld).

    STRICT FACTUAL MODE -- NON-NEGOTIABLE:
    - NO FABRICATION. Record ONLY objects, locations, and outcomes that appear
      VERBATIM in THIS observation. Never invent, infer, or assume.
    - If the observation does not name an object, do NOT claim its location.
    - When uncertain, output 'none' for new_fact.
    - progress states ONLY confirmed counts and what is currently held/placed."""

    task = dspy.InputField()
    last_action = dspy.InputField()
    observation = dspy.InputField()
    current_progress = dspy.InputField()
    new_fact: str = dspy.OutputField(
        desc="ONE world fact stated verbatim from THIS observation "
        "(e.g. 'drawer 3: empty'); every object MUST appear in the observation, else 'none'"
    )
    dead_end: str = dspy.OutputField(
        desc="copy exact last_action if it made NO progress and must not be repeated; else exactly 'none'"
    )
    progress: str = dspy.OutputField(desc="confirmed counts + what's held/placed only; NO location claims")
    plan: str = dspy.OutputField(desc="next concrete subgoals, ordered, terse")


REASON_SYS = (
    "You are an expert agent playing ALFWorld, a text household game. Finish the task in AS FEW STEPS AS POSSIBLE. "
    "Use the Progress and Plan to act with intent; never repeat an action listed as useless. "
    "Call take_action with exactly ONE action from the admissible list, "
    "preferring the one that most directly advances the Plan."
)


class MemoryState:
    def __init__(self) -> None:
        self.progress = "nothing done yet"
        self.plan = "explore to locate the target object(s)"
        self.dead_ends: list[str] = []
        self.taken: list[str] = []
        self.last_action: str | None = None

    def add_dead_end(self, action: str) -> None:
        if action and action not in self.dead_ends:
            self.dead_ends.append(action)


# ---------------------------------------------------------------------------
# ACE Playbook (ported from rung3_ace_alfworld)
# ---------------------------------------------------------------------------
_PRUNE_NET = 2
_PLAYBOOK_CAP = 12


class Playbook:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self._next = 1

    @staticmethod
    def _sim(a: str, b: str) -> float:
        norm = lambda s: set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
        wa, wb = norm(a), norm(b)
        return len(wa & wb) / max(1, len(wa | wb))

    def add(self, text: str) -> None:
        t = text.strip()
        if len(t) < 8:
            return
        if any(self._sim(it["text"], t) > 0.6 for it in self.items):
            return
        self.items.append({"id": self._next, "text": t, "helpful": 0, "harmful": 0})
        self._next += 1

    def bump(self, ids: list[int], key: str) -> None:
        for it in self.items:
            if it["id"] in ids:
                it[key] += 1

    def prune(self) -> None:
        self.items = [
            it for it in self.items if not (it["harmful"] - it["helpful"] >= _PRUNE_NET and it["helpful"] == 0)
        ]

    def render(self, k: int = _PLAYBOOK_CAP) -> str:
        ranked = sorted(self.items, key=lambda it: it["helpful"] - it["harmful"], reverse=True)[:k]
        return "\n".join(f"[{it['id']}] {it['text']}" for it in ranked)

    def render_meta(self, k: int = _PLAYBOOK_CAP) -> str:
        ranked = sorted(self.items, key=lambda it: it["helpful"] - it["harmful"], reverse=True)[:k]
        return (
            "\n".join(f"[{it['id']}] (+{it['helpful']}/-{it['harmful']}) {it['text']}" for it in ranked) or "(empty)"
        )


class Reflect(dspy.Signature):
    """Reflect on ONE ALFWorld episode to improve a STRATEGY PLAYBOOK of GENERAL, reusable tactics
    (e.g. 'for multi-object tasks, keep a placed-count'). NOT facts about this specific room.

    STRICT FACTUAL MODE: ground every claim in what the trajectory actually shows.
    Do not invent. If no general lesson is warranted, output 'none'/empty.

    new_strategies OUTPUT FORMAT (strict): ONLY strategy sentences, ONE imperative sentence per line,
    at most 3 lines. NO headers, NO 'Reasoning:', NO numbering, NO markdown/asterisks."""

    task = dspy.InputField()
    outcome = dspy.InputField(desc="e.g. 'SOLVED in 9 steps' or 'FAILED at 50 steps'")
    trajectory = dspy.InputField(desc="actions and observations from the episode")
    current_playbook = dspy.InputField(desc="existing numbered strategies")
    helpful_ids: str = dspy.OutputField(desc="comma-separated ids of strategies that helped THIS episode, or 'none'")
    harmful_ids: str = dspy.OutputField(desc="comma-separated ids that misled / wasted steps, or 'none'")
    new_strategies: str = dspy.OutputField(
        desc="at most 3 NEW general strategies, ONE imperative sentence per line; empty if none"
    )


_META_RE = re.compile(
    r"(?i)^(reasoning|strateg|note|here|because|this |that |these |the agent|derived|aims|addresses|in summary|explanation|rationale)"
)


def _parse_ids(s: str) -> list[int]:
    out = []
    for tok in (s or "").replace(";", ",").split(","):
        tok = tok.strip().lstrip("[").rstrip("]")
        if tok.isdigit():
            out.append(int(tok))
    return out


def _clean_strategy(line: str) -> str | None:
    line = re.sub(r"[*#`]+", "", line).strip()
    line = line.lstrip("-•*0123456789.) ").strip()
    if not line or line.lower() == "none":
        return None
    if line.endswith(":") or _META_RE.match(line):
        return None
    if len(line.split()) < 4:
        return None
    return line


def _curate(pb: Playbook, refl: Any) -> None:
    pb.bump(_parse_ids(refl.helpful_ids), "helpful")
    pb.bump(_parse_ids(refl.harmful_ids), "harmful")
    added = 0
    for raw in (refl.new_strategies or "").splitlines():
        if added >= 3:
            break
        s = _clean_strategy(raw)
        if s:
            before = len(pb.items)
            pb.add(s)
            added += len(pb.items) > before
    pb.prune()


# ---------------------------------------------------------------------------
# NeMo Relay helpers (ported from ace_relay_trace)
# ---------------------------------------------------------------------------
_RPM_WAIT_BASE = 15.0
_RPM_MAX_RETRIES = 5
_RPM_WAIT_CAP = 60.0


def _is_rpm_error(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s or "rpm" in s


class _LLMSpan:
    """Sync context manager wrapping a single-shot LLM call with nemo_relay.llm.call/call_end."""

    def __init__(self, provider: str, model: str, content: dict, ep_handle: Any = None) -> None:
        self.provider = provider
        self.model = model
        self.request = nemo_relay.LLMRequest({}, content)
        self.ep_handle = ep_handle
        self._handle: Any = None

    def __enter__(self) -> "_LLMSpan":
        self._handle = nemo_relay.llm.call(self.provider, self.request, handle=self.ep_handle, model_name=self.model)
        return self

    def end(self, result: dict | None = None, error: str | None = None) -> None:
        response = {"error": error} if error else (result or {})
        nemo_relay.llm.call_end(self._handle, response)
        self._handle = None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._handle is not None:
            self.end(error=str(exc_val)[:300] if exc_type else None)
        return False


def _reason_traced(
    goal: str,
    mem: MemoryState,
    facts: list[str],
    offered: list[str],
    playbook: str = "",
    common_sense: str = "",
    ep_handle: Any = None,
    step_num: int = 0,
) -> str:
    """Reasoner with per-attempt llm.call/call_end and RPM retry. Mirrors ace_relay_trace._reason_traced."""
    tool = make_action_tool(offered)
    user_parts = []
    if playbook:
        user_parts.append("Learned strategies (apply when relevant):\n" + playbook)
    if common_sense:
        user_parts.append(common_sense)
    user_parts.append(f"Task: {goal}\nProgress: {mem.progress}\nPlan: {mem.plan}")
    if mem.dead_ends:
        user_parts.append("Do NOT repeat these useless actions: " + ", ".join(mem.dead_ends[-8:]))
    if facts:
        user_parts.append("Relevant facts:\n" + "\n".join(f"- {f}" for f in facts))
    user_parts.append("Admissible actions:\n" + "\n".join(f"  - {a}" for a in offered))
    user_msg = "\n".join(user_parts)

    wait = _RPM_WAIT_BASE
    last_exc: Exception | None = None

    for attempt in range(_RPM_MAX_RETRIES):
        llm_req = nemo_relay.LLMRequest(
            {},
            {
                "messages": [{"role": "system", "content": REASON_SYS}, {"role": "user", "content": user_msg}],
                "model": _REASONER_MODEL,
            },
        )
        lh = nemo_relay.llm.call(
            "reasoner",
            llm_req,
            handle=ep_handle,
            model_name=_REASONER_MODEL,
            data={"step": step_num, "attempt": attempt},
        )
        _throttle()
        try:
            out = _reasoner().run(
                messages=[ChatMessage.from_system(REASON_SYS), ChatMessage.from_user(user_msg)],
                tools=[tool],
                generation_kwargs={
                    "tool_choice": "required",
                    "extra_body": {"chat_template_kwargs": {"thinking": True}},
                    "max_tokens": 6000,
                    "temperature": 0.6,
                },
            )
            reply = out["replies"][0]
            action: str | None = None
            if reply.tool_calls:
                action = reply.tool_calls[0].arguments.get("action")
                if action not in offered:
                    action = None
            if action is None:
                for a in offered:
                    if a in (reply.text or ""):
                        action = a
                        break
            if action is None:
                action = offered[0]
            nemo_relay.llm.call_end(lh, {"action_chosen": action})
            return action
        except Exception as e:
            last_exc = e
            nemo_relay.llm.call_end(lh, {"error": str(e)[:300]})
            if _is_rpm_error(e) and attempt < _RPM_MAX_RETRIES - 1:
                nemo_relay.scope.event(
                    "llm_rate_limit_hit",
                    handle=ep_handle,
                    data={
                        "provider": "reasoner",
                        "model": _REASONER_MODEL,
                        "step": step_num,
                        "attempt": attempt + 1,
                        "wait_seconds": wait,
                        "error": str(e)[:300],
                    },
                )
                print(
                    f"  {Fore.YELLOW}[relay] 429/RPM on reasoner step {step_num}, waiting {wait:.0f}s{Style.RESET_ALL}",
                    flush=True,
                )
                time.sleep(wait)
                wait = min(wait * 2, _RPM_WAIT_CAP)
            else:
                print(
                    f"  {Fore.RED}[reason error (attempt {attempt + 1}/{_RPM_MAX_RETRIES}): {e}]{Style.RESET_ALL}",
                    flush=True,
                )
                if not _is_rpm_error(e) and attempt < _RPM_MAX_RETRIES - 1:
                    time.sleep(15 * (attempt + 1))

    not_taken = [a for a in offered if a not in mem.taken]
    explore_new = [a for a in not_taken if a.startswith(("go to", "open"))]
    explore_any = [a for a in offered if a.startswith(("go to", "open"))]
    fallback = (explore_new or explore_any or not_taken or offered)[0]
    nemo_relay.scope.event(
        "reasoner_fallback",
        handle=ep_handle,
        data={
            "step": step_num,
            "fallback_action": fallback,
            "last_error": str(last_exc)[:300] if last_exc else None,
        },
    )
    return fallback


# ---------------------------------------------------------------------------
# Sync helpers called via asyncio.to_thread
# ---------------------------------------------------------------------------
def _sync_memory_step(
    update_fn: Any,
    goal: str,
    mem: MemoryState,
    cur_obs: str,
    store: InMemoryDocumentStore,
    retriever: InMemoryEmbeddingRetriever,
    all_facts: list[str],
    ep_handle: Any,
) -> list[str]:
    """DSPy memory update + Haystack embed + recall. Mutates mem/store/all_facts in-place. Returns recalled facts."""
    mem_content = {
        "messages": [
            {
                "role": "user",
                "content": f"task={goal}\nlast_action={mem.last_action or 'none'}\nobs={cur_obs}\nprogress={mem.progress}",
            }
        ],
        "model": _SUMMARIZER_MODEL,
    }
    with _LLMSpan("memory-updater", _SUMMARIZER_MODEL, mem_content, ep_handle) as span:
        _throttle()
        upd = update_fn(
            task=goal, last_action=mem.last_action or "none", observation=cur_obs, current_progress=mem.progress
        )

        def _first_line(s: str) -> str:
            return (s or "").strip().splitlines()[0].strip()

        mem.progress = _first_line(upd.progress) or mem.progress
        mem.plan = _first_line(upd.plan) or mem.plan
        de = (upd.dead_end or "").strip()
        if de and de == (mem.last_action or ""):
            mem.add_dead_end(de)
        span.end(
            result={"new_fact": upd.new_fact, "dead_end": upd.dead_end, "progress": mem.progress, "plan": mem.plan}
        )

    raw_fact = (upd.new_fact or "").strip().splitlines()[0].strip()
    fact = ground_fact(raw_fact, cur_obs)
    if fact:
        _throttle()
        docs = _doc_embedder().run(documents=[Document(content=fact)])["documents"]
        store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        if fact not in all_facts:
            all_facts.append(fact)

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}{'MEMORY':<9}{Style.RESET_ALL}{Fore.YELLOW}"
        f"progress=[{mem.progress}]  plan=[{mem.plan}]  dead_ends={mem.dead_ends[-5:]}{Style.RESET_ALL}",
        flush=True,
    )

    facts: list[str] = []
    if store.count_documents() > 0:
        _throttle()
        q = _text_embedder().run(text=f"{goal}\n{mem.plan}\n{cur_obs}")["embedding"]
        facts = [d.content for d in retriever.run(query_embedding=q, top_k=_TOP_K_MEMORY)["documents"]]

    recall_str = "  |  ".join(facts) if facts else "(none yet)"
    print(
        f"{Fore.MAGENTA}{Style.BRIGHT}{'RECALL':<9}{Style.RESET_ALL}{Fore.MAGENTA}{recall_str}{Style.RESET_ALL}",
        flush=True,
    )

    return facts


def _sync_reflect(
    reflect_fn: Any,
    reflection_lm: Any,
    goal: str,
    outcome: str,
    trajectory: str,
    current_playbook: str,
    ep_handle: Any,
) -> Any | None:
    """Reflector LLM call with RPM retry. Returns Reflect prediction or None on total failure."""
    ref_content = {
        "messages": [
            {
                "role": "user",
                "content": f"task={goal}\noutcome={outcome}\ntrajectory={trajectory}\nplaybook={current_playbook}",
            }
        ],
        "model": _SUMMARIZER_MODEL,
    }
    wait = _RPM_WAIT_BASE
    for attempt in range(_RPM_MAX_RETRIES):
        ref_req = nemo_relay.LLMRequest({}, ref_content)
        ref_lh = nemo_relay.llm.call(
            "reflector", ref_req, handle=ep_handle, model_name=_SUMMARIZER_MODEL, data={"attempt": attempt}
        )
        try:
            _throttle()
            with dspy.context(lm=reflection_lm):
                refl = reflect_fn(task=goal, outcome=outcome, trajectory=trajectory, current_playbook=current_playbook)
            nemo_relay.llm.call_end(
                ref_lh,
                {
                    "helpful_ids": refl.helpful_ids,
                    "harmful_ids": refl.harmful_ids,
                    "new_strategies": refl.new_strategies,
                },
            )
            return refl
        except Exception as e:
            nemo_relay.llm.call_end(ref_lh, {"error": str(e)[:300]})
            if _is_rpm_error(e) and attempt < _RPM_MAX_RETRIES - 1:
                nemo_relay.scope.event(
                    "llm_rate_limit_hit",
                    handle=ep_handle,
                    data={
                        "provider": "reflector",
                        "model": _SUMMARIZER_MODEL,
                        "attempt": attempt + 1,
                        "wait_seconds": wait,
                        "error": str(e)[:300],
                    },
                )
                print(f"  {Fore.YELLOW}[relay] 429/RPM on reflector, waiting {wait:.0f}s{Style.RESET_ALL}", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, _RPM_WAIT_CAP)
            else:
                print(
                    f"  {Fore.RED}[reflector error (attempt {attempt + 1}/{_RPM_MAX_RETRIES}): {e}]{Style.RESET_ALL}",
                    flush=True,
                )
                if not _is_rpm_error(e) and attempt < _RPM_MAX_RETRIES - 1:
                    time.sleep(15 * (attempt + 1))
    return None


def _make_nemo_response(text: str) -> dict:
    """Wrap a plain-text action string into a minimal NeMoGymResponse dict for /step."""
    return {
        "id": "ace_action",
        "object": "response",
        "created_at": 0.0,
        "status": "completed",
        "model": "tales-ace-agent",
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "output": [
            {
                "id": "ace_msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _extract_admissible(info: dict) -> list[str]:
    """Normalize admissible_commands from TALES step/reset info."""
    raw = info.get("admissible_commands", [])
    if raw and isinstance(raw[0], list):
        return raw[0]
    return list(raw)


# ---------------------------------------------------------------------------
# Gym agent config + request / response types
# ---------------------------------------------------------------------------
class TalesAceAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    max_steps: int = Field(50, ge=1)
    trace_dir: str = "traces"


class TalesAceRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class TalesAceRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    terminated: bool = False
    truncated: bool = False
    info: dict = {}
    steps: int = 0
    playbook_size: int = 0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class TalesAceAgent(SimpleResponsesAPIAgent):
    """ACE agent with NeMo Relay tracing for TALES text-adventure environments.

    The playbook (self._playbook) persists across run() calls so the agent
    learns across episodes during a training run.

    Requires TALES resources server configured with expose_admissible_commands: true.
    """

    config: TalesAceAgentConfig

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._playbook = Playbook()

        api_key = os.environ.get("NVIDIA_API_KEY", "")
        summarizer_lm = dspy.LM(
            f"openai/{_SUMMARIZER_MODEL}",
            api_key=api_key,
            api_base=_NVIDIA_API_BASE,
            temperature=0.4,
            max_tokens=600,
            num_retries=10,
        )
        dspy.configure(lm=summarizer_lm)
        self._update = dspy.Predict(MemoryUpdate)
        self._reflection_lm = dspy.LM(
            f"openai/{_SUMMARIZER_MODEL}",
            api_key=api_key,
            api_base=_NVIDIA_API_BASE,
            temperature=0.7,
            max_tokens=2000,
            num_retries=10,
        )
        self._reflect = dspy.Predict(Reflect)

        trace_dir = os.path.abspath(self.config.trace_dir)
        os.makedirs(os.path.join(trace_dir, "atif"), exist_ok=True)
        os.makedirs(os.path.join(trace_dir, "atof"), exist_ok=True)
        atof_cfg = nemo_relay.AtofExporterConfig()
        atof_cfg.output_directory = os.path.join(trace_dir, "atof")
        atof_cfg.filename = "events.jsonl"
        atof_cfg.mode = nemo_relay.AtofExporterMode.Append
        self._atof = nemo_relay.AtofExporter(atof_cfg)
        self._atof.register("tales-ace-atof")
        self._trace_dir = trace_dir

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise HTTPException(status_code=501, detail="TalesAceAgent handles inference internally. Use /run.")

    async def run(self, request: Request, body: TalesAceRunRequest) -> TalesAceRunResponse:
        env_cookies = request.cookies
        extra = body.model_extra or {}

        # 1. Reset env
        reset_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/reset",
            json=body.model_dump(),
            cookies=env_cookies,
        )
        await raise_for_status(reset_resp)
        reset_data = EnvResetResponse.model_validate(await get_response_json(reset_resp))
        env_cookies = reset_resp.cookies

        initial_obs = reset_data.observation or "(game start)"
        admissible = _extract_admissible(reset_data.info or {})
        goal = parse_goal(initial_obs)
        common_sense = _common_sense_hint(goal)
        print(f"{Fore.CYAN}{Style.BRIGHT}{'GOAL':<9}{Style.RESET_ALL}{Fore.CYAN}{goal}{Style.RESET_ALL}", flush=True)

        # 2. Episode state
        mem = MemoryState()
        store = InMemoryDocumentStore()
        retriever = InMemoryEmbeddingRetriever(document_store=store)
        all_facts: list[str] = []
        traj: list[str] = []
        cur_obs = initial_obs
        total_reward = 0.0
        steps = 0
        done = False

        ep_label = str(uuid.uuid4())[:8]
        atif = nemo_relay.AtifExporter(ep_label, "tales-ace-alfworld", "1.0", model_name=_REASONER_MODEL)
        atif.register(f"atif-ep-{ep_label}")

        with nemo_relay.scope.scope(f"episode-{ep_label}", nemo_relay.ScopeType.Agent) as ep_handle:
            nemo_relay.scope.event(
                "episode_goal",
                handle=ep_handle,
                data={
                    "goal": goal,
                    "common_sense_hint": common_sense or None,
                    "framework": extra.get("framework", "alfworld"),
                    "task_no": extra.get("task_no", 0),
                    "split": extra.get("split", "train"),
                },
            )
            nemo_relay.scope.event(
                "playbook_injected",
                handle=ep_handle,
                data={
                    "bullet_count": len(self._playbook.items),
                    "playbook": self._playbook.render(k=_PLAYBOOK_CAP),
                },
            )

            for _ in range(self.config.max_steps):
                print(f"{Fore.WHITE}{Style.DIM}{'-' * 70}  step {steps + 1}{Style.RESET_ALL}", flush=True)
                print(
                    f"{Fore.WHITE}{Style.DIM}{'OBS':<9}{Style.RESET_ALL}{Fore.WHITE}{Style.DIM}{cur_obs}{Style.RESET_ALL}",
                    flush=True,
                )

                # 3a. Memory update + embed + recall (sync, in thread)
                facts = await asyncio.to_thread(
                    _sync_memory_step,
                    self._update,
                    goal,
                    mem,
                    cur_obs,
                    store,
                    retriever,
                    all_facts,
                    ep_handle,
                )

                # 3b. Choose action via traced reasoner (sync, in thread)
                offered = [
                    a
                    for a in admissible
                    if a not in mem.dead_ends and not (a.startswith(_INFO_ACTIONS) and a in mem.taken)
                ]
                offered = offered or [a for a in admissible if a not in mem.dead_ends] or admissible

                action = await asyncio.to_thread(
                    _reason_traced,
                    goal,
                    mem,
                    facts,
                    offered,
                    self._playbook.render(),
                    common_sense,
                    ep_handle,
                    steps + 1,
                )
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}{'ACTION':<9}{Style.RESET_ALL}{Fore.GREEN}{action}{Style.RESET_ALL}",
                    flush=True,
                )

                # 3c. Step env (async HTTP)
                step_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json=body.model_dump() | {"response": _make_nemo_response(action)},
                    cookies=env_cookies,
                )
                await raise_for_status(step_resp)
                step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
                env_cookies = step_resp.cookies

                new_obs = step_data.observation or cur_obs
                total_reward += step_data.reward
                done = step_data.terminated
                new_admissible = _extract_admissible(step_data.info or {})
                if new_admissible:
                    admissible = new_admissible

                if is_noop(action, cur_obs, new_obs):
                    mem.add_dead_end(action)
                mem.taken.append(action)
                mem.last_action = action
                traj += [f"> {action}", new_obs]
                steps += 1

                # [TRACE 1] per-step event
                nemo_relay.scope.event(
                    "agent_step",
                    handle=ep_handle,
                    data={
                        "step": steps,
                        "observation": cur_obs,
                        "action_taken": action,
                        "progress": mem.progress,
                        "plan": mem.plan,
                        "recalled_facts": facts,
                        "dead_ends": mem.dead_ends[-8:],
                        "result_obs": new_obs[:400],
                        "reward": step_data.reward,
                        "is_done": bool(done),
                    },
                )

                cur_obs = new_obs
                if done or step_data.truncated:
                    break

            outcome = f"SOLVED in {steps} steps" if total_reward > 0 else f"FAILED at {steps} steps"
            nemo_relay.scope.event(
                "episode_outcome",
                handle=ep_handle,
                data={
                    "label": ep_label,
                    "goal": goal,
                    "outcome": outcome,
                    "steps": steps,
                    "total_reward": total_reward,
                    "trajectory_tail": "\n".join(traj)[-2000:],
                },
            )

            # 4. Reflector (sync, in thread)
            traj_str = "\n".join(traj)
            pb_size_before = len(self._playbook.items)
            refl = await asyncio.to_thread(
                _sync_reflect,
                self._reflect,
                self._reflection_lm,
                goal,
                outcome,
                traj_str[-3000:],
                self._playbook.render_meta() or "(empty)",
                ep_handle,
            )

            if refl is not None:
                obs_lines = [line for line in traj_str.splitlines() if not line.startswith(">")]
                last_obs = obs_lines[-1][:200] if obs_lines else ""
                # [TRACE 2] reflection_summary
                nemo_relay.scope.event(
                    "reflection_summary",
                    handle=ep_handle,
                    data={
                        "goal": goal,
                        "outcome": outcome,
                        "steps_taken": steps,
                        "last_observation": last_obs,
                        "helpful_strategy_ids": refl.helpful_ids,
                        "harmful_strategy_ids": refl.harmful_ids,
                        "next_best_strategies": refl.new_strategies,
                        "playbook_size_before": pb_size_before,
                        "current_playbook": self._playbook.render_meta(),
                    },
                )
                _curate(self._playbook, refl)
                nemo_relay.scope.event(
                    "playbook_updated",
                    handle=ep_handle,
                    data={
                        "playbook_size_after": len(self._playbook.items),
                        "bullets_delta": len(self._playbook.items) - pb_size_before,
                        "playbook_snapshot": self._playbook.render_meta(),
                    },
                )
                print(
                    f"{Fore.BLUE}{Style.BRIGHT}[{ep_label}] {outcome}; playbook now {len(self._playbook.items)} bullets{Style.RESET_ALL}",
                    flush=True,
                )
                print(f"{Fore.BLUE}{self._playbook.render_meta()}{Style.RESET_ALL}", flush=True)

        # Export ATIF trajectory for this episode
        atif_path = os.path.join(self._trace_dir, "atif", f"episode-{ep_label}.json")
        with open(atif_path, "w") as f:
            f.write(atif.export_json())
        atif.deregister(f"atif-ep-{ep_label}")

        truncated = not done and steps >= self.config.max_steps
        last_response = NeMoGymResponse.model_validate(_make_nemo_response(mem.last_action or "done"))

        return TalesAceRunResponse(
            responses_create_params=body.responses_create_params,
            response=last_response,
            reward=total_reward,
            terminated=done,
            truncated=truncated,
            info={"steps": steps, "playbook_size": len(self._playbook.items), "goal": goal},
            steps=steps,
            playbook_size=len(self._playbook.items),
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    TalesAceAgent.run_webserver()
