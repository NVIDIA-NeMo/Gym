"""
ace_relay_trace.py — NeMo Relay instrumented version of rung3_ace_alfworld.py

Run:
    .venv/bin/python ace_relay_trace.py --smoke
    .venv/bin/python ace_relay_trace.py --full

=== NeMo Relay API used (v0.4.0) ===
- No nemo_relay.init() — use exporter objects directly.
- ATOF: AtofExporter(AtofExporterConfig) registered once globally; shutdown at end.
- ATIF: AtifExporter created per episode (register → scope runs → export_json()
  → write file → deregister). Unique session_id per episode.
- nemo_relay.llm.call() / call_end() — sync manual LLM lifecycle. Produces
  proper ATIF model steps. Used directly in retry loops (one handle per attempt).
- _LLMSpan helper — thin sync context manager around llm.call/call_end for
  single-shot calls (DSPy Predict) that have no per-attempt retry loop.
- nemo_relay.scope.scope() — sync context manager for episode/phase scopes.
- nemo_relay.scope.event() — sync mark events for all semantic annotations.

=== What gets traced ===
  [1] goal / observation / per-step reasoning
        → episode_goal event, agent_step event after every env.step()
  [2] game state summary / steps taken / next best action
        → reflection_summary event after each reflector call
  [3] LLM call failures (429/RPM) with wait + retry
        → llm_rate_limit_hit event in _reason_traced and reflector block
  [+] cache hit/miss, playbook injection, playbook delta, phase start/end,
      memory-update LLM spans (llm.call/call_end), fallback warnings,
      eval summary, full ATIF trajectory per episode + ATOF event stream
"""

import json
import os
import sys
import time
import uuid

import dspy
import haystack_memory as r2
import nemo_relay
from ace_alfworld import (
    N_TEST,
    N_TRAIN,
    PLAYBOOK_CAP,
    Playbook,
    Reflect,
    _load_cache,
    _save_cache,
    curate,
    reflection_lm,
)
from colorama import Fore, Style
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_memory import (
    GAMMA,
    INFO_ACTIONS,
    MAX_ITERS,
    REASON_SYS,
    REASONER_MODEL,
    SUMMARIZER_MODEL,
    TOP_K_MEMORY,
    MemoryState,
    _common_sense_hint,
    _doc_embedder,
    _reasoner,
    _text_embedder,
    _throttle,
    ground_fact,
    is_noop,
    log_action,
    log_goal,
    log_mem,
    log_obs,
    log_recall,
    log_result,
    log_step,
    make_action_tool,
    parse_goal,
)


# --------------------------------------------------------------------------- #
# NeMo Relay exporters                                                          #
# --------------------------------------------------------------------------- #
_TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
os.makedirs(os.path.join(_TRACE_DIR, "atif"), exist_ok=True)
os.makedirs(os.path.join(_TRACE_DIR, "atof"), exist_ok=True)

# ATOF: raw event stream — registered once, captures every lifecycle event.
_atof_cfg = nemo_relay.AtofExporterConfig()
_atof_cfg.output_directory = os.path.join(_TRACE_DIR, "atof")
_atof_cfg.filename = "events.jsonl"
_atof_cfg.mode = nemo_relay.AtofExporterMode.Append
_atof = nemo_relay.AtofExporter(_atof_cfg)
_atof.register("atof-exporter")


def _atif_register(label: str) -> nemo_relay.AtifExporter:
    """Create and register a fresh ATIF exporter for one episode."""
    atif = nemo_relay.AtifExporter(
        str(uuid.uuid4()),
        "rung3-ace-alfworld",
        "1.0",
        model_name=REASONER_MODEL,
    )
    atif.register(f"atif-ep-{label}")
    return atif


def _atif_export(atif: nemo_relay.AtifExporter, label: str) -> None:
    """Export trajectory to file and deregister."""
    path = os.path.join(_TRACE_DIR, "atif", f"episode-{label}.json")
    with open(path, "w") as f:
        f.write(atif.export_json())
    atif.deregister(f"atif-ep-{label}")


# --------------------------------------------------------------------------- #
# RPM retry constants                                                           #
# --------------------------------------------------------------------------- #
_RPM_WAIT_BASE = 15.0  # conservative floor under 40 RPM / 60 s window
_RPM_MAX_RETRIES = 5
_RPM_WAIT_CAP = 60.0


def _is_rpm_error(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s or "rpm" in s


# --------------------------------------------------------------------------- #
# _LLMSpan — sync context manager for single-shot LLM calls                   #
# Uses nemo_relay.llm.call / call_end. Produces proper ATIF model steps.       #
# For retry loops, call llm.call / call_end directly per attempt instead.      #
# --------------------------------------------------------------------------- #
class _LLMSpan:
    def __init__(self, provider: str, model: str, content: dict, ep_handle=None):
        self.provider = provider
        self.model = model
        self.request = nemo_relay.LLMRequest({}, content)
        self.ep_handle = ep_handle
        self._handle = None

    def __enter__(self):
        self._handle = nemo_relay.llm.call(
            self.provider,
            self.request,
            handle=self.ep_handle,
            model_name=self.model,
        )
        return self

    def end(self, result=None, error: str | None = None):
        response = {"error": error} if error else (result or {})
        nemo_relay.llm.call_end(self._handle, response)
        self._handle = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self.end(error=str(exc_val)[:300] if exc_type else None)
        return False  # never suppress


# --------------------------------------------------------------------------- #
# _reason_traced                                                                #
# Drop-in for rung2._reason(). Uses llm.call/call_end per attempt so each      #
# retry gets its own ATIF model step. Emits llm_rate_limit_hit on 429.         #
# --------------------------------------------------------------------------- #
def _reason_traced(
    goal: str,
    mem: MemoryState,
    facts: list,
    offered: list,
    playbook: str = "",
    common_sense: str = "",
    ep_handle=None,
    step_num: int = 0,
) -> str:
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
                "messages": [
                    {"role": "system", "content": REASON_SYS},
                    {"role": "user", "content": user_msg},
                ],
                "model": REASONER_MODEL,
            },
        )
        lh = nemo_relay.llm.call(
            "reasoner",
            llm_req,
            handle=ep_handle,
            model_name=REASONER_MODEL,
            data={"step": step_num, "attempt": attempt},
        )
        _throttle()
        try:
            out = _reasoner().run(
                messages=[
                    ChatMessage.from_system(REASON_SYS),
                    ChatMessage.from_user(user_msg),
                ],
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
                        "model": REASONER_MODEL,
                        "step": step_num,
                        "attempt": attempt + 1,
                        "wait_seconds": wait,
                        "error": str(e)[:300],
                    },
                )
                print(
                    f"  {Fore.YELLOW}[relay] 429/RPM on reasoner step {step_num}, "
                    f"waiting {wait:.0f}s (attempt {attempt + 1}/{_RPM_MAX_RETRIES})"
                    f"{Style.RESET_ALL}",
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

    # All retries exhausted — heuristic fallback (mirrors rung2)
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


# --------------------------------------------------------------------------- #
# TracedMemoryAgent                                                             #
# Subclasses StructuredMemoryAgent. Overrides forward() to add:                #
#   - episode_goal event                                                        #
#   - _LLMSpan (llm.call/call_end) around every DSPy self.update() call        #
#   - agent_step event after each env.step()   [TRACE 1]                       #
#   - _reason_traced() instead of _reason()                                    #
# --------------------------------------------------------------------------- #
class TracedMemoryAgent(r2.StructuredMemoryAgent):
    def __init__(self, max_iters=MAX_ITERS):
        super().__init__(max_iters=max_iters)
        self._ep_handle = None  # set by episode wrapper before calling forward()

    def forward(self, idx):
        ep_handle = self._ep_handle
        store = InMemoryDocumentStore()
        retriever = InMemoryEmbeddingRetriever(document_store=store)
        mem = MemoryState()
        reward, steps = 0, 0
        all_facts: list[str] = []

        if self.prefill_facts:
            print(
                f"  {Fore.CYAN}[prefill: loading {len(self.prefill_facts)} cached facts "
                f"+ {len(self.prefill_dead_ends)} dead-ends]{Style.RESET_ALL}",
                flush=True,
            )
            _throttle()
            docs = _doc_embedder().run(documents=[Document(content=f) for f in self.prefill_facts])["documents"]
            store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
            all_facts.extend(self.prefill_facts)
        for de in self.prefill_dead_ends:
            mem.add_dead_end(de)

        with r2.alfworld.POOL.session() as env:
            traj = []
            task, info = env.init(idx)
            goal = parse_goal(task)
            common_sense = _common_sense_hint(goal)
            log_goal(goal)

            # [TRACE 1] emit goal at episode start
            if ep_handle:
                nemo_relay.scope.event(
                    "episode_goal",
                    handle=ep_handle,
                    data={
                        "goal": goal,
                        "common_sense_hint": common_sense or None,
                        "prefill_facts": len(all_facts),
                        "prefill_dead_ends": len(mem.dead_ends),
                    },
                )

            cur_obs = "(game start)"

            for _ in range(self.max_iters):
                admissible = info["admissible_commands"][0]
                log_step(steps + 1)
                log_obs(cur_obs)

                # 1. DELTA-update — DSPy memory updater (single-shot, use _LLMSpan)
                mem_content = {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"task={goal}\nlast_action={mem.last_action or 'none'}"
                                f"\nobs={cur_obs}\nprogress={mem.progress}"
                            ),
                        }
                    ],
                    "model": SUMMARIZER_MODEL,
                }
                with _LLMSpan("memory-updater", SUMMARIZER_MODEL, mem_content, ep_handle) as span:
                    _throttle()
                    upd = self.update(
                        task=goal,
                        last_action=mem.last_action or "none",
                        observation=cur_obs,
                        current_progress=mem.progress,
                    )

                    def _first_line(s):
                        return (s or "").strip().splitlines()[0].strip()

                    mem.progress = _first_line(upd.progress) or mem.progress
                    mem.plan = _first_line(upd.plan) or mem.plan
                    de = (upd.dead_end or "").strip()
                    if de and de == (mem.last_action or ""):
                        mem.add_dead_end(de)
                    span.end(
                        result={
                            "new_fact": upd.new_fact,
                            "dead_end": upd.dead_end,
                            "progress": mem.progress,
                            "plan": mem.plan,
                        }
                    )

                # 2. Store grounded world fact
                raw_fact = (upd.new_fact or "").strip().splitlines()[0].strip()
                fact = ground_fact(raw_fact, cur_obs)
                if fact:
                    _throttle()
                    docs = _doc_embedder().run(documents=[Document(content=fact)])["documents"]
                    store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
                    if fact not in all_facts:
                        all_facts.append(fact)
                log_mem(f"progress=[{mem.progress}]  plan=[{mem.plan}]  dead_ends={mem.dead_ends[-5:]}")

                # 3. Recall COMPLEMENTARY facts
                facts = []
                if store.count_documents() > 0:
                    _throttle()
                    q = _text_embedder().run(text=f"{goal}\n{mem.plan}\n{cur_obs}")["embedding"]
                    facts = [d.content for d in retriever.run(query_embedding=q, top_k=TOP_K_MEMORY)["documents"]]
                log_recall(facts)

                # 4. Reason — traced with per-attempt llm.call/call_end + RPM retry
                offered = [
                    a
                    for a in admissible
                    if a not in mem.dead_ends and not (a.startswith(INFO_ACTIONS) and a in mem.taken)
                ]
                offered = offered or [a for a in admissible if a not in mem.dead_ends] or admissible
                action = _reason_traced(
                    goal,
                    mem,
                    facts,
                    offered,
                    playbook=self.playbook,
                    common_sense=common_sense,
                    ep_handle=ep_handle,
                    step_num=steps + 1,
                )
                log_action(action)

                # 5. Step + auto dead-end detection
                obs2, reward, done, info = env.step(action)
                obs2, reward, done = obs2[0], reward[0], done[0]
                if is_noop(action, cur_obs, obs2):
                    mem.add_dead_end(action)
                mem.taken.append(action)
                mem.last_action = action
                traj += [f"> {action}", obs2]

                # [TRACE 1] per-step event: observation + reasoning context + action
                if ep_handle:
                    nemo_relay.scope.event(
                        "agent_step",
                        handle=ep_handle,
                        data={
                            "step": steps + 1,
                            "observation": cur_obs,
                            "action_taken": action,
                            "progress": mem.progress,
                            "plan": mem.plan,
                            "recalled_facts": facts,
                            "dead_ends": mem.dead_ends[-8:],
                            "result_obs": obs2[:400],
                            "is_done": bool(done),
                        },
                    )

                cur_obs = obs2
                steps += 1
                if done:
                    break

        ret = (GAMMA ** (steps - 1)) if reward else 0.0
        log_result(bool(reward), steps, ret)
        return dspy.Prediction(
            goal=goal,
            trajectory="\n".join(traj),
            success=reward,
            steps=steps,
            facts_list=all_facts,
            dead_ends_list=list(mem.dead_ends),
        )


# --------------------------------------------------------------------------- #
# run_episode_traced                                                            #
# Wraps each training episode in a NeMo Relay agent scope. Adds:               #
#   - per-episode ATIF exporter (register before scope, export after)          #
#   - cache_lookup, playbook_injected, episode_outcome events                  #
#   - reflector llm.call/call_end with 429/RPM retry                           #
#   - reflection_summary event  [TRACE 2: state + steps + next best action]    #
#   - playbook_updated event                                                    #
# --------------------------------------------------------------------------- #
def run_episode_traced(agent: TracedMemoryAgent, pb: Playbook, ex, reflect, label: str, cache: dict, cache_key: str):
    cached = cache.get(cache_key, {})
    if cached:
        prev_goal = cached.get("goal", "")
        if prev_goal:
            print(
                f"  {Fore.CYAN}[cache hit: resuming from previous run of "
                f"'{prev_goal[:60]}' ({cached.get('steps', '?')} steps, "
                f"success={cached.get('success', '?')})]{Style.RESET_ALL}",
                flush=True,
            )
        agent.prefill_facts = cached.get("facts", [])
        timeout_fallbacks = set(cached.get("timeout_fallbacks", []))
        agent.prefill_dead_ends = [de for de in cached.get("dead_ends", []) if de not in timeout_fallbacks]
    else:
        agent.prefill_facts = []
        agent.prefill_dead_ends = []

    # ATIF: register before scope opens so scope open event is captured
    atif = _atif_register(label)

    with nemo_relay.scope.scope(f"episode-{label}", nemo_relay.ScopeType.Agent) as ep_handle:
        nemo_relay.scope.event(
            "cache_lookup",
            handle=ep_handle,
            data={
                "cache_key": cache_key,
                "hit": bool(cached),
                "prefill_facts_count": len(agent.prefill_facts),
                "prefill_dead_ends_count": len(agent.prefill_dead_ends),
            },
        )

        agent.playbook = pb.render()
        nemo_relay.scope.event(
            "playbook_injected",
            handle=ep_handle,
            data={
                "bullet_count": len(pb.items),
                "playbook": pb.render(k=PLAYBOOK_CAP),
            },
        )

        agent._ep_handle = ep_handle
        pred = agent(**ex.inputs())
        agent._ep_handle = None

        outcome = f"SOLVED in {pred.steps} steps" if pred.success else f"FAILED at {pred.steps} steps"

        nemo_relay.scope.event(
            "episode_outcome",
            handle=ep_handle,
            data={
                "label": label,
                "goal": pred.goal,
                "outcome": outcome,
                "steps": pred.steps,
                "success": bool(pred.success),
                "trajectory_tail": pred.trajectory[-2000:],
            },
        )

        # Persist cache
        cache[cache_key] = {
            "goal": pred.goal,
            "facts": getattr(pred, "facts_list", []),
            "dead_ends": getattr(pred, "dead_ends_list", []),
            "timeout_fallbacks": [],
            "success": bool(pred.success),
            "steps": pred.steps,
        }
        _save_cache(cache)

        # --- Reflector: llm.call/call_end per attempt + 429/RPM retry --------
        _throttle()
        ref_content_template = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"task={pred.goal}\noutcome={outcome}"
                        f"\ntrajectory={pred.trajectory[-3000:]}"
                        f"\nplaybook={pb.render_meta() or '(empty)'}"
                    ),
                }
            ],
            "model": SUMMARIZER_MODEL,
        }

        wait = _RPM_WAIT_BASE
        refl = None

        for attempt in range(_RPM_MAX_RETRIES):
            ref_req = nemo_relay.LLMRequest({}, ref_content_template)
            ref_lh = nemo_relay.llm.call(
                "reflector",
                ref_req,
                handle=ep_handle,
                model_name=SUMMARIZER_MODEL,
                data={"attempt": attempt},
            )
            try:
                with dspy.context(lm=reflection_lm):
                    refl = reflect(
                        task=pred.goal,
                        outcome=outcome,
                        trajectory=pred.trajectory[-3000:],
                        current_playbook=pb.render_meta() or "(empty)",
                    )
                nemo_relay.llm.call_end(
                    ref_lh,
                    {
                        "helpful_ids": refl.helpful_ids,
                        "harmful_ids": refl.harmful_ids,
                        "new_strategies": refl.new_strategies,
                    },
                )
                break  # success

            except Exception as e:
                nemo_relay.llm.call_end(ref_lh, {"error": str(e)[:300]})
                if _is_rpm_error(e) and attempt < _RPM_MAX_RETRIES - 1:
                    # [TRACE 3] RPM hit on reflector
                    nemo_relay.scope.event(
                        "llm_rate_limit_hit",
                        handle=ep_handle,
                        data={
                            "provider": "reflector",
                            "model": SUMMARIZER_MODEL,
                            "attempt": attempt + 1,
                            "wait_seconds": wait,
                            "error": str(e)[:300],
                        },
                    )
                    print(
                        f"  {Fore.YELLOW}[relay] 429/RPM on reflector, "
                        f"waiting {wait:.0f}s (attempt {attempt + 1}/"
                        f"{_RPM_MAX_RETRIES}){Style.RESET_ALL}",
                        flush=True,
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, _RPM_WAIT_CAP)
                else:
                    raise

        if refl is None:
            # All retries failed — skip curation, still export trajectory
            _atif_export(atif, label)
            return pred

        # [TRACE 2] reflection_summary: game state + steps taken + next best action
        pb_size_before = len(pb.items)
        obs_lines = [line for line in pred.trajectory.splitlines() if not line.startswith(">")]
        last_obs = obs_lines[-1][:200] if obs_lines else ""

        nemo_relay.scope.event(
            "reflection_summary",
            handle=ep_handle,
            data={
                "goal": pred.goal,
                "outcome": outcome,
                "steps_taken": pred.steps,
                "last_observation": last_obs,
                "helpful_strategy_ids": refl.helpful_ids,
                "harmful_strategy_ids": refl.harmful_ids,
                "next_best_strategies": refl.new_strategies,
                "playbook_size_before": pb_size_before,
                "current_playbook": pb.render_meta(),
            },
        )

        curate(pb, refl)

        nemo_relay.scope.event(
            "playbook_updated",
            handle=ep_handle,
            data={
                "playbook_size_after": len(pb.items),
                "bullets_delta": len(pb.items) - pb_size_before,
                "playbook_snapshot": pb.render_meta(),
            },
        )

        print(
            f"{Fore.BLUE}{Style.BRIGHT}[{label}] {outcome}; playbook now {len(pb.items)} bullets{Style.RESET_ALL}",
            flush=True,
        )
        print(f"{Fore.BLUE}{pb.render_meta()}{Style.RESET_ALL}", flush=True)

    # Scope closed — export ATIF trajectory for this episode
    _atif_export(atif, label)
    return pred


# --------------------------------------------------------------------------- #
# Main — phase-level scopes + per-eval episode traces + final summary event    #
# --------------------------------------------------------------------------- #
def main():
    from dspy.datasets.alfworld import AlfWorld

    r2.alfworld = AlfWorld(max_threads=r2.NUM_THREADS)

    reflect = dspy.Predict(Reflect)
    pb = Playbook()
    agent = TracedMemoryAgent()
    cache = _load_cache()

    smoke = "--smoke" in sys.argv
    n_train = 2 if smoke else N_TRAIN
    n_test = 1 if smoke else N_TEST

    # --- training phase -------------------------------------------------------
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}===== RUNG3-RELAY ACE: train {n_train} episodes ====={Style.RESET_ALL}",
        flush=True,
    )
    t0 = time.time()

    with nemo_relay.scope.scope("ace-train", nemo_relay.ScopeType.Agent) as train_handle:
        nemo_relay.scope.event(
            "phase_start",
            handle=train_handle,
            data={
                "phase": "train",
                "n_episodes": n_train,
                "reasoner_model": REASONER_MODEL,
                "summarizer_model": SUMMARIZER_MODEL,
            },
        )
        for i in range(n_train):
            print(
                f"\n{Fore.CYAN}{Style.BRIGHT}--- train episode {i + 1}/{n_train} (trainset[{i}]) ---{Style.RESET_ALL}",
                flush=True,
            )
            run_episode_traced(
                agent,
                pb,
                r2.alfworld.trainset[i],
                reflect,
                f"train{i + 1}",
                cache,
                f"train_{i}",
            )
        nemo_relay.scope.event(
            "phase_end",
            handle=train_handle,
            data={
                "phase": "train",
                "playbook_size": len(pb.items),
                "playbook": pb.render_meta(),
            },
        )

    # --- eval phase -----------------------------------------------------------
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}===== RUNG3-RELAY ACE: eval {n_test} task(s) with "
        f"learned playbook ====={Style.RESET_ALL}",
        flush=True,
    )
    agent.playbook = pb.render()
    succ, steps_solved, rets, records = [], [], [], []
    test_idxs = [2] if smoke else list(range(n_test))

    with nemo_relay.scope.scope("ace-eval", nemo_relay.ScopeType.Agent) as eval_handle:
        nemo_relay.scope.event(
            "phase_start",
            handle=eval_handle,
            data={
                "phase": "eval",
                "n_episodes": len(test_idxs),
                "frozen_playbook_size": len(pb.items),
                "frozen_playbook": pb.render_meta(),
            },
        )

        for i in test_idxs:
            print(
                f"\n{Fore.CYAN}{Style.BRIGHT}--- eval devset[{i}] ---{Style.RESET_ALL}",
                flush=True,
            )
            ep_label = f"eval-{i}"
            cache_key = f"eval_{i}"
            cached = cache.get(cache_key, {})
            agent.prefill_facts = cached.get("facts", [])
            timeout_fallbacks = set(cached.get("timeout_fallbacks", []))
            agent.prefill_dead_ends = [de for de in cached.get("dead_ends", []) if de not in timeout_fallbacks]

            # ATIF: register before scope opens
            atif = _atif_register(ep_label)

            with nemo_relay.scope.scope(f"episode-{ep_label}", nemo_relay.ScopeType.Agent) as ep_handle:
                nemo_relay.scope.event(
                    "cache_lookup",
                    handle=ep_handle,
                    data={
                        "cache_key": cache_key,
                        "hit": bool(cached),
                        "prefill_facts_count": len(agent.prefill_facts),
                        "prefill_dead_ends_count": len(agent.prefill_dead_ends),
                    },
                )
                nemo_relay.scope.event(
                    "playbook_injected",
                    handle=ep_handle,
                    data={
                        "bullet_count": len(pb.items),
                        "playbook": pb.render(k=PLAYBOOK_CAP),
                    },
                )

                agent._ep_handle = ep_handle
                pred = agent(**r2.alfworld.devset[i].inputs())
                agent._ep_handle = None

                outcome = f"SOLVED in {pred.steps} steps" if pred.success else f"FAILED at {pred.steps} steps"
                nemo_relay.scope.event(
                    "episode_outcome",
                    handle=ep_handle,
                    data={
                        "label": ep_label,
                        "goal": pred.goal,
                        "outcome": outcome,
                        "steps": pred.steps,
                        "success": bool(pred.success),
                        "trajectory_tail": pred.trajectory[-2000:],
                    },
                )

            # Scope closed — export ATIF
            _atif_export(atif, ep_label)

            cache[cache_key] = {
                "goal": pred.goal,
                "facts": getattr(pred, "facts_list", []),
                "dead_ends": getattr(pred, "dead_ends_list", []),
                "timeout_fallbacks": [],
                "success": bool(pred.success),
                "steps": pred.steps,
            }
            _save_cache(cache)

            ret = (GAMMA ** (pred.steps - 1)) if pred.success else 0.0
            rets.append(ret)
            succ.append(1.0 if pred.success else 0.0)
            if pred.success:
                steps_solved.append(pred.steps)
            records.append(
                {
                    "devset": i,
                    "goal": pred.goal,
                    "success": int(bool(pred.success)),
                    "steps": pred.steps,
                    "return": round(ret, 4),
                }
            )

        avg_ret = sum(rets) / len(rets)
        succ_pct = 100.0 * sum(succ) / len(succ)
        avg_steps = (sum(steps_solved) / len(steps_solved)) if steps_solved else float("nan")

        # [TRACE 2] final eval summary
        nemo_relay.scope.event(
            "eval_complete",
            handle=eval_handle,
            data={
                "success_pct": succ_pct,
                "avg_return": avg_ret,
                "avg_steps": avg_steps,
                "n_test": len(test_idxs),
                "playbook_final_size": len(pb.items),
                "playbook_final": pb.render_meta(),
            },
        )

    with open("results_rung3_relay.json", "w") as f:
        json.dump(
            {
                "label": "rung3-ace-relay",
                "gamma": GAMMA,
                "playbook": pb.render_meta(),
                "summary": {
                    "return": avg_ret,
                    "success": succ_pct,
                    "avg_steps": avg_steps,
                },
                "games": records,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}\n  RUNG3 ACE (relay-traced)  (playbook = {len(pb.items)} bullets)\n{'=' * 60}")
    print(f"  {'':<10}{'return':>10}{'success%':>11}{'avg_steps':>11}")
    print(f"  {'baseline':<10}{0.485:>10.3f}{60.0:>10.1f}%{12.2:>11.1f}   (from log)")
    print(f"  {'rung3':<10}{avg_ret:>10.3f}{succ_pct:>10.1f}%{avg_steps:>11.1f}")
    print(
        f"{'=' * 60}  ({time.time() - t0:.0f}s)  -> results_rung3_relay.json\n"
        f"  traces/atif/   traces/atof/events.jsonl",
        flush=True,
    )

    _atof.force_flush()
    _atof.shutdown()


if __name__ == "__main__":
    main()
