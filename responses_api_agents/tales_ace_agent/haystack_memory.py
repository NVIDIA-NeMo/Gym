"""
RUNG 2 of the ALFWorld showcase: Haystack as a *structured* agentic memory.
(rung 1 = baseline; rung 3 = add an iterative optimizer on top of this memory.)

The diagnosis from ultra_gepa_run.log: the old memory (per-step free-text
re-summary + cosine top-k) caused "context collapse" -- it recalled a STALE
"cart 1 is empty" long after a bottle had been placed, dropped the progress
count (1 of 2), and let the frozen reasoner loop ("examine drawer 3" x20+).

This rung replaces that with an ACE-style structured playbook (no optimizer yet
-- that's rung 3). Fixed prompts. The ONLY change vs baseline is the memory.
Goal: beat the baseline (60% success, avg 12.2 steps, two-object tasks FAILED at
50) by finishing in fewer steps.

Memory = deterministic state slots + a Haystack world-facts knowledge base:
  * progress  : "soapbottles placed: 1/2; target: cart 1"   <- overwritten each
                 step, so a stale fact can NEVER be recalled (kills the bug)
  * plan      : ordered remaining subgoals
  * dead_ends : actions that produced no change -> injected as "do NOT repeat"
                AND filtered out of the action enum offered to the reasoner
                (structural loop-breaker, not a hope)
  * world_facts (Haystack InMemoryDocumentStore + Nvidia embedders + retriever):
                 large recall base "where did I see object Y", retrieved by
                 (task + plan + obs) to surface COMPLEMENTARY facts, not dups

Same models / endpoint / 40 RPM handling as the baseline run, so the comparison
isolates the memory as the single variable.

    .venv/bin/python rung1_haystack_memory.py            # smoke: 1 hard game
    .venv/bin/python rung1_haystack_memory.py --task 2   # pick devset index
"""

import collections
import math
import os
import re
import sys
import threading
import time

import dspy
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv


colorama_init(autoreset=True)

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import Tool
from haystack.utils import Secret
from haystack_integrations.components.embedders.nvidia import (
    NvidiaDocumentEmbedder,
    NvidiaTextEmbedder,
)
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator


load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Same roles as the baseline run, so memory is the only changed variable.
REASONER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
SUMMARIZER_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

# MAX_ITERS: step budget per episode. 50 gives hard tasks room to explore;
#   lower (e.g. 20) for faster smoke runs, raise if tasks still time out.
# GAMMA: discount factor derived from H so that a 1-step solve scores ~1.0 and
#   an H-step solve scores ~1/e. Do not set manually -- change H instead.
H = MAX_ITERS = 50
GAMMA = math.exp(-1.0 / (H - 1))  # ~0.9798, for the same return metric as baseline

# TOP_K_MEMORY: facts returned from the Haystack vector store each step.
#   3 keeps the prompt tight and avoids redundant context on small stores.
#   Raise (e.g. 5-10) if the agent misses relevant facts in large, long episodes.
TOP_K_MEMORY = 10

# NUM_THREADS: parallel env workers. Keep at 1 unless you have multiple API keys;
#   the 40 RPM free-tier limit serialises everything anyway.
NUM_THREADS = 1

# Rate-limiting: 28 RPM cap gives safe headroom under the 40 RPM free-tier limit.
# Sliding-window limiter covers ALL NVIDIA API calls (DSPy LM + embed + reasoner).
_RPM_CAP = 28
_MIN_GAP = 60.0 / _RPM_CAP  # ~2.14 s minimum gap between consecutive calls
_rate_lock = threading.Lock()
_call_times: collections.deque = collections.deque()  # monotonic timestamps in last 60 s

# Observations that mean "that action changed nothing" -> auto dead-end.
NOOP_MARKERS = ["nothing happens", "nothing special", "already"]
# Purely informational actions: their result is deterministic, so doing one a
# SECOND time yields nothing new -> drop from the offered set once taken. This
# breaks alternating loops (examine drawer 1 / examine drawer 2 / ...) that the
# immediate-repeat + noop-marker checks miss.
INFO_ACTIONS = ("examine", "look", "inventory")

summarizer_lm = dspy.LM(
    f"openai/{SUMMARIZER_MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=0.4,
    max_tokens=600,
    num_retries=10,
)
dspy.configure(lm=summarizer_lm)

# Common-sense object placement filter: objects are never found in these locations.
# Used to de-prioritize implausible "go to" actions for the target object.
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
    """Return a one-line common-sense hint for the reasoner based on the goal."""
    g = goal.lower()
    for obj, unlikely in _UNLIKELY_LOCATIONS.items():
        if obj in g:
            return (
                f"Common sense: {obj}s are NEVER found in {', '.join(unlikely)}. Do NOT navigate to those locations."
            )
    return ""


alfworld = None
_HAYSTACK = {}


# --- color trace ----------------------------------------------------------- #
def _c(tag, text, color, dim=False):
    style = Style.DIM if dim else Style.BRIGHT
    print(f"{color}{style}{tag:<9}{Style.RESET_ALL}{color}{text}{Style.RESET_ALL}", flush=True)


def log_goal(g):
    _c("GOAL", g, Fore.CYAN)


def log_step(n):
    print(f"{Fore.WHITE}{Style.DIM}{'-' * 70}  step {n}{Style.RESET_ALL}", flush=True)


def log_obs(o):
    _c("OBS", o, Fore.WHITE, dim=True)


def log_mem(m):
    _c("MEMORY", m, Fore.YELLOW)


def log_recall(f):
    _c("RECALL", "  |  ".join(f) if f else "(none yet)", Fore.MAGENTA)


def log_action(a):
    _c("ACTION", a, Fore.GREEN)


def log_result(success, steps, ret):
    color = Fore.GREEN if success else Fore.RED
    word = "SOLVED" if success else "FAILED"
    print(
        f"{color}{Style.BRIGHT}{word}{Style.RESET_ALL}{color} in {steps} steps (return={ret:.3f}){Style.RESET_ALL}",
        flush=True,
    )


# --- rate limit + Haystack singletons -------------------------------------- #
def _throttle():
    """Sliding-window rate limiter. Call before EVERY NVIDIA API request
    (DSPy LM, doc-embed, text-embed, Haystack reasoner) to stay under 28 RPM."""
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


def _reasoner():
    if "gen" not in _HAYSTACK:
        _HAYSTACK["gen"] = NvidiaChatGenerator(
            model=REASONER_MODEL,
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            api_base_url=BASE_URL,
            max_retries=8,
        )
    return _HAYSTACK["gen"]


def _doc_embedder():
    if "doc" not in _HAYSTACK:
        e = NvidiaDocumentEmbedder(model=EMBED_MODEL, api_key=Secret.from_env_var("NVIDIA_API_KEY"), api_url=BASE_URL)
        e.warm_up()
        _HAYSTACK["doc"] = e
    return _HAYSTACK["doc"]


def _text_embedder():
    if "txt" not in _HAYSTACK:
        e = NvidiaTextEmbedder(model=EMBED_MODEL, api_key=Secret.from_env_var("NVIDIA_API_KEY"), api_url=BASE_URL)
        e.warm_up()
        _HAYSTACK["txt"] = e
    return _HAYSTACK["txt"]


def parse_goal(task):
    return task.split("Your task is to:")[-1].strip() if "Your task is to:" in task else task.strip()


def make_action_tool(admissible):
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


# --- structured memory ----------------------------------------------------- #
class MemoryState:
    def __init__(self):
        self.progress = "nothing done yet"
        self.plan = "explore to locate the target object(s)"
        self.dead_ends = []  # ordered, de-duplicated
        self.taken = []  # every action taken this episode
        self.last_action = None

    def add_dead_end(self, action):
        if action and action not in self.dead_ends:
            self.dead_ends.append(action)


class MemoryUpdate(dspy.Signature):
    """Maintain a COMPACT running memory for a text household task (ALFWorld).

    STRICT FACTUAL MODE -- NON-NEGOTIABLE:
    - NO FABRICATION. Record ONLY objects, locations, and outcomes that appear
      VERBATIM in THIS observation. Never invent, infer, or assume an object's
      presence, ID, or location.
    - If the observation does not name an object, you do NOT know where it is.
      Do NOT claim an object was "found" anywhere unless this observation lists
      it there.
    - When uncertain, output 'none' for new_fact. Saying "I don't know yet" is
      correct, not a failure.
    - progress states ONLY confirmed counts and what is currently held/placed
      (e.g. 'placed 1/2, holding none'); it must NOT assert any object's location.
    Always keep an explicit count for multi-object tasks (e.g. 'placed 1/2')."""

    task = dspy.InputField()
    last_action = dspy.InputField()
    observation = dspy.InputField()
    current_progress = dspy.InputField()
    new_fact: str = dspy.OutputField(
        desc="ONE world fact stated verbatim from THIS observation "
        "(e.g. 'drawer 3: empty', 'soapbottle 1 on countertop 1'); every "
        "object named here MUST appear in the observation, else output 'none'"
    )
    dead_end: str = dspy.OutputField(
        desc="copy the exact last_action string if it made NO progress and must "
        "not be repeated; otherwise EXACTLY the word 'none'"
    )
    progress: str = dspy.OutputField(desc="confirmed counts + what's held/placed only; NO location claims")
    plan: str = dspy.OutputField(desc="next concrete subgoals, ordered, terse")


REASON_SYS = (
    "You are an expert agent playing ALFWorld, a text household game. Finish the "
    "task in AS FEW STEPS AS POSSIBLE. Use the Progress and Plan to act with "
    "intent; never repeat an action listed as useless. Call take_action with "
    "exactly ONE action from the admissible list, preferring the one that most "
    "directly advances the Plan."
)


def ground_fact(fact, observation):
    """Deterministic anti-hallucination guardrail (no extra LM call -> no RPM
    cost). ALFWorld names objects as '<noun> <id>' (e.g. 'soapbottle 2'). A fact
    is kept only if EVERY object token it mentions also appears in THIS
    observation; otherwise it is a fabrication (e.g. 'soapbottle 2 at bathtub'
    when the obs only shows 'dishsponge 3') and is dropped."""
    if not fact or fact.strip().lower() == "none":
        return None
    norm = re.sub(r"\s+", "", observation.lower())
    objs = re.findall(r"[a-z]+ \d+", fact.lower())
    if objs and not all(re.sub(r"\s+", "", o) in norm for o in objs):
        return None  # names an object not in the observation -> hallucinated
    return fact.strip()


def is_noop(action, before, after):
    if any(m in after.lower() for m in NOOP_MARKERS):
        return True
    # identical observation = stuck, except legitimate re-navigation
    return after.strip() == before.strip() and not action.startswith("go to")


def _reason(goal, mem, facts, offered, playbook="", common_sense=""):
    tool = make_action_tool(offered)
    user = ""
    if playbook:
        user += "Learned strategies (apply when relevant):\n" + playbook + "\n\n"
    if common_sense:
        user += common_sense + "\n\n"
    user += f"Task: {goal}\nProgress: {mem.progress}\nPlan: {mem.plan}\n"
    if mem.dead_ends:
        user += "Do NOT repeat these useless actions: " + ", ".join(mem.dead_ends[-8:]) + "\n"
    if facts:
        user += "Relevant facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n"
    user += "Admissible actions:\n" + "\n".join(f"  - {a}" for a in offered)

    for attempt in range(3):
        _throttle()
        try:
            out = _reasoner().run(
                messages=[ChatMessage.from_system(REASON_SYS), ChatMessage.from_user(user)],
                tools=[tool],
                generation_kwargs={
                    "tool_choice": "required",
                    "extra_body": {"chat_template_kwargs": {"thinking": True}},
                    "max_tokens": 6000,
                    "temperature": 0.6,
                },
            )
            reply = out["replies"][0]
            if reply.tool_calls:
                a = reply.tool_calls[0].arguments.get("action")
                if a in offered:
                    return a
            for a in offered:
                if a in (reply.text or ""):
                    return a
            return offered[0]  # model returned nothing parseable but didn't throw
        except Exception as e:  # noqa: BLE001
            print(f"  {Fore.RED}[reason error (attempt {attempt + 1}/3): {e}]{Style.RESET_ALL}", flush=True)
            if attempt < 2:
                backoff = 15 * (attempt + 1)
                print(f"  {Fore.YELLOW}[backing off {backoff}s before retry]{Style.RESET_ALL}", flush=True)
                time.sleep(backoff)

    # fallback after 3 failures: prefer nav/open actions not yet taken
    not_taken = [a for a in offered if a not in mem.taken]
    explore_new = [a for a in not_taken if a.startswith(("go to", "open"))]
    explore_any = [a for a in offered if a.startswith(("go to", "open"))]
    return (explore_new or explore_any or not_taken or offered)[0]


# --------------------------------------------------------------------------- #
class StructuredMemoryAgent(dspy.Module):
    def __init__(self, max_iters=MAX_ITERS):
        super().__init__()
        self.max_iters = max_iters
        self.update = dspy.Predict(MemoryUpdate)
        self.playbook = ""  # rung3 (ACE) sets this to the evolving strategy text
        self.prefill_facts = []  # rung3: facts discovered in a previous run (same goal+idx)
        self.prefill_dead_ends = []  # rung3: confirmed dead-ends from a previous run

    def forward(self, idx):
        store = InMemoryDocumentStore()
        retriever = InMemoryEmbeddingRetriever(document_store=store)
        mem = MemoryState()
        reward, steps = 0, 0
        all_facts: list[str] = []  # every grounded fact stored this episode (for caching)

        # Pre-populate knowledge from a previous run with the same goal+idx.
        # Batch-embed to save API calls; prefill_dead_ends get injected without game steps.
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

        with alfworld.POOL.session() as env:
            traj = []
            task, info = env.init(idx)
            goal = parse_goal(task)
            common_sense = _common_sense_hint(goal)
            log_goal(goal)
            cur_obs = "(game start)"

            for _ in range(self.max_iters):
                admissible = info["admissible_commands"][0]
                log_step(steps + 1)
                log_obs(cur_obs)

                # 1. DELTA-update the structured memory (throttled: counts as 1 API call)
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

                # 2. store a grounded world fact
                raw_fact = (upd.new_fact or "").strip().splitlines()[0].strip()
                fact = ground_fact(raw_fact, cur_obs)
                if fact:
                    _throttle()
                    docs = _doc_embedder().run(documents=[Document(content=fact)])["documents"]
                    store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
                    if fact not in all_facts:
                        all_facts.append(fact)
                log_mem(f"progress=[{mem.progress}]  plan=[{mem.plan}]  dead_ends={mem.dead_ends[-5:]}")

                # 3. recall COMPLEMENTARY facts
                facts = []
                if store.count_documents() > 0:
                    _throttle()
                    q = _text_embedder().run(text=f"{goal}\n{mem.plan}\n{cur_obs}")["embedding"]
                    facts = [d.content for d in retriever.run(query_embedding=q, top_k=TOP_K_MEMORY)["documents"]]
                log_recall(facts)

                # 4. reason over a loop-broken action set
                offered = [
                    a
                    for a in admissible
                    if a not in mem.dead_ends and not (a.startswith(INFO_ACTIONS) and a in mem.taken)
                ]
                offered = offered or [a for a in admissible if a not in mem.dead_ends] or admissible
                action = _reason(goal, mem, facts, offered, playbook=self.playbook, common_sense=common_sense)
                log_action(action)

                # 5. step + auto dead-end detection
                obs2, reward, done, info = env.step(action)
                obs2, reward, done = obs2[0], reward[0], done[0]
                if is_noop(action, cur_obs, obs2):
                    mem.add_dead_end(action)
                mem.taken.append(action)
                mem.last_action = action
                traj += [f"> {action}", obs2]
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


N_TEST = 10  # full run = same 10 devset tasks the baseline was scored on


def main():
    global alfworld
    import json

    from dspy.datasets.alfworld import AlfWorld

    alfworld = AlfWorld(max_threads=NUM_THREADS)

    # --- full run: all N_TEST tasks, baseline-comparable numbers --------------
    if "--full" in sys.argv:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}===== RUNG2 FULL RUN: devset[:{N_TEST}] ====={Style.RESET_ALL}", flush=True)
        t0 = time.time()
        succ, steps_solved, rets, records = [], [], [], []
        for i in range(N_TEST):
            print(
                f"\n{Fore.CYAN}{Style.BRIGHT}--- task {i + 1}/{N_TEST} (devset[{i}]) ---{Style.RESET_ALL}", flush=True
            )
            pred = StructuredMemoryAgent()(**alfworld.devset[i].inputs())
            ret = (GAMMA ** (pred.steps - 1)) if pred.success else 0.0
            rets.append(ret)
            succ.append(1.0 if pred.success else 0.0)
            if pred.success:
                steps_solved.append(pred.steps)
            records.append(
                {
                    "task": i + 1,
                    "goal": pred.goal,
                    "success": int(bool(pred.success)),
                    "steps": pred.steps,
                    "return": round(ret, 4),
                    "trajectory": pred.trajectory,
                }
            )
        avg_ret = sum(rets) / len(rets)
        succ_pct = 100.0 * sum(succ) / len(succ)
        avg_steps = (sum(steps_solved) / len(steps_solved)) if steps_solved else float("nan")
        with open("results_rung2.json", "w") as f:
            json.dump(
                {
                    "label": "rung2",
                    "gamma": GAMMA,
                    "summary": {"return": avg_ret, "success": succ_pct, "avg_steps": avg_steps},
                    "games": records,
                },
                f,
                indent=2,
            )
        print(f"\n{'=' * 56}\n  RUNG2 (Haystack structured memory) vs baseline\n{'=' * 56}")
        print(f"  {'':<10}{'return':>10}{'success%':>11}{'avg_steps':>11}")
        print(f"  {'baseline':<10}{0.485:>10.3f}{60.0:>10.1f}%{12.2:>11.1f}   (from log)")
        print(f"  {'rung2':<10}{avg_ret:>10.3f}{succ_pct:>10.1f}%{avg_steps:>11.1f}")
        print(f"{'=' * 56}  ({time.time() - t0:.0f}s)  -> results_rung2.json", flush=True)
        return

    # --- smoke: single task (default devset[2], the baseline examine-loop fail)
    idx = 2
    if "--task" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--task") + 1])

    print(f"\n{Fore.CYAN}{Style.BRIGHT}===== RUNG2 smoke: devset[{idx}] ====={Style.RESET_ALL}", flush=True)
    t0 = time.time()
    ex = alfworld.devset[idx]
    pred = StructuredMemoryAgent()(**ex.inputs())
    print(
        f"\n{Fore.CYAN}devset[{idx}]  success={int(bool(pred.success))}  "
        f"steps={pred.steps}  ({time.time() - t0:.0f}s){Style.RESET_ALL}",
        flush=True,
    )


if __name__ == "__main__":
    main()
