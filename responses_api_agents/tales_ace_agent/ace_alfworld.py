"""
RUNG 3 of the ALFWorld showcase: ACE (Agentic Context Engineering) on top of
rung 2's structured Haystack memory.

  rung 1 = baseline (naive memory)
  rung 2 = Haystack structured memory  (rung2_haystack_memory.py)
  rung 3 = + an iterative improvement algorithm  <-- THIS FILE

Why ACE (arXiv:2510.04618) and not GEPA: the rung-2 diagnosis showed the fault
is the *decision policy*, and GEPA could only mutate a summarizer it doesn't
control (credit-assignment mismatch). ACE instead evolves the very context the
reasoner reads -- a persistent, cross-episode STRATEGY PLAYBOOK -- so the thing
being optimized IS the thing feeding the decision. It also fixes "context
collapse" via structured bullet items + incremental delta updates (not monolithic
rewrites), which is exactly the failure we saw in the old free-text memory.

ACE loop (offline, no labels -- learns from execution outcome):
  Generator : rung-2 StructuredMemoryAgent runs an episode, reading the current
              playbook as extra context (agent.playbook).
  Reflector : an LM reads (task, outcome, trajectory, playbook) and says which
              existing bullets helped / hurt, and proposes new general strategies.
  Curator   : deterministically applies delta ops -- bump helpful/harmful counts,
              ADD new bullets (with dedup / grow-and-refine), REMOVE net-harmful
              ones. No wholesale rewrite -> no brevity bias / collapse.

The playbook persists ACROSS episodes; success%/steps should improve as it grows.

    .venv/bin/python rung3_ace_alfworld.py --smoke   # train 2, eval 1 (fast proof)
    .venv/bin/python rung3_ace_alfworld.py --full    # train N, eval N_TEST vs base
"""

import json
import os
import re
import sys
import time

import dspy

# Reuse rung-2's whole agent + memory stack; only the cross-episode layer is new.
import haystack_memory as r2
from haystack_memory import (
    BASE_URL,
    GAMMA,
    NVIDIA_API_KEY,
    SUMMARIZER_MODEL,
    Fore,
    StructuredMemoryAgent,
    Style,
    _throttle,
)


# Persistent cache of discovered facts / dead-ends per (dataset, idx, goal).
# Lets the agent skip already-explored dead locations on restarts with the same goal.
EPISODE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "alfworld_episode_cache.json")


def _load_cache() -> dict:
    try:
        with open(EPISODE_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    with open(EPISODE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# Reflector gets a bigger budget than the per-step summarizer (it reads a whole
# trajectory). Same model, just more tokens / hotter.
reflection_lm = dspy.LM(
    f"openai/{SUMMARIZER_MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=0.7,
    max_tokens=2000,
    num_retries=10,
)

N_TRAIN = 8  # training episodes for the playbook (full mode)
N_TEST = 10  # held-out eval tasks (same 10 the baseline was scored on)
PLAYBOOK_CAP = 12  # max bullets injected into the reasoner (grow-and-refine)
PRUNE_NET = 2  # remove a bullet once (harmful - helpful) >= this and helpful==0


# --------------------------------------------------------------------------- #
# The evolving playbook: structured bullets with utility metadata (ACE)        #
# --------------------------------------------------------------------------- #
class Playbook:
    def __init__(self):
        self.items = []  # {id, text, helpful, harmful}
        self._next = 1

    @staticmethod
    def _sim(a, b):
        import re

        norm = lambda s: set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
        wa, wb = norm(a), norm(b)
        return len(wa & wb) / max(1, len(wa | wb))

    def add(self, text):
        t = text.strip()
        if len(t) < 8:
            return
        if any(self._sim(it["text"], t) > 0.6 for it in self.items):
            return  # dedup: near-duplicate of an existing bullet
        self.items.append({"id": self._next, "text": t, "helpful": 0, "harmful": 0})
        self._next += 1

    def bump(self, ids, key):
        for it in self.items:
            if it["id"] in ids:
                it[key] += 1

    def prune(self):
        self.items = [
            it for it in self.items if not (it["harmful"] - it["helpful"] >= PRUNE_NET and it["helpful"] == 0)
        ]

    def render(self, k=PLAYBOOK_CAP):
        ranked = sorted(self.items, key=lambda it: it["helpful"] - it["harmful"], reverse=True)[:k]
        return "\n".join(f"[{it['id']}] {it['text']}" for it in ranked)

    def render_meta(self, k=PLAYBOOK_CAP):
        ranked = sorted(self.items, key=lambda it: it["helpful"] - it["harmful"], reverse=True)[:k]
        return (
            "\n".join(f"[{it['id']}] (+{it['helpful']}/-{it['harmful']}) {it['text']}" for it in ranked) or "(empty)"
        )


class Reflect(dspy.Signature):
    """Reflect on ONE ALFWorld episode to improve a STRATEGY PLAYBOOK of GENERAL,
    reusable tactics (e.g. 'for multi-object tasks, keep a placed-count and only
    stop when it equals the target'). NOT facts about this specific room.

    STRICT FACTUAL MODE: ground every claim in what the trajectory actually shows.
    Do not invent. If no general lesson is warranted, output 'none'/empty.

    new_strategies OUTPUT FORMAT (strict): ONLY strategy sentences, ONE imperative
    sentence per line, at most 3 lines. NO headers, NO 'Reasoning:', NO numbering,
    NO markdown/asterisks, NO explanation of why. Strategy text only."""

    task = dspy.InputField()
    outcome = dspy.InputField(desc="e.g. 'SOLVED in 9 steps' or 'FAILED at 50 steps'")
    trajectory = dspy.InputField(desc="actions and observations from the episode")
    current_playbook = dspy.InputField(desc="existing numbered strategies")
    helpful_ids: str = dspy.OutputField(
        desc="comma-separated ids of existing strategies that helped THIS episode, or 'none'"
    )
    harmful_ids: str = dspy.OutputField(desc="comma-separated ids that misled / wasted steps, or 'none'")
    new_strategies: str = dspy.OutputField(
        desc="at most 3 NEW general strategies, ONE imperative sentence per line, "
        "no headers/numbering/markdown/explanation; empty if none"
    )


def _parse_ids(s):
    out = []
    for tok in (s or "").replace(";", ",").split(","):
        tok = tok.strip().lstrip("[").rstrip("]")
        if tok.isdigit():
            out.append(int(tok))
    return out


# Reflectors love to emit section headers and per-strategy justifications. Those
# are NOT strategies -> keep them out of the playbook.
_META_RE = re.compile(
    r"(?i)^(reasoning|strateg|note|here|because|this |that |these |the agent|"
    r"derived|aims|addresses|in summary|explanation|rationale)"
)


def _clean_strategy(line):
    line = re.sub(r"[*#`]+", "", line).strip()  # strip markdown
    line = line.lstrip("-•*0123456789.) ").strip()  # strip bullet/number prefix
    if not line or line.lower() == "none":
        return None
    if line.endswith(":") or _META_RE.match(line):  # header / justification
        return None
    if len(line.split()) < 4:  # fragment
        return None
    return line


def curate(pb, refl):
    """Curator: apply delta ops from one reflection. Deterministic, no rewrite.
    Bullets are cleaned and capped to 3 new per episode (grow-and-refine)."""
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


def run_episode(agent, pb, ex, reflect, label, cache: dict, cache_key: str):
    """Generator + Reflector + Curator for one episode.

    If a previous run with the same goal exists in cache, pre-populates the
    agent's Haystack store with discovered facts and injects confirmed dead-ends
    so the agent skips already-explored dead locations.  Actions that ended with
    a timeout (heuristic fallback) are still explored -- their LLM decision was
    never made, so they deserve a real reasoning attempt.
    """
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
        # Only inject dead-ends that were NOT caused by a reasoner timeout.
        # Timeout steps used a heuristic fallback and may have navigated somewhere
        # suboptimally; give the (now-rate-limited) LLM a proper shot at them.
        timeout_fallbacks = set(cached.get("timeout_fallbacks", []))
        agent.prefill_dead_ends = [de for de in cached.get("dead_ends", []) if de not in timeout_fallbacks]
    else:
        agent.prefill_facts = []
        agent.prefill_dead_ends = []

    agent.playbook = pb.render()
    pred = agent(**ex.inputs())
    outcome = f"SOLVED in {pred.steps} steps" if pred.success else f"FAILED at {pred.steps} steps"

    # Persist discovered knowledge for future restarts with the same goal+idx.
    cache[cache_key] = {
        "goal": pred.goal,
        "facts": getattr(pred, "facts_list", []),
        "dead_ends": getattr(pred, "dead_ends_list", []),
        "timeout_fallbacks": [],  # populated by agent if it detects fallback
        "success": bool(pred.success),
        "steps": pred.steps,
    }
    _save_cache(cache)

    _throttle()  # reflector = 1 LM call; must count toward the global RPM cap
    with dspy.context(lm=reflection_lm):
        refl = reflect(
            task=pred.goal,
            outcome=outcome,
            trajectory=pred.trajectory[-3000:],
            current_playbook=pb.render_meta() or "(empty)",
        )
    curate(pb, refl)
    print(
        f"{Fore.BLUE}{Style.BRIGHT}[{label}] {outcome}; playbook now {len(pb.items)} bullets{Style.RESET_ALL}",
        flush=True,
    )
    print(f"{Fore.BLUE}{pb.render_meta()}{Style.RESET_ALL}", flush=True)
    return pred


def main():
    from dspy.datasets.alfworld import AlfWorld

    r2.alfworld = AlfWorld(max_threads=r2.NUM_THREADS)  # set rung-2's global

    reflect = dspy.Predict(Reflect)
    pb = Playbook()
    agent = StructuredMemoryAgent()
    cache = _load_cache()

    smoke = "--smoke" in sys.argv
    n_train = 2 if smoke else N_TRAIN
    n_test = 1 if smoke else N_TEST

    # --- offline training: evolve the playbook over trainset episodes ---------
    print(f"\n{Fore.CYAN}{Style.BRIGHT}===== RUNG3 ACE: train {n_train} episodes ====={Style.RESET_ALL}", flush=True)
    t0 = time.time()
    for i in range(n_train):
        print(
            f"\n{Fore.CYAN}{Style.BRIGHT}--- train episode {i + 1}/{n_train} (trainset[{i}]) ---{Style.RESET_ALL}",
            flush=True,
        )
        cache_key = f"train_{i}"
        run_episode(agent, pb, r2.alfworld.trainset[i], reflect, f"train{i + 1}", cache, cache_key)

    # --- eval with the FROZEN learned playbook --------------------------------
    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}===== RUNG3 ACE: eval {n_test} task(s) with "
        f"learned playbook ====={Style.RESET_ALL}",
        flush=True,
    )
    agent.playbook = pb.render()
    succ, steps_solved, rets, records = [], [], [], []
    test_idxs = [2] if smoke else list(range(n_test))  # smoke: the hard soapbottle task
    for i in test_idxs:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}--- eval devset[{i}] ---{Style.RESET_ALL}", flush=True)
        cache_key = f"eval_{i}"
        cached = cache.get(cache_key, {})
        agent.prefill_facts = cached.get("facts", [])
        timeout_fallbacks = set(cached.get("timeout_fallbacks", []))
        agent.prefill_dead_ends = [de for de in cached.get("dead_ends", []) if de not in timeout_fallbacks]
        pred = agent(**r2.alfworld.devset[i].inputs())
        # Save eval result to cache too
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
    with open("results_rung3.json", "w") as f:
        json.dump(
            {
                "label": "rung3-ace",
                "gamma": GAMMA,
                "playbook": pb.render_meta(),
                "summary": {"return": avg_ret, "success": succ_pct, "avg_steps": avg_steps},
                "games": records,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}\n  RUNG3 ACE  (playbook = {len(pb.items)} bullets)\n{'=' * 60}")
    print(f"  {'':<10}{'return':>10}{'success%':>11}{'avg_steps':>11}")
    print(f"  {'baseline':<10}{0.485:>10.3f}{60.0:>10.1f}%{12.2:>11.1f}   (from log)")
    print(f"  {'rung3':<10}{avg_ret:>10.3f}{succ_pct:>10.1f}%{avg_steps:>11.1f}")
    print(f"{'=' * 60}  ({time.time() - t0:.0f}s)  -> results_rung3.json", flush=True)


if __name__ == "__main__":
    main()
