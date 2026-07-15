# Verifier iteration pain points — onboarding docs

Four self-contained deep-dives into why iterating on verifier design (reward weights, thresholds,
partial credit, judge prompts, aggregation) is slow in NeMo Gym today. Each doc follows the same
shape: **TL;DR → experience it yourself (runnable) → architecture & code flow (file:line) →
gap analysis → where a fix would hook in.** They were written for someone new to the codebase who
needs to implement the fixes.

## The docs

| # | Doc | Pain point | Issue |
|---|---|---|---|
| 1 | [01-config-iteration-feels-like-new-server.md](01-config-iteration-feels-like-new-server.md) | Verifier hyperparameters live in YAML, but every change costs a full stack restart **plus** a full re-rollout — config is snapshotted into each server process at `Popen` time and rewards are computed inside the agent's `/run`. | #1415 |
| 2 | [02-recompute-rewards-from-existing-rollouts.md](02-recompute-rewards-from-existing-rollouts.md) | `rollouts.jsonl` + `*_materialized_inputs.jsonl` already contain everything a stateless verifier needs, and `/verify` is a plain HTTP endpoint — but there is no `gym eval verify` command, so re-scoring stored rollouts is an undocumented hand-rolled script. | #987 |
| 3 | [03-no-cohesive-verifier-iteration-workflow.md](03-no-cohesive-verifier-iteration-workflow.md) | **The umbrella doc.** The six-rung cost ladder (in-process `verify()` test → `gym env test` → override + replay → `gym eval profile` → small re-collection → full re-collection) and a decision table for "what changed → which rung is enough". Start here for the broad picture. | — |
| 4 | [04-judge-only-resume.md](04-judge-only-resume.md) | A hard judge failure crashes the run and **discards the completed policy inference**; a soft one is indistinguishable from "judge said wrong". `--resume` re-pays full inference just to retry judge calls. | — |
| 5 | [05-how-peers-address-pp02-pp04.md](05-how-peers-address-pp02-pp04.md) | **Competitive analysis:** how 11 peer frameworks (verl, Agent Lightning, ART, slime, rLLM, OpenAI Evals, lm-eval-harness, OpenCompass, SWE-bench, AgentBench, Inspect AI) handle pain points 02 and 04, verified against their code at HEAD (2026-07-07). TL;DR: no RL-training peer has first-class support for either; the mature patterns (led by `inspect score`) live in the eval harnesses. | — |

**Proposed fix:** [rfc-gym-eval-verify.md](rfc-gym-eval-verify.md) — a `gym eval verify` command that
replays stored rollouts through `/verify` (addresses pain points 2 and 4).

Suggested reading order for onboarding: **3 → 1 → 2 → 4**. Doc 3 gives you the map; the others
drill into the specific gaps. If you only want the shared code-flow narrative (YAML → Hydra merge →
frozen server config → `/run` → `/verify` → `rollouts.jsonl` → `gym eval profile`), read §3
"Architecture & code flow" of any doc — doc 1 has the most detailed config half, doc 2 the most
detailed artifact half, doc 4 the judge call chain.

## Shared setup for all run examples

All demos use the hosted ECCN model config
[eccn-llama-3.1-8b.yaml](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml)
as the policy (and, in doc 4, also the judge). It is an OpenAI-compatible hosted endpoint — no
local GPU — and only needs:

```bash
cd /path/to/Gym
export INFERENCE_KEY=<your key>
```

The model is an ECCN-classifier fine-tune, so its *answers* to the demo tasks will be poor. That is
intentional and irrelevant: the pain points are about the **workflow**, and any model config can be
substituted. One wiring quirk used everywhere: shipped agent configs reference a model instance
named `policy_model`, but this YAML's top-level key is `eccn-llama-3.1-8b-instruct`, so every
`gym env start` adds an override like:

```bash
"++<agent_instance>.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct"
```

Invoke the CLI as `uv run gym ...` from the repo root. First `gym env start` per environment is
slow (builds a per-server `.venv`); later starts can pass `+skip_venv_if_present=true`.

## The five facts that explain all four pain points

1. **Config is a launch-time snapshot.** The merged config is serialized into the
   `NEMO_GYM_CONFIG_DICT` env var when each server subprocess is spawned
   ([env.py:198-201](../nemo_gym/cli/env.py#L198)) and validated once into a Pydantic object
   ([server_utils.py:368-377](../nemo_gym/server_utils.py#L368)). No hot-reload, no `/reload`.
2. **Verification is welded to inference.** The only built-in caller of `/verify` is the agent's
   `/run`, *after* the model loop
   ([simple_agent/app.py:197-206](../responses_api_agents/simple_agent/app.py#L197)).
3. **Rewards are frozen at collection time.** `gym eval profile` only re-aggregates numeric fields
   already in `rollouts.jsonl` ([reward_profile.py:229-233](../nemo_gym/reward_profile.py#L229));
   nothing downstream ever re-enters `verify()`.
4. **The verify contract is replayable for stateless verifiers.** `BaseVerifyResponse` extends
   `BaseVerifyRequest` ([base_resources_server.py:83-92](../nemo_gym/base_resources_server.py#L83)),
   so a stored rollout row (merged with its materialized-inputs twin to recover dropped fields) is a
   valid `/verify` payload — the basis of every workaround in these docs.
5. **Failures don't persist the expensive half.** Rows are written only after `/run` fully succeeds
   ([rollout_collection.py:513-539](../nemo_gym/rollout_collection.py#L513)); a verify/judge
   exception crashes the run and loses the completed inference.

These docs were generated on 2026-07-06 against `main` @ `8f5a9555`; line numbers drift as the code
moves, but every reference was verified against that commit.
