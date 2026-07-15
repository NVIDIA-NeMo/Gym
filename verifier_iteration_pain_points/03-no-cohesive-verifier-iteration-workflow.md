# Pain point 3: Verifier iteration patterns (tests, config overrides, rollout reuse) are not documented as a cohesive workflow

> This is the umbrella doc for the verifier-iteration pain points. The sibling docs drill into
> individual gaps: [01 (hyperparameter change = full restart)](01-config-iteration-feels-like-new-server.md),
> [02 (no re-verify of stored rollouts)](02-recompute-rewards-from-existing-rollouts.md),
> [04 (judge failures indistinguishable from wrong answers)](04-judge-only-resume.md).

## 1. TL;DR

All the individual pieces for iterating on a verifier exist — in-process `verify()` unit tests, per-server `gym env test`, Hydra `++` config overrides, `gym eval profile` over stored rollouts, `--limit`/`--num-repeats` for small collections — but nothing in the repo or docs says **when each one is sufficient**. A user who changed a reward weight, a grading regex, or a judge prompt cannot tell whether they need zero servers, a server restart, or a fresh model-paid rollout collection. So everyone defaults to the most expensive option: full re-collection through `gym eval run`, re-paying every policy (and judge) inference call to observe a change that in many cases is a pure function of data already on disk. The takeaway: **there is a six-rung cost ladder from "pytest, milliseconds" to "full re-rollout, hours" — this doc makes it explicit, because the codebase doesn't.**

## 2. The iteration ladder (the organizing idea)

Cheapest first. Rungs 1–4 never call the policy model. Rungs 1–2 need no servers at all.

| Rung | Mechanism | Cost | Servers? | Model calls? |
|---|---|---|---|---|
| 1 | In-process `verify()` call (pytest or script) | ms | none | none |
| 2 | `gym env test --resources-server NAME` | sec–min (first run builds a venv) | none (pytest in-process) | none |
| 3 | Config override + restart resources stack, hand-POST stored rollouts to `/verify` | ~1 min restart + manual scripting | yes | judge only (if judge env) |
| 4 | `gym eval profile` re-aggregation of existing rollouts | sec | none | none |
| 5 | Small re-collection: `gym eval run --no-serve --limit N --num-repeats K` | minutes | yes | yes (N×K rollouts) |
| 6 | Full re-collection: `gym eval run` | hours | yes | all of them |

### Decision table: what changed → which rung is enough

| What changed | Sufficient rung | Why |
|---|---|---|
| Reward **weights / aggregation** over already-recorded components | 1 (or a 10-line pandas script over `rollouts.jsonl`) | Components like `correctness`/`schema_valid`/`format` are stored top-level per rollout row ([app.py:145-155](../resources_servers/example_tool_call_multireward/app.py#L145)); re-weighting is arithmetic on frozen data. Caveat: in `example_tool_call_multireward` the weights are hard-coded in code ([app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147)), so *shipping* the change is a code edit — but *evaluating* it needs no rollouts. |
| **verify() logic** (parsing, extraction, matching) | 1 → then 3 to confirm over real stored rollouts | For stateless verifiers, `verify()` is a pure function of the request row; replay stored rollouts through it offline. |
| A **config knob** (threshold, `grading_mode`, labels) | 3 | Config is frozen at server start ([env.py:198-201](../nemo_gym/cli/env.py#L198)); restart with a `++` override, then re-POST stored rollouts to `/verify`. No new model calls needed. |
| **Judge prompt / judge sampling params** | 3 (re-pays judge inference, not policy) or 5 | The judge is called *inside* `verify()` ([equivalence_llm_judge/app.py:461-465](../resources_servers/equivalence_llm_judge/app.py#L461)), so replaying `/verify` re-runs the judge against the *same* policy outputs — the expensive policy half is reused. |
| **Tool behavior** (a resources-server tool the agent calls mid-rollout) | 5, then 6 | Tool outputs are part of the trajectory ([simple_agent/app.py:148-162](../responses_api_agents/simple_agent/app.py#L148)); old trajectories are invalid. Re-collection is genuinely required. |
| **Agent loop / prompts / sampling params** | 5, then 6 | The trajectory itself changes. Nothing stored is reusable. |
| **Statistics only** (pass@k, per-task variance, usage) | 4 | `gym eval profile` recomputes stats from frozen numeric fields; no verifier involved. |

Hard constraint that shapes the whole table: **rung 3's "re-POST stored rollouts" step has no CLI support** — there is no `gym eval verify` / re-score command anywhere in the command registry ([main.py:313](../nemo_gym/cli/main.py#L313)–[main.py:539](../nemo_gym/cli/main.py#L539)). You script it yourself (shown below). That absence is pain point [02](02-recompute-rewards-from-existing-rollouts.md); this doc's pain point is that nobody tells you rungs 1–4 exist at all.

Second constraint: the ladder only fully applies to **stateless** verifiers (mcqa, equivalence_llm_judge, bird_sql, example_tool_call_multireward…). Session-stateful environments (verify reads in-process per-session state, e.g. `example_session_state_mgmt`) and gymnasium-style environments (no `/verify` at all) cannot replay — for those, rungs 3–4 collapse into 5.

## 3. Experience it yourself

We use the hosted ECCN classifier fine-tune as the policy model
([eccn-llama-3.1-8b.yaml](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml)) — it is an
OpenAI-compatible endpoint needing only `INFERENCE_KEY`. It will be *bad* at the weather tool-call task;
that's fine, the workflow is the point. Substitute any model config you like. One wiring quirk: agent
configs reference the model instance named `policy_model`
([example_tool_call_multireward.yaml:16-18](../resources_servers/example_tool_call_multireward/configs/example_tool_call_multireward.yaml#L16)),
but this YAML's top-level key is `eccn-llama-3.1-8b-instruct`
([eccn-llama-3.1-8b.yaml:1](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml#L1)), so we
override the agent's `model_server.name`.

### 3.1 Baseline: start servers, collect a small batch

Terminal 1 (stays occupied — there is no `gym env stop`; shutdown is Ctrl-C on this process,
[env.py:399-404](../nemo_gym/cli/env.py#L399)):

```bash
cd /path/to/Gym
export INFERENCE_KEY=<your key>
uv run gym env start \
  --config resources_servers/example_tool_call_multireward/configs/example_tool_call_multireward.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++example_tool_call_multireward_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct"
```

(Preview the merged config first with the same `--config`s + override via `gym env resolve`,
[env.py:865-885](../nemo_gym/cli/env.py#L865) — verified: the override lands as `model_server.name: eccn-llama-3.1-8b-instruct`.)

Terminal 2 — collect 3 tasks × 2 repeats against the running servers (the committed example dataset has 5 rows):

```bash
uv run gym eval run --no-serve \
  --agent example_tool_call_multireward_simple_agent \
  --input resources_servers/example_tool_call_multireward/data/example.jsonl \
  --output results/multireward_rollouts.jsonl \
  --limit 3 --num-repeats 2 --concurrency 4
```

Every flag is real: `--no-serve` switches dispatch to `collect_rollouts`
([main.py:273-275](../nemo_gym/cli/main.py#L273)), `--limit` → `+limit`
([main.py:495](../nemo_gym/cli/main.py#L495)), `--num-repeats` → `+num_repeats`
([main.py:496](../nemo_gym/cli/main.py#L496)).

Expected artifacts next to the output:

- `results/multireward_rollouts.jsonl` — 6 rows. Each row is the verify response plus stamped identity keys. Keys of a real committed rollout row: `responses_create_params`, `response`, `reward`, `reward_components`, `correctness`, `schema_valid`, `format`, `predicted_calls`, `_ng_task_index`, `_ng_rollout_index`, `agent_ref`. Note what's **missing**: `expected_call`. The verify response model doesn't declare it, so pydantic silently drops it ([app.py:61-71](../resources_servers/example_tool_call_multireward/app.py#L61) vs the request model [app.py:55-58](../resources_servers/example_tool_call_multireward/app.py#L55)) — remember this for rung 3.
- `results/multireward_rollouts_materialized_inputs.jsonl` — the exact expanded `/run` request rows, *including* `expected_call` ([rollout_collection.py:237-240](../nemo_gym/rollout_collection.py#L237), written at [rollout_collection.py:492-494](../nemo_gym/rollout_collection.py#L492)). This file is how you recover dropped per-task fields.
- `results/multireward_rollouts_aggregate_metrics.json` — per-agent metrics from `/aggregate_metrics` ([rollout_collection.py:670](../nemo_gym/rollout_collection.py#L670)).

Look at one row: `reward` is the unweighted sum of the three components, computed at
[app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147). With the ECCN model expect
mostly `reward: 0.0` — irrelevant here.

### 3.2 Now feel the pain: change the reward weighting

Suppose you decide `correctness` should count double. Open
[app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147):

```python
return ToolCallMultiRewardVerifyResponse(
    **body.model_dump(),
    reward=sum(reward_components.values()),
```

There is no config knob — the config class is empty
([app.py:51-52](../resources_servers/example_tool_call_multireward/app.py#L51)). So the "obvious" workflow is:

1. Edit `verify()` in app.py.
2. Ctrl-C terminal 1 (kills all servers + Ray + head server, [env.py:344-386](../nemo_gym/cli/env.py#L344)).
3. Re-run `gym env start` (venv reactivation, Ray bootstrap, health polling).
4. Re-run the whole `gym eval run` — **re-paying every policy model call** to recompute a number that is pure arithmetic over `reward_components` already sitting in `rollouts.jsonl`.

And note: re-running `gym eval profile` on the old rollouts shows *nothing* changed, because profile reads
the frozen `reward` field ([reward_profile.py:141-142](../nemo_gym/reward_profile.py#L141)) — it never
calls `/verify`. That is the trap this doc exists to route you around.

### 3.3 Rung 1 — in-process `verify()`: milliseconds, zero servers

`verify()` is just an async method on a Pydantic object; the HTTP route is only registered at
[base_resources_server.py:138](../nemo_gym/base_resources_server.py#L138). The repo's own tests construct
the server directly ([tests/test_app.py:73-77](../resources_servers/example_tool_call_multireward/tests/test_app.py#L73)):

```python
def _server() -> ToolCallMultiRewardResourcesServer:
    return ToolCallMultiRewardResourcesServer(
        config=ToolCallMultiRewardResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
```

Minimal runnable replay of your **stored rollouts** through a locally edited `verify()` — no servers, no
model. Save as `resources_servers/example_tool_call_multireward/replay.py` and run with that server's venv
(`.venv/bin/python replay.py`, created by rung 2 below), from inside the server dir so `from app import`
resolves:

```python
import asyncio, json
from unittest.mock import MagicMock
from app import (ToolCallMultiRewardResourcesServer, ToolCallMultiRewardResourcesServerConfig,
                 ToolCallMultiRewardVerifyRequest)
from nemo_gym.server_utils import ServerClient

server = ToolCallMultiRewardResourcesServer(
    config=ToolCallMultiRewardResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
    server_client=MagicMock(spec=ServerClient))

inputs = {(r["_ng_task_index"], r["_ng_rollout_index"]): r
          for r in map(json.loads, open("../../results/multireward_rollouts_materialized_inputs.jsonl"))}

async def main():
    for line in open("../../results/multireward_rollouts.jsonl"):
        rollout = json.loads(line)
        key = (rollout["_ng_task_index"], rollout["_ng_rollout_index"])
        row = inputs[key] | rollout  # recover expected_call, dropped from the verify response
        result = await server.verify(ToolCallMultiRewardVerifyRequest.model_validate(row))
        print(key, "old:", rollout["reward"], "new:", result.reward)

asyncio.run(main())
```

The join on `(_ng_task_index, _ng_rollout_index)` mirrors the profiler's own join key
([reward_profile.py:48-49](../nemo_gym/reward_profile.py#L48)). For the pytest form, copy the existing
pattern: request factory [tests/test_app.py:80-88](../resources_servers/example_tool_call_multireward/tests/test_app.py#L80),
assertion style [tests/test_app.py:95-100](../resources_servers/example_tool_call_multireward/tests/test_app.py#L95).
Async tests need no decorator — `asyncio_mode = "auto"` at [pyproject.toml:490](../pyproject.toml#L490).

**Sufficient when:** the change is verify logic or weights over row-local data, on a stateless verifier.
**Not sufficient when:** verify calls out (LLM judge — mock `server_client.post` like
`equivalence_llm_judge`'s tests do) or reads session state.

### 3.4 Rung 2 — `gym env test`: the server's whole test suite in an isolated venv

```bash
uv run gym env test --resources-server example_tool_call_multireward
```

Here `--resources-server` maps to `+entrypoint=resources_servers/<name>`
([main.py:135-140](../nemo_gym/cli/main.py#L135)), dispatch picks single-server `test`
([main.py:278-283](../nemo_gym/cli/main.py#L278)), which runs `setup_env_command && pytest` in the server
dir ([env.py:532-539](../nemo_gym/cli/env.py#L532)). First run is slow — it builds
`resources_servers/<name>/.venv` and installs requirements
([setup_command.py:103-172](../nemo_gym/cli/setup_command.py#L103)); pass `+skip_venv_if_present=true` on
later runs ([setup_command.py:116-117](../nemo_gym/cli/setup_command.py#L116)). The fast inner loop the CLI
itself recommends ([env.py:549](../nemo_gym/cli/env.py#L549)):
`cd resources_servers/example_tool_call_multireward && source .venv/bin/activate && pytest`.

### 3.5 Rung 3 — config override + restart + manual re-verify of stored rollouts

`example_tool_call_multireward` has zero config knobs, so use `mcqa` for the config-knob half.
`grading_mode` is a real server-level knob ([mcqa/app.py:30-38](../resources_servers/mcqa/app.py#L30),
default `null` in [mcqa.yaml:5](../resources_servers/mcqa/configs/mcqa.yaml#L5)) that overrides the per-row
value at [mcqa/app.py:260](../resources_servers/mcqa/app.py#L260). Because CLI overrides merge last
([global_config.py:586-587](../nemo_gym/global_config.py#L586)), no YAML edit is needed:

```bash
uv run gym env start \
  --config resources_servers/mcqa/configs/mcqa.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++mcqa_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct" \
  "++mcqa.resources_servers.mcqa.grading_mode=lenient_boxed"
```

(Both overrides verified with `gym env resolve`.) A **restart is mandatory**: the child process's entire
config view is the `NEMO_GYM_CONFIG_DICT` env var snapshotted at `Popen` time
([env.py:198-201](../nemo_gym/cli/env.py#L198), rebuilt in the child at
[global_config.py:793-800](../nemo_gym/global_config.py#L793), validated once into `self.config` at
[server_utils.py:368-379](../nemo_gym/server_utils.py#L368) /
[server_utils.py:624-629](../nemo_gym/server_utils.py#L624)). uvicorn runs without `reload`
([server_utils.py:698](../nemo_gym/server_utils.py#L698)); there is no `/reload` endpoint, no config-watch,
and no per-request verifier-config override field in `BaseVerifyRequest`
([base_resources_server.py:87-88](../nemo_gym/base_resources_server.py#L87)).

Then re-verify stored rollouts **without new policy calls** by POSTing them at the running resources
server's `/verify` (host/port printed by `gym env start`, or `gym env status`). A stored rollout row is a
superset of the verify request — `BaseVerifyResponse` extends `BaseVerifyRequest`
([base_resources_server.py:83-92](../nemo_gym/base_resources_server.py#L83)) — and for mcqa the extra
grading fields (`options`, `expected_answer`) are echoed into the row
([mcqa/app.py:322-327](../resources_servers/mcqa/app.py#L322)):

```bash
head -1 results/mcqa_rollouts.jsonl > /tmp/row.json
curl -s -X POST http://127.0.0.1:<mcqa_port>/verify \
  -H 'Content-Type: application/json' -d @/tmp/row.json | python3 -m json.tool
```

This is exactly the pattern the repo's own HTTP tests use
([code_gen/tests/test_app.py:93-96](../resources_servers/code_gen/tests/test_app.py#L93)). Two gotchas:
(a) top-level extra keys like `_ng_task_index` are fine, but stray keys *inside*
`responses_create_params` 422 — it's `extra="forbid"`
([openai_utils.py:257](../nemo_gym/openai_utils.py#L257)); (b) if the server's verify-response model
dropped a needed field (multireward's `expected_call`), merge the row with its materialized-inputs twin
first, as in rung 1. **No CLI wraps this loop** — that's [pain point 02](02-recompute-rewards-from-existing-rollouts.md).
For judge environments this same replay re-runs judge inference (verify POSTs the judge model at
[equivalence_llm_judge/app.py:461-465](../resources_servers/equivalence_llm_judge/app.py#L461)) but not the
policy — still a huge saving. Judge knobs are all restart-frozen config: prompt template loaded once in
`__init__` ([equivalence_llm_judge/app.py:269-270](../resources_servers/equivalence_llm_judge/app.py#L269)),
labels/swap/partial-credit at [equivalence_llm_judge/app.py:48-110](../resources_servers/equivalence_llm_judge/app.py#L48).

### 3.6 Rung 4 — `gym eval profile`: re-slice, never re-score

```bash
uv run gym eval profile \
  --inputs results/multireward_rollouts_materialized_inputs.jsonl \
  --rollouts results/multireward_rollouts.jsonl
```

Flags map to `+materialized_inputs_jsonl_fpath` / `+rollouts_jsonl_fpath`
([main.py:527-538](../nemo_gym/cli/main.py#L527)). It loads both files, joins on
`(_ng_task_index, _ng_rollout_index)` ([reward_profile.py:62-89](../nemo_gym/reward_profile.py#L62)), and
aggregates **every top-level numeric/bool field**
([reward_profile.py:229-233](../nemo_gym/reward_profile.py#L229)) — so multireward's `correctness`,
`schema_valid`, `format` each get `mean/`, `std/`, etc. for free. Outputs:
`..._reward_profiling.jsonl` + `..._agent_metrics.json`
([reward_profile.py:289-307](../nemo_gym/reward_profile.py#L289)). Partial collections need the raw Hydra
override `++allow_partial_rollouts=True` — there is no flag for it; the error message itself tells you
([reward_profile.py:80-86](../nemo_gym/reward_profile.py#L80)). **Sufficient when** you want different
statistics over the same rewards. **Never sufficient** after any verifier change: no server is contacted,
no `verify()` runs ([eval.py:421-457](../nemo_gym/cli/eval.py#L421) — pure file I/O + pandas).

### 3.7 Rungs 5–6 — re-collection, small then full

When trajectories themselves are invalidated (tool/agent/model changes), re-collect — but start small:
repeat the 3.1 command with `--limit 2 --num-repeats 1` and a **new** `--output` path. Two sharp edges:

- `--resume` ([main.py:491](../nemo_gym/cli/main.py#L491) → `resume_from_cache`,
  [rollout_collection.py:205-208](../nemo_gym/rollout_collection.py#L205)) only *skips* already-completed
  `(task, rollout)` keys ([rollout_collection.py:409-462](../nemo_gym/rollout_collection.py#L409)); it never
  re-verifies them. Resuming into an old output after a verifier change silently **mixes old-verifier and
  new-verifier rewards in one file.**
- Without `--resume`, the old output is deleted outright
  ([rollout_collection.py:496](../nemo_gym/rollout_collection.py#L496)) — always point re-collections at a
  fresh path.

Rung 6 is the same command with the full dataset (or the serve path: `gym eval run --split ...`, which also
runs data preparation and starts servers, [eval.py:328-402](../nemo_gym/cli/eval.py#L328)).

## 4. Architecture & code flow

Why is verification welded to rollout cost? Follow one row end to end:

1. `gym eval run` argparse translates flags to Hydra tokens and rewrites `sys.argv`
   ([main.py:646](../nemo_gym/cli/main.py#L646), [main.py:73-78](../nemo_gym/cli/main.py#L73)); unknown
   non-dash tokens pass through as raw overrides ([main.py:621-629](../nemo_gym/cli/main.py#L621)).
2. `--no-serve` picks `collect_rollouts` ([main.py:273-275](../nemo_gym/cli/main.py#L273),
   [eval.py:405-410](../nemo_gym/cli/eval.py#L405)); otherwise `e2e_rollout_collection` also prepares data
   and starts servers ([eval.py:328-402](../nemo_gym/cli/eval.py#L328)).
3. Server start serializes the *entire merged config* into each child's environment —
   [env.py:198-201](../nemo_gym/cli/env.py#L198):
   ```python
   command = f"""{setup_env_command(dir_path, global_config_dict, top_level_path)} \\
       && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \\
       {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \\
       python {str(entrypoint_fpath)}"""
   ```
   The child validates its slice once into a frozen Pydantic config
   ([server_utils.py:374-377](../nemo_gym/server_utils.py#L374)) and serves it with uvicorn, no reload
   ([server_utils.py:698](../nemo_gym/server_utils.py#L698)). **Config edits after this point are invisible.**
4. The collector expands rows (limit: [rollout_collection.py:255-258](../nemo_gym/rollout_collection.py#L255);
   repeats: [rollout_collection.py:373-378](../nemo_gym/rollout_collection.py#L373)), persists them as
   materialized inputs ([rollout_collection.py:492-494](../nemo_gym/rollout_collection.py#L492)), and POSTs
   each row to the agent's `/run` ([rollout_collection.py:689](../nemo_gym/rollout_collection.py#L689)).
5. Inside the agent's `/run` ([simple_agent/app.py:176-208](../responses_api_agents/simple_agent/app.py#L176)):
   seed session → the multi-step model/tool loop (`/v1/responses` per step,
   [simple_agent/app.py:87-92](../responses_api_agents/simple_agent/app.py#L87)) → then, **in the same
   request**, verification:
   ```python
   verify_request = SimpleAgentVerifyRequest.model_validate(
       body.model_dump() | {"response": await get_response_json(response)}
   )
   verify_response = await self.server_client.post(
       server_name=self.config.resources_server.name,
       url_path="/verify", ...
   ```
   ([simple_agent/app.py:197-206](../responses_api_agents/simple_agent/app.py#L197)). This is the weld:
   the only built-in caller of `/verify` sits behind the expensive model loop.
6. The resources server's `/verify` route ([base_resources_server.py:138](../nemo_gym/base_resources_server.py#L138))
   runs the env's `verify()` ([app.py:122-155](../resources_servers/example_tool_call_multireward/app.py#L122))
   and its response *is* the `/run` result ([simple_agent/app.py:208](../responses_api_agents/simple_agent/app.py#L208)).
7. The collector stamps `_ng_task_index` / `_ng_rollout_index` / `agent_ref`
   ([rollout_collection.py:513-515](../nemo_gym/rollout_collection.py#L513); constants at
   [global_config.py:118-122](../nemo_gym/global_config.py#L118)) and appends the row to `rollouts.jsonl`
   ([rollout_collection.py:536-539](../nemo_gym/rollout_collection.py#L536)), then triggers
   `/aggregate_metrics` ([rollout_collection.py:577](../nemo_gym/rollout_collection.py#L577)).
8. Everything downstream — `gym eval profile` ([eval.py:444-447](../nemo_gym/cli/eval.py#L444)),
   `gym eval aggregate` — consumes the **frozen** `reward` and other numeric fields
   ([reward_profile.py:141-142](../nemo_gym/reward_profile.py#L141),
   [reward_profile.py:229-233](../nemo_gym/reward_profile.py#L229)). No path re-enters `verify()`.

## 5. Why it hurts / gap analysis

**What exists** (all verified above): in-process test pattern per server; `gym env test`; last-wins `++`
overrides for any nested server field; materialized-inputs + rollouts artifacts that together contain
everything a stateless `verify()` needs; `gym eval profile`; `--limit`/`--num-repeats`/`--resume`.

**What's missing:**

1. **The workflow document.** Nothing in `fern/` or the CLI help orders these by cost or keys them to
   "what changed". The CLI reference documents each command in isolation; the closest thing to guidance is
   an error-message example buried in [env.py:516-519](../nemo_gym/cli/env.py#L516).
2. **A re-verify command** to make rung 3 real (see [02](02-recompute-rewards-from-existing-rollouts.md)): the
   full command registry [main.py:313-539](../nemo_gym/cli/main.py#L313) has `eval prepare|run|aggregate|profile`
   and nothing that replays stored rows through `/verify`.
3. **Misleading affordances that funnel users to rung 6.** `gym eval profile` *sounds* like re-scoring but
   is re-aggregation; `--resume` *sounds* like it helps iteration but skips rather than re-verifies —
   and silently mixes verifier versions if reused after a change; the reference multi-reward environment
   hard-codes its weights ([app.py:51-52](../resources_servers/example_tool_call_multireward/app.py#L51),
   [app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147)), teaching newcomers that
   weight iteration = code edit + restart + re-rollout.
4. **Design choices that make the cheap rungs non-obvious:** verification is synchronous inside `/run`
   (step 5 above), the verify response model silently drops undeclared request fields (the `expected_call`
   trap in §3.1), and config is an immutable per-process snapshot (step 3). Each is individually
   defensible; undocumented and combined, they make "just re-run everything" the only workflow a newcomer
   can discover.

## 6. Where a fix would hook in

Ordered by effort:

1. **Ship this ladder as a docs page** (the literal fix for this pain point). Natural home:
   `fern/versions/latest/pages/build-verifiers/` (exists, has `verification-patterns/`); register the page
   per the repo's docs skill. Content = §2's tables + §3's commands.
2. **Make the reference environment's weights configurable**, so the flagship multi-reward example
   demonstrates rung 3 instead of rung 6: add e.g. `component_weights: Dict[str, float] | None = None` to
   `ToolCallMultiRewardResourcesServerConfig`
   ([app.py:51-52](../resources_servers/example_tool_call_multireward/app.py#L51)) and use it in the sum at
   [app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147). ~6 lines + a test.
3. **Add `gym eval verify` (re-score stored rollouts).** Register a `Command` in the registry next to
   `eval profile` ([main.py:527-538](../nemo_gym/cli/main.py#L527)) targeting a new function in
   [eval.py](../nemo_gym/cli/eval.py) modeled on `reward_profile`
   ([eval.py:421-457](../nemo_gym/cli/eval.py#L421)): load both JSONLs, join per-key using the existing
   aligner ([reward_profile.py:62-89](../nemo_gym/reward_profile.py#L62)), merge input row over rollout row
   (recovering dropped fields), POST each to the resources server's `/verify` via `ServerClient` (mirror
   the dispatcher at [rollout_collection.py:687-701](../nemo_gym/rollout_collection.py#L687) with
   `url_path="/verify"`), and write a fresh rollouts JSONL with updated rewards. The resources-server name
   can come from a `--resources-server` value or be resolved from the agent's config ref
   (`resources_server: ResourcesServerRef`, [simple_agent/app.py:45-46](../responses_api_agents/simple_agent/app.py#L45)).
   Requires only the resources (+ judge model) servers running — a zero-core-change prototype can even ship
   as a `rollout_collection_driver` plugin ([rollout_collection.py:136-145](../nemo_gym/rollout_collection.py#L136)).
4. **Guardrail:** stamp a verifier config/code fingerprint into each rollout row at collection time (hook:
   the stamping block, [rollout_collection.py:513-517](../nemo_gym/rollout_collection.py#L513)) and warn in
   `_load_from_cache` ([rollout_collection.py:409-462](../nemo_gym/rollout_collection.py#L409)) when
   `--resume` would append rows scored by a different verifier than the cached ones.

Fixes 2–4 each collapse one rung of accidental cost; fix 1 is what makes the ladder discoverable at all.
