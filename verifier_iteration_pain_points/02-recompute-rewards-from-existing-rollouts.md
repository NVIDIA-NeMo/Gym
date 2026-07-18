# Pain point 2: No first-class workflow to recompute rewards from existing rollouts (issue #987)

## 1. TL;DR

Rollout collection has two very differently priced halves: the **expensive** half (policy inference plus
the multi-step agent loop) and the **cheap** half (the verifier scoring the finished trajectory). Both are
fused into a single `/run` request, and the reward is frozen into `rollouts.jsonl` at collection time. If
you then change a verifier hyperparameter — a grading mode, a threshold, a judge prompt — the only
supported way to get new rewards is to re-run `gym eval run`, which re-pays all of the policy inference.
The data needed to re-verify is (mostly) already on disk, and every resources server exposes a plain HTTP
`/verify` endpoint, but there is no `gym eval re-verify` command, no library helper, and no documentation
telling users that `rollouts.jsonl` + `*_materialized_inputs.jsonl` together contain everything a
stateless verifier needs. **Takeaway: rewards are recomputable from existing artifacts today only via a
hand-rolled script against `/verify`; Gym should ship that as a first-class command.**

## 2. Experience it yourself

We use the `mcqa` environment because its verifier has a genuine YAML-tunable hyperparameter:
`grading_mode` ([app.py:30-38](../resources_servers/mcqa/app.py#L30)), which controls how the answer
letter is extracted from model output. The policy is the hosted ECCN-classifier fine-tune in
[eccn-llama-3.1-8b.yaml:1-9](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml#L1) — it is
an OpenAI-compatible endpoint that needs an `INFERENCE_KEY` env var. It is *not* good at MCQA, and that is
fine: answer quality is irrelevant here, the workflow is the point. Any model config can be substituted.

### Step 1 — start the full stack (terminal 1)

The eccn config's top-level key is `eccn-llama-3.1-8b-instruct`, not the conventional `policy_model`, so
the agent's `model_server` ref must be overridden (otherwise config parsing fails with
`ServerRefNotFoundError`, [global_config.py:342-345](../nemo_gym/global_config.py#L342)):

```bash
cd /path/to/Gym
export INFERENCE_KEY=<your key>

uv run gym env start \
  --config resources_servers/mcqa/configs/mcqa.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++mcqa_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct"
```

First start is slow (a `uv` venv is built per server). Wait until all servers report healthy
(`uv run gym env status` in another terminal).

### Step 2 — collect rollouts ONCE (terminal 2)

Small on purpose: 3 tasks x 2 repeats = 6 policy trajectories against the committed 5-row example
dataset.

```bash
uv run gym eval run --no-serve \
  --agent mcqa_simple_agent \
  --input resources_servers/mcqa/data/example.jsonl \
  --output results/02/rollouts.jsonl \
  --limit 3 \
  --num-repeats 2 \
  --temperature 1.0 --top-p 0.95
```

``` rerunning with different config
uv run gym eval run --no-serve \
  --agent mcqa_simple_agent \
  --input resources_servers/mcqa/data/example.jsonl \
  --output results/02/rollouts.jsonl \
  --limit 3 \
  --num-repeats 2 \
  --temperature 1.0 --top-p 0.95
```
# Working reverify
```
uv run gym eval reverify \
  --config verifier_iteration_pain_points/mcqa_resources_server.yaml \
  "++mcqa_resources_server.resources_servers.mcqa.grading_mode=lenient_boxed" \
  --inputs outputs/reverify_test/rollouts_materialized_inputs.jsonl \
  --rollouts outputs/reverify_test/rollouts.jsonl \
  --output outputs/reverify_test/rollouts_lenient2.jsonl 

```

Expected artifacts in `results/02`:

| File | What it is |
|---|---|
| `rollouts.jsonl` | 6 rows: one verify response per rollout (`reward`, `response`, `responses_create_params`, `expected_answer`, `extracted_answer`, `_ng_task_index`, `_ng_rollout_index`, `agent_ref`) |
| `rollouts_materialized_inputs.jsonl` | 6 rows: the exact `/run` request bodies, including the dataset fields the verifier grades on (`options`, `grading_mode`, `uuid`) |
| `rollouts_aggregate_metrics.json` | pass@k / mean-reward summary |

Look at one row of each:

```bash
python3 -c "import json; r=json.loads(open('results/02/rollouts.jsonl').readline()); print(sorted(r))"
# ['_ng_rollout_index', '_ng_task_index', 'agent_ref', 'expected_answer', 'extracted_answer',
#  'response', 'responses_create_params', 'reward']
python3 -c "import json; r=json.loads(open('results/02/rollouts_materialized_inputs.jsonl').readline()); print(sorted(r))"
# ['_ng_rollout_index', '_ng_task_index', 'agent_ref', 'expected_answer', 'grading_mode',
#  'options', 'responses_create_params', 'uuid']
```

Note something important: the rollout row does **not** contain `options` or `grading_mode`. The mcqa
verify response only echoes the fields it declares
([app.py:65-67](../resources_servers/mcqa/app.py#L65), built at
[app.py:322-327](../resources_servers/mcqa/app.py#L322)); undeclared extras are silently dropped by
Pydantic. The materialized-inputs file is where the verifier's per-task inputs survive. You need *both*
files to re-verify.

### Step 3 — now change a verifier hyperparameter and feel the cost

Say you want to relax grading from the per-row default `strict_single_letter_boxed` to `lenient_boxed`
(the server-level `grading_mode` overrides the per-row one,
[app.py:260](../resources_servers/mcqa/app.py#L260); the shipped YAML leaves it `null`,
[mcqa.yaml:5](../resources_servers/mcqa/configs/mcqa.yaml#L5)).

**The naive (only documented) path** — restart everything and re-collect:

```bash
# terminal 1: Ctrl-C the running `gym env start` (config is frozen into each child process
# at launch via the NEMO_GYM_CONFIG_DICT env var — see env.py:198-201 — so a restart is mandatory)
uv run gym env start \
  --config resources_servers/mcqa/configs/mcqa.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++mcqa_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct" \
  "++mcqa.resources_servers.mcqa.grading_mode=lenient_boxed"

# terminal 2: re-run the WHOLE thing — all 6 policy trajectories are regenerated
uv run gym eval run --no-serve \
  --agent mcqa_simple_agent \
  --input resources_servers/mcqa/data/example.jsonl \
  --output /tmp/pp2/rollouts_lenient.jsonl \
  --limit 3 --num-repeats 2 --temperature 1.0 --top-p 0.95
```

Every second of this run is model inference you already paid for. At 6 samples it is an annoyance; at
5,000 tasks x 16 repeats against a paid endpoint it is the difference between a 10-second iteration loop
and an hours-long, dollars-expensive one. Worse: with `--temperature 1.0` the new trajectories are
*different* trajectories, so you are no longer measuring "same rollouts, new verifier" — the comparison is
confounded by sampling noise. (`--resume` does not help: it only skips already-completed rows, reusing
their frozen rewards — [main.py:491](../nemo_gym/cli/main.py#L491),
[rollout_collection.py:205-208](../nemo_gym/rollout_collection.py#L205).)

And no, `gym eval profile` does not re-verify — see §3 step 7.

### Step 4 — the manual workaround: verify-only server + POST stored rows to /verify

The resources server is a plain FastAPI service, and mcqa's `verify()` is a pure function of the request
body plus server config. So: start ONLY the resources server with the new hyperparameter, and replay the
stored rows. A resources-server-only config must be hand-written because the shipped YAML bundles an agent
that references `policy_model`:

```bash
cat > /tmp/pp2/mcqa_verify_only.yaml <<'EOF'
mcqa:
  resources_servers:
    mcqa:
      entrypoint: app.py
      domain: knowledge
      grading_mode: lenient_boxed
      host: 127.0.0.1
      port: 9500
EOF

# no model server, no INFERENCE_KEY needed
uv run gym env start --config /tmp/pp2/mcqa_verify_only.yaml
```

Then replay (join the two artifacts on the rollout identity keys, map fields onto the verify request
schema, POST):

```python
# /tmp/pp2/reverify.py  — run with: uv run python /tmp/pp2/reverify.py
import json

import requests

VERIFY_URL = "http://127.0.0.1:9500/verify"
KEY = ("_ng_task_index", "_ng_rollout_index")


def load(path):
    return [json.loads(line) for line in open(path)]


inputs = {tuple(r[k] for k in KEY): r for r in load("/tmp/pp2/rollouts_materialized_inputs.jsonl")}
rollouts = load("/tmp/pp2/rollouts.jsonl")

for roll in sorted(rollouts, key=lambda r: tuple(r[k] for k in KEY)):
    inp = inputs[tuple(roll[k] for k in KEY)]
    # BaseVerifyRequest = the /run request row + the final response:
    #   responses_create_params  <- either file (identical)
    #   response                 <- rollouts.jsonl (the expensive part!)
    #   options / expected_answer / grading_mode / template_metadata
    #                            <- materialized inputs (dropped from the rollout row)
    payload = {**inp, "response": roll["response"]}
    new = requests.post(VERIFY_URL, json=payload).json()
    print(
        f"task={roll['_ng_task_index']} rollout={roll['_ng_rollout_index']} "
        f"old_reward={roll['reward']} new_reward={new['reward']} "
        f"extracted={new['extracted_answer']!r}"
    )
```

This finishes in under a second — zero model calls. Extra top-level keys in the payload (`_ng_*`,
`agent_ref`, `uuid`) are ignored by the Pydantic request model; only the *inside* of
`responses_create_params` is strict (`extra="forbid"`,
[openai_utils.py:250-257](../nemo_gym/openai_utils.py#L250)), and it round-trips unchanged. On this tiny
sample `lenient_boxed` may produce identical rewards (it is a strict superset of strict-boxed extraction);
set `grading_mode: lenient_answer_colon` in the verify-only YAML instead to see every reward visibly
change (the example dataset's answers are `\boxed{}`-style, so `Answer:` extraction finds nothing and all
rewards drop to 0 — a blunt but unmistakable demonstration that the verifier, not the policy, is being
re-run).

That script is the entire feature request of issue #987. It is ~25 lines, it works, and nothing in the
repo tells you it is possible — you have to reverse-engineer the schema mapping in §3 to write it.

## 3. Architecture & code flow

How a reward gets into `rollouts.jsonl`, and why nothing reads it back into `/verify`:

1. **CLI**: `gym eval run --no-serve` dispatches to `collect_rollouts`
   ([main.py:484-490](../nemo_gym/cli/main.py#L484) for the flag,
   [eval.py:405-410](../nemo_gym/cli/eval.py#L405) for the target), which validates
   `RolloutCollectionConfig` ([rollout_collection.py:165-240](../nemo_gym/rollout_collection.py#L165)) and
   calls `RolloutCollectionHelper.run_from_config`.

2. **Input materialization**: each dataset row gets an `agent_ref`
   ([rollout_collection.py:337](../nemo_gym/rollout_collection.py#L337)), a deduplicated
   `_ng_task_index` ([rollout_collection.py:358-359](../nemo_gym/rollout_collection.py#L358)), and one
   copy per repeat with an `_ng_rollout_index`
   ([rollout_collection.py:373-378](../nemo_gym/rollout_collection.py#L373)). The expanded rows are
   written verbatim to `<output_stem>_materialized_inputs.jsonl`
   ([rollout_collection.py:492-494](../nemo_gym/rollout_collection.py#L492); path derived at
   [rollout_collection.py:237-240](../nemo_gym/rollout_collection.py#L237)). Identity key constants live
   at [global_config.py:118-123](../nemo_gym/global_config.py#L118).

3. **Dispatch**: the collector POSTs each materialized row, whole, to the agent's `/run` —
   [rollout_collection.py:689](../nemo_gym/rollout_collection.py#L689):

   ```python
   res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
   ```

   The collector never touches `/verify` itself.

4. **Agent = expensive + cheap fused**: `SimpleAgent.run`
   ([simple_agent/app.py:176-208](../responses_api_agents/simple_agent/app.py#L176)) does, in order:
   `/seed_session` on the resources server (:179-186), its own `/v1/responses` — the multi-step policy
   loop, *this is where all the money goes* (:188-195) — and then builds the verify request by merging the
   original run row with the final response and POSTs it to the resources server:

   ```python
   verify_request = SimpleAgentVerifyRequest.model_validate(
       body.model_dump() | {"response": await get_response_json(response)}
   )
   verify_response = await self.server_client.post(
       server_name=self.config.resources_server.name,
       url_path="/verify",
       json=verify_request.model_dump(),
       cookies=cookies,
   )
   ```

   ([simple_agent/app.py:197-206](../responses_api_agents/simple_agent/app.py#L197)). The verify response
   is returned verbatim as the `/run` result (:208). Dataset extras (`options`, `expected_answer`, ...)
   reach the verifier because the agent's request/response models are `extra="allow"`
   ([simple_agent/app.py:51-60](../responses_api_agents/simple_agent/app.py#L51)).

5. **The verify contract**: `/verify` is registered once, on every resources server, at
   [base_resources_server.py:138](../nemo_gym/base_resources_server.py#L138); `verify()` is abstract
   ([base_resources_server.py:146-148](../nemo_gym/base_resources_server.py#L146)). The base schema is
   exactly three fields — [base_resources_server.py:83-92](../nemo_gym/base_resources_server.py#L83):

   ```python
   class BaseRunRequest(BaseModel):
       responses_create_params: NeMoGymResponseCreateParamsNonStreaming

   class BaseVerifyRequest(BaseRunRequest):
       response: NeMoGymResponse

   class BaseVerifyResponse(BaseVerifyRequest):
       reward: float
   ```

   mcqa extends the request with its grading fields (`options`, `expected_answer`, `grading_mode`,
   `template_metadata` — [app.py:41-62](../resources_servers/mcqa/app.py#L41)) and computes
   `reward = 1.0 if pred == gold else 0.0`
   ([app.py:319-320](../resources_servers/mcqa/app.py#L319)), with the server-config knob winning over the
   row-level one ([app.py:260](../resources_servers/mcqa/app.py#L260)).

6. **Row persistence**: back in the collector, the `/run` JSON is stamped with `_ng_task_index`,
   `_ng_rollout_index`, `agent_ref`
   ([rollout_collection.py:513-517](../nemo_gym/rollout_collection.py#L513)) and appended to the output
   file ([rollout_collection.py:536-539](../nemo_gym/rollout_collection.py#L536); failures go to a
   `_failures.jsonl` sidecar, :531-535).

   **Field-by-field: rollouts.jsonl row vs. what `/verify` needs**

   | `/verify` request field (schema) | Where it lives after collection |
   |---|---|
   | `responses_create_params` ([base_resources_server.py:83-84](../nemo_gym/base_resources_server.py#L83)) | both files (echoed into the rollout row) |
   | `response` ([base_resources_server.py:87-88](../nemo_gym/base_resources_server.py#L87)) | `rollouts.jsonl` only — the expensive artifact |
   | per-server grading fields, e.g. mcqa `options`/`grading_mode`/`template_metadata` ([app.py:41-58](../resources_servers/mcqa/app.py#L41)) | `*_materialized_inputs.jsonl` only, *unless* the server's VerifyResponse re-declares them (mcqa's does not — [app.py:65-67](../resources_servers/mcqa/app.py#L65), and [app.py:322-327](../resources_servers/mcqa/app.py#L322) drops the rest, because `BaseVerifyResponse` is not `extra="allow"`) |
   | join key: `_ng_task_index`, `_ng_rollout_index` ([global_config.py:118-119](../nemo_gym/global_config.py#L118)) | both files, stamped by the collector |

7. **Why `gym eval profile` does NOT re-verify**: it is registered with only `--inputs`/`--rollouts`
   flags — no server config at all ([main.py:527-538](../nemo_gym/cli/main.py#L527)). The implementation
   ([eval.py:422-457](../nemo_gym/cli/eval.py#L422)) loads both JSONLs and hands them to
   `RewardProfiler.profile_from_data`, which is pure pandas: it reads the frozen `reward` field
   ([reward_profile.py:141-142](../nemo_gym/reward_profile.py#L141)) and keeps only numeric/bool
   top-level fields ([reward_profile.py:229-233](../nemo_gym/reward_profile.py#L229)):

   ```python
   for k, v in result.items():
       if isinstance(v, bool):
           numeric_result[k] = int(v)
       elif isinstance(v, (int, float)):
           numeric_result[k] = v
   ```

   No HTTP call, no `ServerClient`, no `/verify`. Changing verifier config changes `gym eval profile`
   output by exactly nothing.

## 4. Why it hurts / gap analysis

**What exists:**

- The artifacts are (almost) sufficient. `rollouts.jsonl` holds the full `response`;
  `*_materialized_inputs.jsonl` holds the full `/run` request row; the two join losslessly on
  `(_ng_task_index, _ng_rollout_index)`. `BaseVerifyResponse` extends `BaseVerifyRequest`
  ([base_resources_server.py:87-92](../nemo_gym/base_resources_server.py#L87)), so the row shape *is*
  (a superset of) the verify request shape.
- `/verify` is a plain, unauthenticated HTTP endpoint on every resources server, addressable as
  `http://{host}:{port}` ([server_utils.py:282-283](../nemo_gym/server_utils.py#L282)).
- An existing exact-join utility: `RewardProfiler.align_rows_and_results`
  ([reward_profile.py:62-89](../nemo_gym/reward_profile.py#L62)).

**What's missing (verified absences — these ARE the pain point):**

- **No re-verify command.** The full CLI surface is `eval prepare|run|aggregate|profile`,
  `env init|resolve|validate|packages|test|start|status`, `dataset ...`, `dev test`, `search`
  ([main.py:474-538](../nemo_gym/cli/main.py#L474) shows the eval group). Nothing replays stored rows
  through `verify()`. Grepping the repo for `re-verify|reverify|rescore|rejudge` finds nothing in core.
- **No verify-only path through the agent.** `/verify` is only ever called from inside agent `run()`
  implementations, *after* the policy loop
  ([simple_agent/app.py:188-206](../responses_api_agents/simple_agent/app.py#L188)); `--resume` reuses
  frozen rewards rather than recomputing them
  ([rollout_collection.py:467-473](../nemo_gym/rollout_collection.py#L467)).
- **Verifier inputs are lossy in rollouts.jsonl.** Grading fields survive only if the server's
  VerifyResponse declares them (mcqa drops `options`; `code_gen` likewise declares its grading input
  `verifier_metadata` only on the *request* — [app.py:66](../resources_servers/code_gen/app.py#L66) — and
  its VerifyResponse ([app.py:69-74](../resources_servers/code_gen/app.py#L69)) omits it, so it is dropped
  too). So "just re-POST the rollout row" fails for many servers unless you also join the materialized
  inputs — undocumented tribal knowledge.
- **No resources-server-only startup path.** Shipped configs bundle an agent whose `model_server` ref
  hard-fails config parsing without a model instance
  ([global_config.py:342-345](../nemo_gym/global_config.py#L342)); there is no `--no-model` flag on
  `gym env start`, so you must hand-write a stripped YAML (§2 step 4).
- **Config is frozen at process start.** Children read the entire merged config from the
  `NEMO_GYM_CONFIG_DICT` env var set at `Popen` time ([env.py:198-201](../nemo_gym/cli/env.py#L198)),
  validated once in `run_webserver`
  ([server_utils.py:624-629](../nemo_gym/server_utils.py#L624), via
  [server_utils.py:368-377](../nemo_gym/server_utils.py#L368)). No hot-reload, no `/reload` endpoint,
  no per-request config override field in `BaseVerifyRequest`. Every hyperparameter change costs a full
  stack restart *and* — absent a re-verify tool — a full re-collection.

The net effect: the system design correctly separates verification (a stateless-ish HTTP endpoint) from
rollout generation, but the only shipped *workflow* fuses them, so users conclude the fused cost is
fundamental. Issue #987's observation is that it isn't — the rollouts file already contains the expensive
half.

## 5. Where a fix would hook in

Minimal viable change: a `gym eval verify` (or `re-verify`) command that joins the two artifacts, POSTs
each merged row to a running resources server's `/verify`, and writes a new rollouts file.

1. **CLI registration** — add an entry to the `COMMANDS` dict next to `"eval profile"`
   ([main.py:527-538](../nemo_gym/cli/main.py#L527)), with `--inputs`, `--rollouts`, `--output`, and
   `--server-name` value flags (the `_value_flag` helper pattern is right there).

2. **Config class** — next to `RewardProfileConfig`
   ([reward_profile.py:37-45](../nemo_gym/reward_profile.py#L37)): same two input paths plus
   `output_jsonl_fpath` and `resources_server_name`.

3. **Target function** — in [eval.py](../nemo_gym/cli/eval.py#L422) beside `reward_profile()`. Core loop
   (reusing existing machinery):

   ```python
   pairs = RewardProfiler().align_rows_and_results(rows, results)   # reward_profile.py:62
   server_client = ServerClient.load_from_global_config()           # rollout_collection.py:714 pattern
   for row, result in pairs:
       payload = row | {"response": result["response"]}             # request = input row + response
       res = await server_client.post(server_name=cfg.resources_server_name,
                                      url_path="/verify", json=payload)
       new_result = await get_response_json(res)
       for k in (TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, AGENT_REF_KEY_NAME):
           new_result[k] = row[k]                                   # restamp, as rollout_collection.py:513-515
       out_file.write(orjson.dumps(new_result) + b"\n")
   ```

   `ServerClient.load_from_global_config` resolves server host/port from the head server the same way
   `--no-serve` collection does ([rollout_collection.py:711-714](../nemo_gym/rollout_collection.py#L711)),
   so the command works against any running stack — including a stripped, verifier-only one. Bound
   concurrency with an `asyncio.Semaphore` like the collector does
   ([rollout_collection.py:498-501](../nemo_gym/rollout_collection.py#L498)), then optionally re-run
   aggregate metrics via the existing `_call_aggregate_metrics`
   ([rollout_collection.py:586](../nemo_gym/rollout_collection.py#L586)) or just point users at
   `gym eval profile` with the new file.

4. **Make verifier-only startup ergonomic** — either a `gym env start --no-model` that injects the dummy
   model config `env validate` already uses (`NO_MODEL_GLOBAL_CONFIG_DICT`,
   [global_config.py:165-172](../nemo_gym/global_config.py#L165)), or documentation of the stripped-YAML
   pattern from §2 step 4.

5. **Schema hardening (follow-up, not MVP)** — the lossy-extras problem in §4 argues for either making
   `BaseVerifyResponse` echo the request extras, or standardizing `verifier_metadata` as a declared
   base-schema field, so a rollout row alone is always a valid verify request.

**Stateless vs. stateful caveat** — a re-verify command is only *correct* for verifiers that are pure
functions of `(request body, server config)`. That covers most graders (mcqa, `example_tool_call_multireward`,
`bird_sql` given its local DBs) and LLM judges — though judges re-pay judge inference at verify time
([equivalence_llm_judge/app.py:461-465](../resources_servers/equivalence_llm_judge/app.py#L461)), which is
still far cheaper than re-running the policy. It is wrong for session-stateful verifiers whose `verify()`
reads in-process, cookie-keyed state seeded during the rollout — replaying
`example_session_state_mgmt` silently returns reward 0
([example_session_state_mgmt/app.py:91-97](../resources_servers/example_session_state_mgmt/app.py#L91)) —
and impossible for the gymnasium family, where `verify()` raises `NotImplementedError` and reward accrues
per `/step` ([gymnasium/base.py:114-115](../resources_servers/gymnasium/base.py#L114)). A real fix should
let servers declare replayability (e.g. a `supports_offline_verify` config/class flag) and have
`gym eval verify` refuse loudly otherwise. Precedent that the need is real: `stirrup_agent` grew its own
bespoke `judge_only` rescoring mode over cached deliverables
([stirrup_agent/app.py:829](../responses_api_agents/stirrup_agent/app.py#L829)) because no Gym-level
facility exists.
