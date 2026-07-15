# Pain point 4: Judge-only resume — re-run judge scoring without re-running inference

## 1. TL;DR

In LLM-judge environments (e.g. `equivalence_llm_judge`), the reward is computed by a *second* LLM call made inside `verify()`, synchronously at the tail end of the same `/run` request that produced the expensive policy inference. If that judge call fails hard (bad key, unreachable endpoint, exhausted retries), the exception propagates all the way up and **crashes the entire `gym eval run` process** — the completed inference for that row is never written anywhere, and there is no row, marker, or error field you could later use to re-score just that sample. The only built-in recovery, `gym eval run --resume`, re-dispatches missing rows through the **full agent loop**, re-paying policy inference for samples whose inference already succeeded. The goal state — "given a job with N judge failures, a resume re-scores exactly those N samples with no new inference, and metrics come out identical to a clean run" — has no supported path today; the closest you can get is a hand-rolled script that re-POSTs stored rollout rows to `/verify`.

## 2. Experience it yourself

We use `resources_servers/equivalence_llm_judge` with the hosted policy model config [eccn-llama-3.1-8b.yaml](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml) (an OpenAI-compatible endpoint; needs `INFERENCE_KEY` in your environment). The model is an ECCN-classifier fine-tune, so both its answers and its judge verdicts will be low quality — that is fine, the *workflow* is the point, and any model config can be substituted.

One wiring quirk to know up front: the eccn config's top-level instance key is `eccn-llama-3.1-8b-instruct` ([eccn-llama-3.1-8b.yaml:1](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml#L1)), but the judge environment references a model instance named `policy_model` in **two** places — the agent's `model_server` ([equivalence_llm_judge.yaml:71-73](../resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml#L71)) and the verifier's `judge_model_server` ([equivalence_llm_judge.yaml:5-7](../resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml#L5)). So we override both refs by name.

### Step 0: start the stack

```bash
export INFERENCE_KEY=<your key>
mkdir -p /tmp/judge_resume_demo

# Terminal 1 (stays in the foreground; Ctrl-C to stop)
uv run gym env start \
  --config resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++equivalence_llm_judge_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_model_server.name=eccn-llama-3.1-8b-instruct"
```

Note: the judge *is* the policy model here (both refs point at the same hosted endpoint) — the default config does the same thing via `policy_model`.

### Step 1: a clean, small run

```bash
# Terminal 2
uv run gym eval run --no-serve \
  --agent equivalence_llm_judge_simple_agent \
  --input resources_servers/equivalence_llm_judge/data/example.jsonl \
  --output /tmp/judge_resume_demo/rollouts.jsonl \
  --num-repeats 2 \
  --concurrency 4
```

The input is 5 committed short-answer tasks ([example.jsonl](../resources_servers/equivalence_llm_judge/data/example.jsonl)), so `--num-repeats 2` gives 10 rollouts total. Expected artifacts in `/tmp/judge_resume_demo/`:

- `rollouts.jsonl` — 10 rows, one per `(task, rollout)`.
- `rollouts_materialized_inputs.jsonl` — the 10 expanded input rows (written at [rollout_collection.py:492-494](../nemo_gym/rollout_collection.py#L492)).
- `rollouts_aggregate_metrics.json` — per-agent metrics (written at [rollout_collection.py:670-671](../nemo_gym/rollout_collection.py#L670)).

Inspect the judge outcome per row:

```bash
jq -c '{t: ._ng_task_index, r: ._ng_rollout_index, reward,
        verdicts: [.judge_evaluations[].verdict_label]}' \
  /tmp/judge_resume_demo/rollouts.jsonl
```

Each row carries the full policy `response`, the `reward`, and `judge_evaluations` — the judge's complete raw response plus the parsed `verdict_label` ([app.py:132-141](../resources_servers/equivalence_llm_judge/app.py#L132)). Three outcomes are possible per row:

| What happened | In `rollouts.jsonl` |
|---|---|
| Judge said "equal" | `reward: 1.0`, `verdict_label: "[[A=B]]"` |
| Judge said "not equal" | `reward: 0.0`, `verdict_label: "[[A!=B]]"` |
| Judge *responded* but emitted neither label ("soft failure") | `reward: 0.0`, `verdict_label: null` — coerced to not-equal at [app.py:492-494](../resources_servers/equivalence_llm_judge/app.py#L492) |
| Judge call *errored* (HTTP/transport, "hard failure") | **no row at all** — see step 2 |

With the eccn fine-tune as judge you will likely see `verdict_label: null` rows: `reward` alone cannot tell you "judge said wrong" from "judge never rendered a verdict".

### Step 2: break the judge — and only the judge — then feel the failure mode

Create a second model-server instance pointing at the same endpoint with a bad key, so policy inference keeps working while every judge call 401s:

```bash
cat > /tmp/judge_resume_demo/broken_judge_model.yaml <<'EOF'
broken_judge_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: https://inference-api.nvidia.com/v1
      api_key: not-a-real-key
      model: nvidia/meta/eccn-llama-3.1-8b-instruct
      return_token_id_information: false
      uses_reasoning_parser: false
EOF
```

Ctrl-C the `gym env start` in Terminal 1 and restart it with the broken judge wired in:

```bash
uv run gym env start \
  --config resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  --config /tmp/judge_resume_demo/broken_judge_model.yaml \
  "++equivalence_llm_judge_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_model_server.name=broken_judge_model"
```

Now re-run the eval to a fresh output:

```bash
uv run gym eval run --no-serve \
  --agent equivalence_llm_judge_simple_agent \
  --input resources_servers/equivalence_llm_judge/data/example.jsonl \
  --output /tmp/judge_resume_demo/rollouts_broken.jsonl \
  --num-repeats 2 \
  --concurrency 4
```

Watch what happens:

1. Policy inference for each row **succeeds** (tokens are spent on the hosted endpoint).
2. The first `/verify` hits the broken judge, the resources server returns HTTP 500, and the whole `gym eval run` process **dies with a traceback** (the dispatcher has no try/except — [rollout_collection.py:511](../nemo_gym/rollout_collection.py#L511)). In-flight rollouts are cancelled.
3. `wc -l /tmp/judge_resume_demo/rollouts_broken.jsonl` — likely 0 rows (rows are only flushed *after* verify succeeds inside `/run`). No `_failures.jsonl` sidecar row is written either: the sidecar only receives rows that carry `_ng_failure_class` ([rollout_collection.py:531-535](../nemo_gym/rollout_collection.py#L531)), which `simple_agent` never sets — it just raises.
4. The completed inference for those rows is **gone**. It existed only in memory inside the agent's `/run` handler.

This is the honest answer to "how does a judge failure manifest in rollouts.jsonl": *it doesn't*. A hard judge failure produces an absent row and a dead job; a soft judge failure produces `reward: 0.0` that is indistinguishable from "judge said incorrect" unless you dig into `judge_evaluations[].verdict_label`.

### Step 3: the only built-in recovery re-pays inference

Fix the judge (Ctrl-C, restart Terminal 1 with the step-0 command), then resume:

```bash
uv run gym eval run --no-serve --resume \
  --agent equivalence_llm_judge_simple_agent \
  --input resources_servers/equivalence_llm_judge/data/example.jsonl \
  --output /tmp/judge_resume_demo/rollouts_broken.jsonl \
  --num-repeats 2 \
  --concurrency 4
```

`--resume` (→ `+resume_from_cache=true`, [main.py:491](../nemo_gym/cli/main.py#L491)) diffs the materialized inputs against the output file on `(_ng_task_index, _ng_rollout_index)` ([rollout_collection.py:409-462](../nemo_gym/rollout_collection.py#L409)) and re-dispatches every missing row — **through the full agent `/run`**, i.e. seed_session + policy inference + verify. There is no "response collected, verification pending" state. For our crashed run that means re-generating all ~10 policy responses just to get 10 judge calls. Scale that to a 10k-sample benchmark where 200 judge calls failed against a rate-limited judge endpoint, and you are re-buying 10k inferences (and 9.8k redundant judge calls) to retry 200 judge calls.

### Step 4: the manual judge-only re-score path (for rows that made it to disk)

For rows that *are* in `rollouts.jsonl` — e.g. soft failures (`verdict_label: null`), or any rows you want re-judged after changing judge config — you can replay them against the live resources server's `/verify` yourself, because this verifier is stateless: everything it grades on (`responses_create_params`, `response`, `expected_answer`) is in the row. Find the verifier's port with:

```bash
uv run gym env status   # look for resources_servers/equivalence_llm_judge → port
```

Then (with the stack from step 0 running):

```python
import json, requests

ROLLOUTS = "/tmp/judge_resume_demo/rollouts.jsonl"
VERIFY_URL = "http://127.0.0.1:<PORT>/verify"  # port from `gym env status`

rows = [json.loads(l) for l in open(ROLLOUTS)]

def judge_soft_failed(row):
    # Judge answered but emitted neither label (app.py:492-494)
    return any(e.get("verdict_label") is None for e in row.get("judge_evaluations", []))

merged = []
for row in rows:
    if not judge_soft_failed(row):
        merged.append(row)
        continue
    payload = dict(row)
    # CRITICAL: strip these two. verify() rebuilds its response via
    # LLMJudgeVerifyResponse(**body.model_dump(), reward=..., judge_evaluations=...)
    # (app.py:326-330); since the request model is extra="allow" (app.py:120),
    # leaving them in produces duplicate keyword args -> TypeError -> HTTP 500.
    payload.pop("reward", None)
    payload.pop("judge_evaluations", None)
    r = requests.post(VERIFY_URL, json=payload)
    r.raise_for_status()
    new = r.json()
    # Re-stamp identity keys: the verify response only serializes declared
    # fields, so the collector's bookkeeping keys are dropped by the server.
    for k in ("_ng_task_index", "_ng_rollout_index", "agent_ref"):
        new[k] = row[k]
    merged.append(new)

with open("/tmp/judge_resume_demo/rollouts_rescored.jsonl", "w") as f:
    for row in merged:
        f.write(json.dumps(row) + "\n")
```

Recompute aggregates offline from the merged file:

```bash
uv run gym eval profile \
  --inputs /tmp/judge_resume_demo/rollouts_materialized_inputs.jsonl \
  --rollouts /tmp/judge_resume_demo/rollouts_rescored.jsonl
```

`gym eval profile` never calls `/verify` — it is pure aggregation over the numeric fields already frozen in the rows ([eval.py:421-457](../nemo_gym/cli/eval.py#L421), numeric pickup at [reward_profile.py:151-157](../nemo_gym/reward_profile.py#L151)) — which is exactly why the re-POST step above is necessary and sufficient before profiling. Note what this workaround **cannot** do: recover the hard-failed rows from step 2. Their responses were never persisted, so there is nothing to re-score; only a full inference re-run can bring them back.

## 3. Architecture & code flow

The judge call chain, end to end, for one rollout:

1. **Dispatcher → agent.** The collector POSTs each materialized input row to the agent's `/run` and raises on any non-2xx — [rollout_collection.py:689](../nemo_gym/rollout_collection.py#L689) and [rollout_collection.py:691](../nemo_gym/rollout_collection.py#L691):

   ```python
   res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
   try:
       await raise_for_status(res)
   except Exception:
       ...
       raise
   ```

2. **Agent runs inference, then verification, in one request.** `SimpleAgent.run()` seeds the session ([app.py:179-185](../responses_api_agents/simple_agent/app.py#L179)), runs the (expensive) model loop via its own `/v1/responses` ([app.py:188-194](../responses_api_agents/simple_agent/app.py#L188)), then builds the verify request and POSTs it to the resources server — [app.py:197-207](../responses_api_agents/simple_agent/app.py#L197):

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
   await raise_for_status(verify_response)
   ```

   The verify response is returned verbatim as the `/run` result ([app.py:208](../responses_api_agents/simple_agent/app.py#L208)). Verification is inseparable from inference in this code path.

3. **Resources server `/verify` → judge model.** The route is registered at [base_resources_server.py:138](../nemo_gym/base_resources_server.py#L138); the base contract is just `{responses_create_params, response}` in, `+ reward: float` out ([base_resources_server.py:83-92](../nemo_gym/base_resources_server.py#L83)). `LLMJudgeResourcesServer.verify()` ([app.py:402-436](../resources_servers/equivalence_llm_judge/app.py#L402)) extracts question/expected/generated from the request body alone, then calls the judge model server by name — [app.py:461-466](../resources_servers/equivalence_llm_judge/app.py#L461):

   ```python
   response = await self.server_client.post(
       server_name=cfg.judge_model_server.name,
       url_path="/v1/responses",
       json=responses_create_params,
   )
   judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
   ```

4. **Judge model server → upstream endpoint, with the only status-code retries in the stack.** The vLLM model server converts to chat-completions and calls the hosted endpoint via `NeMoGymAsyncOpenAI` ([vllm_model/app.py:118-129](../responses_api_models/vllm_model/app.py#L118), [openai_utils.py:539-547](../nemo_gym/openai_utils.py#L539)). Retry policy ([openai_utils.py:470-471](../nemo_gym/openai_utils.py#L470), [openai_utils.py:500-524](../nemo_gym/openai_utils.py#L500)): statuses `429/502/503/504/520` retry with a flat 0.5 s sleep and *bump* `max_num_tries` each time (effectively unbounded — a rate-limited judge stalls rather than fails), plain `500` is capped at `MAX_NUM_TRIES = 3` ([server_utils.py:150](../nemo_gym/server_utils.py#L150)), and anything else (401, 404, 400…) is not retried at all. Separately, the shared `request()` helper retries *transport* errors ([server_utils.py:164-220](../nemo_gym/server_utils.py#L164)); between Gym servers there is **no** HTTP-status retry — a 500 from `/verify` propagates immediately.

5. **Failure propagation.** On a hard judge error, the judge server's exception handler prints a DEBUG line and **re-raises** — it does not return `reward=0` ([app.py:467-472](../resources_servers/equivalence_llm_judge/app.py#L467)):

   ```python
   except Exception as e:
       print(f"DEBUG: LLMJudgeResourcesServer: judge model server HTTP POST error: {type(e).__name__} {e}", flush=True)
       raise e
   ```

   The resources server's exception middleware turns that into HTTP 500 with `repr(e)` as the body ([server_utils.py:511-538](../nemo_gym/server_utils.py#L511)); the agent's `raise_for_status` at [app.py:207](../responses_api_agents/simple_agent/app.py#L207) re-raises; the agent's own middleware turns *that* into a 500 on `/run`; and the dispatcher's `row, result = await future` has no try/except ([rollout_collection.py:511](../nemo_gym/rollout_collection.py#L511)), so `asyncio.run(...)` in `collect_rollouts` ([eval.py:405-410](../nemo_gym/cli/eval.py#L405)) crashes the process. (`math_with_judge` is even barer: its judge POST is not wrapped in try/except at all — [math_with_judge/app.py:348-353](../resources_servers/math_with_judge/app.py#L348).)

6. **What gets persisted, and when.** Only after `/run` returns 2xx does the result get identity-stamped and appended — [rollout_collection.py:513-539](../nemo_gym/rollout_collection.py#L513):

   ```python
   result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
   result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
   result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
   ...
   elif failure_class is not None:
       failures_file.write(serialized + b"\n")     # sidecar (agents that set _ng_failure_class)
   else:
       results_file.write(serialized + b"\n")      # main rollouts.jsonl
   ```

   The failures sidecar exists ([rollout_collection.py:58-86](../nemo_gym/rollout_collection.py#L58), [rollout_collection.py:110-112](../nemo_gym/rollout_collection.py#L110)) but is only fed by agents that catch verify failures and stamp `_ng_failure_class` — today that is `stirrup_agent` alone; `simple_agent` never does.

7. **Resume.** `_load_from_cache` ([rollout_collection.py:409-462](../nemo_gym/rollout_collection.py#L409)) computes `gated = successes_seen | terminal_keys | maxed_out` ([rollout_collection.py:445](../nemo_gym/rollout_collection.py#L445)) and re-dispatches everything else — through step 1 again, full inference included.

8. **Offline metrics.** `gym eval profile` ([main.py:527-538](../nemo_gym/cli/main.py#L527) → [eval.py:421-457](../nemo_gym/cli/eval.py#L421)) joins inputs↔rollouts on the rollout key ([reward_profile.py:62-89](../nemo_gym/reward_profile.py#L62); partial mode via `++allow_partial_rollouts=True`, [reward_profile.py:42-45](../nemo_gym/reward_profile.py#L42)) and aggregates the frozen `reward` ([reward_profile.py:141-142](../nemo_gym/reward_profile.py#L141)). It never calls a server.

## 4. Why it hurts / gap analysis

**What exists:**

- Judge-stateless verification: `equivalence_llm_judge.verify()` needs only the row itself, so re-judging without inference is *architecturally possible* ([app.py:402-436](../resources_servers/equivalence_llm_judge/app.py#L402)).
- Raw judge evidence in rollouts: `judge_evaluations` keeps the judge's full response and parsed verdict ([app.py:132-141](../resources_servers/equivalence_llm_judge/app.py#L132)) — enough to *detect* soft failures after the fact.
- Rollout-level resume with attempt caps and a failures sidecar ([rollout_collection.py:409-462](../nemo_gym/rollout_collection.py#L409)).
- Offline metric recomputation (`gym eval profile`, `gym eval aggregate`).

**What is missing (verified absences — do not go looking for these):**

1. **No re-verify/re-judge command.** The full CLI surface is `eval prepare|run|aggregate|profile`, `env init|resolve|validate|packages|test|start|status`, `dataset ...`, `dev test`, `list`, `search` ([main.py:313-540](../nemo_gym/cli/main.py#L313)). Nothing replays stored rows through `/verify`.
2. **No verify-only path in the agent.** `simple_agent.run()` always runs inference before verification ([app.py:188-194](../responses_api_agents/simple_agent/app.py#L188)); you cannot hand it an existing `response`.
3. **No judge-failure marker in the schema.** `BaseVerifyResponse` is request + `reward: float` ([base_resources_server.py:91-92](../nemo_gym/base_resources_server.py#L91)); `LLMJudgeVerifyResponse` adds `expected_answer` and `judge_evaluations` ([app.py:139-141](../resources_servers/equivalence_llm_judge/app.py#L139)) — no `error`/`judge_status` field anywhere. `reward=0.0` conflates "wrong" with "judge never answered usably".
4. **Hard judge failures destroy the inference.** The response exists only inside the agent's `/run`; on verify failure nothing persists it, and the crash takes the rest of the batch's in-flight rollouts with it ([rollout_collection.py:511](../nemo_gym/rollout_collection.py#L511)).
5. **Resume granularity is whole-rollout.** `--resume` reuses completed rows and re-runs everything else end to end; there is no "inference done, judge pending" state to resume from.

Why the design forces the expensive path: the reward is computed synchronously *inside* the same HTTP request that consumed the model tokens, and persistence happens only after that request fully succeeds. Inference and judgment are fused into one atomic, unrecoverable unit. With a limited judge quota, every judge retry therefore costs a full inference — the exact inversion of what you want when the judge is the flaky, rate-limited component.

## 5. Where a fix would hook in

A first-class "resume judge-failed" needs three pieces: a **failure marker that preserves the response**, an **idempotent re-verify path**, and a **metrics merge**. Minimal viable change, file by file:

1. **Persist inference on verify failure (agent side).** In `SimpleAgent.run()`, wrap the `/verify` POST ([app.py:201-207](../responses_api_agents/simple_agent/app.py#L201)) in try/except; on failure return a payload containing the already-computed `response`, `reward=0.0`, an `error_message`, and `_ng_failure_class="verify"`. The dispatcher's existing routing ([rollout_collection.py:531-535](../nemo_gym/rollout_collection.py#L531)) then lands it in `<stem>_failures.jsonl` instead of crashing the run. `stirrup_agent` already implements exactly this pattern, including transient-vs-legitimate classification ([stirrup_agent/app.py:278-294](../responses_api_agents/stirrup_agent/app.py#L278), [stirrup_agent/app.py:1383-1415](../responses_api_agents/stirrup_agent/app.py#L1383)) — lift it into `simple_agent`. This alone converts "crashed job, lost inference" into "completed job with N recoverable failure rows".

2. **Judge-only resume (dispatcher side).** Teach `_load_from_cache` ([rollout_collection.py:409-462](../nemo_gym/rollout_collection.py#L409)) to recognize sidecar rows with `_ng_failure_class == "verify"` that carry a non-null `response`: instead of re-dispatching the input row to `/run`, dispatch the *failure row* to a verify-only path. Two options for that path:
   - a new agent route (e.g. `/verify_only`) beside `/run` in [base_responses_api_agent.py:50](../nemo_gym/base_responses_api_agent.py#L50) that skips straight to the verify POST; or
   - simpler, have `run()` detect a `response` already present on the (extra-allowing, [app.py:51-52](../responses_api_agents/simple_agent/app.py#L51)) run request and skip the `/v1/responses` call at [app.py:188-194](../responses_api_agents/simple_agent/app.py#L188).

3. **Standalone re-verify CLI (`gym eval verify` or `--rejudge`).** Add a function beside `reward_profile` in [eval.py:421](../nemo_gym/cli/eval.py#L421) that reads a rollouts JSONL, filters rows by a predicate (missing keys against materialized inputs, `_ng_failure_class=="verify"` sidecar rows, or `judge_evaluations[].verdict_label == null`), POSTs each to the resources server's `/verify` via `ServerClient` (resolved from `agent_ref` → agent config → `resources_server` ref), merges results back keyed on `(_ng_task_index, _ng_rollout_index)` (reuse `RewardProfiler._index_by_rollout_key`, [reward_profile.py:53-60](../nemo_gym/reward_profile.py#L53)), rewrites the rollouts file, and re-runs `_call_aggregate_metrics` ([rollout_collection.py:586-673](../nemo_gym/rollout_collection.py#L586)). Register it in the `COMMANDS` dict near [main.py:527](../nemo_gym/cli/main.py#L527). Because the merge is keyed and the aggregate is recomputed from the merged file, metrics come out identical to a clean run — the acceptance criterion.

4. **Make `/verify` replay-safe (verifier side).** Two small fixes in `equivalence_llm_judge`:
   - `_make_response` should pop `reward` and `judge_evaluations` from the echoed payload ([app.py:326-330](../resources_servers/equivalence_llm_judge/app.py#L326)) so re-POSTing a stored rollout row doesn't 500 with a duplicate-kwarg `TypeError` (the gotcha in §2 step 4);
   - optionally catch judge HTTP errors at [app.py:467-472](../resources_servers/equivalence_llm_judge/app.py#L467) and return a marked evaluation (e.g. `verdict_label: "JUDGE_ERROR"` + a new `judge_error: str` field on `LLMJudgeVerifyResponse`, [app.py:139-141](../resources_servers/equivalence_llm_judge/app.py#L139)) instead of raising — turning hard failures into detectable, re-scorable rows even without the agent change. If you do this, keep such rows out of headline metrics by also stamping `_ng_failure_class` so they route to the sidecar.

5. **Schema (optional, ecosystem-wide).** A `verify_error: Optional[str] = None` on `BaseVerifyResponse` ([base_resources_server.py:91-92](../nemo_gym/base_resources_server.py#L91)) would give every environment a standard place to record "verification infrastructure failed" distinctly from "task failed", which `gym eval profile` could then surface (today non-scalar/absent fields are simply invisible to it, [reward_profile.py:151-157](../nemo_gym/reward_profile.py#L151)).

Start with (1) + (4): they are small, local, and immediately stop the bleeding (no more lost inference, failures become identifiable). (2)/(3) then build judge-only resume on top of data that finally exists.
