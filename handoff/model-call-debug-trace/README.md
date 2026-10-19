# Model-call debug trace: handoff for testing with NeMo RL

This folder is a handoff for an engineer or coding agent who wants to understand the
model-call debug trace and test it with their own NeMo RL code. It is not meant to be
merged: drop the `handoff/` commit before opening a PR.

## What it does

With one Gym config block enabled, every Gym model-server process appends one JSON line
per model call to `<dir>/<model_server>.<pid>.jsonl`. Each line holds the request the
agent sent, the response the agent received, and the call's prompt token ids, generated
token ids, and generation log probabilities. This works with NeMo RL token capture
enabled (external staging) or disabled. It is a debug aid: it never fails or delays a
model call, and agents receive the same responses with tracing on or off.

```yaml
# Gym global config. In NeMo RL, put it under env.nemo_gym.
model_call_debug_trace:
  enabled: true
  dir: /absolute/shared/path/model-call-debug
  request_token_data: true   # default; see "How tokens are obtained"
```

From the NeMo RL command line:

```bash
++env.nemo_gym.model_call_debug_trace.enabled=true \
++env.nemo_gym.model_call_debug_trace.dir=/absolute/shared/path/model-call-debug
```

## Where the code is

| Repo | Location | Contents |
|---|---|---|
| Gym | branch `pthombre/model-call-debug-trace`, commit `feat(model-server): add per-model-call debug trace with token data` (on top of main `aa5d277e3`) | The trace itself. |
| NeMo RL | `rl-changes-main-e65b8b8b8.patch` in this folder (applies to RL main `e65b8b8b8`, 2026-10-19) | Worker echo of staged tokens, and rollout ids without token capture. |
| Both, older pins | `gym-changes-pin-17abf2dd.patch`, `rl-changes-pin-50f6b5d6.patch` | The same changes ported to Gym `17abf2dd` and RL `50f6b5d6`, the pins inside the 2026-10-02 `nemo-rl-nightly-gym` image. These are the versions the toy runs below exercised. |

### Gym files

| File | Role |
|---|---|
| `nemo_gym/model_call_debug_trace.py` (new) | Settings model, per-call `CallTrace` context, token extraction, the background JSONL writer, and the `request_engine_token_data` / `note_engine_response` hooks. |
| `nemo_gym/base_responses_api_model.py` | `SimpleResponsesAPIModel` builds the writer in `setup_webserver` and wraps `_invoke_chat_completions` / `_invoke_responses`. Every model server and dialect (Chat Completions, Responses, Anthropic Messages, SSE or not) passes through these two methods. The capture middleware publishes its `model_call_id` in request state, so trace lines join model-call capture records. |
| `responses_api_models/vllm_model/app.py` | Asks the engine for token data (`_preprocess_chat_completion_create_params`) and records the raw engine response before any conversion or stripping (`chat_completions`). |
| `nemo_gym/token_id_capture/sink.py`, `__init__.py`, `external_capture.py` | `NG_DEBUG_TOKENS_FIELD = "ng_debug_tokens"`, exported. External staging strips the field from responses, so it never reaches an agent. |
| `nemo_gym/global_config.py` | `model_call_debug_trace` is a reserved top-level key. Otherwise Gym parses the block as a server instance. |
| `fern/.../model-server/model-call-capture.mdx` | User docs ("Debug trace" section). |

### NeMo RL files (in the patch)

| File | Role |
|---|---|
| `nemo_rl/models/generation/vllm/vllm_worker_async.py` | `NeMoRLChatCompletionRequest` accepts `ng_debug_tokens`. `_finish_request_capture` stages the call as before, then, if the request set the flag, returns the staged prompt ids, generated ids, and log probabilities under `ng_debug_tokens`. |
| `nemo_rl/models/generation/megatron/token_capture.py` | `TQMegatronTokenStager.stage` reads `offload_params["ng_debug_tokens"]` and adds the same object to `response_metadata`. Megatron-LM merges stager metadata into the HTTP response top level (`attach_stage_metadata`), while it drops the payload's own token arrays for offloaded calls. |
| `nemo_rl/environments/nemo_gym.py` | Without token capture, RL never stamped `_ng_rollout_id`, so agents sent model calls without `/ng-rollout/<id>` and the trace had no rollout to group by. When the trace is enabled and capture is not, `NemoGym` stamps `<group_id>_g<generation_index>_t<group_attempt>` on each row. |
| `docs/design-docs/token-capture-ledger.md` | "Debug echo of staged tokens" section. |
| Tests | `tests/unit/models/generation/test_vllm_token_capture_hosting.py`, `tests/unit/data_plane/test_tq_token_sink.py`, `tests/unit/environments/test_nemo_gym_token_capture.py`. |

## Wire contract between Gym and the inference worker

External staging only, admitted calls:

1. Gym's `vllm_model` sets `ng_debug_tokens: true` on the engine request: top-level for the
   vLLM worker, inside `offload_params` for Megatron Inference.
2. The worker stages as usual. It then adds a top-level response field:
   ```json
   "ng_debug_tokens": {
     "prompt_token_ids": [...],
     "generation_token_ids": [...],
     "generation_log_probs": [...],
     "error": null
   }
   ```
   `error` is a string when the payload could not be read. The echo never changes what
   is staged or committed.
3. Gym reads it in `note_engine_response` and removes it before the response reaches the agent.

An RL worker without the patch ignores the flag. vLLM drops unknown request fields with a
warning, and the Megatron stager ignores unknown `offload_params` keys. The trace line then
has `tokens: null` for that call.

Uncaptured calls (token capture disabled, or a call the ledger did not admit) use the
engine's ordinary fields instead: `logprobs=true, top_logprobs=0,
return_tokens_as_token_ids=true`, plus `return_token_ids=true` for vLLM or
`return_tokenized_data=true` for Megatron. The model server removes whatever only exists
because tracing added it. `request_token_data: false` turns this off. It is skipped
anyway when `return_token_id_information: true`, because that path already returns token
data on the served response.

## Record schema

One JSON object per line:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `trace_id` | Unique per line. |
| `rollout_id` | From `/ng-rollout/<id>`; with token capture this is the TQ sample id. |
| `model_call_id` | The capture middleware's id. Equal to the staging key suffix and to the model-call capture record id. |
| `model_server`, `pid`, `path` | `path` is the client-facing route, such as `/v1/responses` or `/v1/messages`. |
| `started_at`, `completed_at`, `latency_ms` | Wall-clock timing of the model call. |
| `external_staging`, `commit_coords` | The worker's commit coordinates (`disposition`, `prev_len`, `delta_len`, `cum_len`, `staging_key`, ...). |
| `request` | Body the agent sent (Responses or Chat params), with `exclude_unset`. |
| `response` | Response the agent received (text, tool calls, reasoning), with token arrays removed. |
| `tokens` | `{prompt_token_ids, generation_token_ids, generation_log_probs}` or `null`. |
| `tokens_source` | `engine_staged` (worker echo), `served_response` (`return_token_id_information`), or `engine_response` (raw engine fields). |
| `token_errors` | Token data that could not be read. |
| `error` | `{type, message}` when the call raised. |

`prompt_token_ids` is the full prompt the engine ran: prefix plus delta in token-in
mode. For a staged call, `len(prompt) + len(generation) == commit_coords.cum_len`.

## Test plan

### 1. Unit tests

Gym, from the Gym checkout with dev dependencies:

```bash
pytest tests/unit_tests/test_model_call_debug_trace.py -q
pytest responses_api_models/vllm_model/tests/test_app.py -k debug_trace -q
```

NeMo RL. These tests are marked `nemo_gym`, so pass `--nemo-gym-only`. Run each in a venv
that has both `nemo_gym` and the code under test:

```bash
pytest --nemo-gym-only tests/unit/models/generation/test_vllm_token_capture_hosting.py
pytest --nemo-gym-only tests/unit/data_plane/test_tq_token_sink.py -k megatron_stager
pytest --nemo-gym-only tests/unit/environments/test_nemo_gym_token_capture.py
```

In the nightly image these venvs are, respectively,
`/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker`,
`/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker`,
and `/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym`. They have no pytest: put the
driver venv's `pytest`, `_pytest`, `pluggy`, and `iniconfig` packages on `PYTHONPATH`, and
add `/opt/nemo-rl` and the Gym checkout to `PYTHONPATH`.

Results on the 2026-10-02 image with the pin patches: Gym 193 passed (trace, base model
server, external capture, messages, streaming suites). `vllm_model` 160 passed, plus one
unrelated failure in `TestEndpointFile::test_unpublished_endpoint_grace_lifecycle`, code
the change does not touch. RL: vLLM hosting 22, Megatron stager 11, NemoGym env 27, all
passed.

### 2. Toy training runs (the real check)

`run-toy-grpo.sh` runs 3 GRPO steps of Qwen3-0.6B on workplace_assistant, a multi-step
tool-calling environment with up to 6 model calls per rollout. It runs on one node with
2 GPUs (1 Megatron train + 1 vLLM generation). It enables the trace and model-call capture,
then runs `validate_trace.py`. Run it twice, once per mode:

```bash
WORK_DIR=/abs/shared/dir MODEL_PATH=/abs/Qwen3-0.6B DATA_PATH=/abs/workplace_toy.jsonl \
  bash run-toy-grpo.sh capture     # token capture on: tokens must come from the worker echo
WORK_DIR=... bash run-toy-grpo.sh nocapture   # off: tokens come from the served response
```

Build the dataset from Gym's example rows. Rows need `agent_ref`, which
`ng_prepare_data` normally stamps:

```python
import json
rows = [json.loads(line) for line in open("resources_servers/workplace_assistant/data/example.jsonl")]
with open("workplace_toy.jsonl", "w") as out:
    for _ in range(4):
        for row in rows:
            row = dict(row, agent_ref={"type": "responses_api_agents", "name": "workplace_assistant_simple_agent"})
            out.write(json.dumps(row) + "\n")
```

`validate_trace.py` fails the run unless every check below holds:

- every call succeeded and has a rollout id, a model-call id, the request, and response text;
- every call has prompt ids, generated ids, and log probabilities. The id and log-prob
  lengths match, and every log prob is finite and at most 0;
- `tokens_source` is `engine_staged` (capture) or `served_response` (nocapture);
- the generated ids decode to the text the agent received, for at least 90% of calls.
  Tool-argument serialization may legitimately differ;
- capture mode: every call is `staged`, and the echoed lengths equal `commit_coords.cum_len`;
- per rollout, trace model-call ids equal model-call capture ids, so no call is missing or extra;
- at least one rollout made several model calls.

It also reports how many later prompts extend the previous prompt plus generation exactly.

Reference results (Slurm, GB300, image `20261002-104438-nemo-rl-nightly-gym.sqsh`, pin patches):

| | capture (job 1334375) | nocapture (job 1334631) |
|---|---|---|
| training | 3 steps, exit 0 | 3 steps, exit 0 |
| traced calls / rollouts | 32 / 26 | 41 / 32 |
| tokens present, expected source | 32/32 `engine_staged` | 41/41 `served_response` |
| decoded generation matches agent text | 32/32 | 41/41 |
| echo equals staged chain length | 32/32 | n/a |
| trace vs capture mismatches | 0 | 0 |
| max calls per rollout; exact-prefix continuations | 4; 6/6 | 4; 9/9 |
| agent-visible responses carrying token data or the echo | 0/32 | not checked |

The run takes about 3 minutes after startup.

### 3. Check by hand

```python
import collections, glob, json
rows = [json.loads(line) for path in glob.glob(f"{trace_dir}/*.jsonl") for line in open(path)]
by_rollout = collections.defaultdict(list)
for row in rows:
    by_rollout[row["rollout_id"]].append(row)
# For a staged multi-call rollout, prev_len of each call equals cum_len of the previous one.
```

## Running in the nightly container (what worked)

- The image bakes RL at `/opt/nemo-rl` (pin `50f6b5d6`), the driver venv at
  `/opt/nemo_rl_venv` (not `/opt/nemo-rl/.venv`), Ray actor venvs in `/opt/ray_venvs/`, and
  Gym server venvs at `/opt/gym_venvs/<kind>/<server>/.venv`. Those venvs install
  `nemo_gym` editable from `/opt/nemo-rl/3rdparty/Gym-workspace/Gym`.
- To test without rebuilding the image, mount the patched Gym over
  `/opt/nemo-rl/3rdparty/Gym-workspace/Gym` and the patched `nemo_rl/` over
  `/opt/nemo-rl/nemo_rl`. Keep the image's Gym and RL revisions (the pin patches), so the
  baked venvs' dependencies still match. Set `NEMO_GYM_VENV_DIR=/opt/gym_venvs` and
  `++env.nemo_gym.skip_venv_if_present=true`.
- RL `main` (2026-10-19) does not run in the 2026-10-02 image: `nemo_rl.telemetry` needs a
  newer `nemo.lens`. Use an image built from your RL revision, or the pin patches.
- Gym `main` with an older image's venvs may hit dependency skew. Prefer matching revisions.

## Limitations

- Engine token requests and the raw-engine read are implemented in `vllm_model`'s chat path,
  which RL training uses. Other model servers, and `vllm_model` with `use_completions_api`
  or `is_responses_native`, record only what their served response carries.
- The Megatron echo is unit-tested only; no end-to-end Megatron-generation run was done.
- Each line stores the full request, so long multi-turn rollouts grow roughly
  quadratically on disk. Lines are flushed, not fsynced.
- Without token capture, rollout ids come from the RL patch (`_stamp_trace_rollout_ids`).
  Other frameworks must send `_ng_rollout_id`, or the task and rollout indices, for
  grouping. Otherwise `rollout_id` is `null`, but tokens are still recorded.
