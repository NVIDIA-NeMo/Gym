# Gym and Switchyard OpenHands MVP

**Status:** Implementation handoff

**Last reviewed:** 2026-07-14

**Gym base:** `main` at `1b7d98a4`

**Gym implementation branch:** `feature/switchyard-openhands-integration`

**Switchyard implementation:** `drifold/feature/token-capture` at `1365ca0`
([NVIDIA-NeMo/Switchyard#63](https://github.com/NVIDIA-NeMo/Switchyard/pull/63))

**OpenHands base:** Gym's pinned
[`nv-OpenHands` commit](https://github.com/sdevare-nv/nv-OpenHands/tree/5f0180054732945df08ad2293903e6873f0492b6)

## Objective

Run Gym's existing OpenHands SWE harness through Switchyard, capture exact vLLM
prompt tokens, generated tokens, and sampled-token log probabilities for every policy
call, and return them in Gym's existing trainer-facing Responses API representation.

The MVP keeps the existing responsibilities:

- OpenHands owns the agent loop, tools, patch production, and completion logs.
- Gym owns SWE evaluation, reward, trace validation, and format adaptation.
- Switchyard owns proxying, exact token capture, durable per-call records, and session
  retrieval.

The expected path is:

~~~text
Gym SWEBenchWrapper
  -> OpenHands agent in Apptainer
  -> Switchyard
  -> one vLLM target
~~~

## Important baseline decision

This implementation starts from Gym `main`. It does not depend on
`feat/model-server-observability`.

Do not use or port:

- `CaptureStore`
- `ModelCallRecord`
- model-call capture middleware
- `/ng-rollout/<id>/...` URL routing
- `ng_model_call_capture`

Those types describe Gym model-server observability. Switchyard is the capture source
for this integration, and its records contain the token data needed for training.

The main branch already has a unique per-run setup path in
`SWEBenchWrapper._setup_params`. The MVP adds a Switchyard UUID there instead of
changing generic rollout collection or importing retry-ID plumbing from the
observability branch.

## Existing code to reuse

### Gym

The current implementation is concentrated in
`responses_api_agents/swe_agents/app.py`:

- `SWEBenchWrapperConfig` owns harness configuration.
- `SWEBenchWrapperInstanceConfig` is serialized into response metadata and returned
  as `instance_config`.
- `OpenHandsHarnessProcessor` builds the agent-container script.
- `SWEBenchWrapper._setup_params` creates unique per-run state.
- `SWEBenchWrapper._inner_responses` builds the current OpenHands trajectory.
- `SWEBenchWrapper.run` creates the final
  `responses_create_params`, `response`, `reward`, and `instance_config`.
- `VLLMConverter(return_token_id_information=True)` converts Chat Completions
  messages into Gym Responses API items.
- `TokenIDLogProbMixin` is the existing training contract for
  `prompt_token_ids`, `generation_token_ids`, and
  `generation_log_probs`.

The current OpenHands configuration pins
`sdevare-nv/nv-OpenHands@5f0180054732945df08ad2293903e6873f0492b6`.
The fork is cloned at setup time; its client is not vendored in Gym.

### Switchyard

PR #63 already provides:

- `target.token_capture_engine: vllm`
- capture activation through `--enable-rl-logging` and `--rl-log-dir`
- correlation through the `proxy_x_session_id` request header
- one JSON completion record per model call
- exact prompt IDs, generation IDs, and generation log probabilities
- buffered upstream responses with a normal synthesized client stream
- `GET /v1/sessions/{session_id}/completions`

The retrieval envelope is `{schema_version, session_id, completions}` — the record
list field is named `completions`. Records are returned ordered by
(`captured_at`, `uuid`). The Gym adapter still validates message-history continuity
and never repairs a broken chain using timestamps.

## MVP scope

The first implementation supports:

- `responses_api_agents/swe_agents`
- `agent_framework: openhands`
- the pinned `nv-OpenHands` fork
- non-replay seed rollouts
- the sequential main-agent session only
- one Switchyard route and one vLLM target
- post-agent HTTP retrieval
- one normal Gym rollout with one token-annotated generated item per model call

The first implementation excludes:

- OpenCode and other Gym harnesses
- replay-prefix or continuation rollouts
- subagent reconstruction
- parallel or branching policy calls
- context compaction or earlier-message rewriting
- SGLang and multiple routed targets
- Polar-style flattened arrays or token-prefix merging
- changes to Gym's generic rollout collection

Unsupported trace shapes fail closed and mask the sample. They do not produce a
partially annotated training trajectory.

## Runtime configuration

### Switchyard

Use the existing Gym model-server name as the Switchyard route ID. Gym's current
OpenHands configs use `policy_model`.

~~~yaml
routes:
  policy_model:
    type: model
    target:
      model: <actual-model-served-by-vllm>
      base_url: http://<vllm-host>:<port>/v1
      api_key: dummy
      format: openai
      token_capture_engine: vllm
~~~

Start the legacy route-bundle path so the Python processor chain is installed:

~~~bash
switchyard \
  --routing-profiles single-vllm.yaml \
  --enable-rl-logging \
  --rl-log-dir /data/switchyard-traces/<job-id> \
  -- serve
~~~

Do not replace this with `serve --config`; that path runs the Rust profile server,
which has no Python processor chain and rejects `--enable-rl-logging` outright.

The session-retrieval endpoint is registered only on this capture-enabled serve
path. Records are plain JSON files under `--rl-log-dir`, so retrieval does not
require the exact process that captured them: any capture-enabled Switchyard
process pointed at the same log directory can serve
`GET /v1/sessions/...`, including after a proxy restart.

### Gym

Add one optional setting to `SWEBenchWrapperConfig`:

~~~yaml
swe_agents:
  responses_api_agents:
    swe_agents:
      agent_framework: openhands
      switchyard_base_url: http://<switchyard-host>:4000
      model_server:
        name: policy_model
        type: responses_api_models
~~~

When `switchyard_base_url` is absent, behavior remains unchanged. When present for
an OpenHands run, policy calls go directly to Switchyard.

The URL must be reachable from both:

- the Gym wrapper process, for retrieval; and
- the OpenHands Apptainer container, for model calls.

## Session identity

Do not derive the Switchyard session from task indices, the SWE instance ID, or the
observability branch's rollout ID helper.

In `SWEBenchWrapper._setup_params`, generate:

~~~python
switchyard_session_id = uuid.uuid4().hex
~~~

Add `switchyard_session_id: Optional[str] = None` to
`SWEBenchWrapperInstanceConfig`. Populate it only when all of these are true:

- `switchyard_base_url` is configured;
- `agent_framework == "openhands"`; and
- this is a normal agent run rather than golden-patch verification
  (`verify_golden_patch` is false).

A lowercase UUID hex string is URL-safe, maps safely through Switchyard's current
session-directory sanitizer, and is unique across retries. It is serialized in the
existing `instance_config`, so `SWEBenchWrapper.run` can retrieve the same session
after `responses` returns without introducing shared mutable state.

## OpenHands transport change

Change the existing `NemoGymClient._post_completion` in the external
`nv-OpenHands` fork. Do not add a second client or a local compatibility proxy.

The direct path is:

1. Build `params` exactly as the client does today.
2. If `SWITCHYARD_BASE_URL` is absent, keep the existing
   `ServerClient.post` behavior.
3. If it is present:
   - require `SWITCHYARD_SESSION_ID`;
   - copy `params`;
   - replace `params["model"]` with `NEMO_GYM_MODEL_SERVER_NAME`;
   - POST to `<base-url>/v1/chat/completions`;
   - send `proxy_x_session_id: <session-id>`;
   - make exactly one HTTP attempt — no transport retries.
4. Reuse the client's existing response-status handling, JSON parsing, cookies,
   metrics, and completion-log writing.

The single-attempt rule exists because an attempt that reaches vLLM writes a
capture record even when the client never receives the response. A transport
retry can therefore leave two records with the same prompt history; strict-history
reconstruction would refuse the session and mask the sample. Gym's
`ServerClient.post` and `nemo_gym.server_utils.request` both retry with backoff,
so the direct path must use the shared aiohttp client without the retry wrapper
and let a failed call fail the run through existing OpenHands error handling.

The model rewrite is required. Switchyard dispatches on the incoming OpenAI
`model` field, while OpenHands normally sends the underlying LLM model string.
`NEMO_GYM_MODEL_SERVER_NAME` is already exported by the Gym harness and exactly
matches the `policy_model` route in the example. `params["model"]` always exists:
`_nemo_gym_llm_kwargs` is built with `model=self.config.model`.

No message-format change is needed. On the direct path the response message never
carries `prompt_token_ids`, `generation_token_ids`, or `generation_log_probs` —
those are attached by Gym's model server, not vLLM — so the client's existing
token-field stripping loop is a no-op and every captured prompt history stays
strictly extending.

Gym must export two additional values only into the agent container:

~~~text
SWITCHYARD_BASE_URL=<configured-base-url>
SWITCHYARD_SESSION_ID=<generated-uuid>
~~~

The evaluator container does not make policy calls and does not receive these values.

After the fork change is pushed, update every OpenHands pin in Gym:

- `responses_api_agents/swe_agents/configs/swebench_openhands.yaml`
- `responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml`
- `responses_api_agents/swe_agents/configs/swebench_multi_tools.yaml`

## Gym retrieval and adaptation

Add a small module at
`responses_api_agents/swe_agents/switchyard_trace.py`. Keep network orchestration
in `SWEBenchWrapper.run`; keep schema validation and conversion in the module.

After `SWEBenchWrapper.responses` returns, `run` already reads the serialized
`instance_config`. If it contains a Switchyard URL and session UUID, request:

~~~text
GET <switchyard-base-url>/v1/sessions/<session-uuid>/completions
~~~

Use Gym's existing async HTTP helper (`nemo_gym.server_utils.request`) with a
bounded timeout; its retries are safe here because the GET is idempotent. No polling
or session-finalization protocol is needed: OpenHands has exited, and Switchyard
durably writes each record (atomic rename) before returning that call's model
response.

### Required session validation

Reject the session unless:

- the envelope has `schema_version == 1`;
- the envelope session ID equals the requested UUID;
- the `completions` list is non-empty;
- every record has `schema_version == 1`;
- every record has the requested session ID and a unique UUID;
- every record has `is_valid == true`;
- every record has non-empty `request_id` and `model`;
- prompt and generation IDs are non-empty integer arrays;
- generation IDs and finite generation log probabilities have equal lengths;
- messages are non-empty and end with an assistant message; and
- model, tools, and tool choice are consistent across the session.

Process records in the endpoint's returned order, but require each later prompt
history to strictly extend the already reconstructed history. Capture time and UUID
are not permitted to repair a divergent or reordered history.

### Reconstruction algorithm

For each record:

1. Split `messages` into `prompt_messages = messages[:-1]` and the final
   assistant message.
2. Copy the record's token triple onto that assistant Chat Completions message.
3. For the first record, initialize the sequence with its prompt plus assistant.
4. For each later record, require its prompt to begin with the entire assembled
   sequence and to contain a non-empty environment-message suffix.
5. Append only that suffix and the new assistant message.
6. Convert the assembled messages with the existing
   `VLLMConverter(return_token_id_information=True)`.
7. Use the existing `split_responses_input_output_items` helper.

Gym's converter places the token triple on the final generated Responses API item
for one assistant message. A message containing tool calls can produce multiple
Responses items, but exactly one generated item retains that call's token triple.

Map Switchyard tools as follows:

| Switchyard | Gym function tool |
|---|---|
| `id` | `name` |
| `description` | `description` |
| `inputSchema.jsonSchema` | `parameters` |
| implicit tool kind | `type: "function"` |

On success:

- replace `responses_create_params.input` with reconstructed input items;
- replace `responses_create_params.tools` with mapped Switchyard tools;
- replace `response.output` with reconstructed output items;
- keep Gym's existing response identity and reward;
- attach compact response metadata containing the Switchyard session ID, source,
  record UUIDs, and captured model. Keep metadata values as strings; JSON-encode the
  UUID list.

Do not attach the raw completion envelope to the rollout. Switchyard remains the
durable diagnostic source.

### Failure behavior

Retrieval, schema, or reconstruction failure must not erase the existing OpenHands
patch, evaluation metrics, response text, or reward.

Instead:

- keep the current OpenHands-derived response;
- set the existing `SWEBenchWrapperInstanceConfig.mask_sample` flag to true — this
  is Gym's established training-mask convention, already used for agent failures;
- add `switchyard_trace_error: Optional[str]` to `SWEBenchVerifyResponse`,
  following the existing `agent_error_kind`-style metrics naming;
- store a concise error containing the session ID and reason; and
- emit no Switchyard token annotations.

This is fail-closed for training and fail-open for rollout diagnostics.

## Trace format rationale

Switchyard emits one lossless capture record per model call. Gym trainers consume
Responses API items carrying `TokenIDLogProbMixin`.

The MVP therefore returns one ordinary top-level Gym rollout:

| Source | Gym destination |
|---|---|
| first record's initial prompt | `responses_create_params.input` |
| later user/tool-result suffixes | ordered `response.output` items |
| each captured assistant generation | generated `response.output` item(s) |
| each record's token triple | final generated item for that assistant message |
| OpenHands evaluation | existing `reward` |
| session and record identity | compact response metadata |

Do not add a nested trainer-facing trace list. Do not retokenize messages. Do not
concatenate token arrays across calls. User and tool-result items remain context and
carry no token triple.

Polar-style prefix merging can be added later at the trainer boundary. It is not part
of this MVP.

## Files to change

### Gym

1. `responses_api_agents/swe_agents/app.py`
   - add `switchyard_base_url`;
   - add and generate `switchyard_session_id`;
   - export the two Switchyard environment variables next to the existing
     `NEMO_GYM_MODEL_SERVER_NAME` export in `OpenHandsHarnessProcessor`'s
     agent command (agent container only; the opencode harness and the eval
     container command are untouched);
   - retrieve and apply the reconstructed trace in `run`;
   - add `switchyard_trace_error`.
2. `responses_api_agents/swe_agents/switchyard_trace.py`
   - DTOs, validation, tool mapping, and reconstruction.
3. `responses_api_agents/swe_agents/tests/test_app.py`
   - configuration, UUID creation, environment export, retrieval integration, and
     masking behavior.
4. `responses_api_agents/swe_agents/tests/test_switchyard_trace.py`
   - focused adapter and validation tests.
5. The three OpenHands YAML files listed above
   - update the fork commit after the external change lands.

Do not change `nemo_gym/rollout_collection.py` or generic model-server code.

### nv-OpenHands

1. Update `openhands/agenthub/nemo_gym_client.py`.
2. Add focused client tests for:
   - fallback to Gym `ServerClient`;
   - direct URL construction;
   - route-model rewrite;
   - session header;
   - missing-session failure;
   - exactly one HTTP attempt per call (no transport retries);
   - response parsing, cookies, metrics, and completion logging.
3. Push the fork commit before updating Gym's pin.

### Switchyard

No Gym-specific Switchyard feature is required beyond PR #63.

Two correctness limitations remain in that branch:

- arbitrary session IDs can collide after directory-name sanitization; the Gym MVP
  avoids this by sending UUID hex only;
- missing `request_id` or `model` does not currently make Switchyard's
  `is_valid` false; Gym explicitly validates both fields.

The parser and target schema already agree on nested
`target.token_capture_engine`; that earlier mismatch is resolved.

## Tests

Minimum coverage:

1. No Switchyard setting preserves current OpenHands behavior exactly.
2. A configured run creates a unique UUID and exports it only to the agent
   container.
3. The OpenHands client selects direct HTTP, rewrites the route model, and sends the
   session header.
4. A valid two-call tool trajectory reconstructs into one Gym rollout with two
   exact token triples and unchanged reward.
5. Invalid schema, wrong session, duplicate UUID, empty tokens, non-finite
   log probabilities, mismatched token lengths, and divergent histories mask the
   sample without partial annotations.
6. One built-in OpenHands example rollout runs end to end through Switchyard and
   preserves its patch/evaluation result.

Focused existing regression suites:

~~~bash
uv run pytest -q tests/unit_tests/test_responses_converter.py
uv run pytest -q responses_api_agents/swe_agents/tests/test_app.py
~~~

Switchyard capture regression suite:

~~~bash
uv run pytest -q \
  tests/test_token_capture.py \
  tests/test_token_capture_target.py \
  tests/test_token_capture_translation.py
~~~

## Implementation order

1. Change and test the pinned `nv-OpenHands` client.
2. Push the fork commit and update Gym's three pins.
3. Add Gym config/session plumbing and agent-only environment export.
4. Add the Gym trace DTO, validation, and reconstruction module.
5. Integrate retrieval and fail-closed masking in `SWEBenchWrapper.run`.
6. Run focused unit tests.
7. Run one example end to end with a fresh Switchyard log directory.

## Done when

The MVP is complete when a built-in Gym OpenHands rollout sends every policy call
through Switchyard under one unique UUID, retrieves all valid records after the agent
finishes, reconstructs one normal Gym training rollout with unchanged token IDs and
log probabilities, preserves the existing patch/evaluation/reward result, and masks
the sample rather than returning partial training data whenever capture is invalid.
