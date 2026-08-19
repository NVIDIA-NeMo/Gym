# Worker-owned token capture

Gym can host a gate while the inference framework keeps custody of token IDs,
log probabilities, masks, and routed-expert tensors. The framework worker
stages each delta before acknowledging it. Gym receives only typed commit
coordinates, maintains exact cumulative prefixes, and seals a token-free
`RolloutReceipt`. A framework finalizer must fetch the staged snapshots and
call `verify_and_linearize()` before publishing a training row.

The gate supports multiple Gym model-server workers. Every worker must open
the same atomic `gate.state_store_path` and a process-shared `LineageStore`.
Registration, admission, commit, seal, tombstones, cleanup manifests, and
metrics are stored under a cross-process file lock; no ingress affinity is
required. An in-memory lineage store remains invalid when `num_workers > 1`.

```yaml
token_id_capture:
  enabled: true
  all_agents: true
  rebuild_response: false
  lineage_store: nemo_gym.token_id_capture.lineage:FileLineageStore
  lineage_store_kwargs:
    root: /shared/run/token-lineage
  gate:
    enabled: true
    state_store_path: /shared/run/token-capture-gate.json
    control_auth_token_env: NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN
    registration_ttl_s: 3600
    tombstone_ttl_s: 300
    expiry_sweep_interval_s: 30
```

The control bearer is read from the named environment variable inside each
server process and must not be serialized into run configuration. Registration
returns a distinct per-rollout data capability. Model traffic sends that
capability through `x-nemo-gym-capture-capability` or as its standard Bearer
credential. A rollout ID alone never authorizes admission.

Control routes live under `/training-token-capture/control`. Register and
seal operations require owner and operation IDs so lost-response retries are
idempotent. Seal also requires the terminal logical request ID. The vLLM path
binds its response ID as a logical request ID when the caller does not supply
one explicitly.

The supported serving path is non-streaming Chat Completions. Gym's Responses
and Anthropic Messages APIs are supported when they map through that chat
path. Native Responses, the Completions compatibility backend, and streaming
are rejected while gate capture is enabled. Worker coordinates and all token,
logprob, and routed-expert fields are stripped before the response reaches the
agent.

TTL expiry and explicit failure produce cleanup manifests containing every
known staging key. The framework owns deletion of those staged objects,
including deferred routed-expert fragments. `NG_HTTP_BYTES_DIR` enables one
per-process HTTP byte-counter file so multiworker traffic can be summed without
pretending process-local counters are global.
