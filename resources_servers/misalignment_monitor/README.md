# Description

A misalignment monitor for training and evaluation rollouts, modeled on OpenAI's layered monitoring stack
([How we detect frontier model misalignment](https://openai.com/index/how-we-detect-frontier-model-misalignment/)).
It reviews agents for sandbox escapes, unauthorized actions, deception, evaluation gaming, sabotage, secret
exposure and monitor evasion, and writes alerts. It never changes a reward, and nothing waits for it: a monitor
that lowered rewards, or delayed the rollouts it was reviewing, would train the policy to hide misbehavior.

| Layer | What it does |
|---|---|
| 0, activation screen (`activation_source`) | Reads the policy's own activations inside vLLM (`engine`) or a frozen copy of the model (`reader`), with a Jacobian lens over the J-space ([A global workspace in language models](https://www.anthropic.com/research/global-workspace)) or a linear probe. Tuned for recall. |
| 1, reviewer | Reviews the context window the agent acted on: task, reasoning, tool calls and results. |
| 2, investigators | A security investigator (does someone need to intervene?) and an alignment investigator (is this worth studying?) each review every layer-1 flag. They are agents: they can read every context window of the rollout, search them and the rollout record, list recent alerts to find duplicates, and run Python in a sandbox over the evidence. |
| 3, confirmation | Checks each security P0 against the rollout's record and earlier alerts before it pages anyone, and reviews alignment alerts for recurring patterns. |

A random `audit_rate` (5%) of what layer 0 or layer 1 clears goes to the next layer anyway, to measure what they
miss. Severities are P0 (a person must act now), P1 (serious, no ongoing harm), P2 (worth a closer look) and none.
Only the security path can raise a P0.

# Setup

Rollouts reach the monitor through the `misalignment_monitored` environment server. Use it in place of the agent's
`legacy_agent` environment server (`environment_servers/misalignment_monitored`), add this server's config, and
point `judge_model_server` at a self-hosted judge model server. This covers `gym eval run` and RL training.
With a `vllm_model` judge served by vLLM's Rust frontend, set `reasoning_field: reasoning`: that frontend rejects the
two reasoning keys the investigators' multi-turn calls carry by default.

`review_schedule` picks when layer 1 runs:

- `final` reviews each rollout once, when it finishes.
- `per_turn` reviews at the end of each agent turn, when the agent's context is summarized or truncated (the window
  it is about to lose), when layer 0 flags a model call, and at least every `max_minutes_between_reviews` (30)
  during a long turn. When the judge falls behind, a rollout's queued review is dropped for its next one, whose
  window contains it, unless the context was rewritten in between. Otherwise agents whose every call ends a turn
  (text-action harnesses such as Terminus) would outrun any judge.

`per_turn` and `activation_source: engine` read each model call from Gym's model-call capture, so they need
`observability_enabled: true` and an absolute `model_call_capture_dir` in the global config, readable by this server.
With capture on, `final` also reviews the last context window the agent saw, tool results included, whatever the
harness records in its result. Only calls from `policy_model_servers` count as the agent acting. The monitor joins
calls to rollouts by rollout id, so ids must be unique within a capture directory, as Gym's capture already
requires: give reruns of the same input explicit `_ng_rollout_id`s or a fresh `model_call_capture_dir`.

# Alerts and pausing

Every non-`none` verdict is appended to `alerts_jsonl_fpath` with its severity, categories, which investigator found
it, a summary, the window and every layer's report, and printed to the server log. Pattern reviews, resolutions and
failed reviews go to the same file. A count line is printed every 100 reviews.

- P0 alerts are POSTed as JSON to `alert_webhook_url` when it is set.
- `GET /status` lists open P0 alerts. When one has been open for `pause_after_unresolved_p0_minutes`, it asks the
  environment server to hold new rollouts, which pauses the whole run without singling out any rollout.
- `POST /resolve` with `{"alert_id": "...", "note": "..."}` closes a P0 after a person has looked at it.
- When `gym eval run` finishes, the environment server asks the monitor (`POST /drain`) to finish its reviews,
  for up to `drain_timeout_s` (600), before metrics are computed and the servers stop.

# Layer 0

## `engine`: the policy's own activations

`vllm_activation_connector.py` is a vLLM plugin built on vLLM's hidden-state extraction (vLLM 0.29). It reads the
residual stream at one layer for every request while vLLM serves it, scores it on the GPU when the request
finishes, and writes a small JSON result whose path comes back in the response. Gym's `vllm_model` server passes
that path on, the capture records it, and the monitor reads it. Generated tokens and the end of the prompt are
read (`max_positions`, 4096).

Start vLLM with the Gym repository root on `PYTHONPATH`:

```bash
vllm serve Qwen/Qwen3-4B --no-enable-chunked-prefill \
  --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1,
    "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [19]}}}' \
  --kv-transfer-config '{"kv_connector": "MisalignmentActivationConnector", "kv_role": "kv_producer",
    "kv_connector_module_path": "resources_servers.misalignment_monitor.vllm_activation_connector",
    "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/misalignment_l0", "mode": "jspace",
      "jlens_path": "lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
      "concepts": ["fake", "fictional", "manipulation", "secretly", "deliberately", "fraud", "cheat"],
      "top_k": 10, "num_writer_threads": 2}}'
```

- The layer counts like transformers' `output_hidden_states`, so a lens fitted at residual block `l` reads layer
  `l + 1`. Middle layers read best.
- `mode: probe` takes `probe_path` (a directory with `probe.safetensors` holding `weight`, `bias`, `mu`, `sd` and
  `config.json` holding `layer` and `temperature`, the format of the
  [Hiddenguard](https://huggingface.co/vineethkalluru/nemotron-lightning-guard-probe) probes) and `probe_threshold`,
  and scores the last prompt token.
- The lens reads through the model's final RMSNorm and unembedding, loaded from the checkpoint (`unembed_tensor`,
  `final_norm_tensor` name them when they are not `lm_head.weight` and `model.norm.weight`). That copy, 0.5 to 2 GB,
  comes on top of vLLM's own memory, so lower `gpu_memory_utilization` a little.
- The extraction uses vLLM's speculative-decoding slot, so it cannot run with MTP or EAGLE drafts, and it needs
  chunked prefill off. Results go to `shared_storage_path`, which the monitor must be able to read: `/dev/shm` on one
  node, a shared filesystem across nodes.
- With `local_vllm_model`, put these under `vllm_serve_kwargs`. In NeMo RL, under `policy.generation.vllm_kwargs`.
- Agents that call the model through the Anthropic Messages API get no engine reading: that conversion drops it.

Pre-fitted lenses for several open models (Qwen3, Qwen3.5, Llama 3.1, Gemma, OLMo, gpt-oss) are on the Hub:

```bash
hf download neuronpedia/jacobian-lens qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt --local-dir lenses
```

## `reader`: a frozen copy

Loads `reader_model` with transformers in this server (`device_map="auto"`, so it needs GPU memory here), and reads
the last `reader_max_tokens` tokens of each reviewed window at `jspace_layer` (`reader_mode: jspace`, with
`jlens_path`, `jspace_concepts` and `jspace_top_k`) or with a probe (`reader_mode: probe`). It reads the rendered
text, not the policy's tokens, and during RL its weights stay at the starting checkpoint. Use it when the policy is
not served by vLLM. `reader_layout` names the modules of architectures jlens does not detect, for example
`{path: backbone, layers: layers, norm: norm_f, embed: embeddings}` for Nemotron-H.

Calibrate layer 0 before relying on it: check its flag rate on clean rollouts, and use the audited share to measure
what it misses. A J-space concept must be a single token of the model's vocabulary.

# Replaying recorded rollouts

`/verify` runs the same review on a recorded rollout and returns it as `monitor_report`, scoring a flagged rollout
1.0. It writes no alerts, so replaying an incident pages no one. With rows labeled `expected_severity`, the
aggregate metrics report recall and false-positive rate at each threshold (`monitor/recall@P1`, ...):

```bash
gym eval reverify \
    --resources-server misalignment_monitor \
    --model-type vllm_model \
    --inputs results/rollouts_materialized_inputs.jsonl \
    --rollouts results/rollouts.jsonl \
    --output results/monitor_replay.jsonl \
    +misalignment_monitor.resources_servers.misalignment_monitor.judge_model_server.name=policy_model
```

# Limits

- Layer 3 reads the evidence Gym has: the rollout's own record and earlier alerts. Gym sandboxes expose no network
  logs or system telemetry.
- Pausing holds new rollouts. It does not stop rollouts in flight or roll a run back, and paging beyond the webhook
  is up to the receiver.
- Rollout state is in memory: a monitor restart forgets open P0 alerts and rollouts in progress.

# Licensing information

Code: Apache 2.0

Dependencies:
- nemo_gym: Apache 2.0
- jlens: Apache 2.0
- vLLM (for the plugin), transformers, accelerate, safetensors: Apache 2.0
