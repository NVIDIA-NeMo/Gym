# Worker-owned token capture

Gym maintains a per-rollout capture ledger while the inference framework keeps
custody of token IDs, log probabilities, masks, and routed-expert tensors. The
framework worker stages each delta before acknowledging it. Gym receives only
typed commit coordinates, records a token-free ledger row with an explicit
`parent_call_id` lineage link, and serves the rollout's manifest through the
capture control routes. A framework finalizer assembles a token-free
`RolloutReceipt` from that manifest, fetches the staged snapshots, and calls
`verify_and_linearize()` before publishing a training row.

The ledger supports multiple Gym model-server workers. Every worker must share
one `LineageStore`; `FileLineageStore` provides cross-process read-after-write
consistency under a file lock, so no ingress affinity is required. An
in-memory lineage store remains invalid when `num_workers > 1`.

```yaml
token_id_capture:
  enabled: true
  all_agents: true
  rebuild_response: false
  lineage_store: nemo_gym.token_id_capture.lineage:FileLineageStore
  lineage_store_kwargs:
    root: /shared/run/token-lineage
```

## Terminal selection precedence

Receipt assembly must name the `terminal_model_call_id` whose ancestry becomes
the training row. Precedence:

1. **Declared.** A harness that reports the response id it actually kept (the
   `x-nemo-gym-logical-request-id` binding, else the response id the vLLM path
   stamps) is authoritative. A declared id that matches no committed row masks
   the rollout — it never falls back to the heuristic, because a wrong or
   stale declaration is evidence the ledger and the harness disagree.
2. **Heuristic.** When no terminal is declared,
   `staging.select_terminal_call()` infers one from the manifest's parent
   links: the earliest-admitted root (rows carry `admitted_at`; an extended
   root beats an abandoned sibling), then a walk that eliminates abandoned
   retries (a childless child loses to an extended sibling). The selection is
   token-free and fail-closed — `verify_and_linearize()` still verifies every
   digest on the chosen chain.
3. **Mask.** Any ambiguous shape (retry of the final call, divergent extended
   branches, orphaned or cyclic rows) selects nothing and the receipt carries
   the selection reason as its failure reason.

`RolloutReceipt.terminal_selection` records which path chose the terminal
(`declared` or `heuristic`) so consumers can meter heuristic reliance.

The supported serving path is non-streaming Chat Completions. Gym's Responses
and Anthropic Messages APIs are supported when they map through that chat
path. Worker coordinates and all token, logprob, and routed-expert fields are
stripped before the response reaches the agent.

Explicit failure produces poison rows in the manifest; the framework owns
deletion of staged objects named by the rows it consumes. `NG_HTTP_BYTES_DIR`
enables one per-process HTTP byte-counter file so multiworker traffic can be
summed without pretending process-local counters are global.
