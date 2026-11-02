# decision_index resources server

Verifier for the [Decision Index benchmark](../../benchmarks/decision_index/README.md). Stateless (no sessions,
`REVERIFY_MODE = STATELESS`).

- `verify()` (in `app.py`, logic in `verifier.py`) reads `verifier_metadata.di_row` (the kit row without `state`)
  and the policy's reply:
  - `answer_mode: decision`: the reply is the `decision_engine_model` envelope `{"di_engine": {status, response,
    error, wall_ms}}`; answers are checked with the kit's `validate` (every question answered, chosen option in
    `criteria`, a full probability distribution).
  - `answer_mode: llm`: reasoning blocks are stripped, the last JSON object (a fenced ```json block wins) must map
    every question key to an option key or true/false. Anything else is an `error` (wrong), never a crash.
  - `answer_mode: auto` (fixture default): decision when the reply is an envelope.
  - The response carries `di_status`, `di_result` (the row as the kit's `results.jsonl` would hold it) and
    `di_score_row` (what the kit's scorers read, without question instructions).
  - A failed engine call (the envelope's `status: error`) returns `mask_sample: true`,
    `failure_kind: decision_index:engine_error`. The kit still counts it as unanswered.
- `aggregate_metrics()` adds the kit's public 0.3 index over every rollout (`decision_index`, areas, per-benchmark
  skill). It overrides `aggregate_metrics` instead of `compute_metrics` because that hook sees only unmasked rows.

The kit is installed from `requirements.txt` (`apolinario/decision-index@9eb2dbe2`, MIT). Its import name is also
`decision_index`, so `verifier.load_kit()` binds it from the installed distribution.

## Tests

```bash
gym env test --resources-server decision_index
```

`tests/verifier_cases.jsonl` is the verifier fixture (both reward endpoints, malformed requests, a masked engine
failure); `tests/test_verifier.py` covers parsing, rewards, masking and kit scoring on the five example rows.
