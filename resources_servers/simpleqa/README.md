# SimpleQA Resources Server

Evaluates short-form factual QA using an LLM judge that grades on a 3-tier
scale (CORRECT / INCORRECT / NOT_ATTEMPTED). Mirrors the
[SimpleQA-Verified](https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code)
judge protocol.

## Verification

The server forwards the model's response and the gold target to an LLM judge,
which returns a single letter:

| Grade | Meaning | Reward |
|-------|---------|--------|
| A: CORRECT | Answer fully contains the gold target with no contradictions | 1.0 |
| B: INCORRECT | Answer contradicts the gold target | 0.0 |
| C: NOT_ATTEMPTED | Model omits / hedges / refuses (no contradiction either) | 0.0 |

When the judge output is unparseable, the verdict defaults to `NOT_ATTEMPTED`,
matching Skills' `DEFAULT_GRADE_IF_UNPARSEABLE = "C"`.

## Metrics

- **`pass@k/correct`**, **`pass@k/incorrect`**, **`pass@k/not_attempted`** —
  per-tier pass-at-k.
- **`pass@1[avg-of-k]/correct`** — primary parity signal vs Skills'
  `pass@1[avg-of-N] correct`.
- **`f1`** = 2·P·R / (P + R) where
  P = correct/(correct+incorrect),
  R = correct/total — the SimpleQA-Verified headline metric.
- **`accuracy_given_attempted`** = correct/(correct+incorrect).

A short reasoning-parser note: the server passes message content to the
judge verbatim and does not inspect `<think>...</think>` blocks. When
running with a vLLM-served reasoning model, start the policy server with
`--reasoning-parser <name>` (e.g. `deepseek_r1` for Nemotron-3 Nano) so
the reasoning trace is split off before the response reaches this server.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simpleqa/configs/simpleqa.yaml"
ng_run "+config_paths=[$config_paths]" \
    +simpleqa.resources_servers.simpleqa.judge_model_server.name=policy_model
```

## Collecting rollouts (5-example smoke test)

```bash
ng_collect_rollouts \
    +agent_name=simpleqa_simple_agent \
    +input_jsonl_fpath=resources_servers/simpleqa/data/example.jsonl \
    +output_jsonl_fpath=results/simpleqa_rollouts.jsonl \
    +num_repeats=1
```

## Data Format

Each JSONL row must have:

```json
{"question": "...", "expected_answer": "..."}
```

`id` is optional but useful for joining against rollouts post-hoc.
