# Description

Checklist judge on **Nemotron Decide**, a decision model that returns a probability for every option of every
question in one forward pass (no generated text). Use it as a drop-in replacement for LLM-judge rewards.

Each checklist item is a yes/no question about the rollout. All items go to Nemotron Decide in one request and the
yes-probabilities become the reward:

| `reward_mode` | reward |
|---|---|
| `probability` (default) | weighted mean of p(yes): dense and calibrated |
| `pass` | weighted fraction of items with p(yes) >= `pass_threshold`: binary-style, like an LLM judge verdict |

- `required: true` items gate the reward: if one is below `pass_threshold` the reward is `required_fail_reward`.
- `weight: 0` items are read and logged only.
- A row may bring its own checklist in `decide_checklist` (`allow_row_checklist`).
- Every verify response carries `decide_checklist_results` (per item: question, weight, p_yes, passed),
  `decide_scores` and `decide_latency_ms`. `compute_metrics` reports mean p(yes) and pass rate per item.
- Endpoint failures (transport, auth, 429/5xx after retries) set `mask_sample: true` and `decide_error`, so they never
  become model-quality zeroes.

The state shown to the model is JSON: `task` (the conversation sent to the policy), `reference` (the row fields named
by `reference_fields`, e.g. `expected_answer`), optional `notes`, and `response` (assistant messages plus, unless
`include_tool_calls: false`, tool calls and outputs). Reasoning items are never included.

## Endpoint

`decide` configures the endpoint (`decide_client.py`, a small async client other components can reuse):

| field | default | |
|---|---|---|
| `base_url` | `http://localhost:8770` (`DECIDE_BASE_URL`) | the model served locally, or any compatible endpoint |
| `api` | `systemone` | `systemone` (`/v1/systemone`) or `decisions` (`/v1/decisions`, OpenAI Decisions shape; same read) |
| `model` | `nemotron-decide` | |
| `api_key_env` | `DECIDE_API_KEY` | name of the env var with the bearer token (unset: no auth header); never put the key in config |
| `timeout_s`, `max_retries`, `max_concurrency` | 30, 3, 32 | |

Serve Nemotron Decide locally (or point `DECIDE_BASE_URL` at any server with the same API). Use `api: decisions`
with a server that exposes `/v1/decisions`.

## Configs

- `configs/decide_judge.yaml`: generic checklist (follows task, correct, complete) with Simple Agent on
  `data/example.jsonl` (includes one row with its own checklist).
- `configs/equivalence_decide_judge.yaml`: the `equivalence_llm_judge` datasets graded by Nemotron Decide instead of
  the LLM judge (`reward_mode: pass` gives the same binary reward; set `DECIDE_REWARD_MODE=probability` for a dense one).

```bash
export DECIDE_BASE_URL=http://localhost:8770   # your Nemotron Decide server
export DECIDE_API_KEY=...   # only if the server needs a bearer token
gym env start --config resources_servers/decide_judge/configs/equivalence_decide_judge.yaml --model-type openai_model
gym eval run --no-serve --agent equivalence_decide_judge_simple_agent \
  --input resources_servers/equivalence_llm_judge/data/example.jsonl --output results/eq_decide.jsonl
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0 (`data/example.jsonl` written for this server)

Dependencies
- nemo_gym: Apache 2.0
