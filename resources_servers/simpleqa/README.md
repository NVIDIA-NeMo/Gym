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
gym env start \
    --model-type vllm_model \
    --resources-server simpleqa \
    +simpleqa.resources_servers.simpleqa.judge_model_server.name=policy_model
```

## Collecting rollouts (5-example smoke test)

```bash
gym eval run --no-serve \
    --agent simpleqa_simple_agent \
    --input resources_servers/simpleqa/data/example.jsonl \
    --output results/simpleqa_rollouts.jsonl \
    --num-repeats 1
```

## Sandboxing

SimpleQA is a pure QA task with a judge grader: the model calls no tools, and the judge call executes nothing. Both
may therefore opt out of the attempt's sandbox group, which saves sandbox capacity on large runs.

| Config | Pairing | Agent loop | Judge call |
|--------|---------|------------|------------|
| `simpleqa.yaml` | `simpleqa_environment_server` | Agent Server process | Resources Server |
| `simpleqa_sandboxed.yaml` | `simpleqa_sandboxed_environment_server` | The attempt's sandbox group (`sandboxed: true`) | Resources Server |

`simpleqa_environment_server` sets `sandbox_opt_out: {task: true, grader: true}` and needs no sandbox provider.
`simpleqa_sandboxed_environment_server` starts one sandbox per attempt from its `sandbox_config` and runs Simple
Agent's loop there; its grader still opts out. Its `model_only` network rule lets the sandbox reach only the policy
model, so it needs a provider that applies that rule, in a provider block named `sandbox`, and a model server address
the sandbox can reach. Add `simpleqa_sandboxed.yaml` and the provider config to the config paths, then run the
sandboxed agent:

```bash
gym eval run --no-serve \
    --agent simpleqa_sandboxed_simple_agent \
    --input resources_servers/simpleqa/data/example.jsonl \
    --output results/simpleqa_sandboxed_rollouts.jsonl \
    --num-repeats 1
```

## Data Format

Each JSONL row must have:

```json
{"question": "...", "expected_answer": "..."}
```

`id` is optional but useful for joining against rollouts post-hoc.
