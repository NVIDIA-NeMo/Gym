# facts_grounding_v2

FACTS Grounding v2 (FACTS Benchmark Suite, Google DeepMind / Google Research / Kaggle; arXiv:2512.10791, section 6):
long-form answers that must stay grounded in a supplied document, judged for eligibility and sentence-level
grounding by Gemini 2.5 Flash and GPT-5 with the v2 prompts.

## Protocol reproduced here

| Component | Official source | This server |
|---|---|---|
| Prompts | Kaggle `deepmind/FACTS-grounding-examples` v17, 856 public rows | `benchmarks/facts_grounding_v2/prepare.py` (pinned SHA-256) |
| Model prompt | starter: `full_prompt` as the single user message | baked into each row (`prompt_config: null`) |
| Judges | `google/gemini-2.5-flash`, then `openai/gpt-5-2025-08-07` | `judge_model_servers` in that order (any OpenAI-compatible endpoints) |
| Eligibility | judge writes its own baseline answer, rates the response vs baseline with the v2 no-context prompt; first judge without `Major Issue(s)` wins; unparseable = eligible | `prompts/eligibility_*.txt` (byte-identical), `extract_instruction_rating`, `is_eligible` |
| Grounding | v2 `UFG_REV21` sentence-level prompt per judge; grounded iff every parsed sentence is supported/no_rad/unknown | `prompts/grounding_*.txt` (byte-identical), `parse_grounding_verdict` (starter port) |
| Score | grounding score = mean of judge verdicts; 0 when ineligible | `reward` in {0, 0.5, 1}; `unadjusted_score` kept (both judges run on ineligible answers too) |

Aggregate metrics: `factuality_score` (primary, adjusted), `unadjusted_factuality_score`, `eligibility_rate`,
per-judge `grounded_rate_eligible/<judge>`, `judge_disagreement_rate_eligible`, eligibility-rating and
parse-validity diagnostics, truncation/empty rates, context-token statistics, and domain/type slices. See
[`benchmarks/facts_grounding_v2/METRICS.md`](../../benchmarks/facts_grounding_v2/METRICS.md).

## Example fixtures

The five committed example rollouts are the matching public prompt rows from the Kimi K3 anchor run. They retain
the actual generated answers and verifier metrics while omitting raw provider-call receipts and performance/token
telemetry. They are smoke fixtures, not a substitute for a full benchmark evaluation.

## Running

```bash
export FACTS_GROUNDING_JUDGE_GEMINI_BASE_URL=https://.../v1 FACTS_GROUNDING_JUDGE_GEMINI_API_KEY=... FACTS_GROUNDING_JUDGE_GEMINI_MODEL=gemini-2.5-flash
export FACTS_GROUNDING_JUDGE_GPT5_BASE_URL=https://.../v1 FACTS_GROUNDING_JUDGE_GPT5_API_KEY=... FACTS_GROUNDING_JUDGE_GPT5_MODEL=gpt-5-2025-08-07

gym eval run --benchmark facts_grounding_v2 --config <policy model config> --split benchmark \
  --temperature 0.0 --top-p 1.0 --max-output-tokens 16384 --output results/facts_grounding_v2.jsonl \
  +route_failures_to_sidecar=true +observability_enabled=true
```

Prompts reach 153k characters; the policy endpoint must accept them without clipping (the verify response records
`policy_input_tokens` so context utilisation can be audited).

## Tests

```bash
gym env test --resources-server facts_grounding_v2
```
