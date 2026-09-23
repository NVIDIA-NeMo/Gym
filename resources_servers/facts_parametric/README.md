# facts_parametric

FACTS Parametric (FACTS Benchmark Suite, Google DeepMind / Google Research / Kaggle; arXiv:2512.10791, section 4):
closed-book factoid questions whose answers are supported by Wikipedia, graded three times by Gemini 2.5 Pro with the
official grader prompt.

## Protocol reproduced here

| Component | Official source | This server |
|---|---|---|
| Questions | Kaggle `kaggle/facts-parametric-public-examples` v2, 1,052 public rows | `benchmarks/facts_parametric/prepare.py` (pinned SHA-256) |
| Model prompt | starter `QUERY_TEMPLATE = "{question}"` | `benchmarks/prompts/generic/default.yaml` (identical template) |
| Grader | Gemini 2.5 Pro, three samples per answer | `judge_model_server` (any OpenAI-compatible endpoint), `judge_samples: 3`, `seed` 0/1/2 |
| Grader prompt | starter `GRADER_TEMPLATE` | `prompts/grader_template.txt` (byte-identical, SHA-256 pinned in `app.py`) |
| Label parse | starter `extract_classification` | `starter_classification` (canonical); closing `Output:` line parse recorded alongside |
| Score | paper: mean of the three grades; starter: all three CORRECT | `reward` = CORRECT fraction; `all_correct` component |

Aggregate metrics (`compute_metrics`) pool the grades: `accuracy` (primary), `hedging_rate`, `mistake_rate`,
`unknown_rate`, `attempted_accuracy` (= accuracy / (1 - hedging)), `f1`, `strict_all_correct_rate`, grader validity
(`judge_valid_rate`, `judge_parse_agreement_rate`), generation truncation/empty rates, a bootstrap 95% CI on accuracy,
and per-topic slices. See [`benchmarks/facts_parametric/METRICS.md`](../../benchmarks/facts_parametric/METRICS.md).

## Running

```bash
# grader endpoint (any OpenAI-compatible Chat Completions endpoint serving Gemini 2.5 Pro)
export FACTS_PARAMETRIC_JUDGE_BASE_URL=https://.../v1
export FACTS_PARAMETRIC_JUDGE_API_KEY=...
export FACTS_PARAMETRIC_JUDGE_MODEL=gemini-2.5-pro   # provider-specific id if needed

gym eval run --benchmark facts_parametric --config <policy model config> --split benchmark \
  --temperature 0.0 --top-p 1.0 --max-output-tokens 16384 --output results/facts_parametric.jsonl \
  +route_failures_to_sidecar=true +observability_enabled=true
```

Grader transport failures are judge failures (rows go to the failures sidecar and are retried on `--resume`); an
empty grader reply is labelled UNKNOWN as the starter does and counted in `judge_empty_grade_rate`.

## Tests

```bash
gym env validate facts_parametric
gym env test facts_parametric
gym env publish facts_parametric
```

Covers both label parsers, prompt construction against the pinned template, verify paths (full/mixed/zero reward,
empty and truncated generations, empty grader replies, transport failures), aggregate metrics, the verifier-fixture
contract (`tests/verifier_cases.jsonl`), the prepare script, and the reporting toolchain. The workload manifest declares
the public 1,052-row benchmark composition and is checked against the resolved Gym configuration.
