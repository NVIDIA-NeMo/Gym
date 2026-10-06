# KIDBench resources server

Scores a child-facing response on the KIDBench rubric — five 1–5 dimensions grounded in
developmental psychology, plus `cultural_alignment` under a country context.

Full benchmark documentation, metric definitions, and protocol deviations live in
[`benchmarks/kidbench/`](../../benchmarks/kidbench/README.md).

## Before first use

```bash
python -m benchmarks.kidbench.prepare
```

The server reads its rubrics and category/country rules from the pinned upstream checkout
that this command creates. It does not fetch them itself: a server booting inside a large
worker pool would otherwise race on a clone, and the rubric text is part of the
measurement, so it should be pinned deliberately rather than acquired at startup.

## Agents

| Agent | Track | Dataset |
|---|---|---|
| `kidbench_simple_agent` | Single turn, `max_steps: 1` | `data/single_turn.jsonl` (5,000 rows) |
| `kidbench_multi_turn_agent` | Five-turn child-actor conversations | `data/multi_turn.jsonl` (200 rows) |

`max_steps: 1` on the single-turn agent is load-bearing: the rubric judges a *first reply*
to a child, so letting the model take another step would score something the child never
saw.

## What `verify()` returns

`reward` is the total quality score — the unweighted mean of the five core dimensions —
rescaled from 1–5 onto [0, 1], because Gym aggregates rewards on that scale. The raw value
is kept in `total_quality_score`, which is the number comparable to published results.

Also on every row: the individual dimensions, the judge's `failure_modes` and `strengths`
tags, its `improvement` revision note, and `judge_parse_failed`. Multi-turn rows add
`turn_scores`, `per_turn`, `degradation_slope`, and `peak_quality_drop`.

A judge call that fails at the transport layer raises `JudgeError` and the row is routed to
the failures sidecar, excluded from aggregates. A judge reply that *arrives* but will not
parse is different — that is a scoring outcome, scored 0.0 and counted in
`judge_parse_failed`.

## Judge wiring

`judge_model_server` should point at the paper's judge. Swapping it is supported but
changes what the scores mean, so `judge_name` is recorded on every verified row.

```yaml
judge_model_server:
  type: responses_api_models
  name: kidbench_judge_model
judge_responses_create_params:
  temperature: 0.0
  top_p: 1.0
  max_output_tokens: 4096
```

## Tests

```bash
gym env test --resources-server kidbench
```

`tests/conftest.py` fetches the pinned upstream checkout before collection, so the prompt
assembly tests compare against the real rubric files rather than fixtures that could drift
from them.

## Use restrictions

The dataset contains safety-sensitive child-facing prompts — self-harm, sexual boundaries,
bullying, online privacy, family conflict. It is for safety evaluation and model
improvement only, never for direct exposure to children, and the child-safety context must
not be stripped when redistributing. See the benchmark README for the full notice.
