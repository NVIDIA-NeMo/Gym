# Indic Arena judge

Resource server for [Indic Arena Hard v2](../../benchmarks/indic/indic-arena-hard-v2/README.md).
It subclasses Gym's unchanged `arena_judge` and keeps Indic-specific behavior in
this component. It uses the standard `simple_agent`, model servers, asynchronous
judge transport, and `legacy_agent` environment server.

## Modes

- `pointwise` (default) evaluates one answer without a baseline using the [custom rubric](prompts/pointwise.yaml).
  Correctness, relevance, completeness, and language quality each receive an
  integer from 0 to 4. Their mean is rescaled to 0–100; reward is that score divided
  by 100. These custom scores are not official Arena Hard win rates.
- `pairwise` (opt-in) validates a nonempty matching-language `baseline_answer`,
  then delegates both answer orderings, category prompts, verdict parsing, and
  Arena-Elo aggregation to the English judge implementation.

Pointwise scoring makes one judge request per nonempty answer. Empty answers get
zero without a judge call. Invalid rubric output is counted and excluded from
rubric means; judge transport failures use Gym's failure routing. Metrics average
repeats within each task, then tasks, and report per-language means. Raw judgments
and parsed scores are retained in the rollout.

## Configuration and isolation

`configs/indic_arena_judge.yaml` defines separate Indic resource, judge-model,
agent, and environment instances. Configure the judge with
`INDIC_ARENA_JUDGE_BASE_URL`, `INDIC_ARENA_JUDGE_API_KEY`, and
`INDIC_ARENA_JUDGE_MODEL`, or override those fields through Gym YAML. The benchmark
selects the mode through `indic_arena_hard_v2_judge_mode`, defaulting to `pointwise`.
Pass `++indic_arena_hard_v2_judge_mode=pairwise` when starting the benchmark
to use baseline comparisons. Standalone resource-server users can set
`evaluation_mode: pairwise` in their Indic resource configuration.

The English `arena_hard_v2` benchmark and the shared `arena_judge` have no changes
in this integration. The optional Indic smoke preset also uses a separate
candidate model and leaves English model settings intact.

Task fields follow the existing Arena contract: `question`, `baseline_answer`,
`category`, and `uid`; `language` is used for pointwise metrics. The dependency-light
[`task_data.py`](task_data.py) extends the existing task schema. Benchmark prompts
are downloaded at runtime; the committed examples are synthetic.

## Validation

```bash
pytest tests/unit_tests/test_indic_arena_hard_v2.py
gym env test +entrypoint=resources_servers/indic_arena_judge +should_validate_data=true
```

The resource remains `verified: false` pending a reviewed benchmark baseline.
