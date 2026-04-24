# IOI (International Olympiad in Informatics)

Resource server for IOI-style competitive programming evaluation.

Subclasses the `competitive_coding_challenges` server to reuse its verify
path (sandbox compile/run, per-subtask scoring via `min(tests) * subtask_cap`)
and adds IOI-shape reporting: `ioi_total_score` (0–600 on the IOI'24 scale)
and `per_problem_subtask_scores` (per-problem breakdown with per-subtask
achieved vs cap).

## Quickstart

Run the server + smoke-test rollouts against the committed
`data/example.jsonl` (5 hand-picked (problem, subtask) rows spanning five
IOI'24 problems):

```bash
ng_run "+config_paths=[resources_servers/ioi/configs/ioi.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

ng_collect_rollouts +agent_name=ioi_simple_agent \
    +input_jsonl_fpath=resources_servers/ioi/data/example.jsonl \
    +output_jsonl_fpath=results/ioi_example_rollouts.jsonl \
    +num_repeats=1
```

The server needs a C++ compiler (the sandbox uses `g++` for grader + solution
compilation) and writes precompiled grader bundles + per-test run dirs under
`shared_dir` (configurable, defaults to `/tmp`).

## Data shape

Each row in the input JSONL is one (problem, subtask) combination. The
`example.jsonl` rows have the question baked into
`responses_create_params.input`; the full benchmark JSONL produced by
`benchmarks/ioi/prepare.py` keeps a free-standing `question` field and pairs
it with `benchmarks/prompts/generic_default.yaml` at rollout time:

```json
{
  "competition_id": "ioi24",
  "problem_id": "nile",
  "subtask": "01-equal",
  "subtask_score": 6,
  "name": "Nile",
  "responses_create_params": {"input": [{"role": "user", "content": "<full IOI problem statement>"}]}
}
```

The server's `test_file` config points at a JSONL-wrapped metadata file:

```json
{"competition_id": "ioi24", "metadata": {"nile": {...}, "message": {...}, ...}}
```

where each problem's value is a subtask-keyed dict — one entry per subtask,
each with `grader_files`, `compile`, `run`, `tests`, `subtask_score`. CCC's
`_normalize_problem_metadata` converts this legacy format into CCC's
normalized shape (`subtasks` + `all_tests`) at load time.

## Metrics

`compute_metrics()` emits, on top of CCC's inherited pass@k stats:

- `ioi_total_score` — sum across problems of max per-subtask score pooled
  across rollouts. On the 0–600 IOI'24 scale.
- `per_problem_subtask_scores` — per-problem breakdown, each with
  `total.{score,max_score}` plus per-subtask `{score, max_score}`.
