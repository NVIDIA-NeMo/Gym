# swe_atlas_qna

Rubric-based LLM-judge resources server implementing the
[SWE-Atlas](https://github.com/scaleapi/SWE-Atlas) **Codebase QnA** verifier.
Each task ships a set of expert-written rubrics; every rubric is graded
independently by an LLM judge, and the reward is the strict "all must-have
rubrics pass" signal from upstream `tests/evaluate_answer.py`.

## What it does

For each rollout, `verify()`:

1. Extracts the candidate answer from the policy/agent response. If the answer
   is wrapped in `<<FINAL_ANSWER>>` tags (the mini-swe-agent QnA harness writes
   its final answer to `/logs/agent/answer.txt` this way), only the wrapped
   content is graded. Single-turn answers are graded as-is.
2. Grades **each rubric** with an independent judge call to
   `/v1/chat/completions` (run concurrently, bounded by `judge_max_concurrency`).
   The judge returns a `YES`/`NO` + `1`/`0` verdict per rubric.
3. Flips the score for `negative`-type rubrics (a matched *undesirable* behavior
   scores 0), mirroring upstream `_apply_negative_flip`.
4. Computes:
   - `reward = 1.0` iff **every scored `must have` rubric passes**, else `0.0`
     (the upstream strict pass).
   - `agg_score` — soft fraction of scored rubrics passed (for analysis).
   - `rubric_scores` — the full per-rubric breakdown (id, title, importance,
     judge verdict + justification).

The grader `system` prompt and `user` template are identical across all
SWE-Atlas QnA tasks, so they live server-side in
[`prompts/`](prompts/) rather than being duplicated into every dataset row.

## Data schema

Each JSONL row carries the question in `responses_create_params.input` and the
grading inputs in `verifier_metadata`:

- `problem_statement` — the codebase question shown to the judge
- `rubrics` — list of `{id, title, annotations: {type, importance}}`
- `instance_id`, `category`, `language`, `repository`, `base_commit`,
  `docker_image`, `sif_basename` — task identity (the image fields are consumed
  by the mini-swe-agent QnA harness, not by the judge)

### Preparing data

The full 124-task benchmark is not committed (it is large and carries benchmark
canary GUIDs). Generate it from the SWE-Atlas source repo with the conversion
script (`scripts/convert_qna_to_gym.py` in that repo):

```bash
python scripts/convert_qna_to_gym.py \
    --qa-dir data/qa \
    --output resources_servers/swe_atlas_qna/data/swe_atlas_qna.jsonl \
    --example-output resources_servers/swe_atlas_qna/data/example.jsonl
```

Only [`data/example.jsonl`](data/example.jsonl) (5 rows) is committed, for smoke
tests.

## Configuring the judge

The judge model is an OpenAI-compatible endpoint resolved from env vars at
config-load time (`${oc.env:...}`):

- `SWE_ATLAS_QNA_JUDGE_BASE_URL` — base URL (must support `/v1/chat/completions`)
- `SWE_ATLAS_QNA_JUDGE_API_KEY` — API key (`MISSING` fallback keeps non-judge
  commands green; live `verify()` fails without a real key)
- `SWE_ATLAS_QNA_JUDGE_MODEL` — model id accepted by the endpoint

SWE-Atlas grades with **Claude Opus 4.5** by default. To run a Gym-managed local
judge instead, swap the `swe_atlas_qna_judge_model` block in the config for a
`local_vllm_model` block (see
`resources_servers/math_with_judge/configs/math_with_local_judge.yaml`).

## Example usage

```bash
# Single-turn smoke test (asks the policy model the question directly, no sandbox)
gym env start \
    --model-type openai_model \
    --resources-server swe_atlas_qna

gym eval run --no-serve \
    --agent swe_atlas_qna_simple_agent \
    --input resources_servers/swe_atlas_qna/data/example.jsonl \
    --output results/swe_atlas_qna_rollouts.jsonl \
    --num-repeats 1
```

The full benchmark runs through the mini-swe-agent QnA harness (added
separately), which explores the repository in an Apptainer sandbox before
answering.

## Metrics

`compute_metrics()` emits pass@k / pass@1[avg-of-k] over `{pass, agg_score}`
per task; `get_key_metrics()` surfaces `mean/reward` and the highest-k pass
rates as the headline numbers.

# Licensing information

- Code: Apache 2.0
- Data: SWE-Atlas (see the [SWE-Atlas repository](https://github.com/scaleapi/SWE-Atlas))

Dependencies
- nemo_gym: Apache 2.0
