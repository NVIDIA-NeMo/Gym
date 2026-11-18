# Agentic Turn Simulation Judge

The policy plays the judge of [`agentic_turn_simulation`](../agentic_turn_simulation) (ATS). Each row is the exact
prompt the ATS judge reads for one recorded agent turn: the system message is the ATS judge system prompt (harness
frame, the four rubrics, the agent's tool definitions, the JSON output schema) and the user message is the rendered
conversation with the turn under evaluation in `<step_under_evaluation>`. The policy writes the judge's JSON verdict.
The reward is the agreement of the verdict's correctness ladder with a reference label built from reference-judge
verdicts on the same prompt (hand labels, where they exist, are reported separately).

Verification is deterministic: no LLM is called at verify time, so the env is cheap to run next to ATS.

## Row format

Task fields sit at the row top level (`task_data.py` declares them; there is no `verifier_metadata`).

```json
{
  "responses_create_params": {"input": [{"role": "system", "content": "<ATS judge system prompt>"},
                                        {"role": "user", "content": "<conversation>...<step_under_evaluation>...</step_under_evaluation>"}]},
  "task": "verdict",
  "row_id": "...",
  "group_id": "...",
  "gold_ladder": "yes",
  "gold_p_yes": 1.0,
  "gold_n": 4,
  "gold_fields": {"addresses_concern": "yes", "correct": "yes", "blocked_by": "none",
                  "request_questionable": "ok", "discard": "no", "discard_reason": "none"},
  "label": {"policy": "...", "tier": "A", "judge": "...", "rubric_tag": "...", "rubric_sha256": "...", "set": "...", "tag": "..."},
  "meta": {"source": "...", "harness": "...", "family": "...", "domain": "...", "split": "...", "prompt_tokens_est": 31000,
           "hand_ladder": "yes", "expect": null, "stratum": null}
}
```

| Field | Meaning |
|---|---|
| `responses_create_params.input` | Exactly `[system, user]`, text content. No `tools` and no `response_format`: the system prompt already says "Return JSON matching this schema". |
| `task` | `verdict` (the only kind; `pairwise` is reserved). |
| `row_id`, `group_id` | Row id; transcript/session key. Rows of one group stay in one split. |
| `gold_ladder` | `yes` / `no`: the reward label (required). |
| `gold_p_yes` | Share of usable reference passes with ladder `yes` (soft reward). |
| `gold_n` | Number of reference passes behind the label. |
| `gold_fields` | Reference verdict fields (majority); used for field matches and the optional field bonus. |
| `label` | Provenance: label policy, tier, judge model, rubric tag and sha256, source set, run tag. |
| `meta` | Provenance used by the metrics: `source`, `harness`, `family`, `split`, `prompt_tokens_est`, and `hand_ladder` (a human ladder label, eval rows only). |

## Reward

`/verify` reads only the assistant `message` items of the response (reasoning items are ignored) and strips any
`<think>...</think>` left in the text, then:

1. **Parse** (`parse_mode`). `strict` (default): after stripping whitespace and one optional ```` ```json ```` fence the
   text must be exactly one JSON object. `lenient`: the ATS `parse_verdict` without its salvage of cut-off output
   (first object anywhere in the text that holds `addresses_concern`, also inside a one-key envelope). A cut-off verdict
   is never read.
2. **Validate** (`validate_schema`, default on): every key the ATS evaluation schema requires is present, enum fields
   take allowed values, string fields are strings. Extra keys are allowed. `max_analysis_words` (default off) fails an
   `analysis` longer than that many words.
3. **Ladder**: `pred_ladder = "no"` if `addresses_concern == "no"`, else `correct`.
4. **Score** (`reward_mode`). `hard` (default): `1` if `pred_ladder == gold_ladder`, else `0`. `soft`: `gold_p_yes` for a
   `yes`, `1 - gold_p_yes` for a `no` (the expected agreement with one reference pass; rows without `gold_p_yes` score
   hard). Optional `field_bonus {weight w, fields [...]}`: `reward = ladder_score * ((1 - w) + w * share of the listed
   fields equal to gold_fields)`, gated by the ladder so matching fields never pay for a wrong ladder (rows without
   gold for the listed fields keep the ladder score).

Unreadable output scores 0 and is **never masked**. `parse_error` names why: `empty`, `no_json`, `multiple_objects`,
`bad_json`, `missing_keys`, `bad_enum`, `too_long` (and `internal` for a verifier exception, which is logged; it also
scores 0 so a bug can never crash the server).

The verify response carries `pred_ladder`, `gold_ladder`, `ladder_match`, `parse_ok`, `parse_error`, the confusion
one-hots `tp`/`fp`/`fn`/`tn` with POSITIVE = ladder `no` (the judge catching a failed turn; all 0 when unparsed),
`analysis_words`, `output_chars`, `pred_fields` (the verdict's enum fields), `field_matches` and `field_match_share`.

## Metrics

`compute_metrics` adds, over every rollout (fractions; unparsed outputs count as wrong, never excluded):

- Gym's `pass@k`, `pass@1[avg-of-k]` and `majority@k` (majority over `pred_ladder`) of `accuracy` (= `ladder_match`,
  percent), with `no_answer`.
- `accuracy`, `balanced_accuracy` (mean of the recalls the data has), `kappa` (Cohen's, over yes/no/unparsed; omitted
  when chance agreement is 1), `fail_recall` (gold `no` caught), `pass_recall` (gold `yes` passed), `pred_yes_rate`,
  `gold_yes_rate`, `parsed_pred_yes_rate`, `leniency_gap` (= yes-rate of the parsed predictions minus the gold yes-rate on
  those rollouts; positive is more lenient than the label; unparsed outputs are left out so format failures do not read
  as strictness).
- `parse_fail` and `parse_fail/<error>`.
- `<slice>/<value>/{accuracy,balanced_accuracy,n}` for `source`, `family`, `harness` (from `meta`) and `tier` (from
  `label`); rows without the field count as `none`, and a field no row carries adds nothing.
- `hand/{accuracy,balanced_accuracy,kappa,fail_recall,pass_recall,n}` against `meta.hand_ladder`, plus `hand/gold_*`:
  the reference label's own agreement with the hand labels on the same rows.
- `expect/*`, the same against `meta.expect` (the expected verdict of a labelled behaviour stratum: a BAD turn should
  fail, a GOOD one pass). The reference judge is known to pass many BAD agentic turns, so `expect/fail_recall` next to
  `expect/gold_fail_recall` shows whether a policy is stricter or more lenient than its label.
- `field_acc/<field>`: agreement with `gold_fields` per field (a rollout without a prediction counts as a miss).

Key metrics: `mean/reward`, `accuracy`, `balanced_accuracy`, `kappa`, `parse_fail`, `leniency_gap`.

## Configuration

| Key | Default | |
|---|---|---|
| `parse_mode` | `strict` | `strict` or `lenient` (above). |
| `validate_schema` | `true` | Off: only the ladder (`addresses_concern`, `correct`) must be decided. |
| `reward_mode` | `hard` | `hard` or `soft`. |
| `field_bonus` | `null` | e.g. `{weight: 0.2, fields: [blocked_by, request_questionable, discard]}`. |
| `max_analysis_words` | `null` | Word cap on `analysis`. |

The agent is `simple_agent` (one model call). The config ships one dataset, `example` (5 synthetic rows, in git); add
`train` / `validation` entries under `agentic_turn_simulation_judge_simple_agent` with the paths of your built row files (other
`.jsonl` files under `data/` are git-ignored).

## Running

Serve the policy with vLLM and its reasoning parser, so the thinking arrives as reasoning items and the message holds
only the verdict (e.g. `vllm serve <model> --reasoning-parser <parser>`). Prompts are long (the static judge text is
about 5k tokens; real rows reach 100k+), so give the server a context that fits the rows plus the output budget, and
fail rows that do not fit loudly with `propagate_context_overflow_errors` (otherwise they come back empty and score 0).

```bash
# Start the servers
gym env start \
  --resources-server agentic_turn_simulation_judge \
  --model-type vllm_model \
  --model-url http://<policy-host>:8000/v1 --model <policy-model-name> --model-api-key EMPTY \
  ++policy_model.responses_api_models.vllm_model.propagate_context_overflow_errors=true

# Collect rollouts
gym eval run --no-serve \
  --agent agentic_turn_simulation_judge_simple_agent \
  --input resources_servers/agentic_turn_simulation_judge/data/example.jsonl \
  --output results/agentic_turn_simulation_judge_example.jsonl \
  --num-repeats 4 --temperature 1.0 --max-output-tokens 32768
```

The deprecated `ng_*` commands take the Hydra form of the same:

```bash
ng_run "+config_paths=[resources_servers/agentic_turn_simulation_judge/configs/agentic_turn_simulation_judge.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +policy_base_url=http://<policy-host>:8000/v1 +policy_api_key=EMPTY +policy_model_name=<policy-model-name> \
  ++policy_model.responses_api_models.vllm_model.propagate_context_overflow_errors=true

ng_collect_rollouts +agent_name=agentic_turn_simulation_judge_simple_agent \
  +input_jsonl_fpath=resources_servers/agentic_turn_simulation_judge/data/example.jsonl \
  +output_jsonl_fpath=results/agentic_turn_simulation_judge_example.jsonl \
  +num_repeats=4 "+responses_create_params={temperature: 1.0, max_output_tokens: 32768}"
```

## Data

Real rows are built outside git from stored reference-judge requests and results (the request messages are the
judge prompt byte for byte); only the synthetic example is committed. Check a built file before use:

```bash
PYTHONPATH=. python resources_servers/agentic_turn_simulation_judge/scripts/validate_rows.py train.jsonl validation.jsonl \
  --report counts.json        # --agent '' for rows without agent_ref
```

It stops at the first row that breaks the schema (TaskData with no undeclared key, `[system, user]` prompt with a
`<step_under_evaluation>` block, no tools, `row_id`/`group_id`, enum values in `gold_fields`, unique `row_id` per file)
and prints counts per file and overall: label balance (overall and per split), rows by tier / source / family /
harness / split, `gold_n`, prompt size, and consistency counts (`gold_p_yes` against `gold_ladder`, `gold_fields`
ladder against `gold_ladder`, duplicate prompts, groups in more than one split).

`data/example.jsonl` is synthetic: five made-up generic-harness turns rendered with the ATS judge prompt builder.
Regenerate it after an ATS prompt change with `PYTHONPATH=. python resources_servers/agentic_turn_simulation_judge/scripts/make_example_rows.py`.
`data/example_rollouts.jsonl` is synthetic too: hand-written verdicts for those rows (model name `synthetic`; three
right, one wrong ladder, one unparseable), scored by this server's `verify`. No model produced them.

## Known limitations

- **Label noise.** The reference is an LLM judge: single passes can disagree with each other, and even unanimous
  multi-pass labels can disagree with human labels. Agreement with the label has a ceiling below 1; read the hand-label
  metrics (`hand/*` against `hand/gold_*`) next to pooled accuracy.
- **Coupling to the ATS rubric.** The rubric, frame and schema are baked into each row's prompt when the row is built;
  a row's label is only valid for the rubric version in `label.rubric_tag` / `label.rubric_sha256`. Do not mix rubric
  versions without a tag. The schema the verifier validates against is a copy of the ATS file
  (`prompts/evaluation_schema.json`); `tests/test_golden.py` fails when the two drift, and checks that the copied
  parser and ladder still agree with ATS.
- **Same yardstick as the ATS reward.** The labels come from the judge that scores ATS rollouts, so higher agreement
  here and higher ATS scores share one yardstick; anchor on hand labels.
- **Benchmark contamination.** Rows rendered from agent-benchmark trajectories contain those benchmarks' tasks; keep
  them out of training data.
- Rows longer than the policy's context score 0 unless context overflow is propagated (above).
- `pairwise` rows are not implemented.
