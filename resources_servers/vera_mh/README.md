# vera_mh

VERA-MH (Validation of Ethical and Responsible AI in Mental Health; Belli et al., arXiv:2605.13318;
https://github.com/SpringCare/VERA-MH at commit `2c9d1fc`, rubric v1.2) evaluates how a chatbot behaves when a user in a
suicide-risk scenario seeks support. A user-simulator model role-plays one of 100 clinician-authored personas, the
chatbot under evaluation replies, and the transcript is judged with a clinical rubric navigated one yes/no question at
a time. Five dimensions are rated: Detects Potential Risk, Confirms Risk, Guides to Human Care, Supportive
Conversation, Follows AI Boundaries.

Two servers implement the benchmark:

- `responses_api_agents/vera_mh_agent` simulates the conversation. The persona (user simulator) and the chatbot under
  evaluation are separately named model servers (`user_model_servers`, `model_server`); the persona speaks first, the
  conversation runs 30 turns (chatbot last) or stops when the persona emits `<END OF CONVERSATION>`; message
  construction, the per-turn role reminder and the transcript format are upstream's.
- `resources_servers/vera_mh` (this server) judges the transcript with the vendored v1.2 rubric (`rubric/`, per-file
  hashes in `rubric/PROVENANCE.md`): one structured `{answer, reasoning}` question per call, upstream's GOTO navigation,
  special cases and fallback matching, severity-based dimension ratings, and the VERA-MH score.

Metric definitions, the upstream-to-Gym mapping and every divergence from upstream are in
[`benchmarks/vera_mh/METRICS.md`](../../benchmarks/vera_mh/METRICS.md).

## Protocol

Upstream's recommended profile (README "Recommended settings"): all 100 personas, one 30-turn conversation each with
GPT 5.2 and one with Claude Opus 4.5 as the user simulator (200 conversations), judged by GPT 5.4 at
`reasoning_effort=low`, pooled over both simulator suites. `gym eval prepare --benchmark vera_mh` downloads the pinned
persona sheet and writes the 200 rows.

## Running

The judge and the two user simulators are reached through OpenAI-compatible chat completions. With the defaults in
`configs/vera_mh.yaml` two keys are needed:

```bash
export OPENAI_API_KEY=...       # judge (gpt-5.4) and the GPT 5.2 user simulator, from OpenAI
export OPENROUTER_API_KEY=...   # the Claude Opus 4.5 user simulator (anthropic/claude-opus-4.5), through OpenRouter
```

Any other route to the same three models is set per role with `VERA_MH_JUDGE_*`, `VERA_MH_USER_GPT52_*` and
`VERA_MH_USER_OPUS45_*` (`_BASE_URL`, `_API_KEY`, `_MODEL`), for example everything through OpenRouter:

```bash
export VERA_MH_JUDGE_BASE_URL=https://openrouter.ai/api/v1 VERA_MH_JUDGE_API_KEY=$OPENROUTER_API_KEY VERA_MH_JUDGE_MODEL=openai/gpt-5.4
export VERA_MH_USER_GPT52_BASE_URL=https://openrouter.ai/api/v1 VERA_MH_USER_GPT52_API_KEY=$OPENROUTER_API_KEY VERA_MH_USER_GPT52_MODEL=openai/gpt-5.2
```

Some routes to GPT models apply a provider-side content filter that rejects suicide-risk transcripts
(`finish_reason=content_filter`); a filtered question fails the judge call for that conversation, which then lands in
the failures sidecar. A direct OpenAI key does not do this. The judge model matters: rubric ratings move with the
judge, so a different judge gives numbers that are not comparable.

The chatbot under evaluation is the policy model. With an OpenAI-compatible endpoint:

```bash
gym eval prepare --benchmark vera_mh
gym eval run --benchmark vera_mh --model-type vllm_model \
    --model <served model id> --model-url <base-url>/v1 --model-api-key $MODEL_API_KEY \
    --output results/vera_mh/<model>.jsonl
```

No sampling parameters or output cap are passed to the chatbot; the endpoint's defaults apply, as with upstream's
provider clients. `--resume` re-runs only the rows that failed. The run writes the rollouts, a failures sidecar,
the materialized inputs and `<stem>_aggregate_metrics.json`; the headline number is `vera_score`.

## What `verify()` returns

`reward` is the conversation's own VERA-MH score over its rated dimensions, rescaled to [0, 1] (0.5 when no dimension
was rated). It is diagnostic only: the official number is the pooled score in the aggregate metrics, and the mean
reward is not the score. Every row also carries `ratings` per dimension, `rating_reasoning`, `yes_question_ids`,
`yes_reasoning`, the full `question_path` with the judge's reasoning, one `judge_receipts` entry per judge call
(model, response id, finish reason, usage, content and hashes), `judge_model`, `transcript_sha256` and
`transcript_matches_agent`.

A simulator or chatbot call that fails aborts the conversation and writes a `vera_mh_simulation_failed` sidecar row; a
judge transport failure or exhausted structured-output retries writes a `judge_failed` row. Both are excluded from all
metrics and retried on `--resume`, mirroring upstream's "skipped due to errors" accounting. Neither is a chatbot safety
failure.

## Tests

```bash
gym env test --resources-server vera_mh
pytest resources_servers/vera_mh/tests/ responses_api_agents/vera_mh_agent/tests/
```

`benchmarks/vera_mh/calibrate.py` checks a run against the pinned upstream code (persona prompts, transcripts and
message construction recomputed by upstream; judge answers replayed through upstream's `LLMJudge`; ratings scored by
upstream's `judge/score.py`). See the benchmark README.

## Terms

The rubric, prompts and persona template are redistributed under Spring Care's VERA-MH license
(`rubric/LICENSE`), which permits use and redistribution with the notice intact and states that a VERA-MH score is
not a certification, endorsement or safety determination. The persona sheet is downloaded at prepare time and is not
committed; only the five example rows and their rollouts are tracked.
