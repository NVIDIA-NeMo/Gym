# Decision Index (DI 0.3)

The [Decision Index](https://github.com/apolinario/decision-index) measures **typed decisions**: a request is a
`state` (context, possibly empty) plus typed `questions`, each a multiple-choice `choice` with explicit options
(`criteria`) or a yes/no `noul`. The public 0.3 suite has 140,178 requests over 43 benchmarks (BFCL, MMLU-Pro, BBH,
HoVer, RAGTruth, When2Call, BANKING77, ...) in five areas. Its headline, the public **Decision Index**, is the
weighted mean of per-benchmark skill: the native metric made coverage-adjusted (unanswered = wrong) and
chance-corrected (0 = random, 100 = perfect), with official area weights (Knowledge & Reasoning 25.8%, Language
Understanding 25.8%, Retrieval & Classification 20.0%, Tools & Automation 18.3%, Arts & Human Taste 10%) and gold
benchmarks weighted 1.2 within their area. 37 benchmarks count; 6 are shown but not counted (MMLU, ARC-Easy,
ARC-Challenge, SimpleBench, RouterBench, SGD).

This Gym benchmark runs the public suite and scores it with the Decision Index kit's own scorers, pinned to
`apolinario/decision-index@9eb2dbe2`, so a Gym run and a kit run of the same answers give the same index.

## Two modes, two configs

| | decision-model mode | LLM mode |
|---|---|---|
| config | `config_decision.yaml` (`decision_index/config_decision`) | `config.yaml` (`decision_index`) |
| policy | a typed decision engine, `/v1/systemone` (or `/v1/decisions`) through [`decision_engine_model`](../../responses_api_models/decision_engine_model/README.md) | any chat model (`--model-type`) |
| what the policy sees | exactly `state` and `questions`, as the kit's `http` engine sends them | the system prompt below plus the same JSON request as the user message |
| answer | a probability for every option of every question, checked with the kit's `validate` | one JSON object mapping each question key to an option key (`choice`) or `true`/`false` (`noul`), at the end of the reply; the chosen option gets probability 1 |
| unanswered | engine capacity rejection = `unsupported`; failed call (after retries) = `error`, masked in Gym's mean reward but counted as wrong in the index, as in the kit | unparseable, missing key or unknown option = `error` (wrong) |

The two modes report separately: each config has its own resources-server instance (`answer_mode: decision` or
`llm`) and agent, so their metrics never mix. Probability-based metrics (retrieval nDCG over 32 candidates, F1
thresholds) see one-hot probabilities in LLM mode, so LLM scores on ToolRet and BRIGHT are ranked with ties.

## Measurement contract

- **Provenance.** Suite built by the kit (`suite rebuild` + `suite import`, all hashes verified), edition 0.3:
  GSM8K rows rebuilt, ForecastBench retired, ToolRet/BRIGHT answerable-query and ACOS/Home-appliance subsets and
  the 442 exclusions applied. Conversion to Gym rows is one row per kit request:
  the request JSON goes into `responses_create_params` and the kit row (questions, gold) into
  `verifier_metadata.di_row`, as in `resources_servers/decision_index/data/example.jsonl`. Split: the suite's test items (`canonical_split: test`).
- **Licences.** The suite mixes 43 upstream datasets with their own terms. Only the 16 benchmarks with an explicit
  MIT, Apache-2.0 or CC BY licence (65,346 rows) are in Gym's GitLab dataset registry. The other 27 (share-alike,
  non-commercial, research-use, gated, no-redistribution, or not verified) are fetch-only: rebuild the suite with
  the public kit and convert it locally (see Data). Never train on any of it; BBH carries the BIG-bench canary.
- **Prompt.** `prompt.yaml`: one fixed system instruction (answer format only) and the JSON request; no
  per-benchmark prompts, no few-shot, no truncation. A decision engine gets only the JSON request.
- **Reward.** Per row, 1.0 when every scored question is right (ChessBench and Habermas accept any listed answer;
  PhishNChips scores `verdict` only; rows without gold, the retrieval candidates, score 1.0 when answered). The row
  reward is a convenience; the benchmark's metric is `decision_index`.
- **Metrics.** `decision_index` (0 to 100, the headline), `decision_index_raw`, `decision_index_area/<area>`,
  `decision_index_skill/<id>:<name>` (counted), `decision_index_not_counted/<id>:<name>`, status counts,
  `decision_index_requests` and `decision_index_full_suite` (1.0 only when all 140,178 requests ran). A partial run's
  index treats missing benchmarks as zero, so it is a lower bound.
- **Run conditions.** One read per request, `num_repeats: 1` (decision engines are deterministic; for sampled LLMs
  raise `num_repeats` until the index varies by less than 1 point; the index scores the first repeat of each
  request, Gym's reward statistics cover all repeats). LLM sampling and token limits come from the model server.

Reference (kit, not Gym): Nemotron Decide 8B, public DI 0.3 = 67.21 (raw 75.34), 140,178/140,178 answered.

## Data

```bash
# Registry rows only (needs mlflow_tracking_uri / mlflow_tracking_token):
gym eval prepare --benchmark decision_index
# Full suite: rebuild with the kit, convert, and point prepare at the result
gym eval prepare --benchmark decision_index ++decision_index_jsonl=/abs/path/di03_public_full.jsonl
# or registry rows + your converted fetch-only rows
gym eval prepare --benchmark decision_index ++decision_index_fetch_only_jsonl=/abs/path/di03_public_fetch_only.jsonl
```

## Run

Serve the decision engine yourself, next to the Gym run (the standard way): start its `/v1/systemone` server on the
job's GPUs and point `DECISION_ENGINE_BASE_URL` at `http://127.0.0.1:<port>`. The endpoint is required and has no
default. For throughput, shard the rows over several local servers (one `gym eval run` per server, each with its own
`head_server.port`) and merge the shards' `di_result` with `verifier.score_index`. Do not point full runs at a shared
production endpoint.

```bash
# Decision engine (served locally)
DECISION_ENGINE_BASE_URL=http://127.0.0.1:8770 DECISION_ENGINE_MODEL=nemotron-decide \
gym eval run --benchmark decision_index/config_decision --split benchmark \
    --output results/di_decision/rollouts.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=$PWD/results/di_decision/capture

# LLM
gym eval run --benchmark decision_index --model-type openai_model --split benchmark \
    --output results/di_llm/rollouts.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=$PWD/results/di_llm/capture
```

The index is in `rollouts_aggregate_metrics.json` (`agent_metrics.decision_index`).
