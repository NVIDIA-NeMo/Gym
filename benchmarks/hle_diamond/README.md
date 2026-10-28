# HLE-Diamond benchmark

[Dataset](https://huggingface.co/datasets/cais/hle-diamond) (MIT; the dataset card asks users not to
publicly share, re-upload or redistribute it) and
[announcement](https://lastexam.ai/blog/hle-diamond) (Center for AI Safety and Scale AI, 2026-09-22).

HLE-Diamond is a 1000-question subset of
[Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) split into two partitions of 500
questions each: `reasoning` (the answer has to be derived) and `knowledge` (the answer has to be
recalled). 272 questions include an image; 796 are exact-match and 204 multiple choice. The model
must produce an explanation, a final answer and a confidence; an LLM judge compares the extracted
final answer with the reference answer.

Preparation pins revision `04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23` <!-- pragma: allowlist secret -->
of the `test` split and downloads it at run time. Nothing from the dataset is committed.

## Scope

| Benchmark | Rows | Prompt | Use |
| --- | --- | --- | --- |
| `hle_diamond` (default) | 1000 (272 with an image) | `prompts/default.yaml`, baked at prepare time (`prompt_config: null`) | the published no-tools setting; needs a vision-capable policy model |
| `hle_diamond/text` | 728 (no image) | `prompts/default.yaml` at rollout time | text-only policy models |
| `hle_diamond/opencode`, `hle_diamond/pi` | 728 | `prompts/opencode.yaml` | sandboxed code tools, no search |
| `hle_diamond/opencode_search`, `hle_diamond/pi_search` | 728 | `prompts/opencode.yaml` | sandboxed code tools plus Tavily search |

Two composable overrides live next to the presets and are not benchmarks (no dataset, not listed by
`gym list benchmarks`): [`judge_openai.yaml`](judge_openai.yaml) (dedicated judge, see Scoring) and
[`sandbox_docker_local.yaml`](sandbox_docker_local.yaml) (sandboxed presets on a local Docker host,
see Quickstart).

**Why the default inverts `hle`'s convention.** `benchmarks/hle` defaults to the text subset and
exposes images as the `hle/config_vision` variant, because HLE's headline figures are text-only.
HLE-Diamond's published numbers are over all 1000 questions, images included, so here the full set is
the default and the text subset is the variant. A `hle_diamond/text` run covers 728/1000 questions
(347 reasoning, 381 knowledge) and is not comparable with the published table; report that coverage.

**Why the sandboxed presets are text-only.** The OpenCode and Pi adapters take a single text user
turn (`input_text` parts only), so image questions cannot be sent through them. The presets therefore
run the 728 text rows, as `benchmarks/hle`'s presets do, and with-tools numbers from them cover
728/1000 questions. The search presets are selectable with `--benchmark` but, exactly like
`hle/opencode_search` and `hle/pi_search`, do not appear in `gym list benchmarks` because they extend
the code-only presets without redeclaring the dataset.

Dropped at prepare time: `rationale`, `author_name`, `canary`, `image_preview`, `rationale_image`.
The `image` data URI appears only inside the materialized `input_image` block of the default set.

## Prompting

The prompt files live in [`prompts/`](prompts/) and carry the HLE authors' `SYSTEM_PROMPT` from
`hle_eval/run_model_predictions.py` in
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) (MIT; each file names the
commit) byte for byte:

```text
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

This is a deliberate departure from `benchmarks/hle` and `benchmarks/hle_verified`, whose
`prompts/default.yaml` files (identical copies, one per benchmark) carry Gym's paraphrase
(`{your explanation for your final answer}`, `{your succinct, final answer}`). The sole purpose of
this benchmark is comparison with the published table, so the measured prompt has to be upstream's;
keeping the Gym paraphrase for consistency with the other HLE benchmarks loses to fidelity. Do not
"fix" the wording back to it.

- `prompts/default.yaml`: the verbatim text as the system message with the bare question as the user
  message, the placement upstream uses (upstream moves the instruction into the user role only for
  `o1` models; this benchmark always sends it as the system message). The braces are doubled in the
  file because `nemo_gym.data.prompt` fills `{placeholders}` with `str.format_map`; the rendered message
  carries the single braces, and `tests/unit_tests/test_hle_diamond.py` asserts that the rendered
  string and the prepared rows equal the upstream text. The default set bakes it into
  `responses_create_params.input` at prepare time (`nemo_gym.data.vision_input.build_image_input`); the
  text variant applies the same file at rollout time.
- `prompts/opencode.yaml`: the same verbatim instruction folded after the question into the single
  user message the OpenCode and Pi adapters accept; the harnesses' native system prompts stay intact.
  Two departures from upstream's placement are inherent to these harnesses and belong next to every
  with-tools number: the instruction arrives as user text rather than as the system message, and the
  harness's own system prompt (several thousand tokens for OpenCode) precedes it.
- Judge: `benchmarks/hle/prompts/judge.txt`, the HLE judge prompt adapted for the
  `equivalence_llm_judge` label parser, is reused unchanged. Upstream's `run_judge_results.py` parses
  a structured object (`extracted_final_answer, reasoning, correct: yes|no, confidence, strict`);
  Gym's file renames the fields to `Reasoning:` / `Judgement:` / `Confidence:` and the verdict is the
  `Judgement: yes|no` string found in free text, with unparseable judge output tracked by
  `judgement_parsing_issue_rate`. The grading criteria are the same.

Image questions: upstream attaches the image as `image_url` with no `detail` setting (the provider
default); Gym's shared builder requests `detail: high`, which changes image tiling and token counts
on the 272 image rows for OpenAI-family models. Making `detail` configurable would be a change to
`nemo_gym.data.vision_input`, not to this benchmark.

Upstream calls the chat-completions API directly (`client.chat.completions.create`); this benchmark
sends a Responses API request through the gateway, which translates it per provider (system-role
handling, the `reasoning` object, what the response echoes; see Reasoning effort for what survives).
Upstream sends no temperature and, by default, no output-token cap (`--max_completion_tokens`
defaults to `None`; the value used for the published table is not stated). When comparing with the
published table, send `++responses_create_params.reasoning.effort=high` and nothing else; on this
gateway a `temperature` is dropped by some models and rejected by others.

### Reasoning effort

The published no-tools table was collected at "reasoning high". What that means through this gateway
was probed per model (Responses API, `reasoning.effort` low against high on the same items):

1. GPT-6 Astra, GPT-6.1 Sol and Gemini 4 Argon honour `reasoning.effort`: reasoning tokens rise
   several-fold from low to high, and the GPT models echo the effort in the response. Label such a
   run "reasoning high".
2. Claude Opus 5.5, Claude Sonnet 5.5 and Claude Opus 5 show no measurable effect of the parameter
   through this gateway: adaptive thinking is always on and Anthropic's `output_config` is dropped by
   the gateway. Label such a run "effort not controllable via this gateway (adaptive default)" and
   never call a Claude run "reasoning high".
3. Every model is still run with the one request shape (`++responses_create_params.reasoning.effort=high`,
   no temperature, top-p or output cap), so runs differ only in what the gateway does with it. Apply
   the label by model family from the probe, not from token counts: `reasoning_tokens > 0` is not
   evidence that the effort was honoured (the Claude models report thousands of reasoning tokens at
   every setting), and only the GPT models echo the effort (`response.reasoning.effort`; Gemini and
   Claude echo `reasoning: null`). Record `response.usage.output_tokens_details.reasoning_tokens` on
   the pilot rows for the mean-output-token comparison only.

On the no-tools path the effort travels on Gym's Responses request. On the sandboxed path it does
not: `++responses_create_params.reasoning.effort` decorates only Gym's run request, while OpenCode
and Pi build their own model calls inside the sandbox (a model-call capture of the stock preset
showed no `reasoning_effort`, so the gateway default, medium, applied). The presets therefore carry
each harness's own knob:

- `opencode.yaml` (inherited by `opencode_search.yaml`):
  `opencode_config.provider.nemo_gym.models.dummy_model.options.reasoningEffort: high`, which
  OpenCode 1.17.11 maps to `reasoning_effort: high` in its chat-completions body (verified in a
  capture). Override per run with
  `++hle_diamond_opencode_agent.responses_api_agents.opencode_sandboxed_agent.opencode_config.provider.nemo_gym.models.dummy_model.options.reasoningEffort=<level>`.
- `pi.yaml` (inherited by `pi_search.yaml`): `thinking: high`, passed to Pi as `--thinking high`
  (Pi's levels: `off, minimal, low, medium, high, xhigh, max`). **Caveat:** the Pi adapter registers
  its provider with `compat.supportsReasoningEffort: false`, and Pi's `openai-completions` path emits
  `reasoning_effort` only when that flag is true, so Pi runs through this adapter do not transmit
  the level today (established by inspecting `pi_agent/app.py`, whose provider block the sandboxed
  adapter reuses, and Pi 0.85.1's `openai-completions` path; no Pi request has been captured yet).
  The knob is the right value and takes effect once the adapter exposes the flag;
  until then label Pi runs "effort not transmitted (adapter)".

## Dataset format

Every prepared row carries:

| Field | Meaning |
| --- | --- |
| `question` | question text, templated into the prompt |
| `expected_answer` | reference answer; read by the judge |
| `answer_type` | `exactMatch` or `multipleChoice`; carried for analysis, never read by the judge |
| `uuid` | upstream `id` |
| `partition` | `reasoning` or `knowledge`; read by the accuracy slices |
| `category`, `raw_subject` | upstream subject labels; carried for analysis, never read |
| `has_image` | whether the question has an image |
| `responses_create_params` | default set only: the materialized `input` messages; image rows carry an `input_image` block with `detail: high` (upstream sends no `detail`; see Prompting) |

`prepare.py` loads the whole split, validates it (1000 rows, 500 + 500 partitions, 272 image rows,
1000 unique ids) and only then writes, so an incomplete download never leaves a file behind.
`--limit N` (positive integers only; rejected at parse time otherwise) is the one supported way to
prepare a subset, and it says loudly that the completeness check was skipped; the limit applies after
the text filter and the summary line says `limited to N`. `prepare_text.py` is a thin wrapper for
`prepare(text_only=True)`; its script form accepts `--limit N` and nothing else, and
`gym eval prepare --benchmark hle_diamond/text ++prepare_script_args.limit=N` reaches the same
function.

## Scoring

`reward` is 1.0 when the judge answers `Judgement: yes` and 0.0 otherwise: `equivalence_llm_judge`
with the HLE labels, no swap check, no regex extraction and no per-record regex, the same judge
settings as `benchmarks/hle/config.yaml` except for the case rule below (see Prompting for how the
judge prompt differs from upstream's structured output). An empty final answer scores 0 without a
judge call (`failure_reason: empty_final_answer`). The judge defaults to the policy model, as in
`benchmarks/hle`, so the model under test grades itself unless `judge_openai.yaml` is composed in;
self-judged numbers are not comparable with the published table (see Quickstart).

The verdict labels are matched case-insensitively: `judge_labels_case_insensitive: true` on the single
judge server instance every preset shares (`str.casefold` on the judge text and on the labels; the
recorded `verdict_label` is always the configured spelling). Upstream reads a structured
`correct: yes|no` field whose case cannot vary, whereas Gym finds the label in free text, and on the
gold sweep (Harness validation) o3-mini wrote `Judgement: Yes` on 1 of 1000 rows, which the byte-exact
default scored 0 as `no_verdict`. The knob is off everywhere else, so `hle` and `hle_verified` are
unchanged; when it is on, labels that differ only in case are rejected at config time.

Judge request: the judge call carries no `max_output_tokens` and no reasoning parameter
(`judge_responses_create_params: {input: []}`), so the judge model runs at its default effort and its
verdict is never truncated. Upstream caps the judge at `max_completion_tokens=4096`. The divergence is
deliberate: a long reasoning trace from an o3-mini-class judge must not cut the verdict off and turn
a graded row into a parsing failure.

Headline metric: `mean/reward`, accuracy over every evaluated row. Additional metrics:

- `slices/partition/reasoning/mean/reward` and `slices/partition/knowledge/mean/reward`: accuracy per
  partition, also promoted to `key_metrics`, with `question_count` and `rollout_count` next to each
  and `slices/partition/missing_label_rollouts` counting rows without a label (never assumed). They
  come from the judge server's `report_slice_fields: [partition]` knob, which is off by default
  everywhere else. The hook reads the label from the saved rollout row, so `LLMJudgeVerifyResponse`
  now echoes the request's extra row fields (`uuid`, `partition`, `category`, `raw_subject`,
  `has_image`); before this change the judge's `/verify` reply dropped them and every slice would
  have reported `missing_label_rollouts` only. Open PR
  [#4658](https://github.com/NVIDIA-NeMo/Gym/pull/4658) adds `category`, `raw_subject` and modality
  slices under `domains/...` for HLE; the two hooks are additive, the fields this benchmark carries
  are the ones that PR reads, and that PR needs the same echo to see them in a real run.
- `judgement_parsing_issue_rate` and its per-kind rates: the fraction of judged rollouts whose judge
  output had no, conflicting or repeated verdicts.
- Coverage: the upstream harness divides by all 1000 questions and counts a failed API call or a
  missing judge verdict as incorrect, whereas `mean/reward` is over the rows present in the rollouts
  file (a failed `/run` ends the run by default, or is dropped from the score with
  `++route_failures_to_sidecar=true`). To use the upstream denominator, run with
  `++count_missing_rollouts_as_zero=true` (and `'++count_failure_classes_as_zero=[agent_run_error]'`
  when failures are routed to the sidecar) and report `coverage/imputed`; otherwise compare the
  rollouts file's row count with 1000 (728 for the text configs) and report rows evaluated / 1000
  next to every accuracy.

Upstream also reports a calibration error from the model's stated confidence; it is not computed here.

### Dedicated judge

`judge_openai.yaml` is a composable override, not a benchmark: it declares no dataset and does not
appear in `gym list benchmarks`. It defines `hle_diamond_judge_model`, an `openai_model` on the
policy gateway (`policy_base_url` / `policy_api_key` from `env.yaml`) serving
`HLE_DIAMOND_JUDGE_MODEL` (default `azure/openai/o3-mini`; the GPT route returns well-formed
reasoning items, so the judge needs none of the repair the policy server does, Quickstart > Model
server), and repoints the single
`hle_diamond_equivalence_llm_judge_resources_server` that every preset shares:

```bash
gym eval run --benchmark hle_diamond --split benchmark '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]' ...
```

Judge identity is a documented divergence, next to the `judge.txt` format adaptation and the uncapped
judge call: upstream pins the judge to the snapshot `o3-mini-2025-01-31`, whereas `azure/openai/o3-mini`
is a gateway alias whose served snapshot is not pinned and cannot be read from
`judge_evaluations[*].response.model` (the gateway echoes the requested id). Record the alias and the
run date next to every number.

Composition order is load-bearing. The presets set `judge_model_server` directly and a later entry in
`config_paths` overrides an earlier one (`GlobalConfigDictParser.load_extra_config_paths`), so the
override has to merge after the preset. The raw Hydra token above does: the CLI appends raw tokens
after every translated flag (`--benchmark`, `--model-type`), wherever the token stands on the command
line. `--config <file>` does not: in this Gym version the CLI translates `--config` before
`--benchmark`, so a `--config` override merges first and silently loses every key the preset sets
(`judge_model_server`; for the search presets also `network_access` and the environment note). Never
pass `judge_openai.yaml` or `sandbox_docker_local.yaml` with `--config`;
`tests/unit_tests/test_hle_diamond.py` composes the commands in this README through the CLI's own flag
translation to keep that true.

## Harness validation

Model-free checks that have been run (commands below, from the repository root):

- `tests/unit_tests/test_hle_diamond.py`: preparation against an in-memory arrow fixture (field set,
  the `input_image` block, the text filter, fail-closed on a wrong row count / partition split /
  image count / duplicated id, `--limit` truncation after the text filter and above the corpus size,
  the `prepare_text.py` script form, no download on invalid arguments); the shipped prompt files
  rendering exactly the upstream `SYSTEM_PROMPT` and `prepare.py` baking exactly it; config
  discovery and composition, the judge prompt path, manifests, the reasoning-effort knobs of all
  four sandboxed presets, `sandbox_docker_local.yaml` resolving to exactly the Docker provider with
  inherited networking while the unused harness's block stays inert; the run commands of this README
  parsed from the file and composed through the CLI's own flag translation (the judge and Docker
  overrides merge after the preset and win, the `--config` form merges first and loses, every
  documented command carries `--split benchmark`); the partition slices through
  `compute_aggregate_metrics`; and the production row shape (`/verify` reply through
  `SimpleAgentVerifyResponse` into the slices). Mutation-checked: changing one word of the prompt,
  lowering either effort knob to `medium`, breaking the count check, the limit truncation, the
  response field echo or the slice hook, switching a README command to `--config` or dropping its
  `--split benchmark` each makes tests fail. The `gym env start` / `--no-serve` pair of Running the
  full set is composed through the same CLI translation: the three `--config` files land in
  command-line order so the judge override wins, and the collection command carries the
  sidecar routing, `reasoning.effort=high`, one repeat and no sampling flag.
- `resources_servers/equivalence_llm_judge/tests/test_app.py`, the case rule: `Judgement: Yes` scores
  1.0 with `judge_labels_case_insensitive` and `no_verdict` / 0.0 without; a text holding both
  `Judgement: yes` and `JUDGEMENT: No` still follows the last-verdict rule with `conflicting_verdicts`;
  matches return the configured labels; legacy rollouts are re-parsed with the same setting; labels
  differing only in case are rejected when the knob is on. Mutation-checked: removing the folding,
  either call-site pass-through, the validator or the knob in `config.yaml` each fails a test.
- `tests/unit_tests/test_hle_diamond_search_exclusions.py`: the generator's normalisation, dedup and
  audit trail; the rendered and the committed list loading through the real Tavily parser and
  blocking load-bearing hosts; the committed list being at least as strict as `benchmarks/hle`'s;
  and the upstream pin being a full 40-character revision (the `--download` URL depends on it).
- Real preparation printed `Wrote 1000 problems ... (272 with an image)` and
  `Wrote 728 problems ... (skipped 272 image questions)`; the written rows were re-counted
  (500 / 500 partitions, 272 `input_image` blocks, 1000 and 728 unique ids, no dropped field leaked)
  and every one of the 1000 materialized system messages equals the upstream text byte for byte.
- Both manifests validate against the prepared files:

  ```bash
  gym env validate --manifest benchmarks/hle_diamond/manifest.yaml        # hle_diamond: 1000 rows (benchmark)
  gym env validate --manifest benchmarks/hle_diamond/text/manifest.yaml   # hle_diamond_text: 728 rows (benchmark)
  ```

### Gold-as-prediction sweep and negative controls (model-free)

Run on 2026-10-16 over the real HTTP `/verify` path (`gym env start` with `config.yaml`, then a
`--no-serve` collection and `gym eval aggregate`), with every row's `expected_answer` fed back as the
model's `Answer:` line. No policy model is involved: these are properties of the harness. The judge
was an oracle stub (a local `/v1/responses` server that grades the filled HLE judge prompt by
normalised exact match of the final `Answer:` line against `[correct_answer]`), so only Gym's plumbing
is measured:

| Input fed back as the answer | Rows scored 1.0 |
| --- | --- |
| the reference answer | 1000/1000 (reasoning 500/500, knowledge 500/500; exactMatch 796/796, multipleChoice 204/204) |
| empty final answer | 0/1000 (`failure_reason: empty_final_answer`, no judge call) |
| `Answer: None` | 0/1000 |
| a wrong answer (`<gold>_wrong` on exactMatch rows, another letter on multiple-choice rows) | 0/1000 |
| the reference followed by a contradicting second `Answer:` line | 0/1000 |

The last two rows show plumbing only: the oracle grades the last `Answer:` line, so the 0/1000 on
the double-answer control proves that both lines reach the judge intact and the verdict flows back,
not how a real judge treats a contradicting second answer (not measured with o3-mini); and the
oracle's "wrong answer" is the same `_wrong` decoration whose weakness against a real judge is
stated under Judge agreement on gold.

On the gold file `slices/partition/reasoning/mean/reward` and `slices/partition/knowledge/mean/reward`
were 1.0 and 1.0 with `question_count` 500 each; a mixed file (gold on the reasoning rows, wrong on
the knowledge rows) gave `mean/reward` 0.5 with slices 1.0 / 0.0; `uuid` and `partition` were echoed
on every saved row.

### Judge agreement on gold (a judge property, not a model score)

The same gold sweep through the dedicated judge (`judge_openai.yaml`, alias `azure/openai/o3-mini`,
2026-10-16, default effort, no output cap) scored 997/1000 (reasoning 497/500, knowledge 500/500)
with `judgement_parsing_issue_rate` 0.001, mean judge output 180 tokens, mean latency 3.1 s and 100 s
wall at concurrency 32. The three misses:

1. `675f556ede1b5c57f5efbf6a`: the judge wrote `Judgement: Yes`; the byte-exact label match returned
   `no_verdict` and reward 0. A verifier-side false negative, since removed by
   `judge_labels_case_insensitive` (Scoring).
2. `671360871b12bd5183fec91a`: the judge dropped a parenthesis while extracting a lambda-calculus
   answer and judged its own extraction wrong. A judge-model error.
3. `67c0f062afa64ff0bf4ecf55`: rubric-style reference ("Any five 5-letter words that are headwords
   ...") that even the reference text fed back verbatim fails. A dataset property (Known limitations).

Negative controls through the same judge on a stratified 100-row sample (80 exactMatch, 20
multipleChoice):

- `Answer: None`: 2/100 scored 1.0, both multiple-choice items whose correct option begins with
  "None" / "none" (Known limitations); the control is confounded on such items and clean on the
  rest.
- another sampled row's reference of the same answer type (rows grouped by `answer_type`, the donor
  being the next row by `uuid` whose reference differs from the row's own): 0/100 scored 1.0
  (exactMatch 0/80, multipleChoice 0/20; reasoning 0/50, knowledge 0/50), 100 judge calls with
  `judge_labels_case_insensitive` on, 0 parsing issues, mean judge output 241 tokens, 22 s wall at
  concurrency 16 (2026-10-16). This is the judge's false-positive rate on genuinely wrong answers
  for this sample.
- the reference with a `_wrong` suffix: 14/100 scored 1.0, all on exactMatch rows (14/80); o3-mini
  treats the suffix as an annotation of the right answer. A weak control, kept because 14/80 bounds
  the judge's leniency towards decorated right answers from above; it is not a wrong-answer rate.
- a different option letter on the multiple-choice rows: 0/20 scored 1.0.

## Known limitations

- `67c0f062afa64ff0bf4ecf55` carries a rubric-style reference ("Any five 5-letter words that are
  headwords ..."): the judge prompt asks whether the extracted answer matches `[correct_answer]`, so
  even the reference fed back verbatim scores 0 with o3-mini. The row is kept because upstream keeps
  it; expect at most 999/1000 on gold with that judge.
- At least 11 of the 204 multiple-choice rows have a correct option that begins with "None",
  matched case-insensitively (read from the prepared rows: the option line of the reference letter
  starts with "None" on ten rows and with "none" on `66ebefa090db075818df99a9`; the judge treats
  both alike; 9 reasoning, 2 knowledge: `6736c2e82db9d8cc7cd8e607`, `69a738a50a19a2f7b803ca97`,
  `66fb75ded83ed7a299fdd136`, `67c133211c3841e3c6b8d938`, `66fde2de0b21f205c121aca7`,
  `672422631fac8550679c4bd6`, `66e8e1056c0b8ace336996ea`, `67253c33ad98e94b47ea3477`,
  `66ebefa090db075818df99a9`, `670d122102e9032e4a224628`, `67cbde9e89f4b433f1de1ada`). A model
  answering "None" on them is right, so an `Answer: None` control over the whole set has a floor of
  about 11/1000, and the two false positives of the 100-row control were two of these rows. The
  carried `answer_type` and `uuid` let you separate them.
- On the sandboxed presets (`opencode`, `opencode_search`, `pi`, `pi_search`) the aggregate carries
  neither the partition slices nor `judgement_parsing_issue_rate`: the adapters' inherited
  `aggregate_metrics` never calls the judge resources server's `compute_metrics`. The rollouts carry
  `partition` and `judge_evaluations`, so compute both post hoc (Reading sandboxed results); the
  durable fix is for those adapters to proxy aggregation to their resources server, a Gym
  shared-component change outside this benchmark.
- The Pi presets do not transmit the thinking level: the Pi adapter registers its provider with
  `compat.supportsReasoningEffort: false`, so Pi's own request carries no `reasoning_effort` whatever
  `thinking:` says (Prompting > Reasoning effort). Label Pi runs "effort not transmitted (adapter)";
  the fix is a change to `responses_api_agents/pi_agent`, outside this benchmark.
- The search presets' leakage policy binds only the Tavily tools: under the Docker provider (open
  egress) the agent can search and fetch from bash around it, and Tavily pre-filters at most 150 of
  the 220 excluded domains, the rest being enforced locally (Sandboxed agents (with tools), Search
  exclusions). A reproduction-grade with-tools run needs the OpenSandbox provider with
  `network_access: model_and_tools`.
- Gateway timeout: the LiteLLM gateway used for the pilot enforces a 360 s per-attempt limit
  (HTTP 408, `litellm.Timeout ... Timeout passed=360.0`) and the model server (`litellm_model`
  inherits `openai_model`'s retry layer) retries a 408 with a fresh generation, so an answer that
  needs more than six minutes is lost rather than slow (Running the full set says how to report the
  loss). Not fixable from Gym configuration; ownership of the limit sits with the gateway operators.
- `litellm_model` refuses to start unless the proxy's `/health/readiness/details` reports
  `litellm_version >= 1.83.0` (the guard added after the March 2026 LiteLLM supply-chain incident;
  `_check_proxy_version` in its `app.py`). A key scoped to the model routes gets HTTP 403 on that
  route, so the server fails before the first request (`Could not verify LiteLLM proxy version at
  <base>/health/readiness/details`). The runs recorded here did not meet the guard because they ran
  `openai_model` with an equivalent normaliser (Quickstart > Model server). A follow-up change
  (branch `rshahbazyan/litellm-model-version-header-fallback`) reads it from the
  `x-litellm-version` header of a one-token chat-completion probe when the health route answers 403
  (the `/v1/models` route carries no such header on real deployments), keeping the minimum-version
  check; until it lands, a scoped key needs the gateway operators to open the health route to it. `openai_model` is not a workaround for the Claude and Gemini routes:
  it does not repair their reasoning items, so every such rollout fails validation.

## Quickstart

Commands that were run from the repository root with `hf_token`, `policy_base_url`,
`policy_api_key` and `policy_model_name` in `env.yaml`:

```bash
gym eval prepare --benchmark hle_diamond        # Wrote 1000 problems to benchmarks/hle_diamond/data/hle_diamond_benchmark.jsonl (272 with an image)
gym eval prepare --benchmark hle_diamond/text   # Wrote 728 problems to benchmarks/hle_diamond/data/hle_diamond_benchmark_text.jsonl (skipped 272 image questions)
gym list benchmarks | grep hle_diamond          # hle_diamond, hle_diamond/text, hle_diamond/opencode, hle_diamond/pi
```

`cais/hle-diamond` is gated: request access on the dataset page and put the token in `env.yaml` as
`hf_token`. When that key is absent, `prepare.py` passes `token=None` and `datasets` falls back to the
cached `huggingface-cli login` / `HF_TOKEN` credentials (verified: the gated parquet resolves with
`token=None` and fails with `401` for `token=False`).

**Model server.** The commands below pass `--model-type litellm_model`, the server for LiteLLM-proxy
gateways such as the one these runs went through: the Claude and Gemini routes return reasoning
items whose `summary` is null or absent and whose parts are `output_text`, which `NeMoGymResponse`
rejects, and `litellm_model` repairs exactly that shape before validation (`_normalize_to_response`
in `responses_api_models/litellm_model/app.py` sets a missing or null `summary` to `[]` and renames
`output_text` parts of reasoning items to `reasoning_text`; it extends `openai_model`, so the retry
layer and the server knobs are shared, and its config file wires `openai_base_url`, `openai_api_key`
and `openai_model` from the same `policy_*` keys of `env.yaml`). Use `--model-type openai_model` for a native
OpenAI-compatible server (OpenAI itself, a vLLM Responses endpoint), which emits well-formed items
and needs no repair; it has none. `litellm_model` also checks the proxy's version at start through
`/health/readiness/details` (Known limitations). The runs recorded in this README were made with
`openai_model` plus a normaliser equivalent to `litellm_model`'s repair, which an earlier revision
of this branch carried as `normalize_reasoning_items` and which was dropped in favour of the
existing `litellm_model` path; both change the same two fields and nothing else, so the numbers
stand.

### No-tools runs

**Not run as written**: the pilot and the full-set collection used the two-command shape of Running
the full set, where `gym env start --config` composes the judge override. This one-process form is
kept as the shortest path to a `--benchmark` run; every part of it is load-bearing:

- `--split benchmark` is required: the dataset is `type: benchmark` and the CLI has no default split
  (without it: `gym eval run: error: missing required configuration: split`).
- `'+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]'` makes the dedicated judge grade; without it
  the policy grades itself and the number must be labelled self-judged. It has to be the raw Hydra
  token, which the CLI appends after the `--benchmark` and `--model-type` configs so the override
  merges last and wins; `--config <file>` merges before the preset and loses (see Scoring >
  Dedicated judge).
- `++responses_create_params.reasoning.effort=high` is the one sampling setting; no `--temperature`,
  `--top-p` or `--max-output-tokens` (see Prompting > Reasoning effort for how to label the run).
- `--num-repeats 1`; `--limit 20` for the pilot, then the full set.

```bash
gym eval run --benchmark hle_diamond --split benchmark --model-type litellm_model \
    '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]' \
    --num-repeats 1 --limit 20 \
    ++policy_model_name=<gateway model id> \
    ++responses_create_params.reasoning.effort=high \
    --output results/hle_diamond_pilot.jsonl
```

Use `--benchmark hle_diamond/text` with the same flags for a text-only policy model; its 728/1000
coverage is not comparable with the published table.

Before treating a run as comparable, check on the pilot rows:

- The effort label, by model family (Prompting > Reasoning effort): GPT-6 Astra, GPT-6.1 Sol and
  Gemini 4 Argon are "reasoning high" (for the GPT models also confirm the echoed
  `response.reasoning.effort == "high"`); Claude Opus 5.5, Claude Sonnet 5.5 and Claude Opus 5 are
  "effort not controllable via this gateway (adaptive default)". `reasoning_tokens > 0` is not
  evidence either way.
- The judge is the dedicated one: `judge_evaluations[*].response.model` echoes the requested alias
  (`azure/openai/o3-mini` by default; the served snapshot is not verifiable, so record the alias and
  the run date), and `judgement_parsing_issue_rate` is near zero.
- Coverage: rows evaluated / 1000, or `coverage/imputed` with the upstream denominator (see Scoring).

### Sandboxed presets on a local Docker host

**Run on 2026-10-16 for `hle_diamond/opencode`** with this recipe plus a third composed file that
uncommented `network: host` and pointed the dataset at 5 text rows (GPT-6 Astra, judge o3-mini,
exit 0, 187 s wall; 5/5 rows scored, 3/5 made real tool calls, all 15 OpenCode requests carried
`reasoning_effort: high`, and `urlopen("https://example.com")` from inside the image returned HTTP
200, hence the label below). The bridge-networking default and the 20-row form are **not run as
written**: on the verifying workstation bridge networking timed out (ufw drops docker0 to host
traffic and the hostname resolves to `127.0.1.1`). The committed override keeps bridge networking
as the default because it exposes the least, but that form is unverified; if the first rollout
hangs on the model call, uncomment `network: host`. Every run through this recipe is **comparison
infrastructure only: egress is unrestricted**, and its results must carry the label "local Docker;
egress not restricted".

```bash
# The presets always compose nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml, whose
# ${oc.env:...} values must resolve even though the Docker provider is used: placeholders suffice.
export OPENSANDBOX_API_KEY=unused-local-docker
export OPENSANDBOX_DOMAIN=unused.invalid
export OPENCODE_SANDBOX_IMAGE=gym-opencode-science:local      # PI_SANDBOX_IMAGE=gym-pi-science:local for the pi presets
export OPENCODE_ARTIFACTS_DIR=/abs/path/oc-artifacts          # PI_ARTIFACTS_DIR for pi: the harness's native transcripts
mkdir -p /abs/path/model-calls

gym eval run --benchmark hle_diamond/opencode --split benchmark --model-type litellm_model \
    '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml,benchmarks/hle_diamond/sandbox_docker_local.yaml]' \
    --num-repeats 1 --limit 20 \
    ++policy_model_name=<gateway model id> \
    ++responses_create_params.reasoning.effort=high \
    ++observability_enabled=true ++model_call_capture_dir=/abs/path/model-calls \
    --output results/hle_diamond_opencode_pilot.jsonl
```

- Both overrides travel in one raw `+config_paths` token so they merge after the preset (see
  Scoring > Dedicated judge); with `--config` the search presets' `network_access: model_and_tools`
  would win and fail closed on the Docker provider, and the model would be told internet is disabled.
- `sandbox_docker_local.yaml` defines a second provider block, `sandbox_docker` (the content of
  `nemo_gym/sandbox/providers/docker/configs/docker.yaml`), because the composed `sandbox:` block
  cannot be removed or re-typed by an override, and repoints `hle_diamond_opencode_agent` and
  `hle_diamond_pi_agent` at it with `network_access: inherit` (restricted policies fail closed on
  any provider but OpenSandbox). Only the agent the preset defines is started; the other block stays
  a partial, inert entry. It also replaces the presets' environment note, which says internet is
  disabled, with the truth.
- Uncomment `network: host` (the verified form) in the override on hosts whose firewall drops
  bridge (docker0) to host traffic or whose hostname resolves to a loopback alias such as
  `127.0.1.1` under the presets' `use_absolute_ip: true`: the container must reach the Gym model
  server at the advertised address.
  Host networking also lets the container reach every service on the host; where the firewall allows
  it, prefer bridge networking with `++use_absolute_ip=false ++default_host=<routable host IP>`.
- The Docker provider ignores `sandbox_config.ttl_s` when an entrypoint is set, and the presets set
  `entrypoint: [sleep, infinity]`, so container lifetime relies on the agent's cleanup. Check
  `docker ps -a --filter label=nemo-gym.sandbox` after every run.
- The first run builds per-server `uv` virtual environments and needs PyPI access from the host.
- The search presets additionally need `TAVILY_API_KEY`.

### Reading sandboxed results

- `response.status` is `null` on sandboxed rows (the adapter synthesises the response) unless the
  output limit was hit, when the adapter sets `incomplete` with
  `incomplete_details.reason = max_output_tokens`. Read success or failure from
  `opencode_finished`, `opencode_failed`, `opencode_exit_code` and
  `opencode_export_found` (OpenCode) or `finished_naturally`, `pi_failed`, `pi_exit_code` and
  `pi_error_type` (Pi); the aggregate metrics carry `mean/opencode_failed` and `mean/opencode_finished`.
- The aggregate of a sandboxed run carries neither `slices/partition/...` nor
  `judgement_parsing_issue_rate`, although every rollout row carries `partition` and
  `judge_evaluations`: the collector posts `/aggregate_metrics` to the agent server, and the
  OpenCode and Pi adapters inherit `BaseResponsesAPIAgent.aggregate_metrics`, which never calls
  the judge resources server's `compute_metrics` where both live (observed on the 5-row
  `hle_diamond/opencode` pilot: no `slices/*` and no parsing key in its aggregate). Compute them
  post hoc from the rollouts with the slice hook's rule (rewards grouped by `partition` with their
  counts; rows with `judgement_parsing_issues`) until the adapters proxy aggregation to their
  resources server (Known limitations).
- `ng_trajectory`, `ng_perf` and `ng_model_call_capture` are attached only with
  `++observability_enabled=true ++model_call_capture_dir=<absolute dir>`; without them every
  trajectory health check reports "unobserved". Use a distinct capture directory per run.
- Confirm tool use: at least one pilot row must show `ng_perf.num_tool_calls > 0` (or a `tool` part
  in the harness transcript under the artifacts directory). Otherwise the sandbox was never
  exercised and the run is a no-tools run through a different prompt path, not a with-tools result.
- Confirm effort: the captured OpenCode request body carries `reasoning_effort: high` (all 15
  requests of the 5-row `hle_diamond/opencode` pilot capture). Pi's request will not carry it, by
  inspection of the adapter's `compat.supportsReasoningEffort: false` and Pi 0.85.1's
  `openai-completions` path rather than from a capture (no Pi run has been captured yet); confirm
  on the first Pi capture (Prompting > Reasoning effort).
- Token and latency figures are not comparable with the announcement's mean output tokens: the
  harness's system prompt and tool schema dominate the input, and the call is chat-completions.

## Running the full set

The comparable run is split into servers and collection, so that one process holds the model servers
for the whole run and the judge override composes with `--config`: under `gym env start` every
`--config` lands in `config_paths` in command-line order, so `judge_openai.yaml` merges after
`config.yaml` and wins, whereas the `--benchmark ... --split benchmark` form of the Quickstart needs
the raw `+config_paths=[...]` token (Scoring > Dedicated judge). Under `--no-serve` the dataset is
named by `--input`, `--split` is not read, and the servers must already be up. Run as written on
2026-10-16 for the Claude Opus 5.5 and Gemini 4 Argon 20-row pilot collections (`--input` pointing
at the sample, `--concurrency 20`); the GPT-6 Astra pilot ran the same shape without
`++route_failures_to_sidecar=true`, which is why one gateway timeout ended it and why the flag is
part of the shape. The GPT-6 Astra full-set collection ran to completion with exactly this shape
(946 of 1000 rows scored, 54 routed to the sidecar) from Gym commit `6421f1457` of the
pre-rewrite branch (`c99ffe134` in the current history), before `judge_labels_case_insensitive`
existed and with `openai_model` as the policy server (the GPT route needs no repair): servers load
code and configuration once at start, so that run's verdicts are byte-exact (Reporting a full run):

```bash
RUN=/abs/path/hle-diamond-runs/<model-slug>/<UTC timestamp>   # outside the repository
mkdir -p "$RUN/model-calls"

# 1. Servers: policy, hle_diamond judge resources server, agent, dedicated judge. Blocks until stopped.
gym env start \
    --config responses_api_models/litellm_model/configs/litellm_model.yaml \
    --config benchmarks/hle_diamond/config.yaml \
    --config benchmarks/hle_diamond/judge_openai.yaml \
    ++policy_model_name=<gateway model id> \
    ++observability_enabled=true ++model_call_capture_dir="$RUN/model-calls"

# 2. Collection, from a second shell. No --split, no temperature, top-p or output cap.
gym eval run --no-serve \
    --agent hle_diamond_equivalence_llm_judge_simple_agent \
    --input benchmarks/hle_diamond/data/hle_diamond_benchmark.jsonl \
    --output "$RUN/rollouts.jsonl" \
    --num-repeats 1 --concurrency 32 \
    ++responses_create_params.reasoning.effort=high \
    ++route_failures_to_sidecar=true \
    ++observability_enabled=true ++model_call_capture_dir="$RUN/model-calls"

# 3. Stop: SIGINT the `gym env start` process; its KeyboardInterrupt path shuts every server down.
```

For the 728-row text variant (not run as written) pass `text.yaml` in place of `config.yaml`,
`--agent hle_diamond_text_equivalence_llm_judge_simple_agent` and the text file as `--input`.

Claude and Gemini models through LiteLLM-style gateways return reasoning items whose `summary` is
null or absent (the pilot's wire error was `output.0.reasoning.summary Field required`; the `openai`
client's dump of the same bodies shows `summary: null`) with `output_text` parts, which
`NeMoGymResponse` rejects; `litellm_model` repairs exactly that shape before validation
(Quickstart > Model server), which is why the server command above loads `litellm_model.yaml`.
Without the repair every such rollout that reached Gym was lost to a validation error (pilot:
Claude 20/20 and Gemini 16/20 rows rejected; Gemini's other 4 were gateway timeouts, Known
limitations). The Claude Opus 5.5, Gemini 4 Argon and Nemotron 3.5 Super VL collections recorded
here ran `openai_model` with the equivalent normaliser of an earlier branch revision, so their
numbers stand; a new run of those families needs `litellm_model`, and so a key that passes its
version check (Known limitations).

### Reporting a full run

Every full run passes `++route_failures_to_sidecar=true`, so a row whose model call fails (the
gateway timeout, an HTTP 500) goes to the failure sidecar instead of ending the run. Report, next to
every number:

- Coverage: scored rows / 1000, and lost rows by failure kind from the sidecar (the 408
  `litellm.Timeout` rows are the gateway's 360 s per-attempt limit).
- Two accuracies: `mean/reward` over the scored rows, and accuracy with the lost rows counted as
  wrong, which is upstream's denominator (all 1000 questions). Lost rows are the longest generations,
  so the over-scored accuracy is biased upward and the mean output tokens downward; say so beside
  each figure.
- The judge alias and run date (Scoring > Dedicated judge), the effort label by model family
  (Prompting > Reasoning effort), and the partition slices with their `question_count` (from the
  aggregate on the no-tools path; computed post hoc from the rollouts on the sandboxed presets, see
  Reading sandboxed results).
- The Gym commit the servers loaded, because servers read code and configuration once at start: it
  decides whether `judge_labels_case_insensitive` (verdicts matched case-insensitively, otherwise
  byte-exact) was in effect; name the policy model server next to it (`litellm_model`, or
  `openai_model` with the earlier normaliser). The GPT-6 Astra full run launched on 2026-10-16 from
  `6421f1457` (pre-rewrite id; `c99ffe134` now) matched verdicts byte-exact and ran `openai_model`
  without any normaliser.
- When rows lost to the gateway timeout are rescued outside the collector (the identical request
  from `rollouts_materialized_inputs.jsonl` sent with `background: true` and polled, then scored
  through the same `/verify` endpoint and merged before `gym eval aggregate`), report a third view
  next to the two above: accuracy over all 1000 with the rescued rows included, stating that a
  rescued row differs only in transport; a row still `incomplete` or `failed` after two background
  attempts stays lost and is listed.

## Sandboxed agents (with tools)

`hle_diamond/opencode` and `hle_diamond/pi` give the model sandboxed Python tools without search;
`hle_diamond/opencode_search` and `hle_diamond/pi_search` add Tavily search and page tools. All four
run the 728 text rows with the verbatim HLE instructions in a single user message and one repeat;
keep collection repeats at one. The published with-tools column was produced under upstream's
`docs/evaluation-with-tools.md` protocol, from which these presets depart in concrete ways, so they
support a **comparison, not a reproduction**, of that column:

- Harness: OpenCode or Pi inside Gym's sandbox instead of each vendor's own agent (Claude Code,
  Codex CLI, Gemini CLI, Muse Code), with the harness's own system prompt ahead of the instruction.
- Sandbox: the `gym-*-science` images carry Python 3.13 with SageMath and Lean/Mathlib preinstalled,
  and the environment note tells the model so; upstream requires Python 3.11.16 with pinned
  requirements and forbids specialist solvers, chess engines, OCR and other task libraries.
- Web tools: Tavily search and page tools with up to 200,000 cached page characters and 2,000-word
  scrolling, instead of exactly `web_search` (at most 20 results) and `web_fetch` (Markdown capped
  at 60,000 characters, no media).
- Leakage policy: two of upstream's seven blocklist families are enforced in full and the
  search-engine result-page family approximately; the search provider pre-filters at most 150 of the
  220 excluded domains and the rest are enforced locally (see Search exclusions).
- Rows: 728/1000 text rows; image questions cannot be sent through these adapters.
- Reasoning effort: requested through the harness's own knob; transmitted by OpenCode, not by Pi
  (see Prompting > Reasoning effort).

State these departures next to every number collected with these presets.

**First run of the search presets.** A 10-row `hle_diamond/opencode_search` pilot on 2026-10-16
(GPT-6 Astra, judge o3-mini, local Docker with host networking, egress unrestricted) was the first
execution of either search preset. With the code of that day every `web_search` failed on Tavily's
150-domain limit (Search exclusions > Provider-side cap on excluded domains), so its 9 scored rows
(8 correct) measure a search tool that returned nothing plus unrestricted bash egress, not a
with-tools result; the same run exposed the search-engine result-page gap that the SERP rules now
close. No search-preset number from the committed code exists yet. Before reading any statistic from
the next pilot, confirm in the Tavily server log that `web_search` returns results: one cap warning
at start and no `Tavily HTTP 400`.

See the shared
[OpenCode and Pi setup guide](../../responses_api_agents/opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)
for images, credentials, network policies, token limits and failure handling. Gates specific to
these presets:

- Sandbox images: `OPENCODE_SANDBOX_IMAGE` and `PI_SANDBOX_IMAGE` must point at images with the
  scientific stack described in the guide.
- Search credentials: the search presets read `TAVILY_API_KEY`; without it the Tavily resource does
  not start.
- Egress: the presets request `network_access: model_only` (code-only) or `model_and_tools`
  (search). Only the OpenSandbox provider enforces those policies; the Docker provider has no
  network policy, so a local Docker run goes through `sandbox_docker_local.yaml`, which switches to
  `network_access: inherit`, and its egress is **not** blocked (verified: an HTTPS request to a
  public site succeeds inside such a sandbox). Label any such run accordingly. With open egress the
  exclusion policy binds only the Tavily tools: in the first `opencode_search` pilot the agent, once
  `web_search` failed, queried `google.com/search`, `bing.com/search`, `html.duckduckgo.com` and
  `search.yahoo.com/search` from bash with `python -c "import requests ..."`, downloaded PDFs and
  images from arbitrary hosts, and used `tavily_scroll_page` on DuckDuckGo result pages as a search
  engine (one row: 198 `scroll_page` calls, 154 of them on result pages). None of those fetches
  passes through the policy. A reproduction-grade with-tools run therefore needs the OpenSandbox
  provider with `network_access: model_and_tools`, under which the sandbox reaches only the model and
  tool servers, so every fetch goes through the Tavily resource and its policy.
- Answer exposure: the search presets supply the HLE-Diamond
  [exclusion list](search_exclusions.json), derived from the upstream `docs/blocklist.json` domains
  and URL prefixes by `build_search_exclusions.py` (see Search exclusions below for what is and is
  not enforced). Deployments requiring an additional policy must supply their combined policy file
  through the Tavily resource's `exclude_domains_file_path`; keep private policies outside the public
  repository. The list reduces answer exposure but cannot guarantee that every answer-bearing page is
  excluded.

### Search exclusions

`search_exclusions.json` is generated, never hand-edited: `build_search_exclusions.py` maps the
upstream `docs/blocklist.json` at the revision pinned in the script (`UPSTREAM_COMMIT`, with the
snapshot's SHA-256 in `UPSTREAM_BLOCKLIST_SHA256`) onto Gym's format and unions it with
`benchmarks/hle/search_exclusions.json`, so the Diamond list is at least as strict as Gym's HLE list.
Gym's Tavily resource (`URLExclusionPolicy`) expresses two rule types, so two of upstream's seven rule
families are enforced in full, one is approximated and four are not enforced; the file holds 220
`domain` and 118 `url_substring` rules:

| Upstream family | Rules | Gym rule | Enforced |
| --- | --- | --- | --- |
| `domains` | 220 | `domain`: exact hostname or any subdomain | yes |
| `url_prefixes` | 66 | `url_substring`: case-insensitive substring of the URL, scheme and leading `www.` removed; a prefix becomes a substring, so at least as strict | yes |
| `serp_url_patterns` | 12 | 14 `url_substring` rules naming each engine's host and search path (`SERP_PATTERN_SUBSTRINGS`) | approximately (below) |
| `url_patterns` | 22 | none | no |
| `patterns_case_sensitive` | 4 | none | no |
| `answer_title_patterns` | 5 | none | no |
| `answer_title_allow_hosts` | 7 | none | no |

The unrepresented families are recorded by name and count under `unrepresented_upstream_rules` and
the approximation under `approximated_upstream_rules`, keys the parser ignores. The consequences make
the search presets a **weaker leakage policy than upstream's protocol**
(`docs/evaluation-with-tools.md`), and a with-tools number from them must be labelled as such:

- Only hostnames and URL substrings are inspected. No search query is refused (upstream refuses
  queries with a blocked `site:` domain or a long verbatim span of the question), and no result title,
  snippet or fetched Markdown is matched against the regex families (upstream rejects pages whose
  text matches `url_patterns` or `patterns_case_sensitive`, e.g. "Humanity's Last Exam" in several
  languages, `cais/hle`, specific arXiv ids and answer-key phrasing).
- Answer-key style titles (`answer_title_patterns`) are not filtered.
- The parser percent-decodes a URL and re-checks it, so a `url_substring` embedded in a proxy or
  archive link is caught; a `domain` rule applies to the outer hostname only, so an archive or
  proxy page that wraps a blocked host in its path passes unless a substring rule matches.
- Upstream blocks GitHub wholesale (`github.com`, `githubusercontent.com`, `gist.github.com`,
  `api.github.com`, `github.io`), and so does this list: the agent cannot fetch any GitHub page.

#### Search-engine result pages

Upstream's `serp_url_patterns` are twelve regexes anchored at the start of the scheme-less URL
(`^([a-z0-9-]+\.)*bing\.com/search`, ...). The generator approximates each by the `url_substring`
rules in `SERP_PATTERN_SUBSTRINGS`, keyed by the exact upstream regex so that a changed or new pattern
is reported and dropped rather than approximated silently: `google.com/search`,
`scholar.google.com/scholar`, `bing.com/search`, `duckduckgo.com/html`, `duckduckgo.com/lite`,
`duckduckgo.com/?q=`, `html.duckduckgo.com`, `search.yahoo.com/search`, `baidu.com/s?`, `yandex.`,
`startpage.com`, `search.brave.com/search`, `ecosia.org/search`, `mojeek.com/search`. With them
`find_in_page` and `scroll_page` refuse a result page (the pilot's "scroll DuckDuckGo as a search
engine" pattern) and a result page returned by `web_search` is dropped. Remaining differences:

- Narrower than upstream: Google and Google Scholar are blocked on `.com` only, where upstream
  matches every Google top-level domain (`google.co.uk/search`, `scholar.google.de/scholar?`, ...).
- Wider than upstream: `yandex.` blocks every Yandex host and path and any URL containing that text
  (a page titled "Yandex.Taxi", say), and `startpage.com` blocks the whole host, because a substring
  cannot express "any path ending in `/search`"; `scholar.google.com/scholar` also covers the longer
  Scholar paths upstream leaves alone (`/scholar_lookup`, `/scholar_case`). Like every
  `url_substring` rule, each of these matches anywhere in the URL, so `mybing.com/search` and a
  result-page URL embedded in another URL (`example.com/?u=google.com/search`) are blocked too.
  Other non-result pages of the engines are left alone (`google.com/maps`, `duckduckgo.com/about`).
- Not covered: result pages of engines upstream does not list, and anything fetched outside the
  Tavily tools (the egress gate above).

#### Provider-side cap on excluded domains

Tavily's `/search` accepts at most 150 `exclude_domains` and answers HTTP 400 ("Max 150 domains are
allowed") beyond that; this list has 220. The first `opencode_search` pilot (2026-10-16) sent every
domain, so all 14 `web_search` calls failed with `Tavily HTTP 400 after 1 attempts` (11 in the 9
exported rows, 3 in the stopped row's model-call capture) while `find_in_page`/`scroll_page` (Tavily
`/extract`, which takes no domain list) worked; Gym's HLE list has 9 domains and never reached the
limit. The Tavily resource now sends a capped subset:

- `tavily_exclude_domains_limit` (default 150, also the largest value the config accepts, since a
  longer list fails every search) bounds the provider-side list; `0` or `null` sends no
  `exclude_domains` at all. Whatever it is set to, the full `domain` and `url_substring` policy is
  enforced locally on every URL the agent requests and on every URL the provider returns, so the cap
  changes which results Tavily withholds before ranking, never which pages the agent can see.
- The subset is deterministic: the domains matching `tavily_exclude_domains_priority` first (default
  `huggingface.co`, `hf.co`, `lastexam.ai`, `agi.safe.ai`, `safe.ai`, `scale.com`, `*.scale.com`,
  `github.com`, `githubusercontent.com`, `archive.*`, in that order; matched case-insensitively
  against the normalised hostnames), then the rest in file order, cut at the limit. On this list that
  sends the 15 priority hosts and the next 135 in file order. Every priority host already sits inside
  the first 150 entries, so here the priority block only reorders: the 70 not sent are upstream's
  last 70 `domains` entries, namely 38 unlabelled hostnames (`mistraltms.it` ... `autoklarheit.de`;
  upstream's list carries no annotation or reason for any entry), the social networks
  (`instagram.com`, `facebook.com`, `x.com`, `reddit.com`, `tiktok.com`, `linkedin.com`, ...),
  `quora.com`, the StackExchange family, `autobenchaudit.com` and the jsDelivr CDN. A result from one
  of those is dropped locally, which costs the query a result slot but never shows the page.
- The server logs one warning per start naming the number of domains not sent (70 here).

Two `url_substring` rules inherited from `benchmarks/hle/search_exclusions.json`
(`openreview.net/pdf/<40 hex characters>.pdf`) carry OpenReview document ids. They are not secrets,
they pass the repository's secrets scan unaltered, and they are kept rather than widened to the
`openreview.net/pdf/` prefix so the list does not block more than upstream does.

Refresh procedure when upstream changes its blocklist: update `UPSTREAM_COMMIT` and
`UPSTREAM_BLOCKLIST_SHA256` together in `build_search_exclusions.py`, run
`python benchmarks/hle_diamond/build_search_exclusions.py --download`, review the diff of
`search_exclusions.json` (a rule family new to the generator is reported as a warning; a changed or
new `serp_url_patterns` regex is reported as dropped until `SERP_PATTERN_SUBSTRINGS` maps it), and run
`pytest tests/unit_tests/test_hle_diamond_search_exclusions.py`.

## Tests

```bash
pytest tests/unit_tests/test_hle_diamond.py tests/unit_tests/test_hle_diamond_search_exclusions.py \
    resources_servers/equivalence_llm_judge/tests/test_app.py \
    resources_servers/tavily_search/tests/test_search_policy.py
```

## Licensing

Code: Apache 2.0.

`cais/hle-diamond` data: MIT, with the dataset card's request not to publicly share, re-upload or
redistribute it; the data is downloaded at run time and never committed. The prompt files in
`prompts/` reproduce the HLE authors' `SYSTEM_PROMPT` from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) (MIT) verbatim; the judge prompt
is reused from `benchmarks/hle/prompts/judge.txt`, adapted from the same harness.
