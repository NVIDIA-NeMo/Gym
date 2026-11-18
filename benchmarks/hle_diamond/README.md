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
| `hle_diamond/opencode`, `hle_diamond/pi` | 728 | `prompts/opencode.yaml` | sandboxed code tools, no search (`opencode_agent` / `pi_agent` agent sessions) |
| `hle_diamond/opencode_search`, `hle_diamond/pi_search` | 728 | `prompts/opencode.yaml` | sandboxed code tools plus Tavily search (agent sessions) |
| `hle_diamond/opencode_reference` | 728 | `prompts/opencode.yaml` | `opencode_search` on the reference sandbox image (upstream's Python 3.11.16 environment) under upstream's egress policy; needs the OpenSandbox provider (see Reference preset) |
| `hle_diamond/opencode_full`, `hle_diamond/opencode_search_full` | 1000 (272 with an image) | `prompts/opencode.yaml`, baked at prepare time by `prepare_opencode_legacy.py` | deprecated: the OpenCode presets over the full set, images attached through `opencode run -f`, on `opencode_sandboxed_agent`; needs a vision-capable policy model |
| `hle_diamond/opencode_hle_tools` | 728 | `prompts/opencode.yaml` | upstream's with-tools protocol on every point Gym controls: `opencode_reference`'s agent, reference image and enforced egress with the Tavily tools replaced by upstream's `web_search` and `web_fetch` behind its blocklist (`resources_servers/hle_web_tools`, Tavily backend); needs the OpenSandbox provider; see Upstream web tools for the fidelity table |
| `hle_diamond/opencode_hle_tools_full` | 1000 (272 with an image) | `prompts/opencode.yaml`, baked at prepare time by `prepare_opencode_legacy.py` | `opencode_hle_tools` over the full set on the same agent session: the comparable denominator for the published with-tools column; image rows reach the model through OpenCode's file attachment (Image questions through OpenCode); needs a vision-capable policy model and the OpenSandbox provider |
| `hle_diamond/opencode_legacy_agent`, `hle_diamond/opencode_search_legacy_agent`, `hle_diamond/pi_legacy_agent`, `hle_diamond/pi_search_legacy_agent`, `hle_diamond/opencode_reference_legacy_agent` | 728 | `prompts/opencode.yaml` | deprecated: the text presets above as they ran before agent sessions, on `opencode_sandboxed_agent` / `pi_sandboxed_agent` |

Two composable overrides live next to the presets and are not benchmarks (no dataset, not listed by
`gym list benchmarks`): [`judge_openai.yaml`](judge_openai.yaml) (dedicated judge, see Scoring) and
[`sandbox_docker_local.yaml`](sandbox_docker_local.yaml) (sandboxed presets on a local Docker host,
see Quickstart).

**Why the default inverts `hle`'s convention.** `benchmarks/hle` defaults to the text subset and
exposes images as the `hle/config_vision` variant, because HLE's headline figures are text-only.
HLE-Diamond's published numbers are over all 1000 questions, images included, so here the full set is
the default and the text subset is the variant. A `hle_diamond/text` run covers 728/1000 questions
(347 reasoning, 381 knowledge) and is not comparable with the published table; report that coverage.

**Text and full sandboxed presets.** `opencode`, `opencode_search`, `opencode_hle_tools`, `pi` and
`pi_search` run the 728 text rows, as `benchmarks/hle`'s presets do, so with-tools numbers from them
cover 728/1000 questions. They run `opencode_agent` and `pi_agent` in agent sessions (Sandboxed agents
(with tools) > Agent sessions) with the shared settings of `benchmarks/hle`'s presets.
`opencode_hle_tools_full` runs `opencode_hle_tools` over all 1000 rows on the same agent session: the
`opencode_agent` session accepts the user turn's `input_image` parts, stages each decoded image into the
sandbox and attaches it with `opencode run --file`, so the image reaches the model call as an `image_url`
part (Sandboxed agents (with tools) > Image questions through OpenCode). `opencode_full` and
`opencode_search_full` run all 1000 the same way on the deprecated `opencode_sandboxed_agent`. The
deprecated Pi adapter accepts `input_text` parts only, so no Pi preset covers the image rows. The search
presets and `opencode_hle_tools` are selectable with `--benchmark` but, exactly like `hle/opencode_search`
and `hle/pi_search`, do not appear in `gym list benchmarks` because they extend another preset without
redeclaring the dataset; `hle_diamond/opencode_reference` restates it (its header says why) and is listed,
and `hle_diamond/opencode_hle_tools_full` declares its own dataset and is listed too.

**Deprecated relay presets.** A preset named `*_legacy_agent` runs a deprecated sandboxed agent
(`opencode_sandboxed_agent` or `pi_sandboxed_agent`) behind the `legacy_agent` relay and goes away
with the relay. `opencode_legacy_agent`, `opencode_search_legacy_agent`, `pi_legacy_agent`,
`pi_search_legacy_agent` and `opencode_reference_legacy_agent` are the text presets as they ran
before agent sessions, kept under the agent names of the presets that replace them; the two harnesses
are configured differently, so do not compare their scores as one measurement (Agent sessions lists
what changed). `opencode_full` and `opencode_search_full` keep their names and run the same deprecated
`opencode_sandboxed_agent` behind the relay over the full set; the session preset over the full set is
`opencode_hle_tools_full`, and the two relay presets go away with the relay too.

Dropped at prepare time: `rationale`, `author_name`, `canary`, `image_preview`, `rationale_image`.
The `image` data URI appears only inside the materialized `input_image` block of the default set and
of the OpenCode full set (`prepare_opencode_legacy.py`).

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
  carries the single braces, and `resources_servers/equivalence_llm_judge/tests/test_hle_diamond.py` asserts that the rendered
  string and the prepared rows equal the upstream text. The default set bakes it into
  `responses_create_params.input` at prepare time (`nemo_gym.data.vision_input.build_image_input`); the
  text variant applies the same file at rollout time.
- `prompts/opencode.yaml`: the same verbatim instruction folded after the question into the single
  user message the OpenCode and Pi adapters accept; the harnesses' native system prompts stay intact.
  The text presets apply it at rollout time; `prepare_opencode_legacy.py` bakes it into the full set for
  `opencode_hle_tools_full`, `opencode_full` and `opencode_search_full` (the default set's file cannot
  serve them: it carries the instruction as a system message, which the relay adapter does not forward
  and the session agent folds into OpenCode's instructions rather than the user turn). Two departures from
  upstream's placement are inherent to these harnesses and belong next to every with-tools number:
  the instruction arrives as user text rather than as the system message, and the harness's own
  system prompt (several thousand tokens for OpenCode) precedes it.
- Judge: `benchmarks/hle/prompts/judge.txt`, the HLE judge prompt adapted for the
  `equivalence_llm_judge` label parser, is reused unchanged. Upstream's `run_judge_results.py` parses
  a structured object (`extracted_final_answer, reasoning, correct: yes|no, confidence, strict`);
  Gym's file renames the fields to `Reasoning:` / `Judgement:` / `Confidence:` and the verdict is the
  `Judgement: yes|no` string found in free text, with unparseable judge output tracked by
  `judgement_parsing_issue_rate`. The grading criteria are the same.

Image questions: upstream attaches the image as `image_url` with no `detail` setting (the provider
default); Gym's shared builder requests `detail: high`, which changes image tiling and token counts
on the 272 image rows for OpenAI-family models on the default set. Making `detail` configurable would
be a change to `nemo_gym.data.vision_input`, not to this benchmark. On `opencode_hle_tools_full`,
`opencode_full` and `opencode_search_full` the `detail` field never reaches the model: the session agent
(`nemo_gym.core.media.split_text_and_media` reads only the part's data URI) or the relay adapter writes
the decoded image into the sandbox and OpenCode builds its own chat-completions request, whose `image_url` part
carries only the URL, as upstream's does (Sandboxed agents (with tools) > Image questions through
OpenCode).

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
showed no `reasoning_effort`, so the gateway default, medium, applied). How each preset requests
high effort:

- The agent-session presets (`opencode`, `opencode_search`, `opencode_hle_tools`, `opencode_hle_tools_full`,
  `opencode_reference`, `pi`, `pi_search`): through the model server, not the preset. Neither session agent can put an effort on its harness's
  calls (a session's `opencode_config` takes no model options, and `pi_agent`'s generated provider
  declares no `reasoning_effort` support, so Pi's `--thinking` level never leaves the sandbox), and
  both refuse a request `reasoning` field with 422, so `++responses_create_params.reasoning.effort` must
  not be passed to them. Add the effort to every chat-completions body the policy model server sends
  with its `extra_body`, for example
  `++policy_model.responses_api_models.vllm_model.extra_body.reasoning_effort=high` or
  `++policy_model.responses_api_models.litellm_model.extra_body.reasoning_effort=high` (the key follows
  `--model-type`). A value the harness sends itself would win over `extra_body`; neither sends one.
  The override covers every call through that model server, so compose `judge_openai.yaml` (as the
  documented commands do): with the default judge, which is the policy model, the judge calls would
  run at high effort too.
- `opencode_full.yaml` (inherited by `opencode_search_full.yaml`), on
  `opencode_sandboxed_agent`, and the text `*_legacy_agent` presets on the same agent:
  `opencode_config.provider.nemo_gym.models.dummy_model.options.reasoningEffort: high`, which
  OpenCode 1.17.11 maps to `reasoning_effort: high` in its chat-completions body (verified in a
  capture). Override per run with
  `++<agent>.responses_api_agents.opencode_sandboxed_agent.opencode_config.provider.nemo_gym.models.dummy_model.options.reasoningEffort=<level>`,
  where `<agent>` is the preset's agent instance: `hle_diamond_opencode_full_agent` on the full
  presets, `hle_diamond_opencode_reference_agent` on `opencode_reference_legacy_agent.yaml`, and
  `hle_diamond_opencode_agent` on `opencode_legacy_agent.yaml` and `opencode_search_legacy_agent.yaml`.
  Under any other name the override adds an agent block that nothing uses, and the effort stays
  `high`.
- `pi_legacy_agent.yaml` (inherited by `pi_search_legacy_agent.yaml`), on the deprecated
  `pi_sandboxed_agent`: `thinking: high`, which that agent passes as `--thinking high`. Pi's
  `openai-completions` path emits `reasoning_effort` only when its provider declares support, and the
  generated provider does not, so the level is not transmitted.

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
| `responses_create_params` | default and OpenCode full sets only: the materialized `input` messages; image rows carry an `input_image` block with `detail: high` (upstream sends no `detail`; see Prompting). The default set has a system and a user message, the OpenCode full set a single user message |

`prepare.py` loads the whole split, validates it (1000 rows, 500 + 500 partitions, 272 image rows,
1000 unique ids) and only then writes, so an incomplete download never leaves a file behind.
`--limit N` (positive integers only; rejected at parse time otherwise) is the one supported way to
prepare a subset, and it says loudly that the completeness check was skipped; the limit applies after
the text filter and the summary line says `limited to N`. `prepare_text.py` is a thin wrapper for
`prepare(text_only=True)`; its script form accepts `--limit N` and nothing else, and
`gym eval prepare --benchmark hle_diamond/text ++prepare_script_args.limit=N` reaches the same
function. `prepare_opencode_legacy.py` is the same wrapper for `prepare(prompt="opencode")`, which writes
all 1000 rows with `prompts/opencode.yaml` to `hle_diamond_benchmark_opencode_full.jsonl`.

## Scoring

`reward` is 1.0 when the judge answers `Judgement: yes` and 0.0 otherwise: `equivalence_llm_judge`
with the HLE labels, no swap check, no regex extraction and no per-record regex, the same judge
settings as `benchmarks/hle/config.yaml` except for the case rule below (see Prompting for how the
judge prompt differs from upstream's structured output). An empty final answer scores 0 without a
judge call (`failure_reason: empty_final_answer`). The judge defaults to the policy model, as in
`benchmarks/hle`, so the model under test grades itself unless `judge_openai.yaml` is composed in;
self-judged numbers are not comparable with the published table (see Quickstart).

The verdict labels are matched case-insensitively: `judge_labels_case_insensitive: true` on the single
judge server instance every preset shares (`str.casefold` on the judge text and on the labels, with bold,
italic and code marks and the spacing around colons ignored, so `Judgement: **no**` reads as
`Judgement: no`; the recorded `verdict_label` is always the configured spelling). Upstream reads a structured
`correct: yes|no` field whose case cannot vary, whereas Gym finds the label in free text, and on the
gold sweep (Harness validation) o3-mini wrote `Judgement: Yes` on 1 of 1000 rows, which the byte-exact
default scored 0 as `no_verdict`. The knob is off by default; `hle` and `hle_verified` turn it on as well.
When it is on, labels that differ only in case or markup are rejected at config time.

Judge request: the judge call carries no `max_output_tokens` and no reasoning parameter
(`judge_responses_create_params: {input: []}`), so the judge model runs at its default effort and its
verdict is never truncated. Upstream caps the judge at `max_completion_tokens=4096`. The divergence is
deliberate: a long reasoning trace from an o3-mini-class judge must not cut the verdict off and turn
a graded row into a parsing failure. `opencode_hle_tools` (and `opencode_hle_tools_full`, which inherits its
judge) is the exception: its judge runs as upstream's, `max_output_tokens: 4096` and a `question_extract_regex`
that hands the judge the bare question rather than the question plus the folded answer-format instruction;
on the full set's image rows the judge joins the text parts of the multimodal user turn before the regex
applies (Sandboxed agents (with tools) > Upstream web tools).

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
`resources_servers/equivalence_llm_judge/tests/test_hle_diamond.py` composes the commands in this README through the CLI's own flag
translation to keep that true.

## Harness validation

Model-free checks that have been run (commands below, from the repository root):

- `resources_servers/equivalence_llm_judge/tests/test_hle_diamond.py`: preparation against an in-memory arrow fixture (field set,
  the `input_image` block, the text filter, fail-closed on a wrong row count / partition split /
  image count / duplicated id, `--limit` truncation after the text filter and above the corpus size,
  the `prepare_text.py` script form, no download on invalid arguments); the shipped prompt files
  rendering exactly the upstream `SYSTEM_PROMPT` and `prepare.py` baking exactly it; config
  discovery and composition, the judge prompt path, manifests, the reasoning-effort knobs of the
  relay presets and the absence of one on the session presets, `sandbox_docker_local.yaml` resolving to
  exactly the Docker provider with inherited networking while the unused harness's block stays inert;
  the run commands of this README
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
- On the relay presets (`*_legacy_agent`, `opencode_full`, `opencode_search_full`), which go
  through the `legacy_agent` relay, the aggregate carries
  neither the partition slices nor `judgement_parsing_issue_rate`: the sandboxed adapter's inherited
  `aggregate_metrics` never calls the judge resources server's `compute_metrics`. The rollouts carry
  `partition` and `judge_evaluations`, so compute both post hoc (Reading sandboxed results). The
  agent-session presets aggregate through the Environment Server, which forwards to the judge
  resources server.
- Output budget on the OpenCode session presets: 32,768 tokens per model call within a 229,376-token
  context (the shared profile, as in `benchmarks/hle`). OpenCode asks for its output cap on every
  call, so the sandboxed presets' `remaining_context` policy, which left the cap to the model server,
  has no equivalent in `opencode_agent`, and asking for the whole 262,144-token window makes vLLM
  refuse the call. The Pi session presets keep `remaining_context`.
- Step cap and compaction on the OpenCode session presets: the shared profile leaves unset two
  settings of the sandboxed benchmark profile, the build agent's 400-step cap and
  `compaction.auto: false`, so compaction, the step limit and session titles follow OpenCode's
  defaults, and the run budget (`timeout`, 14,400 s) bounds the episode. `opencode_agent` accepts
  both in a session's `opencode_config`, so a run can restore them with
  `++hle_diamond_opencode_agent.responses_api_agents.opencode_agent.opencode_config.compaction.auto=false`
  and
  `++hle_diamond_opencode_agent.responses_api_agents.opencode_agent.opencode_config.agent.build.steps=400`
  (`hle_diamond_opencode_reference_agent` on `opencode_reference`). Either can move scores. The
  profile keeps these presets and `benchmarks/hle`'s alike, so a change there belongs in the profile.
- Failed runs: a session run that stops at its output, context or time limit is graded on what it
  produced, as in `benchmarks/hle`. A harness crash (a nonzero exit, no result) is retried and then
  reported as a failed rollout, so it leaves the denominator; the relay presets'
  `execution_failure_reward_zero` scored it 0. The crash leaves the agent route as an error (HTTP
  502 from OpenCode, an exception from Pi), so the Environment Server handles it as a dependency
  failure before `agent_outcome_policy` runs, and no `by_failure_kind` setting makes it gradable. A
  failed response that carries an agent outcome does reach the policy. Report the failed-rollout
  count next to every score.
- The OpenCode session presets start without network access to npm. With any plugin configured
  (the adapter always loads its own), OpenCode 1.17.11 installs `@opencode-ai/plugin` into each
  session's fresh config directory at every start unless the bundle is staged
  (`sandbox_plugin_dependencies_path`, opencode_agent README, Owner extensions); offline, the start
  stalls about 70 s before that install fails. Neither science image ships the bundle yet, so expect
  that delay on every row of `opencode`, `opencode_search`, `opencode_hle_tools`, `opencode_hle_tools_full`
  and `opencode_reference`.
- OpenCode's permissions on the session presets are opencode_agent's `permissions/default.yaml`:
  everything is allowed except listed bash commands (process killing, `rm -rf` of system paths, raw
  disk writes, git network commands and others). The sandboxed presets allowed every command and
  denied only web fetch, web search and subagents.
- Web fetch and subagents in the OpenCode session presets: the shared profile denies `task` under
  `permission`, and opencode_agent's `permissions/default.yaml` denies `webfetch`, `websearch` and
  `skill`, all after its `"*": allow`. A two-row probe on a public image with open egress (2026-11-10)
  told the model to fetch `https://example.com` with `webfetch` and to run a `task` subagent: every
  OpenCode request offered only `bash`, `edit`, `glob`, `grep`, `read`, `todowrite` and `write`, and the
  one `task` call the model made anyway came back as "Model tried to call unavailable tool 'task'".
  Before opencode_agent.yaml's `tools` map was removed, the same probe fetched the page and ran the
  subagent.
- A session's sandbox is created at seed, before the agent's own concurrency limit (`concurrency:
  512`) applies. `pi_agent` waits for a slot only when the activation starts, outside the run
  budget, while the sandbox idles and its TTL runs, so a long wait can use up the 1,800 s of TTL
  headroom and lose the sandbox before the deadline. `opencode_agent` does the same: a first
  activation's budget starts once it holds a slot. The sandboxed Pi agent took the slot before
  creating the sandbox. Keep `--concurrency` at or below 512 on these presets so that no session
  queues.
- The search presets' leakage policy binds only the Tavily tools: under the Docker provider (open
  egress) the agent can search and fetch from bash around it, and Tavily pre-filters at most 150 of
  the 220 excluded domains, the rest being enforced locally (Sandboxed agents (with tools), Search
  exclusions). A reproduction-grade with-tools run needs the OpenSandbox provider with
  `network_access: model_and_tools`; `opencode_reference.yaml` requests exactly that and refuses the
  Docker override unless the run is acknowledged as egress-unrestricted (Reference preset), but no
  OpenSandbox run of it exists yet.
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
gym eval prepare --benchmark hle_diamond/opencode_full  # Wrote 1000 problems to benchmarks/hle_diamond/data/hle_diamond_benchmark_opencode_full.jsonl (272 with an image)
gym list benchmarks | grep hle_diamond          # hle_diamond, hle_diamond/text, hle_diamond/opencode, hle_diamond/pi, hle_diamond/opencode_reference, hle_diamond/opencode_full, hle_diamond/opencode_hle_tools_full, and the listed *_legacy_agent presets
```

`gym eval prepare --benchmark hle_diamond/opencode_hle_tools_full` prepares the same
`hle_diamond_benchmark_opencode_full.jsonl` (same `prepare_script`); the 2026-11-18 pilot below ran the
same composition through a dataset override before this file existed (Upstream web tools > Acknowledged Docker
pilot, 2026-11-18).

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

The runs recorded in this section and under Reference preset predate the move of the text presets
to agent sessions: they ran `opencode_sandboxed_agent` through the `legacy_agent` relay (the presets
now named `*_legacy_agent`), and their rows carry that adapter's fields (`opencode_finished` and so on).

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

**Run on 2026-10-26 for `hle_diamond/opencode_full`, image rows only**, with this recipe
plus a third composed file that uncommented `network: host` and pointed the dataset at 5 image rows of the
prepared OpenCode full set (3 PNG, 1 JPEG, 1 WebP; `uuid`s `6729b153b7eb78fd8315f689`,
`67c158c8f98cdae328e0ff70`, `67a67d05a563d51fa3d38b3e`, `671f2870e38f776acdad8a93`,
`67a55a7242db6a9442ed34a2`), `--model-type openai_model` with GPT-6 Astra, judge o3-mini,
`++default_host=<host IP>` (the workstation's hostname resolves to `127.0.1.1`) and concurrency 5:
exit 0, about 14 min wall, 5/5 rows scored with reward 1.0, `opencode_attachment_count` 1 and
`opencode_failed` 0 on every row, no adapter or model error, 2/5 rows made real tool calls (21 and
62, bash with a few `read`/`glob`), all requests carried `reasoning_effort: high`, and the first
model call of every row carried one `image_url` part whose data URI equals the dataset's byte for
byte (no `detail` key) after the two synthetic Read-tool text parts. None of the five is among the
26 image rows larger than 2000 px, so that run says nothing about OpenCode's resizer; the one-row
smokes below do. A first attempt of the same
smoke, before the adapter declared the model's image modality, scored 0/5: OpenCode had replaced
each image with its "this model does not support image input" text and the model answered "unable
to determine" (Sandboxed agents (with tools) > Image questions through OpenCode). Label as for
every run through this recipe: "local Docker; egress not restricted; comparison infrastructure
only"; 5 rows say nothing about the full-set score.

**Run on 2026-10-26 for `hle_diamond/opencode_full`, one image row larger than 2000 px,
twice**, with this recipe plus a third composed file (`network: host`, dataset pointed at the single row
`66f1b940f8bb4425f74e4299`, a 14950x2780 PNG of 743,849 bytes), `--model-type openai_model` with
GPT-6 Astra, judge o3-mini, `++default_host=<host IP>`, `--limit 1 --concurrency 1`, each with a
model-call capture:

- Control, OpenCode's default image limits restored through
  `++...opencode_config.attachment.image.max_width=2000 ...max_height=2000 ...max_base64_bytes=5242880`:
  every captured `image_url` carried a 2000x372 PNG of 105,698 bytes, not the dataset's bytes, which
  shows the resizer loads and runs in the science image.
- As committed (limits lifted by the adapter): captured by the 272-image-row run of
  `hle_diamond/opencode_search_full` on 2026-10-26 (GPT-6 Astra, judge o3-mini, same
  recipe). The first policy call of every row carried exactly one `image_url` part whose data URI has the
  dataset's length and media type on 272/272 rows, with no `detail` key; for this row the
  991,822-character data URI reached the gateway unchanged on every call. The same run shows the cost
  of the lifted limits: the provider rejected this image on every call with HTTP 400
  `invalid_request_error` ("The image you provided requires 40716 patches after processing, exceeding
  the limit of 30000"), and OpenCode re-issued the identical request until the sandbox timeout
  (2,400 s in that run) killed the container, so the row scored 0 without a judge call. With the
  committed `sandbox_timeout` of 14,400 s the same row would retry for four hours before scoring 0.
  That run scored 73.2 % (199/272) over the image rows, with this row inside as a zero.

Both are labelled as every run through this recipe ("local Docker; egress not restricted; comparison
infrastructure only") and say nothing about the score.

```bash
# The presets always compose nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml, whose
# ${oc.env:...} values must resolve even though the Docker provider is used: placeholders suffice.
export OPENSANDBOX_API_KEY=unused-local-docker
export OPENSANDBOX_DOMAIN=unused.invalid
export OPENCODE_SANDBOX_IMAGE=gym-opencode-science:local      # PI_SANDBOX_IMAGE=gym-pi-science:local for the pi presets
export OPENCODE_ARTIFACTS_DIR=/abs/path/oc-artifacts          # full presets only: opencode_sandboxed_agent's native transcripts
mkdir -p /abs/path/model-calls

gym eval run --benchmark hle_diamond/opencode --split benchmark --model-type litellm_model \
    '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml,benchmarks/hle_diamond/sandbox_docker_local.yaml]' \
    --num-repeats 1 --limit 20 \
    ++policy_model_name=<gateway model id> \
    ++policy_model.responses_api_models.litellm_model.extra_body.reasoning_effort=high \
    ++observability_enabled=true ++model_call_capture_dir=/abs/path/model-calls \
    --output results/hle_diamond_opencode_pilot.jsonl
```

- Both overrides travel in one raw `+config_paths` token so they merge after the preset (see
  Scoring > Dedicated judge); with `--config` the search presets' `network_access: model_and_tools`
  would win and fail closed on the Docker provider, and the model would be told internet is disabled.
- The effort travels on the model server's `extra_body`, not on `responses_create_params`: the
  session agents refuse a `reasoning` request field with 422, and their harnesses' own calls carry no
  effort (Prompting > Reasoning effort). The key follows `--model-type`.
- `sandbox_docker_local.yaml` defines a second provider block, `sandbox_docker` (the content of
  `nemo_gym/sandbox/providers/docker/configs/docker.yaml`), because the composed `sandbox:` block
  cannot be removed or re-typed by an override, and repoints `hle_diamond_opencode_agent`,
  `hle_diamond_pi_agent` and the full presets' `hle_diamond_opencode_full_agent` at it with
  `network_access: inherit` (restricted policies fail closed on any provider but OpenSandbox). Only
  the agent the preset defines is started; the other blocks stay partial, inert entries. The text
  `*_legacy_agent` presets keep those agent names on the relay agents, so the override refuses to
  compose with them (an almost-server error): run them on OpenSandbox. It also
  replaces the presets' environment note (the session agents' `system_prompt`, the full presets'
  `/tmp/nemo-gym-environment.txt`), which says internet is disabled, with the truth.
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
- The search presets, `opencode_hle_tools` and `opencode_hle_tools_full` additionally need `TAVILY_API_KEY`;
  `opencode_hle_tools_full` also needs a vision-capable policy model (`--benchmark hle_diamond/opencode_hle_tools_full`
  in the acknowledged command below, otherwise unchanged).
- `hle_diamond/opencode_reference`, `hle_diamond/opencode_hle_tools` and `hle_diamond/opencode_hle_tools_full`
  (all three on the reference agent, which the override addresses by name) are
  refused by this override until the run is acknowledged as egress-unrestricted: without `HLE_DIAMOND_ACK_UNRESTRICTED_EGRESS=egress-not-restricted` the
  agent's `network_access` resolves to a refusal string and the agent server stops at start with
  `Input should be 'inherit', 'model_only' or 'model_and_tools' [... input_value='refused - ...']`
  (verified 2026-10-26; no rollout was written). With it the reference agent runs on local Docker
  with `network_access: inherit`, the reference image and a note that says egress is not restricted,
  so the pilot is labelled by construction. Set `OPENCODE_REFERENCE_SANDBOX_IMAGE` (default
  `gym-opencode-hle-reference:local`), not `OPENCODE_SANDBOX_IMAGE`:

  ```bash
  export HLE_DIAMOND_ACK_UNRESTRICTED_EGRESS=egress-not-restricted
  export OPENCODE_REFERENCE_SANDBOX_IMAGE=gym-opencode-hle-reference:local
  gym eval run --benchmark hle_diamond/opencode_reference --split benchmark --model-type litellm_model \
      '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml,benchmarks/hle_diamond/sandbox_docker_local.yaml]' \
      --num-repeats 1 --limit 5 --concurrency 5 \
      ++policy_model_name=<gateway model id> \
      ++policy_model.responses_api_models.litellm_model.extra_body.reasoning_effort=high \
      ++observability_enabled=true ++model_call_capture_dir=/abs/path/model-calls \
      ++route_failures_to_sidecar=true \
      ++use_absolute_ip=false ++sandbox_docker.docker.create.network=host \
      --output results/hle_diamond_opencode_reference_pilot.jsonl
  ```

  The last override pair is the host-networking form described above, written as overrides instead
  of editing the file (`use_absolute_ip=false` is correct only because the container shares the
  host's network namespace). Reference preset below records the run.

### Reading sandboxed results

- Agent-session rows (`opencode`, `opencode_search`, `opencode_hle_tools`, `opencode_hle_tools_full`, `pi`,
  `pi_search`, `opencode_reference`) carry the agent's own `response.status` (`completed`, or `incomplete` at a limit), its observations in
  `ng_agent_observations`, and `failure_kind` / `failure_reason` on an episode the Environment Server
  could not grade. Their aggregate carries the partition slices and `judgement_parsing_issue_rate`.
  The rest of this list describes the sandboxed-agent rows of the relay presets (`*_legacy_agent`,
  `opencode_full`, `opencode_search_full`).
- `response.status` is `null` on sandboxed rows (the adapter synthesises the response) unless the
  output limit was hit, when the adapter sets `incomplete` with
  `incomplete_details.reason = max_output_tokens`. Read success or failure from
  `opencode_finished`, `opencode_failed`, `opencode_exit_code` and
  `opencode_export_found` (OpenCode) or `finished_naturally`, `pi_failed`, `pi_exit_code` and
  `pi_error_type` (Pi); the aggregate metrics carry `mean/opencode_failed` and `mean/opencode_finished`.
- The aggregate of a full-preset run carries neither `slices/partition/...` nor
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
- Confirm tool use: at least one pilot row must show `ng_perf.num_tool_calls > 0` (or, on the full
  presets, a `tool` part in the harness transcript under the artifacts directory; session presets
  write none there). Otherwise the sandbox was never
  exercised and the run is a no-tools run through a different prompt path, not a with-tools result.
- Confirm effort. On the full presets the captured OpenCode request bodies carry `reasoning_effort:
  high` (all 15 requests of the 5-row `hle_diamond/opencode` pilot capture on
  `opencode_sandboxed_agent`). On the agent-session presets the model-call capture records the
  harness's request before the model server adds its `extra_body`, so it shows no effort; check the
  resolved config (`rollouts_resolved_config.yaml`, the policy model's `extra_body`) instead. A smoke
  over synthetic questions with a logging proxy in front of the endpoint (2026-11-10) saw
  `reasoning_effort: high` on every body the model server sent: 3 of 3 OpenCode calls, 7 of 7 Pi
  calls, and the 5 judge calls, which went to the same model server there. Check it on every pilot
  (Prompting > Reasoning effort).
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
`hle_diamond/opencode_search` and `hle_diamond/pi_search` add Tavily search and page tools;
`hle_diamond/opencode_hle_tools` adds upstream's own two web tools behind its blocklist (Upstream web
tools below). These five run the 728 text rows with the verbatim HLE
instructions in a single user message and one repeat; `hle_diamond/opencode_hle_tools_full` runs
`opencode_hle_tools` over all 1000 rows, image rows included (Upstream web tools). Keep collection repeats
at one. The published
with-tools column was produced under upstream's
`docs/evaluation-with-tools.md` protocol, from which these presets depart in concrete ways, so they
support a **comparison, not a reproduction**, of that column:

- Harness: OpenCode or Pi inside Gym's sandbox instead of each vendor's own agent (Claude Code,
  Codex CLI, Gemini CLI, Muse Code), with the harness's own system prompt ahead of the instruction.
- Sandbox: the `gym-*-science` images carry Python 3.13 with SageMath and Lean/Mathlib preinstalled,
  and the environment note tells the model so; upstream requires Python 3.11.16 with pinned
  requirements and forbids specialist solvers, chess engines, OCR and other task libraries.
  `opencode_reference.yaml`, `opencode_hle_tools.yaml` and `opencode_hle_tools_full.yaml` close this departure
  with the reference image (Reference preset).
- Web tools (Tavily search presets): Tavily search and page tools with up to 200,000 cached page
  characters and 2,000-word scrolling, instead of exactly `web_search` (at most 20 results) and
  `web_fetch` (Markdown capped at 60,000 characters, no media). `opencode_hle_tools` serves exactly
  those two tools, with the residual differences listed under Upstream web tools.
- Leakage policy (Tavily search presets): two of upstream's seven blocklist families are enforced in
  full and the search-engine result-page family approximately; the search provider pre-filters at
  most 150 of the 220 excluded domains and the rest are enforced locally (see Search exclusions).
  `opencode_hle_tools` enforces all seven (Upstream web tools).
- Rows: 728/1000 text rows on `opencode`, `opencode_search`, `opencode_hle_tools`, `pi` and
  `pi_search`; all 1000 on `opencode_hle_tools_full` (agent session) and on `opencode_full` and
  `opencode_search_full` (relay), whose image rows go through OpenCode's file-attachment path rather
  than as a direct `image_url` input (see Image questions through OpenCode below). Both harnesses lift
  OpenCode's image limits so that all 272 images, the 26
  larger than 2000 px on a side included, are sent as the dataset's bytes, as upstream sends them; an
  endpoint with an image-size limit can still reject one (GPT-6 Astra rejects the 14950x2780 PNG of
  `66f1b940f8bb4425f74e4299`, which then scores 0, see Image questions through OpenCode). The
  deprecated Pi adapter accepts text parts only.
- Reasoning effort: requested through the model server's `extra_body` on the session presets
  (`opencode_hle_tools` included) and through OpenCode's own option on the full presets (see
  Prompting > Reasoning effort).

State these departures next to every number collected with these presets. `opencode_hle_tools` closes
every departure Gym can close; its fidelity table (Upstream web tools) states, requirement by requirement,
what it matches and what still cannot be matched.

### Agent sessions

The text presets (`opencode`, `opencode_search`, `opencode_hle_tools`, `pi`, `pi_search`,
`opencode_reference`) and the full-set `opencode_hle_tools_full` run `opencode_agent` or `pi_agent` through
the benchmark's own Environment Server,
`hle_diamond_equivalence_llm_judge_environment_server` (`single_agent_turn_legacy`), instead of the
deprecated sandboxed agents behind a `legacy_agent` relay. Each episode seeds the judge resources
server, opens an agent session, runs one activation with the prepared prompt, closes the session and
verifies. The presets compose the agents' shared offline science profiles
([`opencode_agent`](../../responses_api_agents/opencode_agent/configs/profiles/offline_science.yaml),
[`pi_agent`](../../responses_api_agents/pi_agent/configs/profiles/offline_science.yaml)), the same as
`benchmarks/hle`'s presets, and add only the HLE-Diamond judge and dataset; a test holds the agents
equal to `hle`'s apart from those two keys:

- Sandbox: the judge lends none, so every session creates its own through the `sandbox` (OpenSandbox)
  provider, with the image, resources and entrypoint of the sandboxed presets, and destroys it on
  close. Its lifetime (`ttl_s`) is 16,200 s: it starts when the seed creates the sandbox, while the
  14,400 s run budget starts only after the seed, which may wait up to 1,200 s for the sandbox to
  be ready.
- Egress: `network_access: model_only` (code-only) or `model_and_tools` (search), as before; the
  policy allows the model server's host and, for `model_and_tools`, each granted tool server's host.
  Such a sandbox cannot download a runtime, so OpenCode runs the science image's
  `/usr/local/bin/opencode` at the digest its Dockerfile checks (`sandbox_executable_path`,
  `sandbox_executable_sha256`) and Pi the image's `/opt/node` and `/opt/pi` (`sandbox_runtime_dir: /opt`,
  `pi_version: 0.85.1`).
- Search: the agent's `tool_servers` lists the Tavily server (`opencode_hle_tools`: the `hle_tools`
  instance of `hle_web_tools`); the Environment Server seeds a session on that server for every
  episode and grants its MCP tools to the agent. The judge serves no tools, so it grants none of its
  own.
- Environment note: the session agents carry it as `system_prompt`, which OpenCode reads as its
  instructions file and Pi appends to its system prompt.
- Token limits: OpenCode asks for at most 32,768 output tokens per call within a 229,376-token
  context; Pi has the 262,144-token window with `output_token_policy: remaining_context`, which drops
  each request's output cap so the model server sizes it from the context the request leaves.
- Failed runs: a run that stops at its output, context or time limit is graded on what it produced; a
  harness crash is retried and then reported as a failed rollout (Known limitations).

What changes against the sandboxed presets is listed under Known limitations (OpenCode's output
budget, step cap, compaction setting, permissions and offline plugin install, failed runs, when a
session's sandbox is created, and web fetch and subagents). The `*_legacy_agent` presets,
`opencode_full` and `opencode_search_full` still run the sandboxed agents, so their text-question scores
come from a different harness than `opencode` and are not comparable with it.

**First run of the search presets.** A 10-row `hle_diamond/opencode_search` pilot on 2026-10-16
(GPT-6 Astra, judge o3-mini, local Docker with host networking, egress unrestricted) was the first
execution of either search preset. With the code of that day every `web_search` failed on Tavily's
150-domain limit (Search exclusions > Provider-side cap on excluded domains), so its 9 scored rows
(8 correct) measure a search tool that returned nothing plus unrestricted bash egress, not a
with-tools result; the same run exposed the search-engine result-page gap that the SERP rules now
close. No search-preset number from the committed code exists yet. Before reading any statistic from
the next pilot, confirm in the Tavily server log that `web_search` returns results: one cap warning
at start and no `Tavily HTTP 400`.

### Reference preset

`opencode_reference.yaml` is `opencode_search.yaml` with two of the departures above removed, so
that the remaining ones can be measured one at a time:

- Sandbox: the
  [reference image](../../responses_api_agents/opencode_sandboxed_agent/hle_reference_image/README.md)
  (`OPENCODE_REFERENCE_SANDBOX_IMAGE`, default `gym-opencode-hle-reference:local`) carries upstream's
  Python 3.11.16, a shell, the twelve packages of upstream's `docs/requirements-tools.txt` (copied
  verbatim, SHA-256 pinned in `tests/test_hle_reference_image.py`) and the same OpenCode 1.17.11 install
  as the science image, nothing else; the environment note names exactly those packages and no
  SageMath, Lean or `/opt/science/README.md`.
- Egress: `network_access: model_and_tools`, upstream's "no direct internet access" (section 1 of
  `docs/evaluation-with-tools.md`). Only the OpenSandbox provider enforces it, and the agent fails
  closed on any other; the preset renames the agent (`hle_diamond_opencode_reference_agent`) so that
  `sandbox_docker_local.yaml` cannot lift the policy silently and refuses unless the run is
  acknowledged with `HLE_DIAMOND_ACK_UNRESTRICTED_EGRESS=egress-not-restricted` (Sandboxed presets on
  a local Docker host).

What stays, and belongs next to every number: the OpenCode harness and its system prompt, the Tavily
search and page tools in place of exactly `web_search` and `web_fetch` (the note names neither;
`opencode_hle_tools` extends this preset and swaps the Tavily tools for the `hle_web_tools` server that
provides exactly those two tools, keeping this agent, image and egress policy (Upstream web tools)), the
weaker leakage policy (Search exclusions), 728/1000 text rows, and the judge alias. The preset restates the dataset block
because Gym's `_inherit_from` is a single-step rename (a chained `_inherit_from` leaves the agent
without an entrypoint), which is also why it appears in `gym list benchmarks`; a test holds its agent
equal to `opencode_search`'s apart from the image and the note.

**OpenSandbox gate: open.** No OpenSandbox run of this preset, of `opencode_hle_tools`, of `opencode_hle_tools_full` or of the image exists; the only run is the
acknowledged local Docker pilot below, whose egress was not restricted. The reference image has been
verified offline (`--network=none`) but not under a provider network policy.

**Acknowledged Docker pilot, 2026-10-26** (the command of Sandboxed presets on a local Docker host
with `--model-type openai_model`, GPT-6 Astra `azure/openai/gpt-6-astra`, judge `azure/openai/o3-mini`,
5 text rows, concurrency 5, host networking; label: "local Docker; egress not restricted; comparison
infrastructure only"). Exit 0, 242 s wall, 0 sidecar rows, 0 containers left. OpenCode started inside
the 785 MB reference image and finished on 5/5 rows (`opencode_finished` 5/5, `opencode_failed` 0/5,
exit code 0, export found); 5/5 scored 1.0 (all five rows were `reasoning` rows). Tool use: 29 calls
in all (per row 2, 2, 9, 2, 14): 10 `bash` calls, every one running `python` against the pinned stack
(SymPy, NumPy, SciPy), plus `tavily_web_search` and `tavily_find_in_page`; the Tavily log carried the
one 70-domain cap warning and no `Tavily HTTP 400`. Every captured OpenCode request carried
`reasoning_effort: high` (5/5 capture files). No image-related failure: no missing binary, no
`ModuleNotFoundError`, no download attempt by OpenCode's own tools. The same run shows why the gate
matters: with egress unrestricted, one row fetched SageMath source from `raw.githubusercontent.com`
and called `sagecell.sagemath.org` from `bash` with `requests` (3 such commands), reaching a host the
exclusion list blocks for the Tavily tools and a service the reference sandbox must not have; under
`network_access: model_and_tools` on OpenSandbox those calls fail. The same command without
`HLE_DIAMOND_ACK_UNRESTRICTED_EGRESS` stopped at agent start with the refusal (no rollout written).

See the shared
[OpenCode and Pi setup guide](../../responses_api_agents/opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)
for images, credentials, network policies, token limits and failure handling. Gates specific to
these presets:

- Sandbox images: `OPENCODE_SANDBOX_IMAGE` and `PI_SANDBOX_IMAGE` must point at images with the
  scientific stack described in the guide. The agent-session presets also need the OpenCode binary
  and the Pi runtime those images install (Agent sessions).
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

### Image questions through OpenCode

`opencode_hle_tools_full` (agent session), `opencode_full` and `opencode_search_full` (relay) are the
sandboxed presets that cover the 272 image rows; both harnesses hand the image to OpenCode as a file.
The session agent (`responses_api_agents/opencode_agent`, PR #4922) splits the one user message into
text and media with `nemo_gym.core.media.split_text_and_media`, which decodes each `input_image` part's
base64 data URI and reads nothing else of the part, writes the bytes to
`/tmp/nemo-gym-media/<sha256>.<ext>` in the sandbox (`nemo_gym.sandbox.stage_media`) and passes each
path with `--file` to `opencode run`; it declares image input on the Gym model and lifts OpenCode's
image limits for the session (its README > session path). The relay adapter
(`responses_api_agents/opencode_sandboxed_agent/app.py`) reads the same parts, uploads the bytes to
`/tmp/nemo-gym-attachments/image-<n>.<ext>` and passes each path with `-f`; its rows report the
number of attached images in `opencode_attachment_count`. In both the `input_text` part stays the
query. How OpenCode 1.17.11 then carries the image, by inspection of its source
at tag `v1.17.11` (`packages/opencode/src/cli/cmd/run.ts`, `session/prompt.ts`, `tool/read.ts`,
`session/message-v2.ts`, `image/image.ts`) and confirmed by the 5-row capture smoke below:

- `opencode run -f <path>` in local mode labels the file `text/plain` and routes it through OpenCode's
  Read tool, which sniffs the bytes (PNG, JPEG, GIF, WebP) and attaches a `data:` file part; the
  model's first message therefore holds two synthetic text parts (`Called the Read tool with the
  following input: {"filePath": ...}` and `Image read successfully`) and the image part ahead of the
  question text, a placement upstream's direct `[text, image_url]` content does not have.
- OpenCode forwards an image only to a model whose config declares `modalities.input` with `image`
  (`unsupportedParts` in `provider/transform.ts` otherwise substitutes `ERROR: Cannot read "image-1.png"
  (this model does not support image input)`, which the first 5-row smoke hit: every row answered
  "unable to determine"). The adapter adds that declaration to the generated config whenever the
  turn carries images and never otherwise, so a text-only run's config is unchanged.
- The `@ai-sdk/openai-compatible` provider sends the image as an `image_url` part with only the URL;
  the dataset's `detail: high` is not transmitted, which matches upstream's request.
- OpenCode's `image/image.ts` downscales any image larger than 2000 px on a side or 5 MiB of base64
  with its bundled photon resizer and re-encodes it (PNG first, else JPEG at quality 80 down to 40);
  `auto_resize: false` fails the turn on such an image instead of sending it, and only a resizer that
  fails to load leaves it untouched. The resizer is active in `gym-opencode-science:local`: with
  OpenCode's default limits, the one-row control smoke below sent the 14950x2780 PNG of
  `66f1b940f8bb4425f74e4299` (743,849 bytes) as a 2000x372 PNG of 105,698 bytes. 26 of the 272 image
  rows (12 PNG, 14 JPEG; sides from 2042 to 14950 px; none over 5 MiB of base64, none with a declared
  type that differs from its bytes) exceed that threshold. The adapter therefore sets
  `attachment.image.max_width`, `max_height` and `max_base64_bytes` to 2^31 - 1 in the generated
  config whenever the turn carries images, so every image is sent as the dataset's bytes, which is
  what upstream's `run_model_predictions.py` sends; override those keys through `opencode_config` to
  restore a limit. The cost: an endpoint with its own image-size limit rejects an oversized image
  outright, and OpenCode retries the rejected request until `sandbox_timeout`, so that row scores 0
  (one row on GPT-6 Astra in the 272-row run; see Sandboxed presets on a local Docker host).
  Restoring OpenCode's limits for a run trades that loss for downscaled images and changes the label.
- Formats: PNG (186 rows), JPEG (81), WebP (4) and GIF (1) are all sniffable. Any other media type, a
  remote `image_url` (the sandbox has no internet) or a `file_id` input fails the row before anything
  runs, with the reason in the error.
- The reconstructed `response.output` of such a row drops the file part (the adapter's export reader
  keeps text parts only) and the harness transcript under the artifacts directory keeps the synthetic
  Read-tool text; the model-call capture is where the `image_url` part is visible.

### Upstream web tools (`opencode_hle_tools`)

`hle_diamond/opencode_hle_tools` is `opencode_reference` with the Tavily search and page tools replaced
by [`resources_servers/hle_web_tools`](../../resources_servers/hle_web_tools/README.md), a tool server
that implements section 3 of upstream's `docs/evaluation-with-tools.md` (commit
`22ed3074b1e7b134bcbc09028d0ba320839b0655` <!-- pragma: allowlist secret -->) and is the only tool server of the preset. It is the preset that
follows upstream's protocol on every point Gym controls: the reference image (section 2), the enforced
egress policy (section 1) and the two tools behind the blocklist (section 3). What it cannot match is in
the fidelity table below. It runs the 728 text rows; `hle_diamond/opencode_hle_tools_full` is the same
preset with only the dataset swapped for the full set, all 1000 questions with the 272 image rows, which is
the denominator of the published with-tools column (Coverage below).

- Exactly two tools over MCP, `web_search(query)` (URL, title and snippet of at most 20 results per call before
  filtering) and `web_fetch(url)` (main content as Markdown, capped at 60,000 characters with a
  truncation indicator, no image, audio or video payloads; PDF rendering is Tavily's and unverified live). OpenCode
  names them `hle_tools_web_search` and `hle_tools_web_fetch`; its own `webfetch` and `websearch` stay denied.
- The upstream `docs/blocklist.json`, byte-exact (digest pinned in the server and checked at start),
  with **all seven rule families** enforced with the semantics the protocol describes, plus its query
  refusals (a blocked `site:` domain, a verbatim span of the question: 8 consecutive words, the
  server's choice where upstream states no length; a search from a session not seeded with the
  question, such as a plain-HTTP call from the sandbox carrying no session, is refused outright rather
  than served without the span rule; a session the sandbox seeds itself is not covered, see below),
  its fetch-URL checks after percent-decoding
  archive and proxy wrappers, its title-and-Markdown re-check of fetched and cached pages, and a log
  line per blocked call and dropped result naming the family and rule. This closes the four unenforced
  and one approximated families of the `opencode_search` preset (Search exclusions below), which stays
  as the Tavily-native comparison.
- The environment note (the agent's `system_prompt`) is the reference preset's: internet disabled, no
  downloads and no package installation, external information only through the two tools (named as the
  model sees them), Python 3.11 with the twelve packages and nothing else. A test holds it to exactly
  the pinned package list and to none of SageMath, Lean, `/opt/science` or Tavily.

**Agent session.** The preset extends `opencode_reference.yaml`, so its agent is the reference agent
(`hle_diamond_opencode_reference_agent`, Reference preset): an `opencode_agent` session on the
reference image (`OPENCODE_REFERENCE_SANDBOX_IMAGE`, default `gym-opencode-hle-reference:local`) under
`network_access: model_and_tools`, with the Tavily resource removed (`tavily: null`) and `hle_tools`
granted instead. The Environment Server seeds the `hle_tools` server for every episode with the row
(`/seed_session` reads the question from `task_data`) and grants its MCP tools from the agent's
`tool_servers`; the sandbox reaches the tool server by address (`absolute_ip_server_types` adds the
Resources servers) while the judge resources server and, with `judge_openai.yaml`, the judge model pin
`host: 127.0.0.1`, so the sandbox reaches exactly the model server's host and the tool server's;
`web_fetch` of a long page or PDF gets `mcp_tool_timeout_seconds: 300`; the effort travels on the model
server's `extra_body` (Prompting > Reasoning effort); and the rows and aggregate are the session presets'
(Reading sandboxed results). The judge, on this preset only, runs as upstream's: `[question]` is the bare
question (`question_extract_regex` cuts the folded instruction off) and the call is capped at
`max_output_tokens: 4096` (Scoring). Run it with `TAVILY_API_KEY` exported in the shell that starts the
servers; the `++count_missing_rollouts_as_zero=true` flag gives upstream's denominator (every row, a
failed call counted as wrong; add `'++count_failure_classes_as_zero=[agent_run_error]'` when failures
are routed to the sidecar) and `coverage/imputed` says how many rows it imputed:

```bash
export TAVILY_API_KEY=...   # the hle_tools server's search and extraction backend
export OPENCODE_REFERENCE_SANDBOX_IMAGE=gym-opencode-hle-reference:local   # the default
gym eval run --benchmark hle_diamond/opencode_hle_tools --split benchmark --model-type litellm_model \
    '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]' \
    --num-repeats 1 \
    ++policy_model_name=<gateway model id> \
    ++policy_model.responses_api_models.litellm_model.extra_body.reasoning_effort=high \
    ++count_missing_rollouts_as_zero=true \
    ++observability_enabled=true ++model_call_capture_dir=/abs/path/model-calls \
    --output results/hle_diamond_opencode_hle_tools.jsonl
```

**Coverage: 1000 against 728.** The published numbers (lastexam.ai/blog/hle-diamond, "1,000 questions,
reasoning high"; GPT-6 Astra 82.9 % with tools) divide by all 1000 questions, so a number from
`opencode_hle_tools` covers 728/1000 and is not comparable with them. `hle_diamond/opencode_hle_tools_full`
([`opencode_hle_tools_full.yaml`](opencode_hle_tools_full.yaml)) composes this preset and swaps only the
reference agent's dataset for the full set that `prepare_opencode_legacy.py` writes
(`hle_diamond_benchmark_opencode_full.jsonl`, `prompt_config: null`): one user message per row with the
question and the folded instruction, plus an `input_image` block on the 272 image rows. Same agent,
image, egress policy, tool server, note and judge (a test holds the two compositions equal apart from the
dataset); it needs a vision-capable policy model, and the same command as above runs it with
`--benchmark hle_diamond/opencode_hle_tools_full`. The image travels as described under Image questions
through OpenCode: the session agent stages the decoded bytes and attaches them with `opencode run --file`,
the dataset's `detail: high` is never read (upstream's `run_model_predictions.py` sends `image_url` with no
`detail` either), and the judge sees the bare question because `question_extract_regex` applies to the
joined text parts of the multimodal turn. The placement is not upstream's: OpenCode's Read tool puts two
synthetic text parts, the image and the staged `/tmp/nemo-gym-media/<sha256>.<ext>` path ahead of the
question text, where upstream sends one `[text, image_url]` user message (fidelity table, harness analogue). The 2026-11-18
acknowledged Docker pilot below ran this composition over the 272 image rows and confirmed in every one of
the 272 model-call captures that the first policy call carries one `image_url` part with the dataset's
bytes and no `detail` key, as the 2026-10-26 relay run had for 272/272 rows (Sandboxed presets on a local
Docker host). Report rows evaluated / 1000 next to the number as for every full run (Reporting a full run).

A reproduction-grade run needs the OpenSandbox provider, which enforces the egress policy, in egress
mode `dns+nft` (preflight: confirm the deployment's mode, and inside a sandbox that
`python -c 'import requests; requests.get("https://example.com")'` fails while the `hle_tools` MCP
endpoint answers); keep `head_server_serve_credentials` unset. Like `opencode_reference`, this preset is
refused by `sandbox_docker_local.yaml` unless `HLE_DIAMOND_ACK_UNRESTRICTED_EGRESS=egress-not-restricted`
is exported, and such a run is comparison infrastructure only (Quickstart > Sandboxed presets on a local
Docker host).

**Fidelity table.** One row per requirement of upstream's protocol and harness, sections 1 to 4 of
`docs/evaluation-with-tools.md` and then everything else in `hle_eval/` that moves a with-tools number,
all at upstream commit `22ed3074b1e7b134bcbc09028d0ba320839b0655`: <!-- pragma: allowlist secret -->

| Upstream requirement (`docs/evaluation-with-tools.md` at `22ed3074b1e7b134bcbc09028d0ba320839b0655` <!-- pragma: allowlist secret -->, and `hle_eval/` at the same commit) | Status | Where, and why |
| --- | --- | --- |
| §1 Disable internet access from the code sandbox, including downloads and package installation | matched (OpenSandbox) | `network_access: model_and_tools` becomes a deny-all OpenSandbox network policy that allows the model server's host and the `hle_tools` host; any other provider fails closed (`sandbox_docker_local.yaml` lifts it only behind `HLE_DIAMOND_ACK_UNRESTRICTED_EGRESS`, label the run). The image has no `curl`, `git` or Node; `pip` is present and has no network. Gate: the OpenSandbox server must run egress mode `dns+nft` (in `dns` mode a hard-coded IP bypasses the policy); that is a deployment setting, not Gym config, and no OpenSandbox run of this preset exists yet. |
| §1 External information is available only through `web_search` and `web_fetch` | matched | `hle_tools` is the only tool server; OpenCode's `webfetch`, `websearch` and `task` are denied. Residual: `/seed_session` is unauthenticated, so a sandbox can seed a fresh session and search without the span rule for the real question (NVIDIA-NeMo/Gym#5845); the blocklist families do not depend on the seed. |
| §1 Network access for the evaluated model is separate from task internet access | matched | The sandbox reaches the model server's host and the `hle_tools` host, nothing else: the judge resources server and the judge model pin `host: 127.0.0.1`, which wins over `absolute_ip_server_types`. OpenSandbox's policy is per host, so every port of those two hosts is open. |
| §2 Python 3.11.16 with a shell | matched | The reference image: `python:3.11.16-slim-bookworm` at a pinned digest, `python --version` asserted at build, bash from the base image. |
| §2 Install the pinned requirements (`docs/requirements-tools.txt`, twelve packages) | matched | `hle_reference_image/requirements-tools.txt` is byte-identical (SHA-256 `c1c626be…7068`), installed from a hashed lock with `pip check`; the test pins every version. Transitive versions are Gym's lock (upstream resolves them at install time, so theirs are unknowable). |
| §2 No specialist solvers, chess engines, OCR tools or task libraries | matched | Only `ripgrep` is added; a test scans the Dockerfile and lock for forbidden tokens, and the offline verification found none. The OpenCode binary and its plugin files are harness-side, as upstream's vendor CLIs are. |
| §2 (implicit) The model is told nothing beyond what the harness says | cannot match exactly | A two-sentence environment note is appended to OpenCode's instructions: internet disabled, the two tools by the names OpenCode shows (`hle_tools_web_search`, `hle_tools_web_fetch`), Python 3.11 and the twelve packages. It names nothing the image lacks (tested); state it next to every number. |
| §3 Expose only `web_search(query)` and `web_fetch(url)` | matched | The MCP token lists exactly the two; the server refuses to start with any other set. OpenCode shows them as `<grant>_<tool>` (`hle_tools_web_search`, `hle_tools_web_fetch`); upstream states no naming requirement for its wrapper. |
| §3 `web_search`: URLs, titles, snippets; at most 20 results per call before filtering | matched | Tavily `/search` with `max_results` 20 (bounded `le=20`), no answer, no page bodies, no images; survivors at most 20. |
| §3 `web_fetch`: main content of a page or PDF as Markdown, capped at 60,000 characters with a truncation indicator; no image, audio or video payloads | matched | Tavily `/extract` in Markdown; cut at 60,000 with `[Content truncated at 60,000 characters; N characters omitted]`; a page of any length is served cut (no size refusal); `data:` URIs removed, `include_images: false`; `max_fetch_chars` cannot exceed 60,000. Unverified: PDF rendering is Tavily's and no live PDF fetch is recorded. |
| §3 Load `docs/blocklist.json` | matched | Byte-exact copy (SHA-256 `3abaf35e…6575`), digest checked at start; all seven families present, unknown families rejected. |
| §3 `domains`: exact hostnames and all their subdomains | matched | Host or `.`-suffix match after UTS #46/IDNA normalisation, on requested, returned, redirected and wrapped URLs and `site:` operands. |
| §3 `url_prefixes`: URL prefixes, ignoring case, scheme and leading `www.` removed | matched | As written plus the canonical form; a prefix with a query also matches its `key=value` pairs in any order (a superset). |
| §3 `serp_url_patterns`: raw search-engine result URLs | matched | Case-insensitive regexes on the scheme-less URL and its canonical forms, on fetch and result URLs. |
| §3 `url_patterns`: case-insensitive regexes on queries, URLs, result titles/snippets, fetched titles/Markdown | matched | Applied to the query and its Unicode-folded readings, every URL candidate, result titles and snippets, the fetched title and the Markdown as served. |
| §3 `patterns_case_sensitive`: the same checks, case-sensitive | matched | Compiled without flags; same surfaces. |
| §3 `answer_title_patterns`: result titles only, also with trailing site names removed | matched | Only on search-result titles; the title and each cut before a trailing `\|`, `-`, `–`, `—`, `·`, `•`, `»`, `::` segment. |
| §3 `answer_title_allow_hosts`: exempt from `answer_title_patterns` only | matched | Host or subdomain short-circuits the answer-title check only; every other family still applies. |
| §3 `web_search`: check the query before the backend; filter every result by URL, title and snippet; refuse a blocked `site:` domain or a long verbatim span of the question | matched | Query check before the Tavily call; per-result filter; `site:` operands parsed to hostnames (a non-hostname operand is refused); span rule at 8 consecutive words (upstream states no length; the 8 is Gym's choice); a session without a seeded question fails closed. |
| §3 `web_fetch`: check the URL before fetching, including blocked URLs inside proxy/archive links after percent-decoding; after fetching check title and Markdown; apply to cached responses too | matched | Breadth-first expansion through percent-decoding, WHATWG normalisation and archive/proxy wrappers before any fetch; the final URL, title and Markdown as served are checked on fresh fetches and on cache hits (a blocklist change evicts a cached page). |
| §3 Log blocked calls and the matching rule for auditing | matched | Every refusal and every dropped result at WARNING with family, rule, where and a 200-character subject, emitted without `--verbose`. |
| §4 Run each model through its provider's harness (Claude Code 2.1.278/2.1.280, Codex CLI 0.155.1, Gemini CLI 0.60.0, Muse Code 1.3.0) | cannot match | OpenCode 1.17.11 (binary digest checked in the sandbox and at image build) for every model, with OpenCode's own system prompt ahead of the instruction. The harness mapping below says what each row would take in Gym. |
| `run_model_predictions.py`: `SYSTEM_PROMPT` verbatim | matched (text) / cannot match (placement) | The text is byte-exact (tested). Upstream sends it as the system message with the bare question as the user message; here it follows the question in one user message behind OpenCode's native prompt. Upstream's with-tools runs went through vendor CLIs whose prompt delivery is undocumented, so no placement is provably theirs; this one is stated. |
| `run_judge_results.py`: `JUDGE_PROMPT`, structured `correct: yes\|no` | matched (criteria) / cannot match (parse) | `benchmarks/hle/prompts/judge.txt` carries the same criteria with three label renames (`Judgement:`, `Reasoning:`, `Confidence:`), upstream's `0|\%|` / `100|\%|` spelled `0%` / `100%`, and one blank line fewer; the verdict is the last `Judgement: yes\|no` label in free text, case-insensitive, not a schema-enforced field. |
| Judge model `o3-mini-2025-01-31` | cannot match (snapshot) | Compose `judge_openai.yaml` (every documented run line does); `HLE_DIAMOND_JUDGE_MODEL` defaults to the gateway alias `azure/openai/o3-mini`, whose served snapshot is neither pinned nor echoed. Record alias and run date. Without the override the policy grades itself. |
| Judge call: `max_completion_tokens=4096`, one user message, no system message; `[question]` is `question['question']` | matched (this preset) | `judge_responses_create_params.max_output_tokens: 4096`, `judge_system_message: ""`, and `question_extract_regex` cuts the folded instruction so the judge sees the bare question. A verdict cut off by the cap scores 0 as `no_verdict`, as upstream. |
| `[response]` is the model's whole completion | harness analogue | The last assistant message of the OpenCode session (upstream does not say what text was judged for the CLI harnesses). |
| Confidence extracted by the judge; calibration error reported | cannot match | Not computed; `reward` is the accuracy only. |
| Accuracy over every question (`n` = all rows; a failed call counts as wrong); Wald 95 % half-width | matched (denominator) / own interval | The documented run line passes `++count_missing_rollouts_as_zero=true` (add `'++count_failure_classes_as_zero=[agent_run_error]'` with `++route_failures_to_sidecar=true`) and reports `coverage/imputed`; Gym's reward profile emits its own confidence interval, not upstream's Wald half-width. |
| HLE-Diamond: 1,000 questions, 272 with an image | harness analogue (`opencode_hle_tools_full`, run 2026-11-18 on local Docker) / cannot match (`opencode_hle_tools`) | `opencode_hle_tools_full` runs all 1000 rows on this agent with the dataset's bytes and no `detail`, but not with upstream's `[text, image_url]` content: OpenCode routes the staged file through its Read tool, so the model's first message carries two synthetic text parts, the image and the staged path ahead of the question text (Image questions through OpenCode). A vision-capable policy model is required. `opencode_hle_tools` runs the 728 text rows: report 728/1000 next to every number from it. |
| No temperature sent | matched | The preset and its chain set none; OpenCode forwards one only when the request carries it. On `vllm_model`, confirm `sampling_overrides` carry none. |
| Output budget (no with-tools figure; `>= 8192` recommended for reasoning models) | matched where stated | 32,768 output tokens per call in a 229,376 context, 14,400 s run budget, OpenCode's default step cap and compaction. |
| "All models are evaluated with reasoning high" | matched (OpenAI-style) / cannot assert (Claude via gateway) | `++policy_model.responses_api_models.<model>.extra_body.reasoning_effort=high` on the run line, verified in the model-call capture; Claude models through this gateway ignore the parameter. |
| One prediction per question, no repeats | matched | `num_repeats: 1`, `--num-repeats 1`. |
| Single judge pass; correct iff `yes` | matched | `check_twice_swap: false`, `use_per_record_regex: false`; the last label decides. |

Harness mapping for §4, from the agents Gym has today:

| Upstream harness | Gym status |
| --- | --- |
| Claude Code 2.1.278 / 2.1.280 | `responses_api_agents/claude_code_agent` pins an exact version but rejects `network_access: model_and_tools` and remote MCP tool grants, so it cannot carry `hle_tools`: adapter change needed before a preset. |
| Codex CLI 0.155.1 | `responses_api_agents/codex_agent` supports sandboxed sessions, MCP grants and `model_and_tools`; a preset needs the Codex runtime baked into the reference image (a restricted sandbox cannot download it) and a parsing smoke at 0.155.1, which is outside its tested version (0.144.4). |
| Gemini CLI 0.60.0 | no Gym agent. |
| Muse Code 1.3.0 | no Gym agent. |

Two residual differences from upstream's wrapper remain in every row above, and a with-tools number from
this preset should state both:

- Backend: the protocol names no search or extraction provider, and this server uses Tavily (`/search`
  at `max_results: 20`, `search_depth: advanced`; `/extract` in Markdown). Which pages a query
  surfaces, their ranking and the page-to-Markdown rendering are therefore Tavily's, and a query longer
  than 400 characters (Tavily's limit) is refused ("The query is longer than 400 characters.") before the blocklist check, a refusal
  upstream's protocol does not have.
- Sandbox-seeded sessions: `/seed_session` is unauthenticated and `network_access: model_and_tools`
  admits the tool-server host, so a process in the sandbox can `POST /seed_session` for a fresh
  session with a question of its choice and search from it without the span rule applying to the
  real question. The server refuses re-seeding an existing session (HTTP 409), which does not cover a
  new one or one its session cap has forgotten, and the session and MCP-token signing key derives from
  public config. Closing it needs an
  authenticated seed channel; tracking issue: NVIDIA-NeMo/Gym#5845. Until then the span rule binds a
  cooperative harness, not the sandboxed model.

**Pilot evidence, relay harness only.** The one pilot of this preset (2026-10-26, 5 text rows, GPT-6
Astra, judge o3-mini, local Docker with host networking and `++use_absolute_ip=false`, egress
unrestricted; comparison infrastructure only) ran the preset as it then was, `opencode_sandboxed_agent`
behind the `legacy_agent` relay on the science image: exit 0, 5/5 rows scored (3 correct), 4/5 rows made
tool calls, among them two `web_search` calls that returned 20 and 16 results with no Tavily error; all 14
captured OpenCode requests carried `reasoning_effort: high` (OpenCode's own option on that harness) and
offered exactly `hle_tools_web_search` and `hle_tools_web_fetch` as MCP tools next to OpenCode's native
file and shell tools. A manual `web_search` through the server's MCP endpoint, in a session seeded with a
real question and quoting ten consecutive words of it, was refused with the span message and logged
as `family=question_span rule='8 consecutive words of the question' where=query subject='8 words'`; a
manual `web_fetch` of a Hugging Face dataset page was refused as a blocked site. The server-side
checks hold on the session harness (the server's tests cover the episode seed and the MCP grant), but
that pilot ran another harness on another image with open egress, so its 3/5 says nothing about this
preset. The pilot below ran it as it now is on the session harness and the reference image, on local Docker
with open egress; no OpenSandbox run exists.

**Acknowledged Docker pilot, 2026-11-18** (the acknowledged command of Sandboxed presets on a local Docker
host with `--model-type litellm_model`, GPT-6 Astra `azure/openai/gpt-6-astra` with `reasoning_effort: high`
in the model server's `extra_body`, judge `azure/openai/o3-mini` through `judge_openai.yaml`, Docker
`network: host`, run timeout 2,400 s and 120 OpenCode steps per row, one repeat, concurrency 16; label:
local Docker; egress not restricted; comparison infrastructure only). `hle_diamond/opencode_hle_tools`
over the 728 text rows: **624/728 = 85.7 %** (knowledge 325/381, reasoning 299/347), 728/728 scored, two
rows at the timeout. The 272 image rows through the same composition with the dataset swapped for the
`has_image` rows of the full set (what `opencode_hle_tools_full` declares): **203/272 = 74.6 %**
(reasoning 125/153, knowledge 78/119), 272/272 scored after one `--resume` for a row whose OpenCode
process exited; every first policy call carried the image as described above. Composed over all 1,000
questions, two runs of one configuration: **827/1000 = 82.7 %** against the published 82.9 % for GPT-6
Astra with tools (knowledge 403/500 = 80.6 %, reasoning 424/500 = 84.8 %). Captured tool calls were
`hle_tools_web_search` and `hle_tools_web_fetch` plus OpenCode's native tools, no other web tool; the
reference note was in every request, SageMath, Lean and `/opt/science` in none. 48 of the 1,000 rows end
with the agent asking the user a question instead of an `Answer:` line, as an interactive coding agent
does when it cannot verify; upstream's protocol adds no nudge and neither does this preset. One run, one
repeat, no confidence interval, open egress: a comparison, not a reproduction (Reference preset >
OpenSandbox gate).

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
pytest resources_servers/equivalence_llm_judge/tests/test_hle_diamond.py \
    resources_servers/tavily_search/tests/test_hle_diamond_search_exclusions.py \
    resources_servers/equivalence_llm_judge/tests/test_app.py \
    resources_servers/tavily_search/tests/test_search_policy.py \
    responses_api_agents/opencode_sandboxed_agent/tests/test_hle_reference_image.py \
    resources_servers/hle_web_tools/tests/test_app.py \
    responses_api_agents/opencode_sandboxed_agent/tests/test_app.py   # the image-attachment path of the adapter
```

## Licensing

Code: Apache 2.0.

`cais/hle-diamond` data: MIT, with the dataset card's request not to publicly share, re-upload or
redistribute it; the data is downloaded at run time and never committed. The prompt files in
`prompts/` reproduce the HLE authors' `SYSTEM_PROMPT` from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) (MIT) verbatim; the judge prompt
is reused from `benchmarks/hle/prompts/judge.txt`, adapted from the same harness.
