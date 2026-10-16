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

This is a deliberate departure from `benchmarks/hle` and `benchmarks/hle_verified`, which share
Gym's `benchmarks/hle/prompts/default.yaml`, a paraphrase (`{your explanation for your final answer}`,
`{your succinct, final answer}`). The sole purpose of this benchmark is comparison with the published
table, so the measured prompt has to be upstream's; one prompt shared across the three HLE benchmarks
loses to that. Do not "fix" the wording back to the shared file.

- `prompts/default.yaml`: the verbatim text as the system message with the bare question as the user
  message, the placement upstream uses (upstream moves the instruction into the user role only for
  `o1` models; this benchmark always sends it as the system message). The braces are doubled in the
  file because `nemo_gym.prompt` fills `{placeholders}` with `str.format_map`; the rendered message
  carries the single braces, and `tests/unit_tests/test_hle_diamond.py` asserts that the rendered
  string and the prepared rows equal the upstream text. The default set bakes it into
  `responses_create_params.input` at prepare time (`nemo_gym.vision_input.build_image_input`); the
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
`nemo_gym.vision_input`, not to this benchmark.

The upstream harness sends no temperature and no output-token cap. When comparing with the published
table, send `++responses_create_params.reasoning.effort=high` and nothing else; on this gateway a
`temperature` is dropped by some models and rejected by others.

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
   no temperature, top-p or output cap), so runs differ only in what the gateway does with it. Check
   `response.usage.output_tokens_details.reasoning_tokens` on the pilot rows and record which label
   applies before comparing with the published mean output tokens.

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
  the level today. The knob is the right value and takes effect once the adapter exposes the flag;
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
settings as `benchmarks/hle/config.yaml` (see Prompting for how the judge prompt differs from
upstream's structured output). An empty final answer scores 0 without a judge call
(`failure_reason: empty_final_answer`). The judge defaults to the policy model, as in
`benchmarks/hle`, so the model under test grades itself unless `judge_openai.yaml` is composed in;
self-judged numbers are not comparable with the published table (see Quickstart).

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
`HLE_DIAMOND_JUDGE_MODEL` (default `azure/openai/o3-mini`, the upstream judge family), and repoints
the single `hle_diamond_equivalence_llm_judge_resources_server` that every preset shares:

```bash
gym eval run --benchmark hle_diamond --split benchmark '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]' ...
```

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
  `--split benchmark` each makes tests fail.
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

Not yet run: a gold-as-prediction sweep (every `expected_answer` fed back as the response) through
the judge with the dedicated judge model, which needs a judge endpoint and belongs to the pilot run.

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

### No-tools runs

**Not run as written** while writing this README: rollout collection belongs to the pilot. Every
part of the command is load-bearing:

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
gym eval run --benchmark hle_diamond --split benchmark --model-type openai_model \
    '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]' \
    --num-repeats 1 --limit 20 \
    ++policy_model_name=<gateway model id> \
    ++responses_create_params.reasoning.effort=high \
    --output results/hle_diamond_pilot.jsonl
```

Use `--benchmark hle_diamond/text` with the same flags for a text-only policy model; its 728/1000
coverage is not comparable with the published table.

Before treating a run as comparable, check on the pilot rows:

- Reasoning effort was honoured or is not controllable (Prompting > Reasoning effort): confirm
  `response.usage.output_tokens_details.reasoning_tokens` and label the run accordingly.
- The judge is the dedicated one (`judge_evaluations[*].response.model`) and
  `judgement_parsing_issue_rate` is near zero.
- Coverage: rows evaluated / 1000, or `coverage/imputed` with the upstream denominator (see Scoring).

### Sandboxed presets on a local Docker host

**Not run as written** for `hle_diamond`. The same recipe (Docker provider, host networking, the
OpenCode effort knob, the dedicated-judge composition aside) was run end to end on `hle/opencode`
for two rows before these presets existed, which is where the facts below come from. Every run
through this recipe is **comparison infrastructure only: egress is unrestricted**, and its results
must carry the label "local Docker; egress not restricted".

```bash
# The presets always compose nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml, whose
# ${oc.env:...} values must resolve even though the Docker provider is used: placeholders suffice.
export OPENSANDBOX_API_KEY=unused-local-docker
export OPENSANDBOX_DOMAIN=unused.invalid
export OPENCODE_SANDBOX_IMAGE=gym-opencode-science:local      # PI_SANDBOX_IMAGE=gym-pi-science:local for the pi presets
export OPENCODE_ARTIFACTS_DIR=/abs/path/oc-artifacts          # PI_ARTIFACTS_DIR for pi: the harness's native transcripts
mkdir -p /abs/path/model-calls

gym eval run --benchmark hle_diamond/opencode --split benchmark --model-type openai_model \
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
- Uncomment `network: host` in the override on hosts whose firewall drops bridge (docker0) to host
  traffic or whose hostname resolves to a loopback alias such as `127.0.1.1` under the presets'
  `use_absolute_ip: true`: the container must reach the Gym model server at the advertised address.
  Host networking also lets the container reach every service on the host; where the firewall allows
  it, prefer bridge networking with `++use_absolute_ip=false ++default_host=<routable host IP>`.
- The Docker provider ignores `sandbox_config.ttl_s` when an entrypoint is set, and the presets set
  `entrypoint: [sleep, infinity]`, so container lifetime relies on the agent's cleanup. Check
  `docker ps -a --filter label=nemo-gym.sandbox` after every run.
- The first run builds per-server `uv` virtual environments and needs PyPI access from the host.
- The search presets additionally need `TAVILY_API_KEY`.

### Reading sandboxed results

- `response.status` is `null` on sandboxed rows (the adapter synthesises the response). Read the
  outcome from `opencode_finished`, `opencode_failed`, `opencode_exit_code` and
  `opencode_export_found` (OpenCode) or `finished_naturally`, `pi_failed`, `pi_exit_code` and
  `pi_error_type` (Pi); the aggregate metrics carry `mean/opencode_failed` and `mean/opencode_finished`.
- `ng_trajectory`, `ng_perf` and `ng_model_call_capture` are attached only with
  `++observability_enabled=true ++model_call_capture_dir=<absolute dir>`; without them every
  trajectory health check reports "unobserved". Use a distinct capture directory per run.
- Confirm tool use: at least one pilot row must show `ng_perf.num_tool_calls > 0` (or a `tool` part
  in the harness transcript under the artifacts directory). Otherwise the sandbox was never
  exercised and the run is a no-tools run through a different prompt path, not a with-tools result.
- Confirm effort: the captured OpenCode request body carries `reasoning_effort: high`; a captured Pi
  request does not (Prompting > Reasoning effort).
- Token and latency figures are not comparable with the announcement's mean output tokens: the
  harness's system prompt and tool schema dominate the input, and the call is chat-completions.

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
- Leakage policy: two of upstream's seven blocklist families are enforced (see Search exclusions).
- Rows: 728/1000 text rows; image questions cannot be sent through these adapters.
- Reasoning effort: requested through the harness's own knob; transmitted by OpenCode, not by Pi
  (see Prompting > Reasoning effort).

State these departures next to every number collected with these presets.

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
  public site succeeds inside such a sandbox). Label any such run accordingly.
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
families are enforced:

| Upstream family | Rules | Gym rule | Enforced |
| --- | --- | --- | --- |
| `domains` | 220 | `domain`: exact hostname or any subdomain | yes |
| `url_prefixes` | 66 | `url_substring`: case-insensitive substring of the URL, scheme and leading `www.` removed; a prefix becomes a substring, so at least as strict | yes |
| `serp_url_patterns` | 12 | none | no |
| `url_patterns` | 22 | none | no |
| `patterns_case_sensitive` | 4 | none | no |
| `answer_title_patterns` | 5 | none | no |
| `answer_title_allow_hosts` | 7 | none | no |

The unrepresented families are recorded by name and count under `unrepresented_upstream_rules` in
the JSON, a key the parser ignores. The consequences make the search presets a **weaker leakage
policy than upstream's protocol** (`docs/evaluation-with-tools.md`), and a with-tools number from
them must be labelled as such:

- Only hostnames and URL substrings are inspected. No search query is refused (upstream refuses
  queries with a blocked `site:` domain or a long verbatim span of the question), and no result title,
  snippet or fetched Markdown is matched against the regex families (upstream rejects pages whose
  text matches `url_patterns` or `patterns_case_sensitive`, e.g. "Humanity's Last Exam" in several
  languages, `cais/hle`, specific arXiv ids and answer-key phrasing).
- Raw search-engine result pages (`serp_url_patterns`) and answer-key style titles
  (`answer_title_patterns`) are not filtered.
- The parser percent-decodes a URL and re-checks it, so a `url_substring` embedded in a proxy or
  archive link is caught; a `domain` rule applies to the outer hostname only, so an archive or
  proxy page that wraps a blocked host in its path passes unless a substring rule matches.
- Upstream blocks GitHub wholesale (`github.com`, `githubusercontent.com`, `gist.github.com`,
  `api.github.com`, `github.io`), and so does this list: the agent cannot fetch any GitHub page.

Two `url_substring` rules inherited from `benchmarks/hle/search_exclusions.json`
(`openreview.net/pdf/<40 hex characters>.pdf`) carry OpenReview document ids. They are not secrets,
they pass the repository's secrets scan unaltered, and they are kept rather than widened to the
`openreview.net/pdf/` prefix so the list does not block more than upstream does.

Refresh procedure when upstream changes its blocklist: update `UPSTREAM_COMMIT` and
`UPSTREAM_BLOCKLIST_SHA256` together in `build_search_exclusions.py`, run
`python benchmarks/hle_diamond/build_search_exclusions.py --download`, review the diff of
`search_exclusions.json` (a rule family new to the generator is reported as a warning), and run
`pytest tests/unit_tests/test_hle_diamond_search_exclusions.py`.

## Tests

```bash
pytest tests/unit_tests/test_hle_diamond.py tests/unit_tests/test_hle_diamond_search_exclusions.py \
    resources_servers/equivalence_llm_judge/tests/test_app.py
```

## Licensing

Code: Apache 2.0.

`cais/hle-diamond` data: MIT, with the dataset card's request not to publicly share, re-upload or
redistribute it; the data is downloaded at run time and never committed. The prompt files in
`prompts/` reproduce the HLE authors' `SYSTEM_PROMPT` from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) (MIT) verbatim; the judge prompt
is reused from `benchmarks/hle/prompts/judge.txt`, adapted from the same harness.
