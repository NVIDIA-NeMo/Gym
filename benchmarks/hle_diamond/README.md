# HLE-Diamond benchmark

[Dataset](https://huggingface.co/datasets/cais/hle-diamond) (MIT; the dataset card asks users not to
publicly share, re-upload or redistribute it) and
[announcement](https://lastexam.ai/blog/hle-diamond) (Center for AI Safety and Scale AI, 2026-09-22).

HLE-Diamond is a 1000-question subset of
[Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) split into two partitions of 500
questions each: `reasoning` (the answer has to be derived) and `knowledge` (the answer has to be
recalled). 272 questions include an image; 796 are exact-match and 204 multiple choice. The model
must produce an explanation, a succinct final answer and a confidence; an LLM judge compares the
extracted final answer with the reference answer.

Preparation pins revision `04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23` <!-- pragma: allowlist secret -->
of the `test` split and downloads it at run time. Nothing from the dataset is committed.

## Scope

| Benchmark | Rows | Prompt | Use |
| --- | --- | --- | --- |
| `hle_diamond` (default) | 1000 (272 with an image) | prepared at prepare time (`prompt_config: null`) | the published no-tools setting; needs a vision-capable policy model |
| `hle_diamond/text` | 728 (no image) | `benchmarks/hle/prompts/default.yaml` at rollout time | text-only policy models |
| `hle_diamond/opencode`, `hle_diamond/pi` | 728 | `benchmarks/hle/prompts/opencode.yaml` | sandboxed code tools, no search |
| `hle_diamond/opencode_search`, `hle_diamond/pi_search` | 728 | `benchmarks/hle/prompts/opencode.yaml` | sandboxed code tools plus Tavily search |

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

The prompts are Gym's shared HLE prompts, referenced from `benchmarks/hle/prompts/` so that no
prompt file lives in this directory. They are adaptations of the
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) harness, not verbatim copies:

- `default.yaml`: Gym's HLE system prompt (Explanation / Answer / Confidence) with the bare question
  as the user turn, the placement upstream's `run_model_predictions.py` uses. The wording differs:
  upstream's `SYSTEM_PROMPT` asks for `{your explanation for your answer choice}` and
  `{your chosen answer}`; Gym's file asks for `{your explanation for your final answer}` and
  `{your succinct, final answer}`. The default set bakes this text into
  `responses_create_params.input` with `nemo_gym.vision_input.build_image_input`; the text variant
  applies the same file at rollout time. Every prepared row and every sandboxed preset therefore
  carries the paraphrase, which is a comparability caveat to state next to any number compared with
  the published table. Keeping the shared file is deliberate (one HLE prompt across `hle`,
  `hle_verified` and `hle_diamond`); a verbatim-upstream prompt would be a separate decision.
- `opencode.yaml`: the same instructions folded into one user message for the sandboxed harnesses,
  whose native system prompts stay intact.
- `judge.txt`: the HLE judge prompt adapted for the `equivalence_llm_judge` label parser. Upstream's
  `run_judge_results.py` parses a structured object
  (`extracted_final_answer, reasoning, correct: yes|no, confidence, strict`); Gym's file renames the
  fields to `Reasoning:` / `Judgement:` / `Confidence:` and the verdict is the `Judgement: yes|no`
  string found in free text, with unparseable judge output tracked by `judgement_parsing_issue_rate`.
  The grading criteria are the same.

Image questions: upstream attaches the image as `image_url` with no `detail` setting (the provider
default); Gym's shared builder requests `detail: high`, which changes image tiling and token counts
on the 272 image rows for OpenAI-family models. Making `detail` configurable would be a change to
`nemo_gym.vision_input`, not to this benchmark.

The upstream harness sends no temperature and no output-token cap. When comparing with the published
table, send `++responses_create_params.reasoning.effort=high` and nothing else; on this gateway a
`temperature` is dropped by some models and rejected by others.

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
prepare a subset, and it says loudly that the completeness check was skipped. `prepare_text.py` is a
thin wrapper for `prepare(text_only=True)`.

## Scoring

`reward` is 1.0 when the judge answers `Judgement: yes` and 0.0 otherwise: `equivalence_llm_judge`
with the HLE labels, no swap check, no regex extraction and no per-record regex, the same judge
settings as `benchmarks/hle/config.yaml` (see Prompting for how the judge prompt differs from
upstream's structured output). An empty final answer scores 0 without a judge call
(`failure_reason: empty_final_answer`). The judge defaults to the policy model, as in
`benchmarks/hle`, so the model under test grades itself unless `judge_openai.yaml` is composed in;
self-judged numbers are not comparable with the published table (see Quickstart).

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
- Coverage: the upstream harness divides by all questions, so compare the rollouts file's row count
  with 1000 (728 for the text configs) and report rows evaluated / 1000 next to every accuracy.

Upstream also reports a calibration error from the model's stated confidence; it is not computed here.

### Dedicated judge

`judge_openai.yaml` is a composable override, not a benchmark: it declares no dataset and does not
appear in `gym list benchmarks`. It defines `hle_diamond_judge_model`, an `openai_model` on the
policy gateway (`policy_base_url` / `policy_api_key` from `env.yaml`) serving
`HLE_DIAMOND_JUDGE_MODEL` (default `azure/openai/o3-mini`, the upstream judge family), and repoints
the single `hle_diamond_equivalence_llm_judge_resources_server` that every preset shares:

```bash
gym eval run --benchmark hle_diamond '+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]' ...
```

## Harness validation

Model-free checks that have been run (commands below, from the repository root):

- `tests/unit_tests/test_hle_diamond.py`: preparation against an in-memory arrow fixture (field set,
  the `input_image` block, the text filter, fail-closed on a wrong row count / partition split /
  image count / duplicated id, `--limit` truncation after the text filter and above the corpus size,
  the `prepare_text.py` script form, no download on invalid arguments), config discovery and
  composition, the judge prompt path, manifests, the partition slices through
  `compute_aggregate_metrics`, and the production row shape (`/verify` reply through
  `SimpleAgentVerifyResponse` into the slices). The count check, the limit truncation, the response
  field echo and the slice hook were mutation-checked: breaking any of them makes tests fail.
- `tests/unit_tests/test_hle_diamond_search_exclusions.py`: the generator's normalisation, dedup and
  audit trail; the rendered and the committed list loading through the real Tavily parser and
  blocking load-bearing hosts; the committed list being at least as strict as `benchmarks/hle`'s;
  and the upstream pin being a full 40-character revision (the `--download` URL depends on it).
- Real preparation printed `Wrote 1000 problems ... (272 with an image)` and
  `Wrote 728 problems ... (skipped 272 image questions)`; the written rows were re-counted
  (500 / 500 partitions, 272 `input_image` blocks, 1000 and 728 unique ids, no dropped field leaked).
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

Rollout collection has **not** been run while writing this README. The intended no-tools command,
matching the published setting (reasoning effort high, no temperature, no token cap, one repeat),
is:

```bash
gym eval run --benchmark hle_diamond --model-type openai_model \
    ++policy_model_name=<gateway model id> \
    ++responses_create_params.reasoning.effort=high \
    --num-repeats 1 --output results/hle_diamond_rollouts.jsonl
```

Add `--split benchmark` if the CLI asks for a split, and
`'+config_paths=[benchmarks/hle_diamond/judge_openai.yaml]'` to grade with the dedicated judge. Use
`--benchmark hle_diamond/text` for a text-only policy model.

## Sandboxed agents (with tools)

`hle_diamond/opencode` and `hle_diamond/pi` give the model sandboxed Python tools without search;
`hle_diamond/opencode_search` and `hle_diamond/pi_search` add Tavily search and page tools. All four
run the 728 text rows with the Explanation / Answer / Confidence instructions in a single user
message and one repeat; keep collection repeats at one. The published with-tools column used each
vendor's own harness with a different tool set, so these presets support a **comparison, not a
reproduction**, of that column.

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
  network policy, so a local Docker run needs `network_access: inherit` and its egress is **not**
  blocked. Label any such run accordingly.
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
redistribute it; the data is downloaded at run time and never committed. Prompts are reused from
`benchmarks/hle/prompts/` and are adapted from the HLE authors'
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) harness (MIT).
