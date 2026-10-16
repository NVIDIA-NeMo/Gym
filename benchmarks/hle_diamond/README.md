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

Upstream's prompts are reused unchanged from `benchmarks/hle/prompts/`, so no prompt file lives in
this directory:

- `default.yaml`: the `centerforaisafety/hle` system prompt (Explanation / Answer / Confidence) with
  the bare question as the user turn. The default set bakes it into `responses_create_params.input`
  with `nemo_gym.vision_input.build_image_input`; the text variant applies it at rollout time.
- `opencode.yaml`: the same instructions folded into one user message for the sandboxed harnesses,
  whose native system prompts stay intact.
- `judge.txt`: the official HLE judge prompt.

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
| `responses_create_params` | default set only: the materialized `input` messages; image rows carry an `input_image` block with `detail: high` |

`prepare.py` loads the whole split, validates it (1000 rows, 500 + 500 partitions, 272 image rows,
1000 unique ids) and only then writes, so an incomplete download never leaves a file behind.
`--limit N` (positive integers only; rejected at parse time otherwise) is the one supported way to
prepare a subset, and it says loudly that the completeness check was skipped. `prepare_text.py` is a
thin wrapper for `prepare(text_only=True)`.

## Scoring

`reward` is 1.0 when the judge answers `Judgement: yes` and 0.0 otherwise: `equivalence_llm_judge`
with the HLE labels, no swap check, no regex extraction and no per-record regex, exactly as
`benchmarks/hle/config.yaml`. An empty final answer scores 0 without a judge call
(`failure_reason: empty_final_answer`). The judge defaults to the policy model, as in
`benchmarks/hle`, so the model under test grades itself unless `judge_openai.yaml` is composed in.

Headline metric: `mean/reward`, accuracy over every evaluated row. Additional metrics:

- `slices/partition/reasoning/mean/reward` and `slices/partition/knowledge/mean/reward`: accuracy per
  partition, also promoted to `key_metrics`, with `question_count` and `rollout_count` next to each
  and `slices/partition/missing_label_rollouts` counting rows without a label (never assumed). They
  come from the judge server's `report_slice_fields: [partition]` knob, which is off by default
  everywhere else. Open PR [#4658](https://github.com/NVIDIA-NeMo/Gym/pull/4658) adds `category`,
  `raw_subject` and modality slices under `domains/...` for HLE; the two hooks are additive and the
  fields this benchmark carries are the ones that PR reads.
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
  image count / duplicated id, `--limit` boundaries, no download on invalid arguments), config
  discovery and composition, the judge prompt path, manifests, and the partition slices through
  `compute_aggregate_metrics`. The count check and the slice hook were mutation-checked: breaking
  either makes the tests fail.
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
`hf_token` (a `huggingface-cli login` token is not read by `gym eval prepare`).

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
  and URL prefixes by `build_search_exclusions.py`. Deployments requiring an additional policy must
  supply their combined policy file through the Tavily resource's `exclude_domains_file_path`; keep
  private policies outside the public repository. The list reduces answer exposure but cannot
  guarantee that every answer-bearing page is excluded.

## Tests

```bash
pytest tests/unit_tests/test_hle_diamond.py resources_servers/equivalence_llm_judge/tests/test_app.py
```

## Licensing

Code: Apache 2.0.

`cais/hle-diamond` data: MIT, with the dataset card's request not to publicly share, re-upload or
redistribute it; the data is downloaded at run time and never committed. Prompts are reused from
`benchmarks/hle/prompts/` and originate from the HLE authors'
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) harness.
