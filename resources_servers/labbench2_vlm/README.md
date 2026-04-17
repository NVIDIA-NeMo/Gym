# LABBench2 VLM Resources Server

## Overview

Evaluates vision-language models on scientific figure and table question answering
from [LABBench2](https://huggingface.co/datasets/EdisonScientific/labbench2).
The model receives a scientific figure or table alongside a free-form question.
Verification uses an LLM-as-judge to compare the model's answer against the gold
answer. Binary reward (1.0 = equivalent, 0.0 = not equivalent).

- Task type: single-turn VLM QA
- Domain: `knowledge`
- Subtasks: `figqa2-img`, `figqa2-pdf`, `tableqa2-img`, `tableqa2-pdf`
- Grading: LLM judge with configurable equivalence labels (`[[A=B]]` / `[[A!=B]]`)

## Server Composition

Use LABBench2 VLM with:

- `responses_api_agents/labbench2_vlm_agent` — custom agent (see below)
- `responses_api_models/*` — policy model (e.g. `openai_model`)
- `resources_servers/labbench2_vlm` — this resource server
- A separate judge model instance (configured via `judge_model_server`)

### Custom Agent: `labbench2_vlm_agent`

JSONL rows are lightweight (text-only, no base64). The custom agent
`labbench2_vlm_agent` extends `simple_agent` and overrides `run()` to embed
images/PDFs from disk before sending the request to the model. It imports
`embed_media_into_row` from `prepare_data.py` and resolves
`verifier_metadata.media_dir` against the configured `media_base_dir`.

`pymupdf` (for PDF rendering) is only required in the agent's venv
(`responses_api_agents/labbench2_vlm_agent/requirements.txt`), not in the
resource server or core `nemo_gym`.

## Dataset Format

JSONL rows are **lightweight** — they contain text and a media reference, not
base64 images. The `labbench2_vlm_agent` resolves and embeds media at rollout time.

Each JSONL row contains:

- `responses_create_params.input[0].content`: an `input_text` block with the
  question (no images — those are injected by the agent).
- `verifier_metadata.ideal`: gold answer string.
- `verifier_metadata.tag`: subtask identifier (e.g. `figqa2-img`, `tableqa2-pdf`).
- `verifier_metadata.id`: unique task identifier.
- `verifier_metadata.media_dir`: relative path to the directory containing the
  image or PDF for this question (e.g. `media/figs/imgs/<uuid>` or
  `test_media/figs/imgs/<uuid>`), resolved against the agent's `media_base_dir`.

See `data/example.jsonl` for concrete examples.

### Validation splits

| Dataset | File | Tasks |
|---------|------|-------|
| figqa2-img | `data/figqa2_img_validation.jsonl` | Figure QA (image input) |
| figqa2-pdf | `data/figqa2_pdf_validation.jsonl` | Figure QA (PDF-rendered input) |
| tableqa2-img | `data/tableqa2_img_validation.jsonl` | Table QA (image input) |
| tableqa2-pdf | `data/tableqa2_pdf_validation.jsonl` | Table QA (PDF-rendered input) |

### Media storage

Media files (images and PDFs) live in `data/media/` (gitignored, downloaded by
`prepare_data.py`). A small subset for smoke tests lives in `data/test_media/`
(committed to git).

```
data/
  media/                                  # gitignored, full dataset
    figs/imgs/<uuid>/figure.png
    figs/pdfs/<uuid>/paper.pdf
    tables/imgs/<uuid>/table.png
    tables/pdfs/<uuid>/paper.pdf
  test_media/                             # committed, 5 examples
    figs/imgs/<uuid>/figure.png
    ...
  example.jsonl                           # points to test_media/
  figqa2_img_validation.jsonl             # points to media/
```

## Key Config Fields

- `judge_model_server`: reference to the judge model (type `responses_api_models`).
- `judge_responses_create_params`: parameters forwarded to the judge model.
- `judge_prompt_template_fpath`: path to the judge prompt template (default:
  `prompt_templates/judge.txt`). Placeholders: `{question}`, `{expected_answer}`,
  `{generated_answer}`.
- `judge_equal_label` / `judge_not_equal_label`: verdict labels the judge must
  output. Defaults: `[[A=B]]` / `[[A!=B]]`.
- `judge_endpoint_max_concurrency`: semaphore limit for concurrent judge calls
  (default: 64). Set to `null` to disable.
- `media_base_dir` (on the agent): base directory for resolving `media_dir`
  references, relative to the Gym root (default: `resources_servers/labbench2_vlm/data`).
- `dpi` (on the agent): DPI for PDF page rendering (default: 170).

## Metrics

`compute_metrics` produces:

- **Overall**: `pass@k/accuracy`, `pass@1[avg-of-{k}]/accuracy`, per-sample aggregates.
- **Per-tag breakdown**: the same metrics namespaced by subtask tag
  (e.g. `figqa2-img/pass@1/accuracy`, `tableqa2-pdf/pass@1/accuracy`).
- `judge_no_verdict`: fraction of tasks where the judge returned neither label.

`get_key_metrics` surfaces: `mean/input_tokens`, `mean/output_tokens`, and
the highest-k values for `pass@1[avg-of-{k}]/accuracy` and `pass@{k}/accuracy`.

## Data Download

Media files (scientific figures and table images/PDFs) are downloaded from a
public GCS bucket by `prepare_data.py`. Validation JSONL can also be
downloaded via `gitlab_identifier` (for internal NVIDIA users) using
`ng_prepare_data +should_download=true +data_source=gitlab`.

Media files must always be downloaded separately via `prepare_data.py` --
the `gitlab_identifier` mechanism only handles the lightweight JSONL files.

`data/test_media/` contains a small subset of 5 media files (2 figures,
2 PDFs, 1 table image) committed to git. Combined with `data/example.jsonl`,
this enables smoke tests immediately after `git clone` without any download.

## Preprocessing

`prepare_data.py` downloads media from GCS into `data/media/` and writes
lightweight JSONL (no base64 embedding). From the repository root:

```bash
python3 resources_servers/labbench2_vlm/prepare_data.py
python3 resources_servers/labbench2_vlm/prepare_data.py --example  # also populates test_media/
```

## Usage

### Judge setup

The judge model lives in a separate config (`configs/judge_model_openai.yaml`)
so a user who wants a non-OpenAI judge (e.g. a local vLLM) can drop that file
and supply their own `responses_api_models` instance named `judge_model` (see
`responses_api_models/vllm_model/configs/vllm_model.yaml` for the vLLM form).

Credentials go in `env.yaml` at the **repository root** (the parser loads
`$CWD/env.yaml` first, then falls back to `$PARENT_DIR/env.yaml`). The file
is gitignored — create it if missing.

```yaml
# env.yaml — policy and judge credentials
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: <your-api-key>
policy_model_name: openai/openai/gpt-5.2
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: <your-api-key>
judge_model_name: openai/openai/gpt-5-mini
```

`judge_model_openai.yaml` reads these three keys via
`${oc.select:judge_*,…}` — you can also override them inline on the CLI
(e.g. `+judge_base_url=… +judge_api_key=… +judge_model_name=…`) instead of
putting them in `env.yaml`.

Start the servers:

```bash
ng_run "+config_paths=[resources_servers/labbench2_vlm/configs/labbench2_vlm.yaml,resources_servers/labbench2_vlm/configs/judge_model_openai.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

Collect rollouts:

```bash
ng_collect_rollouts \
  +agent_name=labbench2_vlm_simple_agent \
  "+config_paths=[resources_servers/labbench2_vlm/configs/labbench2_vlm.yaml,resources_servers/labbench2_vlm/configs/judge_model_openai.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
  +input_jsonl_fpath=resources_servers/labbench2_vlm/data/figqa2_img_validation.jsonl \
  +output_jsonl_fpath=results/figqa2_img_rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 2048, reasoning: {effort: high}}"
```

`ng_collect_rollouts` writes sidecar files next to `output_jsonl_fpath`:

- `*_materialized_inputs.jsonl`
- `*_aggregate_metrics.json`

Run the example data for a quick smoke test (works immediately after `git clone`
because `data/test_media/` and `data/example.jsonl` are committed):

```bash
ng_collect_rollouts \
  +agent_name=labbench2_vlm_simple_agent \
  "+config_paths=[resources_servers/labbench2_vlm/configs/labbench2_vlm.yaml,resources_servers/labbench2_vlm/configs/judge_model_openai.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
  +input_jsonl_fpath=resources_servers/labbench2_vlm/data/example.jsonl \
  +output_jsonl_fpath=resources_servers/labbench2_vlm/data/example_rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 2048, reasoning: {effort: high}}"
```

`example_rollouts.jsonl` is gitignored (42MB+ due to embedded base64 in the
rollout output). Regenerate it locally with the command above.

### One-shot alternative

`ng_e2e_collect_rollouts` starts the server stack, preprocesses, and collects
rollouts in a single command (don't run `ng_run` separately). Input path and
agent ref are auto-derived from the dataset entry in the chained config
(`++split` picks which one):

```bash
ng_e2e_collect_rollouts \
  "+config_paths=[resources_servers/labbench2_vlm/configs/labbench2_vlm.yaml,resources_servers/labbench2_vlm/configs/judge_model_openai.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
  ++split=validation \
  ++output_jsonl_fpath=results/labbench2_vlm_validation.jsonl \
  +num_samples_in_parallel=16
```

For a fast smoke test, add `+limit=10 +num_repeats=1`.

## Throttling

Each in-flight sample fans out to one policy call + one judge call, so the
endpoints see roughly `2 × num_samples_in_parallel` concurrent requests.
On a hosted endpoint you'll likely hit rate limits or socket errors
(`Hit N global ClientOSError`) well before saturating your machine.

Cap concurrency with `+num_samples_in_parallel=<N>`:

```bash
ng_collect_rollouts ... +num_samples_in_parallel=16
```

Start around 16 and bump up if it holds.

## Licensing

Code: Apache 2.0
Dataset: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
(sourced from [EdisonScientific/labbench2](https://huggingface.co/datasets/EdisonScientific/labbench2))
