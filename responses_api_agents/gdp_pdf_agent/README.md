# GDP.pdf Agent

A thin `simple_agent` subclass that embeds a pre-rendered document into the model request at
rollout time, with a reactive DPI backoff for policy models that reject oversized requests.

GDP.pdf rows reference their document by a prepare-time manifest
(`verifier_metadata.document_manifest`) rather than inlining anything: the corpus averages ~46
pages per task, so base64 in the JSONL would make the dataset unusable. The manifest (page text +
one 150 DPI PNG screenshot per page) is produced **once**, at prepare time
(`benchmarks/gdp_pdf/prepare.py`'s `prepare_document()`, via LiteParse OCR), not on every rollout
— GDP.pdf's 100 source PDFs are static, so re-parsing the same documents on every row, every
repeat, every model would redundantly repeat the same expensive OCR pass. This agent just reads
the cached manifest and resizes the cached page PNGs for its DPI backoff, rather than
re-rendering from the PDF.

On `run()` the agent resolves the manifest against `documents_base_dir` and rewrites the first
user message's content to:

```
[{"type": "input_text",  "text": "<prompt>"},
 {"type": "input_text",  "text": "<document_delivery>...compositing/coverage notice...</document_delivery>"}, # when needed
 {"type": "input_text",  "text": "<document>...extracted text...</document>"},   # include_text
 {"type": "input_image", "image_url": "data:image/png;base64,..."},              # one per page/batch
 ...]
```

This matches the published setup: parsed document text, with page images for models that accept
vision input. A manifest's page image paths must stay inside the manifest's own directory (as `prepare.py`
writes them, `pages/<file>`); the agent refuses a manifest that names an image anywhere else.

## Config

| Field | Default | Meaning |
|---|---|---|
| `documents_base_dir` | `resources_servers/gdp_pdf/data` | Base for resolving `document_manifest`, relative to the Gym root |
| `dpi` | 150 | Starting DPI for the backoff. Must match the DPI documents were rendered at during prepare (`source_dpi` in manifest.json) -- pages can only be resized down from the cached screenshot, never rendered fresh at a higher DPI |
| `min_dpi` | 72 | Floor DPI the reactive backoff will reduce to |
| `max_images` | `null` | Cap on images per request; when exceeded, multiple pages composite into one labeled image (up to 4 pages/image) instead of dropping pages. `null` disables compositing unless the policy model itself reports an image-count limit |
| `max_pages` | `null` | Cap on pages read from the manifest per document; `null` uses every page |
| `include_text` | `true` | Attach the manifest's extracted text |
| `include_images` | `true` | Attach the manifest's rendered page images |
| `strip_images_from_output` | `true` | Drop base64 from the saved rollout artifacts |
| `regenerate_on_incomplete_output` | `false` | Regenerate an incomplete answer with a smaller image footprint instead of keeping it as the result. Only for reproducing the earlier behaviour; counted in `document_delivery.output_truncation_regenerations` |
| `mask_failed_attempts` | `false` | Training only: mark attempts that end without a gradable answer with `mask_sample` (see below) |

`max_steps: 1` — GDP.pdf is single-turn with no tools.

## Reactive DPI backoff and page compositing

Matches [Artificial Analysis's GDP.pdf methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf),
which publishes the DPI endpoints (150, 72) and describes page compositing under tight per-request
image-count limits — not a decrement schedule or a token-budget estimate. Every document starts at
`dpi` (150, matching the manifest's cached resolution). If the policy model's response signals a
problem, the agent reacts and retries:

- **An explicit rejection** (`ClientResponseError`) is classified by `_input_limit()` into
  `image_count` (server reported a numeric image cap — widen compositing, up to 4 pages/image),
  `payload`/`context` (request or context too large — reduce DPI by 20%, toward the `min_dpi`
  floor), or `None` (unclassified — see below).
- **An incomplete answer** (`status == "incomplete"`, no exception raised) is the row's result: the
  model spent its output budget, while an input that does not fit is rejected up front. The answer
  goes to the verifier as is and is not generated again, since regenerating would re-sample a model
  result. `regenerate_on_incomplete_output: true` restores the earlier behaviour for comparison:
  reduce DPI and regenerate while images are part of the payload.
- **Auth/rate-limit errors** (401/403/429) are systemic, not per-document, so they propagate
  instead of being retried — every other row would fail identically.
- **Any other model error** (an unclassified HTTP error such as a 5xx with a body, or a transport
  failure) is not a limit of the document: `run()` raises its typed model failure, so the row is
  masked and collection retries it within its budget, instead of scoring it 0.

When compositing is active, the agent adds a `document_delivery` notice to the model input stating
how many labeled pages each image may contain. If the four-pages-per-image ceiling cannot fit every
page within the endpoint's image cap, the notice also gives the last page covered by images and
explains that later pages remain in the extracted text.

`document_delivery` in each rollout row records the DPI/compositing actually used, plus every
rejected attempt along the way. It includes `image_pages_covered`, `image_pages_omitted`, and
`image_coverage_end_page`, so a degraded or failed delivery is visible rather than silently mistaken
for a model failure. Two counts summarize the rejected attempts: `input_limit_rejections` (requests
the endpoint rejected for their input size, each answered by a smaller delivery) and
`output_truncation_regenerations` (nonzero only with `regenerate_on_incomplete_output`). The row
also carries them as `agent_adaptations`, which the run's failure facts sum per run under
`failures.agent_adaptations`. In an agent session, gdp_pdf's `verify()` writes both fields from the response's
metadata (see [Agent sessions](#agent-sessions)). `strip_images_from_output` keeps rollout JSONL readable; redaction is recorded
as a `multimodal_history_redacted` gap on the trajectory.

## Failed attempts in training

An attempt can end without a gradable answer when the input still does not fit after the backoff
(a context, image-count or payload limit). It then scores 0 and carries a `failure_reason`.
Evaluation should count that zero, so `mask_failed_attempts` is off by default. For training the
zero says nothing about an answer, so set `mask_failed_attempts: true`: such attempts get
`mask_sample: true` and a `failure_kind` (`model_context_limit` for a context limit,
`model_request_error` for an image-count or payload limit), and the trainer leaves them out of the
loss. A failed policy call or `/verify` is masked in either mode, and rows whose judge call failed
are already masked by `judge_failsafe`. In an agent session the Environment Server returns the row, so the agent
marks the attempt in its delivery record and gdp_pdf's `verify()` sets the same `mask_sample`, `failure_kind` and
`failure_reason`.

## Agent sessions

Under an agent session (`single_agent_turn` or `single_agent_turn_legacy`), the agent reads the manifest
from the row's `agent_context.agent` (`artifacts.document_manifest`, written by `benchmarks/gdp_pdf/prepare.py`)
and resolves it under `documents_base_dir` when the session is seeded. A row without `agent_context`, a
missing, unreadable or malformed manifest, a manifest whose page images leave its directory, or a manifest
rendered at a DPI other than `dpi` is refused with HTTP 422, so a session never answers without its document.
The manifest is resolved through symlinks and refused when it lands outside `documents_base_dir`: when the
documents live on other storage behind a symlink (for example `data/documents -> /shared/gdp_pdf/documents`),
set `documents_base_dir` to the real directory the manifest paths are relative to (`/shared/gdp_pdf`);
otherwise every session is refused while `run()` still works. The activation builds the same request as `run()`
and runs the same backoff:

- In a session, Simple Agent returns a model HTTP error as a failed response instead of raising it. The agent
  reads the input limit from that response and retries with a smaller delivery, all within the one activation.
- A document that does not fit even as text returns an empty failed response without an outcome, which the
  Environment Server verifies as a zero attempt, as `run()` scores it. gdp_pdf's `verify()` reads the delivery
  record and adds the row's `failure_reason`, and with `mask_failed_attempts` its `mask_sample` and `failure_kind`.
- Any other model error (a 5xx that is not an empty-body crash, 401, 403, 429) stays the
  `model_request_error` failure Simple Agent reports, for the Environment Server's outcome policy to decide;
  `run()` raises the same typed failure.
- An empty-body 5xx is taken for an oversized request and backed off like one, since that is how vLLM fails
  under an oversized multi-image prompt (Gym's model server relays the empty body). If it persists through
  text only, it may be an outage rather than a limit, so the session returns the last `model_request_error`
  failure for the outcome policy instead of the zero attempt `run()` scores.
- The delivery record goes into the response's `document_delivery` metadata as a JSON string, with
  `pages_truncated` in it. gdp_pdf's `verify()` copies it to the row's `document_delivery` and its counts to
  `agent_adaptations`, so a session's row carries the fields a `/run` row does.
- Verification and its failures belong to the Environment Server, including `skip_verification`.

## Text-only models

Set `include_images: false` to run a text-only policy model. This is a config change, not a code
change — but scores are not comparable to the multimodal configuration, since layout, figures, and
scanned content are lost.
