# GDP.pdf Agent

A thin `simple_agent` subclass that embeds a pre-rendered document into the model request at
rollout time, with a reactive DPI backoff for policy models that reject or truncate oversized
requests.

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
vision input.

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
- **A silent truncation** (`status == "incomplete"`, no exception raised) is treated the same as a
  size problem: some servers accept an oversized request and simply run out of output room instead
  of rejecting it outright. The agent reduces DPI and retries rather than keeping the truncated
  answer, as long as images are actually part of the payload.
- **Auth/rate-limit errors** (401/403/429) are systemic, not per-document, so they propagate
  instead of being retried — every other row would fail identically.
- **Any other exception** (an unclassified HTTP error, or a non-HTTP exception like an upstream
  library assertion) is treated as a terminal, per-document failure: the row scores 0 (an empty
  answer earns no rubric credit) instead of crashing the whole evaluation run.

When compositing is active, the agent adds a `document_delivery` notice to the model input stating
how many labeled pages each image may contain. If the four-pages-per-image ceiling cannot fit every
page within the endpoint's image cap, the notice also gives the last page covered by images and
explains that later pages remain in the extracted text.

`document_delivery` in each rollout row records the DPI/compositing actually used, plus every
rejected attempt along the way. It includes `image_pages_covered`, `image_pages_omitted`, and
`image_coverage_end_page`, so a degraded or failed delivery is visible rather than silently mistaken
for a model failure. `strip_images_from_output` keeps rollout JSONL readable; redaction is recorded
as a `multimodal_history_redacted` gap on the trajectory.

## Text-only models

Set `include_images: false` to run a text-only policy model. This is a config change, not a code
change — but scores are not comparable to the multimodal configuration, since layout, figures, and
scanned content are lost.
