# GDP.pdf Agent

A thin `simple_agent` subclass that embeds the source PDF into the model request at rollout time.

GDP.pdf rows reference their PDF by relative path (`verifier_metadata.pdf_relpath`) rather than
inlining it: the corpus averages ~46 pages per task, so base64 in the JSONL would make the dataset
unusable. On `run()` the agent resolves the path against `media_base_dir` and rewrites the first
user message's content to:

```
[{"type": "input_text",  "text": "<prompt>"},
 {"type": "input_text",  "text": "<document>...extracted text...</document>"},   # include_text
 {"type": "input_image", "image_url": "data:image/png;base64,..."},              # one per page
 ...]
```

This matches the published setup: parsed document text, with page images for models that accept
vision input.

## Config

| Field | Default | Meaning |
|---|---|---|
| `media_base_dir` | `resources_servers/gdp_pdf/data` | Base for resolving `pdf_relpath`, relative to the Gym root |
| `dpi` | 150 | Base page render resolution ([AA's GDP.pdf methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf)) |
| `min_dpi` | 72 | Floor `select_dpi` reduces toward under budget pressure |
| `max_total_image_tokens` | 200000 | Per-document image-token budget driving DPI reduction; `null` disables it (fixed `dpi`) |
| `max_pages` | `null` | Cap on pages per document; `null` renders every page |
| `include_text` | `true` | Attach extracted PDF text |
| `include_images` | `true` | Attach rendered page images |
| `strip_images_from_output` | `true` | Drop base64 from the saved rollout artifacts |

`max_steps: 1` — GDP.pdf is single-turn with no tools.

## Payload size

A fixed 150 DPI on every document would alone approach ~1M tokens for the longest (200-page) task,
so `dpi` is a *base*, not a fixed value: `select_dpi` reduces it -- toward `min_dpi` -- whenever a
document's estimated image-token cost would exceed `max_total_image_tokens`. When it reduces, the
agent logs a warning and records `verifier_metadata.rendered_dpi`, the same way `max_pages`
truncation records `pages_truncated` -- a lower-resolution render is visible in the rollout, not
silently mistaken for a model failure. `max_pages` remains available as a coarser, separate cap
(drop trailing pages entirely) for when even the DPI floor isn't enough.

`strip_images_from_output` keeps rollout JSONL readable; redaction is recorded as a
`multimodal_history_redacted` gap on the trajectory.

## Text-only models

Set `include_images: false` to run a text-only policy model. This is a config change, not a code
change — but scores are not comparable to the multimodal configuration, since layout, figures, and
scanned content are lost.
