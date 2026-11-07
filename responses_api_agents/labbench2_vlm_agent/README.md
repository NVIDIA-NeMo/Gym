# LABBench2 VLM Agent

Custom agent extending `simple_agent` for the `labbench2_vlm` resource server.

Overrides `run()` to resolve `verifier_metadata.media_dir` references, read
image/PDF files from disk, base64-encode them, and inject `input_image` blocks
into `responses_create_params` before sending the request to the model.

The media reaches the model only through `run()`, so the agent refuses agent sessions (a seed
answers 422) rather than asking about images the model never sees. Run it through a
`legacy_agent` environment server, as the shipped configs do.

See `resources_servers/labbench2_vlm/README.md` for full documentation.

## Config Fields

- `media_base_dir`: base directory for resolving media paths (relative to Gym root)
- `dpi`: DPI for PDF page rendering (default: 170)
- `strip_images_from_output`: remove image blocks from the response, observations, and `ng_trajectory` (default: true);
  `CaptureStore` retains the original request when model-call capture is enabled
