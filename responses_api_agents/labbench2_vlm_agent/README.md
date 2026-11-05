# LABBench2 VLM Agent

Custom agent extending `simple_agent` for the `labbench2_vlm` resource server.

Overrides `run()` to resolve `verifier_metadata.media_dir` references, read
image/PDF files from disk, base64-encode them, and inject `input_image` blocks
into `responses_create_params` before sending the request to the model.

Agent sessions read the media from the row's `agent_context` (`artifacts.media`, `values.tag`), embed it in
the session's episode, and remove the images from the session's observations and trajectory. A session whose row
has no `agent_context`, or whose media directory is missing under `media_base_dir`, is refused with HTTP 422.

See `resources_servers/labbench2_vlm/README.md` for full documentation.

## Config Fields

- `media_base_dir`: base directory for resolving media paths (relative to Gym root)
- `dpi`: DPI for PDF page rendering (default: 170)
- `strip_images_from_output`: remove image blocks from the response, observations, and `ng_trajectory` (default: true);
  `CaptureStore` retains the original request when model-call capture is enabled
