# OpenCode Vision Agent

[`opencode_agent`](../opencode_agent/README.md) in a server venv that also carries pillow,
numpy, scipy and opencv. OpenCode's `bash` tool inherits the server's environment, so the
model's `python3` is this venv and can crop, measure and plot the task images.

Turn on the multimodal options of `opencode_agent`:

- `attach_images: true` writes the data-URL images of the last user message into the run
  directory as `image_<i>.<ext>` and attaches them with `-f`.
- `vision: true` declares the generated `model_server` model entry image-capable, so OpenCode
  sends the attachments and its `read` tool returns image files as images.

The model's shell is not sandboxed (`no_sandbox_runtime`): it runs with the agent server's
permissions.

Environment: `resources_servers/visual_coding_opencode` (benchmark example: `benchmarks/babyvision_opencode`).
