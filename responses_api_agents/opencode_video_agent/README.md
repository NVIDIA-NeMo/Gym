# Optional sandboxed OpenCode video

This subclass reuses the stock sandbox agent's run loop, model routing, native
OpenCode prompt/compact, verification and teardown. It adds private source-video
binding, a SHA256-pinned self-contained runtime bundle, the `watch_video` and
`view_image` plugin/provider, and media-preserving transcript export. It never
puts gold labels or real host paths into the user prompt.

The runtime must contain stock OpenCode 1.17.11, its pinned JS dependencies,
Python 3.13.14, FFmpeg/FFprobe and the lightweight media CLI. The bundle carries
the four NeMo-RL media modules (contract, archive, tools, CLI) under their
`nemo_rl` namespace; this is not a host training-stack dependency.

Configure dedicated sandbox workdir, source roots, bundle hash, matching media
profile/context/output limits, and `execution_failure_reward_zero: true`.
Optional: `opencode_model` (`<provider>/<model>`) renames the stock provider and
model, which OpenCode's system prompt quotes; `media_root` puts bound videos and
the media tools' workspace in a directory containing the workdir, so the video
can sit outside OpenCode's working directory; `media_tool_timeout_seconds` bounds
one visual tool call.
Use the optional `vllm_video_model` model adapter and validated serving support.
No custom compact prompt or pre-request token-counting endpoint is installed.

CPU tests are not real GPU/sandbox acceptance. Source mutation is detected after
execution and scored as an execution failure; this is not an OS-level guarantee
that arbitrary terminal commands cannot write files. Use an appropriately
unprivileged/isolated image for actual evaluation. Media capture is private and
opt-in. No data, binary bundle, credentials or checkpoint is included in Git.
