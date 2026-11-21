# Code-assisted VQA and Agent-X (AGA original training)

The original training rows and answer graders of the AGA Code-assisted VQA and Agent-X tasks, run under five native
harnesses through `resources_servers/aga_visual_tasks/configs/multiharness.yaml`. No public VTC/VStar data replaces
the training rows.

Original Gym: https://github.com/katerynaCh/Gym/tree/168e95ee80780737273dda011d31d1d5c58f35ce (Apache-2.0).
`visual_prefix.txt` is copied unchanged from that revision's
`responses_api_agents/opencode_multimodal_agent/prompts/visual_prefix.txt`. The private data export has no verified
dataset-license declaration, so no private rows or images are committed, and a train descriptor would use the Gym `TBD`
value. Code-assisted VQA is the rows with `source_task=opencv_visual_question_answering`; Agent-X is the rows with
`metadata.category=agentx`. Pass a preselected original subset; preparation rejects other categories.

## Prepare and run

```bash
python benchmarks/aga_visual_tasks/prepare.py --source ORIGINAL.jsonl --output OUT_DIR \
  --family code_assisted_vqa --image-root ORIGINAL_IMAGES --harness pi
# Claude Code takes only text in its prompt, so it reads the files:
python benchmarks/aga_visual_tasks/prepare.py ... --harness claude_code --file-input
```

`--harness` is one of `opencode`, `pi`, `codex`, `claude_code`, `openclaw`; each row routes to `aga_<harness>_agent` and
`aga_<harness>_environment_server`. Set `AGA_IMAGE_ROOT` to the original image root and `AGA_SANDBOX_IMAGE` to an image
with the harness runtimes, Python3/Pillow/numpy/scipy/OpenCV and a `/images` directory the sandbox user can write
(otherwise every seed fails with "Sandbox write_file failed"). The lent sandbox reaches only the policy model, so the
OpenCode, Pi, Codex and OpenClaw runtimes must already be in the image; Claude Code needs nothing there, because its
agent host uploads the pinned binary. Only public image bytes are lent; gold and grader fields stay outside the sandbox.
Do not blend `multiharness.yaml` with a config that edits a stock agent block, such as `benchmarks/hle/pi.yaml`: the
`aga_<harness>_agent` blocks inherit the stock ones, so they would take its settings.
`--include-image-paths` appends the original `/images` paths to the user message as a separate input-delivery protocol;
`--file-input` implies it.

## What differs from the original training task

- **Answer grading only.** The original task added an auxiliary reward for the model's shell calls (bonuses for correct
  calls, penalties for failed or repeated ones). That reward reads each shell call's exit code from the harness's
  observations, which the stock harness adapters do not report, so it is not part of this recipe. The reward is the
  original answer grader's.
- **No generation budget.** The original task allowed 8 model calls and then asked for a final answer. Each harness's own
  turn limit and the episode timeout bound a run here (`max_turns: 8` for Claude Code, `build.steps: 8` for OpenCode);
  Codex, Pi and OpenClaw have none.
- The native harnesses differ in how they pass images; see `prepare.py`.

This is a recipe over the stock harnesses, not a claim of training parity with the original task.
