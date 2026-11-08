# Visual coding under OpenCode

A generic multi-turn visual-coding environment: the OpenCode CLI, with a Python shell that can
crop, measure and plot images, answers a question about an image over as many tool turns as it
needs, and a GPT judge grades the final `\boxed{}` answer. Nothing in the environment is
dataset-specific. Rows carry the image and prompt in `responses_create_params`, plus `question`
and `expected_answer` for the judge.

Users of it:

- `benchmarks/babyvision_opencode`: BabyVision (388 puzzles) with a visual-reasoning instruction.
- `benchmarks/babyvision_opencode_question_only`: the same puzzles with only the question, for
  models trained on OpenCode's own system prompt.
- RL training sets in the same row format (e.g. synthetic visual-tracking puzzles: mazes,
  metro maps, tangled lines).

## Harness

`responses_api_agents/opencode_vision_agent` (OpenCode 1.17.11):

- The question image is written to the run directory as `image_0.<ext>` and attached with `-f`.
- The prompt is one user message: the visual-reasoning instruction, `---`, the question, and
  "Please answer the question and put the final answer within \boxed{}."
- Tools are OpenCode's own: bash, read, glob, grep, write, edit, todowrite. `python3` in the
  shell has pillow, numpy, scipy and opencv. `read` on an image returns it to the model as an
  image, which is how the model looks at the crops and plots it makes. webfetch is denied.
- There is no turn cap. Each task stops after 1800 s, and a timed-out task scores 0.
- `private_tmp` gives each task its own `/tmp`.

## Sandbox sessions

`configs/sandbox_sessions.yaml` runs the same rows, prompt and judge with `responses_api_agents/opencode_agent` in
a sandbox session (an episode that the Environment Server opens, runs and closes on the agent). The differences
from the harness above:

- Each task runs OpenCode in its own sandbox, not on the agent's host. The sandbox reaches only the model server, so
  the model cannot look answers up. `private_tmp` is not needed, because no two tasks share a sandbox.
- The prompt image is uploaded into the sandbox and attached with `opencode run --file`. The model entry declares
  image input, so `read` returns images to the model, as with `vision`.
- Gym's token capture records each model call. The config turns off automatic compaction and sets no step limit,
  because capture cannot join a compaction summary or a step-limit message to the scored calls. The same config
  serves evaluation and RL training.
- The OpenCode settings match the harness above (OpenCode 1.17.11, 1800 s per task, a 131072-token context window
  and 32768 output tokens), except that thinking is on, for a reasoning policy. Set `thinking: false` for a model
  without reasoning. The permissions allow every path and repeated tool calls, because
  `opencode run` ends the run with no answer when it rejects a request. `experimental.continue_loop_on_deny` is not
  needed. Subagents (`task`), `webfetch` and `websearch` are off.

The session needs the following:

1. Build the sandbox image from `sandbox_image/Dockerfile` and set `VISUAL_CODING_OPENCODE_SANDBOX_IMAGE` to it. The
   image has the same Python toolkit as `opencode_vision_agent` and OpenCode under `/opt/opencode/bin`.
2. Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the OpenSandbox provider.
3. Run on a host whose address the sandbox can reach. The config sets `use_absolute_ip`, and only the model servers
   advertise that address.

```bash
gym env start --config resources_servers/visual_coding_opencode/configs/sandbox_sessions.yaml \
    --model-type vllm_model ++policy_base_url=<url> ++policy_api_key=<key> ++policy_model_name=<model>
```

For BabyVision, use `benchmarks/babyvision_opencode/sandbox_sessions.yaml` or
`benchmarks/babyvision_opencode_question_only/sandbox_sessions.yaml` in place of this config.

## Scoring

The reward is a GPT judge's verdict (`gpt-4o-mini` at temperature 0 by default; any
OpenAI-compatible endpoint via `visual_coding_opencode_judge_base_url` and
`visual_coding_opencode_judge_model_name`, with the key in `OPENAI_API_KEY`). The judge sees the question text, the gold answer, string_match's extraction
(the last "answer:", else the last `\boxed{}`), and the last 4000 characters of the model's
text. It cannot see the image. Only `"equivalent"` scores 1; `"different"`, `"unsure"`, a judge
error and an empty response score 0. `string_match_reward` is kept as a diagnostic.

## BabyVision data

```bash
LMUData=/path/to/LMUData python benchmarks/babyvision_opencode/prepare.py
```

`prepare.py` reads BabyVision in VLMEvalKit's TSV format (`BabyVision.tsv`; set `LMUData` to its
directory, or `BABYVISION_TSV` to the file). Two copies are accepted:

- md5 `b3a93182…`
- md5 `4216333e…`

They have identical questions and answers. In the `b3a93182…` copy, 175 of the 388 images are
JPEG re-encodes of the `4216333e…` copy's PNGs.
