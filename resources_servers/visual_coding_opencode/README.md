# Visual coding under OpenCode

A generic multi-turn visual-coding environment: OpenCode, with a Python shell that can
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

`responses_api_agents/opencode_agent` (OpenCode 1.17.11) runs each task in a sandbox session: the Environment
Server opens an agent session, sends the task and closes the session.

- Each task runs OpenCode in its own sandbox. The sandbox reaches only the model server, so the model cannot look
  answers up.
- The question image is uploaded into the sandbox and attached with `opencode run --file`. The model entry declares
  image input, so `read` also returns images to the model. This is how the model looks at the crops and plots that
  it makes.
- The prompt is one user message: the visual-reasoning instruction, `---`, the question, and
  "Please answer the question and put the final answer within \boxed{}."
- Tools are OpenCode's own: bash, read, glob, grep, write, edit and todowrite. `python3` in the shell has pillow,
  numpy, scipy and opencv. Subagents (`task`), `webfetch` and `websearch` are off.
- The permissions allow every path and repeated tool calls, because `opencode run` ends the run with no answer when
  it rejects a request.
- There is no turn cap. Each task stops after 1800 s. The context window is 131072 tokens, with 32768 output tokens.
- Thinking is on, for a reasoning policy. Set `thinking: false` for a model without reasoning.
- Gym's token capture records each model call, so the same config serves evaluation and RL training. The config
  turns off automatic compaction and sets no step limit, because capture cannot join a compaction summary or a
  step-limit message to the scored calls.

## Run

The config needs the following:

1. Build the sandbox image from `sandbox_image/Dockerfile` and set `VISUAL_CODING_OPENCODE_SANDBOX_IMAGE` to it. The
   image has the Python toolkit and OpenCode under `/opt/opencode/bin`.
2. Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the OpenSandbox provider.
3. Run on a host whose address the sandbox can reach. The config sets `use_absolute_ip`, and only the model servers
   advertise that address.

```bash
gym env start --config resources_servers/visual_coding_opencode/configs/visual_coding_opencode.yaml \
    --model-type vllm_model ++policy_base_url=<url> ++policy_api_key=<key> ++policy_model_name=<model>
```

The policy must accept images.

## Scoring

The reward is a GPT judge's verdict (`gpt-4o-mini` at temperature 0 by default; any
OpenAI-compatible endpoint via `visual_coding_opencode_judge_base_url` and
`visual_coding_opencode_judge_model_name`, with the key in `OPENAI_API_KEY`). The judge sees the question text, the
gold answer, string_match's extraction (the last "answer:", else the last `\boxed{}`), and the last 4000 characters
of the model's text. It cannot see the image. Only `"equivalent"` scores 1. `"different"`, `"unsure"` and an empty
response score 0. A judge failure (a failed call, or a reply without a valid verdict) is not a score: the row comes
back masked (`mask_sample`, `failure_kind: judge_failed`), so RL does not train on it as a wrong answer.
`string_match_reward` is kept as a diagnostic.

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
