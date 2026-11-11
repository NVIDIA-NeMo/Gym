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

- Each task runs OpenCode in its own sandbox. The resources server lends it: the episode's seed starts the sandbox
  and the episode's close stops it. With `sandbox_network_access: model_only`, the sandbox can reach only the policy
  model server's host, on every port. This keeps the model off the internet only on an OpenSandbox server that
  enforces egress at the IP layer (`dns+nft`). The config gives every other Gym server the address `127.0.0.1`, so the
  sandbox cannot reach the judge, the agent or the resources and Environment Servers. The sandbox size (2 CPUs, 8 GiB)
  is in the resources server's `sandbox_config`.
- Under NeMo RL, the sandbox can still reach Gym's head server. NeMo RL sets the head server's host to the node
  address, which is the policy host whose every port `model_only` opens. The head server's `/global_config_dict_yaml`
  returns the merged config, including any key passed literally. Other environments' servers in an RL blend that do
  not set `127.0.0.1` get the node address too. Allowing only the model server's port needs a change in Gym's core.
- The question image is uploaded into the sandbox and attached with `opencode run --file`. The agent sets
  `image_input: true`, so the model entry declares image input and `read` also returns images to the model. This is
  how the model looks at the crops and plots that it makes.
- The prompt is one user message: the visual-reasoning instruction, `---`, the question, and
  "Please answer the question and put the final answer within \boxed{}."
- Tools are OpenCode's own: bash, read, glob, grep, write, edit and todowrite. `python3` in the shell has pillow,
  numpy, scipy and opencv. Subagents (`task`), `webfetch` and `websearch` are off.
- The permissions allow every path and repeated tool calls, because `opencode run` ends the run with no answer when
  it rejects a request.
- There is no turn cap. Each task stops after 1800 s. The context window is 131072 tokens, with 32768 output tokens.
- `thinking: true` adds `opencode run --thinking`, which only shows reasoning in OpenCode's own output. The policy's
  serving setup decides whether the model reasons.
- The Environment Server admits 64 episodes at a time, the agent's concurrency, because each seed starts a sandbox.
  An episode has 5400 s, below the sandbox's `ttl_s` of 7200 s.
- Gym's token capture records each model call, so the same config serves evaluation and RL training. The config
  turns off automatic compaction, because capture cannot join a compaction summary to the scored calls. It sets no
  step limit, so a task ends at its answer, its timeout or the context limit.
- The images that `read` returns come between model calls. On NeMo RL's single controller the vLLM worker stages
  them, so training sees them. Gym's local capture records them only with #5837. Saved rollouts keep the prompt image
  in `responses_create_params`, but not the images that `read` returns: their tool outputs keep only text.
- The pairing is OpenCode-only: the instruction names OpenCode's tools, the benchmarks pair with `opencode_agent`, and
  the sandbox image ships only OpenCode. A Pi pairing is a follow-up.

## Run

The config needs the following:

1. Build the sandbox image from `sandbox_image/Dockerfile` and set `VISUAL_CODING_OPENCODE_SANDBOX_IMAGE` to it. The
   image has the Python toolkit, OpenCode under `/opt/opencode/bin` and OpenCode's plugin dependencies, which the
   config pins by SHA-256. The plugin dependencies come from the lockfile in `sandbox_image/opencode/`, a copy of
   `benchmarks/agentic_if/opencode/`.
2. Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the OpenSandbox provider.
3. Run on a host whose address the sandbox can reach. The config sets `use_absolute_ip`, and only the policy model
   server advertises that address. The egress rule of the lent sandbox allows that host only.

```bash
gym env start --config resources_servers/visual_coding_opencode/configs/visual_coding_opencode.yaml \
    --model-type vllm_model ++policy_base_url=<url> ++policy_api_key=<key> ++policy_model_name=<model>
```

The policy must accept images.

## Scoring

The reward is a GPT judge's verdict (`gpt-4o-mini` at temperature 0 by default; any
OpenAI-compatible endpoint via `visual_coding_opencode_judge_base_url` and
`visual_coding_opencode_judge_model_name`, with the key in `OPENAI_API_KEY`). The judge sees the question text, the
gold answer, the last `\boxed{}` of the model's text, and the last 4000 characters of that text. It cannot see the
image. The judge prompt marks the model's text as data, and only the judge's own verdict, the last one in its reply,
counts. Only `"equivalent"` scores 1. `"different"`, `"unsure"` and an empty response score 0. A run that does not
complete scores 0 without a judge call, even if its text holds an earlier `\boxed{}` guess: an `incomplete` run (a
timeout, an output cap or a context limit), or a `failed` run that the Environment Server's outcome policy still sends
to verify (a model 400, 413 or 422, an out-of-memory kill, or no outcome metadata). A judge failure (a failed call, or
a reply without a valid verdict) is not a score: the row comes back masked (`mask_sample`, `failure_kind:
judge_failed`), so RL leaves it out of the loss. Under NeMo RL's default `grpo.masked_reward_policy: include`, its 0.0
still counts in its group's baseline. `exclude` leaves it out there too.

## BabyVision data

```bash
gym eval prepare --benchmark babyvision_opencode
```

`prepare.py` downloads BabyVision from the [`UnipatAI/BabyVision`](https://huggingface.co/datasets/UnipatAI/BabyVision)
dataset (MIT license) at revision `849cbeea2b1d00ad334e9e1e2b49a1960c9ff3f3`: one parquet file,
[`data/train-00000-of-00001.parquet`](https://huggingface.co/datasets/UnipatAI/BabyVision/resolve/849cbeea2b1d00ad334e9e1e2b49a1960c9ff3f3/data/train-00000-of-00001.parquet),
with all 388 tasks. It checks the file's SHA-256 (`7f6ce0d1…`). On a host without internet access, download that
file elsewhere and set `BABYVISION_PARQUET` to it.

For a multiple-choice task, the question lists the options as `(A)`, `(B)`, … after `Choices:`, and the answer is
the option's letter. Every image becomes an RGB PNG data URL. The release stores PNGs, JPEGs and WebPs; decoding
them and saving them as PNG keeps every pixel.

## Example data

`data/example.jsonl` has five rows. The first three (`example_0` to `example_2`) are small synthetic puzzles. The
last two are BabyVision tasks 4624 (Spatial Perception, Count 3D blocks) and 487 (Visual Pattern Recognition, Logic
Patterns), taken from the same `UnipatAI/BabyVision` revision and written in `prepare.py`'s format.
`data/example_rollouts.jsonl` holds one OpenCode rollout of each row.
