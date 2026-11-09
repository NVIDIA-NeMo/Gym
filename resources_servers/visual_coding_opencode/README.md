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

## Scoring

The reward is a GPT judge's verdict (`gpt-4o-mini` at temperature 0 by default; any
OpenAI-compatible endpoint via `visual_coding_opencode_judge_base_url` and
`visual_coding_opencode_judge_model_name`, with the key in `OPENAI_API_KEY`). The judge sees the question text, the gold answer, string_match's extraction
(the last "answer:", else the last `\boxed{}`), and the last 4000 characters of the model's
text. It cannot see the image. Only `"equivalent"` scores 1. Received invalid or unsure verdicts and
empty policy answers score zero. Failed judge calls use the shared retry policy, then enter the
failures sidecar and are excluded from accuracy until recovered. `string_match_reward` is kept as a
diagnostic.

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
