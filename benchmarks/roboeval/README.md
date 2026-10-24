# RoboEval (CodeBotler)

[RoboEval](https://github.com/ut-amrl/codebotler) evaluates code generation for service mobile robots. It has 16 tasks
with 5 prompt paraphrases each, for example greeting people in every lab or delivering an item. The model
writes `task_program()` against a small robot API (`get_current_location`, `get_all_rooms`, `is_in_room`, `go_to`,
`ask`, `say`, `pick`, `place`). See [`resources_servers/roboeval`](../../resources_servers/roboeval/README.md) for how
programs are scored.

**Evaluation only.** RoboEval's license (BSD-3-Clause with a machine-learning restriction) forbids using its contents as
training data. `prepare.py` writes the rows locally, and `data/.gitignore` keeps them out of version control.

## Prepare data

```bash
gym eval prepare --benchmark roboeval
```

`prepare.py` and the resources server fetch CodeBotler at a pinned commit into `resources_servers/roboeval/.codebotler`
(git and network required) unless `ROBOEVAL_ROOT` points at a checkout. The prompts are written into the rows
(`prompt_source: prepared`), because Gym prompt templates cannot express the few-shot conversation. Preparing writes `data/roboeval_benchmark.jsonl` (80 rows: 16 tasks x 5 paraphrases) and `data/example.jsonl` (the first
paraphrase of each task). Each row holds CodeBotler's default chat prompt, a system message with the robot API followed
by few-shot examples (`code_generation/openai_chat_completion_prefix.py`), plus the task paraphrase. Sampling follows
RoboEval's defaults (temperature 0.2, top-p 0.95), with `max_output_tokens` raised from 512 to 16384 so reasoning models
can finish.

## Run

```bash
gym env start --benchmark roboeval --model-type vllm_model
gym eval run --no-serve --agent roboeval_simple_agent \
    --input benchmarks/roboeval/data/roboeval_benchmark.jsonl --num-repeats 20 \
    --output results/roboeval.jsonl ++observability_enabled=true ++model_call_capture_dir=$PWD/results/model_calls
```

RoboEval reports pass@1 over 20 samples per prompt: `pass@1[avg-of-20]/accuracy` in the aggregate metrics.

## Known issue in RoboEval's chat prompt

FindBackpack, GetDrink, MailDelivery, and StaplerDelivery need `pick()`/`place()`, which CodeBotler's chat prompt does
not document. Chat models practically never call them: Nemotron-3-Nano 0 of 400 samples, and CodeBotler's own saved
o3-mini run 0 of 400. The prompt is kept verbatim for comparability, so these four tasks score near 0 for chat models.

## Differences from RoboEval's harness

- CodeBotler's chat path stops generation at `"\n#"`, `"\nclass"`, and a code fence. Here the model answers freely, and
  the verifier takes the last fenced code block, so a reasoning model's thinking and comments cannot truncate its program.
- The simulation timeout is 5 s instead of 1 s, because RoboEval's 1 s includes interpreter start-up and misfires under
  concurrent load. On CodeBotler's saved GPT-4 completions, both settings reproduce RoboEval's stored verdicts exactly.
