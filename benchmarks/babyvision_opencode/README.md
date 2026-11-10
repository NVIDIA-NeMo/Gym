# BabyVision under OpenCode

BabyVision (388 visual-reasoning puzzles: counting, coordinates, tracing paths) solved by
OpenCode with a Python shell that can crop, measure and plot the image. The environment is
the generic [`visual_coding_opencode`](../../resources_servers/visual_coding_opencode/README.md)
resources server, with `opencode_agent` running each task in its own sandbox session; this benchmark only adds the
data.

Each row is one user message: the image, a visual-reasoning instruction, `---`, the question,
and "Please answer the question and put the final answer within \boxed{}." A GPT judge grades
the final answer against the gold answer (see the resources server's README for the scoring).
`mean/reward` leaves out rows where the judge failed (a failed call, or a reply without a valid
verdict). Those rows go to the run's failures file, not into the average.

Scores are not comparable to the BabyVision leaderboard. The official scorer gives its judge
(GPT-5.2 in its run script) only the last `\boxed{}` content. This judge (`gpt-4o-mini` by
default) also reads the end of the model's text, so an answer without a box can score 1.

[`babyvision_opencode_question_only`](../babyvision_opencode_question_only/README.md) is the same
benchmark without the instruction.

## Prepare data

`prepare.py` downloads BabyVision from [`UnipatAI/BabyVision`](https://huggingface.co/datasets/UnipatAI/BabyVision)
on Hugging Face at a pinned revision and checks the file's SHA-256. The resources server's README (section
"BabyVision data") names the file and how each task becomes a row.

```bash
gym eval prepare --benchmark babyvision_opencode
```

On a host without internet access, download
[the pinned parquet file](https://huggingface.co/datasets/UnipatAI/BabyVision/resolve/849cbeea2b1d00ad334e9e1e2b49a1960c9ff3f3/data/train-00000-of-00001.parquet)
elsewhere and set `BABYVISION_PARQUET` to it.

## Run

The judge defaults to `gpt-4o-mini` on the OpenAI API with `OPENAI_API_KEY`; set
`visual_coding_opencode_judge_base_url` and `visual_coding_opencode_judge_model_name` to use
another OpenAI-compatible endpoint.

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark babyvision_opencode \
    ++output_jsonl_fpath=results/benchmarks/babyvision_opencode.jsonl \
    ++split=benchmark \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```

The policy must accept images. Each task runs OpenCode in its own sandbox, and Gym's token capture records the model
calls, so the same config serves RL training. The resources server's README (section "Run") lists what the sandbox
needs: the sandbox image, the OpenSandbox provider settings and an address that the sandbox can reach.
