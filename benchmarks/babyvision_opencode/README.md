# BabyVision under OpenCode

BabyVision (388 visual-reasoning puzzles: counting, coordinates, tracing paths) solved by
OpenCode with a Python shell that can crop, measure and plot the image. The environment is
the generic [`visual_coding_opencode`](../../resources_servers/visual_coding_opencode/README.md)
resources server, with `opencode_agent` running each task in its own sandbox session; this benchmark only adds the
data.

Each row is one user message: the image, a visual-reasoning instruction, `---`, the question,
and "Please answer the question and put the final answer within \boxed{}." A GPT judge grades
the final answer against the gold answer (see the resources server's README for the scoring).

[`babyvision_opencode_question_only`](../babyvision_opencode_question_only/README.md) is the same
benchmark without the instruction.

## Prepare data

`prepare.py` reads BabyVision in VLMEvalKit's TSV format (`BabyVision.tsv`) and checks it is one
of the two known copies by md5.

```bash
export LMUData=/path/to/LMUData   # the directory holding BabyVision.tsv; or BABYVISION_TSV=/path/to/BabyVision.tsv
gym eval prepare --benchmark babyvision_opencode
```

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
