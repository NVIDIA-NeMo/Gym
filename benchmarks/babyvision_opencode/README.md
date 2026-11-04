# BabyVision under OpenCode

BabyVision (388 visual-reasoning puzzles: counting, coordinates, tracing paths) solved by the
OpenCode CLI with a Python shell that can crop, measure and plot the image. The environment is
the generic [`visual_coding_opencode`](../../resources_servers/visual_coding_opencode/README.md)
resources server with the [`opencode_vision_agent`](../../responses_api_agents/opencode_vision_agent/README.md)
harness; this benchmark only adds the data.

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

The policy must accept images, and the OpenCode CLI (1.17.11) must be installable on the agent
host. The model's shell runs with the agent server's permissions; each task gets its own `/tmp`
(`private_tmp`, which needs unprivileged user namespaces).
