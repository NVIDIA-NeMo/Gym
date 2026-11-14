# ARC-AGI-2 (public evaluation set)

120 grid-transformation puzzles. Each test input is one row; the model must write the exact output grid. Three
configs share the rows and the `arc_code` resources server (exact match, reward 1 or 0):

| Config | Harness |
| --- | --- |
| `arc_agi_2` | No tools: `simple_agent` with `prompts/no_tools.yaml`; the model reasons and writes the grid in `<answer>` |
| `arc_agi_2/configs/arc_agi_2_tools` | `arc_code_agent`: digits plus one image per grid, Python through a native or text tool, auto-submitted answer once a transform passes every example; code runs in OpenSandbox sandboxes (`OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`) |
| `arc_agi_2/configs/arc_agi_2_tools_local` | The same harness with the code run in a local process on the agent host (development only) |

## Data

```bash
python benchmarks/arc_agi_2/prepare.py   # arcprize/ARC-AGI-2 data/evaluation (167 test inputs), commit in data/source.json
python benchmarks/arc_agi_2/prepare.py --challenges arc-agi_evaluation_challenges.json \
    --solutions arc-agi_evaluation_solutions.json   # ARC Prize 2026 Kaggle files (172 test inputs)
```

The two sources share the 120 task ids; the 2026 files add a test input to five tasks and differ in seven grids.
Report which one was used.

## Run and score

Four samples per test input (`num_repeats: 4`; do not pass `--num-repeats` as well, it multiplies):

```bash
gym eval run --benchmark arc_agi_2 --model-type vllm_model --model <name> --model-url <url>/v1 \
    --split benchmark --concurrency 32 --output outputs/arc_agi_2/rollouts.jsonl
python benchmarks/arc_agi_2/score.py --rows benchmarks/arc_agi_2/data/arc_agi_2_no_tools.jsonl \
    --rollouts outputs/arc_agi_2/rollouts.jsonl --output outputs/arc_agi_2/score.json
```

`score.py` scores the way ARC Prize does: a test input counts as solved when one of two attempts matches every cell,
a task scores the fraction of its test inputs solved, and the benchmark is the mean over tasks. With 4 samples it
reports pass@1 and the unbiased pass@2, plus the mean number of output tokens per rollout.

## Reference: Nemotron 3.5 Super

`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` (revision `a1ccf94`), served with vLLM 0.32.0 (TP4, MTP3 speculative
decoding, float32 Mamba SSM cache, `qwen3_coder` tool parser, `nemotron_v3` reasoning parser, 262,144-token context,
`--limit-mm-per-prompt '{"image":64,"video":1}'` for the grid images),
thinking on, temperature 1, top-p 1; ARC Prize 2026 evaluation files, 4 samples per test input.

| Config | pass@1 | pass@2 | Mean output tokens |
| --- | ---: | ---: | ---: |
| `arc_agi_2` (no tools) | 4.8% | 7.5% | 40,191 |
| `arc_agi_2_tools` (OpenSandbox) | 17.8% | 24.8% | 53,620 |
| `arc_agi_2_tools_local` | 17.4% | 23.5% | 53,311 |

For the tools runs the agent used `thinking_template_kwargs: {enable_thinking: true}`,
`final_template_kwargs: {enable_thinking: false}`, `native_tools: true`, `nocode_nudge: 1`.
