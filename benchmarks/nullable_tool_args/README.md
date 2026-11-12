# Nullable tool args

Measures how often a model writes JSON `null` for required nullable tool arguments instead of the string `"None"`. See `resources_servers/nullable_tool_args/README.md`.

# Prepare data
```bash
gym eval prepare --benchmark nullable_tool_args
# or: python benchmarks/nullable_tool_args/prepare.py --num-samples 5000 --seed 0
```

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark nullable_tool_args \
    --output results/benchmarks/nullable_tool_args.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <>
```
