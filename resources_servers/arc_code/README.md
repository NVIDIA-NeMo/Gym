# ARC code resources server

Verifier for ARC-style grid puzzles that a model solves by writing and running Python. It is paired with
`responses_api_agents/arc_code_agent`, which runs the multi-turn harness; this server only scores the final answer.

## Verification

`verify` reads the last assistant message of the response and takes the **last** `<answer> ... </answer>` block in it. The
block must hold one grid row per line, digits separated by single spaces (rows of equal length, no blank lines, single
digits only). The reward is `1.0` when the grid equals `expected_output` cell for cell (both are normalised to packed
digit text before comparing), otherwise `0.0`. A missing or unreadable block gives `0.0` with
`extraction_successful: false`. The server holds no state; the code sandbox lives in the agent.

Response fields beyond `reward`: `predicted_output` (the parsed grid or null) and `extraction_successful`.

## Row schema

```json
{"task_id": "puzzle-001",
 "examples": [{"input": [[0, 1]], "output": [[1, 0]]}],
 "test_input": [[0, 1]],
 "expected_output": [[1, 0]],
 "responses_create_params": {"input": [{"role": "user", "content": "arc"}]}}
```

Grids are `list[list[int]]` with digits 0-9. The agent builds the prompt from `examples` and `test_input` and ignores the
row's own `input`. `data/example.jsonl` holds three small puzzles.

## Running

```bash
gym env start \
    --config resources_servers/arc_code/configs/arc_code.yaml \
    --config responses_api_agents/arc_code_agent/configs/arc_code_agent.yaml \
    --model-type vllm_model
gym env test +entrypoint=resources_servers/arc_code
```

## Licensing

Example rows are puzzles from the public ARC-AGI-2 task collection (Apache 2.0, https://github.com/arcprize/ARC-AGI-2),
under neutral task ids.
