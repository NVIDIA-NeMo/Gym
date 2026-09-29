# GraphWalks resources server

OpenAI's [GraphWalks](https://huggingface.co/datasets/openai/graphwalks)
long-context benchmark. Each task provides an adjacency list (often
massive) and asks the model either to:

- **parents**: list every parent of a target node, or
- **bfs**: list every node reachable at exactly depth N via BFS from a
  source node.

## Scoring

1. The model must end its response with a line of the form
   `Final Answer: [n1, n2, ...]`. If the format is missing,
   `parse_failed=True` and reward=0.
2. Otherwise reward is the **F1 score** between the predicted node
   set and the expected node set (continuous in [0, 1]):
   - both empty → 1.0
   - one empty (the other non-empty) → 0.0
   - else `2·P·R / (P + R)`

Grader ported from
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/graphwalks.py.

## Start environment

```bash
gym env start \
    --resources-server graphwalks \
    --model-type vllm_model
```

## Collect example rollouts

```bash
gym eval run --no-serve \
    --agent graphwalks_simple_agent \
    --input resources_servers/graphwalks/data/example.jsonl \
    --output resources_servers/graphwalks/data/example_rollouts.jsonl
```

For the full benchmark run see
[`benchmarks/graphwalks/README.md`](../../benchmarks/graphwalks/README.md).

## Licensing

- Code: Apache 2.0
- Data ([openai/graphwalks](https://huggingface.co/datasets/openai/graphwalks)): see upstream license
