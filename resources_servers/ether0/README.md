# ether0 benchmark environment

[Benchmark](https://huggingface.co/datasets/futurehouse/ether0-benchmark) and [paper](https://arxiv.org/pdf/2506.17238).

325 chemistry reasoning questions across 14 task types. All answers are a molecule. Around 25 questions per task, including:

- Completing SMILES fragments
- Designing molecules adhering to molecular formula and functional group constraints
- Predicting reaction outcomes
- Proposing one-step synthesis pathways
- Editing the solubility of a molecule
- Converting IUPAC name to SMILES
- Answering multiple-choice questions about safety, ADME properties, BBB permeability, toxicity, scent, and pKa

Note that retro-synthesis and oracle-solubility require an additional verifier server (see `ether0-serve` in the [ether0 repo](https://github.com/Future-House/ether0/)).

## Quickstart 

Create `env.yaml`:
```
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: futurehouse/ether0
```

Start servers and collect rollouts
```bash
# start vllm and nemo gym servers
vllm serve futurehouse/ether0 & 
gym env start \
    --resources-server ether0 \
    --model-type vllm_model &

# wait for above to be ready
gym eval run --no-serve \
    --agent ether0_simple_agent \
    --input resources_servers/ether0/data/example.jsonl \
    --output resources_servers/ether0/data/ether0_rollouts.jsonl

tail -n 1 resources_servers/ether0/data/ether0_rollouts.jsonl | jq | less
```

See `scripts/prepare_ether0.py` to prepare the full dataset.