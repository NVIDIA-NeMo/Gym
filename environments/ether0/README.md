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

Retro-synthesis and oracle-solubility are graded by three chemistry models running in-process
(`local_oracles: true`). Missing MolTrans checkpoints and the default ZINC20 catalog download
on first use. See the [resources server setup](../../resources_servers/ether0/README.md)
for cache locations, pinned checksums, path overrides, offline setup, and remote-oracle options.

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
gym env start --environment ether0 --model-type vllm_model &

# wait for above to be ready
gym eval run --no-serve \
    --agent ether0_simple_agent \
    --input environments/ether0/data/example.jsonl \
    --output environments/ether0/data/ether0_rollouts.jsonl

tail -n 1 environments/ether0/data/ether0_rollouts.jsonl | jq | less
```

See `prepare.py` to prepare the full dataset, or run the `ether0` benchmark: `gym eval prepare --benchmark ether0`.