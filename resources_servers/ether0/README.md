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

Retro-synthesis and oracle-solubility are graded by three chemistry models (solubility, purchasability, forward reaction). By default the resources server runs them in-process (`local_oracles: true`) and needs:

| File | SHA-256 | Setting |
| --- | --- | --- |
| `USPTO480k_model_step_400000.pt` (95 MB, linked from ether0's [`packages/remotes/README.md`](https://github.com/Future-House/ether0/blob/b77e0d8eb06f62c8528da72bff22d4c7fee11371/packages/remotes/README.md)) | `1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587` | `moltrans_model_path`, or `ETHER0_REMOTES_MOLTRANS_MODEL_PATH`; if unset, uses the cache described below |
| `zinc20.bloom` (2 GB, the `zinc20` catalog of `molbloom` 2.3.4) | `7c6939251f2c86b7ee996faa727478bb6ceeffc675ab27f7f7e3a181a288b766` | `zinc20_bloom_path`; if unset, `molbloom` downloads it to `~/.cache/molbloom` |

MolTrans downloads automatically on server startup if missing, including at an explicitly configured path.
Without a path, it uses `$XDG_CACHE_HOME/nemo-gym/ether0/USPTO480k_model_step_400000.pt`
(or `~/.cache/nemo-gym/ether0/USPTO480k_model_step_400000.pt` when `XDG_CACHE_HOME` is unset).
The download is locked across workers sharing that path, checked against the pinned SHA-256,
and published atomically. Existing files are checksum-checked before loading and are never overwritten
on mismatch. For offline jobs, stage the file and set `moltrans_model_path` in advance.
For cluster jobs, point this setting or `XDG_CACHE_HOME` at writable persistent storage visible to the container.
First use requires network access to download MolTrans and, if not already supplied, ZINC20.

The server refuses to start if downloading fails, an existing MolTrans file has the wrong checksum,
an explicitly configured ZINC20 file is missing, or a model fails its control query. Alternatives:

- set `ETHER0_REMOTES_API_BASE_URL` to a running `ether0-serve` (see the [ether0 repo](https://github.com/Future-House/ether0/)); the local oracles are then not loaded;
- set `local_oracles: false`; with no remote either, those rows are returned with `mask_sample=true` instead of a reward of zero.

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

See `scripts/prepare_ether0.py` to prepare the full dataset, or run the `ether0` benchmark: `gym eval prepare --benchmark ether0`.