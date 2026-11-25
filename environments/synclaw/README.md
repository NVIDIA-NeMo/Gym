# Description

BioNeMo-SynClaw is a multi-turn, tool-using
environment for chemical retrosynthesis planning. The agent is given a target molecule
(as a SMILES string) and must propose a synthesizable pathway using building-block
search, reaction-template search, and forward/reverse reaction simulation tools,
then submit a final route with `submit_pathway`.

The model must:
- Search for purchasable building blocks and applicable reaction templates
- Use `simulate_reaction`/`simulate_reverse_reaction` to explore and validate
  candidate steps
- Explain the proposed route in natural language
- Submit a final pathway (a sequence of reaction steps, or a single building block
  for direct purchase)

If exact synthesis of the target isn't found, the model should submit
the highest-similarity analogue it can verify while preserving the
largest recognizable scaffold and key functional groups.

This environment pairs the `synclaw` resources server (`resources_servers/synclaw`),
which hosts the synthesis tools and verifier, with the `synclaw_agent` agent harness
(`responses_api_agents/synclaw_agent`), which drives the multi-turn tool-calling loop.

# Example usage

Create an `env.yaml` file in the Gym root directory specifying
`policy_model_name`. For a hosted endpoint, also set `policy_base_url` and
`policy_api_key`. See the
[documentation](https://docs.nvidia.com/nemo/gym/latest/reference/configuration#local-configuration-envyaml)
for details.

## Building the BioNeMo-SynClaw database

The tools read one SQLite database that holds three things:

- **Reaction templates:** BioNeMo-SynClaw's 501 bundled reactions, the union of its SynFormer, PaRoutes,
  ORD and NVIDIA libraries.
- **Building-block stock:** the [eMolecules](https://www.emolecules.com/) orderable
  catalog plus BioNeMo-SynClaw's 37 common reagents (water, methanol, methyl iodide, ...), which purchasable
  catalogs omit but many reaction templates need.
- **Search index** over the building blocks.

Build the database before starting the servers. Build time, memory use and disk space depend on
the catalog release and your hardware:

```bash
python resources_servers/synclaw/prepare_db.py
```

The script clones BioNeMo-SynClaw at the revision pinned in `resources_servers/synclaw/pyproject.toml`, downloads
the eMolecules release, loads the templates, the eMolecules building blocks and the common
reagents, and builds the index. It needs `git` and `uv`. The result is written to
`resources_servers/synclaw/data/synclaw_emolecules.sqlite`, the resources server's default `db_path`,
so no override is needed.

By default it downloads the `2026-09-01` eMolecules release.
eMolecules publishes dated releases and may retire old ones; if the download fails, pass a newer
release date:

```bash
python resources_servers/synclaw/prepare_db.py --release YYYY-MM-DD
```

Downloads and intermediate files stay under `resources_servers/synclaw/.synclaw_build/` and, like the
database, are gitignored; don't commit or redistribute the eMolecules files.
Rebuild the database whenever the BioNeMo-SynClaw pin changes. To serve a different database, override
`++synclaw_resources_server.resources_servers.synclaw.db_path=/path/to/synclaw.sqlite`.

## Running servers

### vLLM replicas (recommended for throughput)

Use [vllm-router](https://github.com/vllm-project/router) to share inference load
while preserving useful prompt prefixes. The vLLM preset connects Gym to one
router endpoint and caps the run at 64 live rollouts. The reusable agent config
does not set a global rollout limit. Replica deployment remains external: start
identical vLLM replicas with prefix caching enabled using your usual serving tools.
Serve them with pinned BioNeMo-SynClaw's context limit,
`--max-model-len "$(bash environments/synclaw/max_model_len.sh)"`.

Gym's built-in `number_of_instances`/Ray Serve gateway distributes requests without
prefix-cache awareness. The router favors replicas that already cache a rollout's
prefix while balancing load, reducing repeated prefill of growing histories.

Install the patched router wheel built from commit
`e7dda77989fc75d878da8071031de58e519d2415` plus BioNeMo-SynClaw's
[`vllm_router_compat.patch`](vllm_router_compat.patch). Build it with
[`build_vllm_router_wheel.sh`](build_vllm_router_wheel.sh), a build helper for Slurm clusters
with container support. Its header shows the required container and output-directory settings.
The tested build includes load-accounting and cache-affinity fixes; the package
version `0.1.15` alone does not identify it. Install the wheel in the environment
used to run the router (once, before allocating GPUs):

```bash
uv pip install /path/to/patched/vllm_router-*.whl
bash environments/synclaw/router.sh --worker-urls http://127.0.0.1:810{0..7}
```

This example assumes eight existing local replicas on ports 8100–8107; replace
these with your own worker URLs **without `/v1`**. The router runs in the foreground.
In another terminal, start Gym with `policy_model_name` set in `env.yaml` to the replicas'
served model name. Remove any old `policy_base_url` override from `env.yaml`, or
set it to `http://127.0.0.1:8090/v1`:

```bash
gym env start --config environments/synclaw/vllm.yaml
```

`router.sh` accepts native router options, including `--host` and `--port`.
If the router runs on another host or port, override `++policy_base_url=http://router:8090/v1`.
All adapter workers must use that same router endpoint. The router balances
individual inference requests; it does not require sticky rollout assignments.
For a single replica or an already balanced endpoint, point `policy_base_url`
directly at that endpoint instead.

Keep the model's sampling, tool parser, and context limit configured for your deployment.
The agent announces pinned BioNeMo-SynClaw's context limit to the model for budget feedback, and
rollouts fail if the endpoint's `/v1/models` reports a different one. To serve another limit,
set the agent's `max_model_len` to match it, or to `null` to turn context feedback off.
Match `++num_samples_in_parallel` (or `gym eval run --concurrency`)
to how many rollout histories your replicas can keep cached.

The patched router accepts the agent's `reasoning_effort` values `low`, `medium`, `high` and
`max`.

### Hosted model endpoint

```bash
gym env start --environment synclaw --model-type openai_model
```

The agent drives the model server's `/v1/responses` endpoint and replays the model's full
output (reasoning items included) on every turn, so a rollout stays on-policy for training.

Model servers that downconvert to Chat Completions (`vllm_model`, `inference_provider`) return
reasoning items via the reasoning parser, so this works out of the box. Responses-native hosted
providers are different: they return no `reasoning` output item unless the request asks for
`reasoning.summary`, so reasoning tokens are billed but nothing is echoed back. Requesting a
summary is not supported on the downconverting path (`reasoning.summary` has no Chat Completions
representation), so it would have to be set per-environment rather than on the agent.

## Collecting rollouts

To evaluate a model on the ChEMBL-1000 test set, follow
[`benchmarks/synclaw_chembl_1k`](../../benchmarks/synclaw_chembl_1k/README.md).

Rollouts can be collected using the example dataset as follows:

```bash
gym eval run --no-serve \
    --agent synclaw_agent \
    --input environments/synclaw/data/example.jsonl \
    --output results/example_rollouts.jsonl \
    --limit 5
```

`data/example.jsonl` contains five ChEMBL-1000 example tasks. The corresponding model
responses, tool calls and verification scores are in `data/example_rollouts.jsonl`.

Each input row provides the task the agent seeds its session with:
- `question`: task instructions describing the target molecule and available tools
- `system_prompt`: system prompt for the agent
- `available_tools`: tools exposed to the model for this task
- `target_smiles`: the target molecule the submitted pathway is scored against

See the *Task sets* section of the main `README.md` in the
[BioNeMo-SynClaw](https://github.com/NVIDIA-BioNeMo/SynClaw) repository for generating input data.

# Verification Logic

The resources server scores the best accepted pathway retained by the agent from
`submit_pathway`. Its final molecule comes from the RDKit-generated reaction product
or a direct-purchase building block and is compared with `target_smiles`.

- **Reward = 1.0** for accepted pathways whose final molecule matches the target
  under BioNeMo-SynClaw's exact-match rules. These account for equivalent tautomers and allow
  compatible stereoisomers where the target leaves stereochemistry unspecified, while
  requiring agreement with stereochemistry the target specifies.
- **Reward = `max_similarity_reward` × similarity** for other accepted pathways.
  Fingerprint similarity `1.0` alone does not establish exact reconstruction.
  Similarity is BioNeMo-SynClaw's stereo-aware fingerprint similarity. `max_similarity_reward`
  defaults to `0.5`; set it to `1.0` for reward to equal similarity alone.
- **Reward = 0.0** if no valid submission was made, the pathway was not accepted,
  or the submission could not be resolved to a molecule

# Licensing information

Code: Apache 2.0

Example targets: ChEMBL (CC BY-SA 3.0), redistributed by
[SynFormer](https://github.com/wenhao-gao/synformer).

Dependencies:
- nemo_gym: Apache 2.0
- synclaw: Apache 2.0
