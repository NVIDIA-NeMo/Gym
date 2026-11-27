# BioNeMo-SynClaw ChEMBL-1000

Retrosynthesis planning for the 1,000 public ChEMBL target molecules from
[SynFormer](https://github.com/wenhao-gao/synformer) (`chembl_filtered_1k.txt`), run with the
[BioNeMo-SynClaw environment](../../environments/synclaw/README.md). This is a held-out test set; there is no
training split.

For each target the model searches a purchasable-building-block stock and reaction templates,
simulates reactions, and submits a pathway. The stock is the eMolecules orderable catalog plus
BioNeMo-SynClaw's 37 common reagents.

Three targets are radiotracers (`[11CH3]`, `[18F]`, `[125I]`). Exact-match scoring keeps
isotopes but no stock building block carries them, so the prepare step strips isotope labels from
all targets. Changed rows keep the source SMILES in `original_target_smiles`.

## Run it

1. Build the BioNeMo-SynClaw database (reaction templates, building-block stock and search index).
   Build time and resource requirements depend on the catalog release and your hardware:

   ```bash
   python resources_servers/synclaw/prepare_db.py
   ```

2. Prepare the 1,000 tasks:

   ```bash
   gym eval prepare --benchmark synclaw_chembl_1k
   ```

3. Collect rollouts against your model:

   ```bash
   gym eval run --benchmark synclaw_chembl_1k \
       --model-type vllm_model \
       --model <served-model-name> \
       --model-url <base-url>/v1 \
       --model-api-key <key> \
       --split benchmark \
       --concurrency 64 \
       --output results/synclaw_chembl_1k.jsonl \
       ++policy_model.responses_api_models.vllm_model.propagate_context_overflow_errors=true
   ```

   Use `--model-type vllm_model` for Chat Completions endpoints (vLLM, NIM and most
   OpenAI-compatible providers), and `--model-type openai_model` for endpoints that serve the
   Responses API natively. See
   [Hosted model endpoint](../../environments/synclaw/README.md#hosted-model-endpoint) for how each
   handles reasoning. With `vllm_model`, keep the `propagate_context_overflow_errors` override:
   without it, a context overflow comes back as an empty truncated turn instead of ending the
   rollout cleanly with its best route.

   Serve the model with pinned BioNeMo-SynClaw's context limit,
   `--max-model-len "$(bash environments/synclaw/max_model_len.sh)"`: the agent warns the model as
   it approaches that limit, and rollouts fail if the endpoint's `/v1/models` reports a
   different one. Endpoints that report none (most hosted APIs) are not checked. For a deployment
   with another limit, pass
   `++synclaw_chembl_1k_agent.responses_api_agents.synclaw_agent.max_model_len=<tokens>`; the score is
   then not comparable with default runs. The example explicitly keeps
   up to 64 rollouts live; lower `--concurrency` for a small deployment. Add
   `--limit 10` for a quick smoke test.

   To spread load across several vLLM replicas, start the router as described in the
   [environment README](../../environments/synclaw/README.md#vllm-replicas-recommended-for-throughput)
   and pass its URL as `--model-url`. Don't add `environments/synclaw/vllm.yaml` to a benchmark run;
   it also starts the environment's own servers.

## Protocol

A score is comparable only with runs that share:

- the BioNeMo-SynClaw revision pinned in `resources_servers/synclaw/pyproject.toml` (reaction templates, prompts,
  tools and scoring), with the resources server's and agent's default settings;
- the eMolecules release in the database (`2026-09-01` by default; see `prepare_db.py --release`);
- the task file from `gym eval prepare` (isotope labels stripped) and one rollout per task.

Changing any of these changes results, so bump `version` in `manifest.yaml` when the pin or the
default release changes.

## Scoring

Each rollout's reward is:

- **1.0** if the accepted pathway's final product matches the target under BioNeMo-SynClaw's
  exact-match rules (`exact_reconstruction`), accounting for equivalent tautomers and
  stereochemistry left unspecified by the target while preserving specified stereochemistry.
- **0.5 × similarity** for any other accepted pathway (an analogue).
- **0.0** if nothing was submitted or the pathway was not accepted.

Each output row also records `submitted`, `accepted`, `similarity` and `exact_reconstruction`, so
the exact-reconstruction rate is the mean of `exact_reconstruction`. See
[Verification logic](../../environments/synclaw/README.md#verification-logic).

## License

The targets are ChEMBL structures (CC BY-SA 3.0), redistributed by SynFormer.
