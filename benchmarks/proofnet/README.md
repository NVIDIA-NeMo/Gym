# ProofNet

Lean4 formal proof benchmark bound to the
[`math_formal_lean`](../../resources_servers/math_formal_lean/) resources server
and `simple_agent` (single-turn, matching NeMo-Skills' evaluation protocol).

- **Tasks**: 186 theorems (test split; 185 `valid` rows are discarded)
- **Source**: [deepseek-ai/DeepSeek-Prover-V1.5 `proofnet.jsonl`](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/2c4ba9119eef74d0d611f494261b2c5bae98c69a/datasets/proofnet.jsonl), pinned to commit `2c4ba9119eef74d0d611f494261b2c5bae98c69a`.
- **Prompt**: shared `benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml` (same as miniF2F and MOBench). **Intentionally differs from NeMo-Skills' upstream choice of `lean4/formal-proof`** — using the deepseek-prover-v2 variant for consistency across all ported Lean benchmarks.
- **Reward**: binary; 1.0 iff the Lean4 compiler accepts the proof with no `sorry`.

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/proofnet/config.yaml]"
```

Downloads the source JSONL, filters to `split == "test"`, and writes
`data/proofnet_benchmark.jsonl`. Rows pass through with the upstream schema
(`name`, `split`, `header`, `informal_prefix`, `formal_statement`, `goal`).

## Running

Verification shells out to the NeMo-Skills Lean4 sandbox over HTTP. Bring up the
sandbox container separately (see
[`resources_servers/math_formal_lean/README.md`](../../resources_servers/math_formal_lean/README.md))
and set `NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT` before starting
the server.

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/proofnet/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=proofnet_math_formal_lean_simple_agent \
    +input_jsonl_fpath=benchmarks/proofnet/data/proofnet_benchmark.jsonl \
    +output_jsonl_fpath=results/proofnet_rollouts.jsonl \
    +num_repeats=32 \
    +prompt_config=benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```
