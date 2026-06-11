# Original CVDP Report Example

Load this reference only when the user asks for a concrete example of how to
understand benchmark reports produced by NeMo Gym. It uses the original CVDP
code-generation environment, not a multi-turn sandbox navigation variant.

## Where The Example Lives

- Environment README: `resources_servers/cvdp/README.md`
- Resource server config: `resources_servers/cvdp/configs/cvdp.yaml`
- Agent config: `responses_api_agents/cvdp_agent/configs/cvdp_agent.yaml`
- Example rollout artifacts already in this repo: `resources_servers/cvdp/data/`
- Report generator: `resources_servers/cvdp/scripts/cvdp_pass_at_k_report.py`

The original CVDP flow asks the model to produce SystemVerilog from a hardware
design specification. The verifier runs the generated RTL against a cocotb-style
test harness and returns binary reward.

## Command Shape

Start the CVDP and model servers:

```bash
ng_run "+config_paths=[resources_servers/cvdp/configs/cvdp.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Collect repeated rollouts:

```bash
ng_collect_rollouts \
  +agent_name=cvdp_agent \
  +input_jsonl_fpath=resources_servers/cvdp/data/<dataset>.jsonl \
  +output_jsonl_fpath=results/cvdp_rollouts.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=4 \
  "+responses_create_params={max_output_tokens: 4096, temperature: 0.2, top_p: 0.7}" \
  "+config_paths=[resources_servers/cvdp/configs/cvdp.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Generate the CVDP report layout:

```bash
python resources_servers/cvdp/scripts/cvdp_pass_at_k_report.py \
  --rollouts results/cvdp_rollouts.jsonl \
  --output results/cvdp_report/ \
  --model <model-or-agent-name> \
  --dataset <original-cvdp-dataset>.jsonl \
  --k 1
```

## Report Files

`cvdp_pass_at_k_report.py` splits rollouts by `_ng_rollout_index` and produces:

- `composite_report.txt`: top-level human-readable result across repeat samples.
- `composite_report.json`: machine-readable composite metrics.
- `sample_<n>/report.txt`: per-repeat human-readable result.
- `sample_<n>/report.json`: per-repeat machine-readable metrics.
- `sample_<n>/raw_result.json`: per-task pass/fail details.
- `sample_<n>/<task>/prompts/`: prompt collateral for that task, when available.
- `sample_<n>/<task>/reports/`: verifier output for that task, when available.

The composite report is the first file to read for pass rates and category
breakdowns. Per-sample reports are useful when repeat-to-repeat variance is high.
Per-task verifier outputs are the evidence source for failure explanations.

## How To Interpret The Report

Use the generic BLADE workflow from the main skill, then add CVDP-specific
questions:

- Which CVDP category has the largest pass-rate drop?
- Are failures dominated by compile errors, simulation assertion failures,
  timeouts, malformed modules, or missing outputs?
- Do repeated samples fail consistently, or does pass@k reveal high instability?
- Are failures tied to unsupported SystemVerilog constructs or to wrong design
  behavior under hidden tests?
- Does self-contained example data reproduce the same failure class?

For original CVDP, the key distinction is whether generated RTL compiles,
simulates, and satisfies the hidden task harness.

## Sanitization Notes

When sharing a CVDP-based report, keep aggregate metrics, task ids, categories,
and verifier summaries that are cleared for release. Remove local filesystem
paths, credentials, private endpoints, user names, and raw source-bearing fields
unless the data owner explicitly cleared them.
