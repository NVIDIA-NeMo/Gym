# physics_with_judge

A resources server for verifying physics answers using unit-aware numerical comparison and an optional LLM-as-a-judge fallback.

## How it works

Verification runs in two stages:

**Stage 1 — Library (pint)**

The server extracts the content of the last `\boxed{…}` in the model output, preprocesses common LaTeX notation into a pint-parseable string, and compares the resulting quantities after unit conversion.

Key features:
- **Unit conversion**: 1300 W and 1.3 kW are treated as equivalent. Any dimensionally compatible pair of units is handled (e.g. m/s ↔ km/h, J ↔ kJ).
- **Relative tolerance**: numerical comparison uses a configurable `rtol` (default 0.1%), so 9.80 and 9.81 are accepted as equal within tolerance.
- **Dimensionality check**: incompatible units (e.g. W vs J) always return reward 0.

The preprocessor handles the most common LaTeX patterns:

| Input | Output |
|---|---|
| `1.3 \text{ kW}` | `1.3 kW` |
| `1.3 \times 10^{5} \text{ km/s}` | `1.3e5 km/s` |
| `9.81 \text{ m/s}^{2}` | `9.81 m/s**2` |
| `100 \Omega` | `100 ohm` |
| `\frac{1}{2} \text{ J}` | `0.5 J` |
| `100 °C` | `100 degC` |

If library reward > 0.5, the result is returned immediately without calling the judge.

**Stage 2 — LLM judge (optional)**

When the library stage fails (parse error, dimensionality mismatch, or wrong value), an LLM judge evaluates physical equivalence. The judge is called twice — (A→B) then (B→A) — and both calls must agree the answers are equivalent before reward 1.0 is granted, eliminating positional bias.

The judge is disabled by default (`should_use_judge: false`). Enable it by overriding the config.

## Configuration

| Field | Default | Description |
|---|---|---|
| `should_use_judge` | `false` | Enable LLM judge fallback for cases the library cannot verify |
| `rtol` | `1e-3` | Relative tolerance for numerical comparison (0.1%) |
| `judge_model_server` | — | Server reference for the judge model |
| `judge_responses_create_params` | — | Inference parameters for judge calls |

## Usage

### Start servers (library-only verification)

```bash
ng_run "+config_paths=[resources_servers/physics_with_judge/configs/physics_with_judge.yaml,\
    responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

### Start servers with LLM judge enabled

```bash
ng_run "+config_paths=[resources_servers/physics_with_judge/configs/physics_with_judge.yaml,\
    responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    ++physics_with_judge.resources_servers.physics_with_judge.should_use_judge=true
```

### Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=physics_with_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/physics_with_judge/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/physics_with_judge/results/example_rollouts.jsonl \
    +num_repeats=5 \
    "+responses_create_params={max_output_tokens: 8192, temperature: 1.0}"
```

### Profile rewards

```bash
ng_reward_profile \
    +input_jsonl_fpath=resources_servers/physics_with_judge/data/example.jsonl \
    +rollouts_jsonl_fpath=resources_servers/physics_with_judge/results/example_rollouts.jsonl \
    +output_jsonl_fpath=resources_servers/physics_with_judge/results/example_profiled.jsonl \
    +pass_threshold=1.0
```

### Print aggregate metrics

```bash
python scripts/print_aggregate_results.py \
    +jsonl_fpath=resources_servers/physics_with_judge/results/example_profiled.jsonl
```

### Use with ns_tools (tool-augmented agent)

Add `physics_with_judge` to the ns_tools verifiers map and set `verifier_type` in your data samples:

```bash
ng_run "+config_paths=[resources_servers/physics_with_judge/configs/physics_with_judge.yaml,\
    resources_servers/ns_tools/configs/ns_tools.yaml,\
    responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Baseline results

Model: `Qwen/Qwen3-8B`, `num_repeats=5`, `max_output_tokens=8192`, `temperature=1.0`, library-only verification.

| Task | pass@1 | pass@5 |
|---|---|---|
| 1300 W → kW | 0.00 | 0 |
| 30 m/s → km/h | 0.00 | 0 |
| KE of 2 kg at 10 m/s | 0.00 | 0 |
| Speed of light in km/s | 0.20 | 1 |
| Weight of 5 kg on Earth | 0.40 | 1 |
| **Overall** | **0.12** | — |

## Licensing

Code: Apache 2.0

Dependencies:
- nemo_gym: Apache 2.0
- pint: BSD 3-Clause
