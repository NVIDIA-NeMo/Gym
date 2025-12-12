(gym-rl-framework-integration-success-criteria)=

# Success Criteria

Use these criteria to validate that your Gym integration is working correctly. A successful integration must pass all validation benchmarks.

:::{tip}
These success criteria may evolve as new integration challenges are discovered. Check this page for updates when troubleshooting integration issues.
:::

## Validation Checklist

### 1. Component Form Factor

Verify that your integration implements all required components as specified in {doc}`gym-integration-footprint-and-form-factor`:

- [ ] OpenAI-compatible HTTP server
- [ ] On-policy token ID fixes
- [ ] Gym spinup and integration
- [ ] Rollout orchestration
- [ ] Training loop integration

### 2. Environment Configuration

Verify that your integration can load and run arbitrary Gym training environments through configuration:

- [ ] Environment configuration loads from YAML
- [ ] Multiple environments can be selected at runtime
- [ ] Environment parameters are configurable without code changes

### 3. Math Reasoning Benchmark

Train on the DAPO17k math training environment and verify model improvement on AIME24.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Parameter
  - Value
* - Training environment
  - [DAPO17k math environment](https://github.com/NVIDIA-NeMo/Gym/blob/299e8c04f4a3bbf0f6069139092225f2fe3aa70f/resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml)
* - Base model
  - [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
* - Minimum training steps
  - 1,000
* - Validation set
  - AIME24 (included with training environment)
* - Target accuracy
  - ≥85%
```

### 4. Workplace Assistant Benchmark

Train on the workplace assistant environment and verify validation set improvements.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Parameter
  - Value
* - Training environment
  - [Workplace assistant environment](https://github.com/NVIDIA-NeMo/Gym/tree/299e8c04f4a3bbf0f6069139092225f2fe3aa70f/resources_servers/workplace_assistant)
* - Base model
  - [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
* - Minimum training steps
  - 100
* - Success criterion
  - Observable validation set improvement
```

## Troubleshooting

If your integration fails to meet the success criteria:

1. **Training crashes**: Check for off-policy issues. Refer to {doc}`openai-compatible-http-server-on-policy-correction`
2. **No improvement**: Verify rollout orchestration is correctly tracking token IDs
3. **Environment errors**: Verify OpenAI-compatible HTTP server endpoints match the specification

## Related Topics

- {doc}`gym-integration-footprint-and-form-factor` - Required integration components
- {doc}`openai-compatible-http-server-on-policy-correction` - On-policy training fixes
