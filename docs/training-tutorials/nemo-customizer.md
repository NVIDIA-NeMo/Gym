(training-nemo-customizer)=
# NeMo Customizer

:::{admonition} 🚧 Integration Not Yet Available
:class: warning

NeMo Customizer integration is blocked by an external service dependency.

**Status**: No integration code exists in NeMo Gym. See [GitHub Issue #550](https://github.com/NVIDIA-NeMo/Gym/issues/550).

**What you CAN do now**: Collect rollouts and prepare training data for manual upload.
:::

---

## What Is NeMo Customizer?

[NeMo Customizer](https://docs.nvidia.com/nemo/nemo-microservices/latest/) is NVIDIA's managed fine-tuning microservice. It provides:

- Managed infrastructure—no GPU cluster setup required
- NVIDIA-optimized training recipes
- API-driven workflow for automation
- Built-in checkpointing and model versioning

## Integration Status

| Component | Status | Notes |
|-----------|--------|-------|
| Rollout collection | ✅ Available | Use `ng_collect_rollouts` |
| Training data preparation | ✅ Available | Use `ng_prepare_data` |
| NeMo Customizer API client | ❌ Not implemented | Requires external coordination |
| Data upload to Customizer | ❌ Not implemented | Blocked on API access |
| Training job management | ❌ Not implemented | Blocked on API access |

## What You Can Do Now

While NeMo Customizer integration is pending, you can collect and prepare training data using existing NeMo Gym tools. The output is compatible with most training frameworks and can be manually uploaded to NeMo Customizer.

### Complete Example: Collect Rollouts for Training

This example uses the `example_single_tool_call` resource server included with NeMo Gym:

```bash
# Step 1: Start servers (in terminal 1)
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

```bash
# Step 2: Collect rollouts (in terminal 2)
ng_collect_rollouts +agent_name=example_single_tool_call_simple_agent \
    +input_jsonl_fpath=resources_servers/example_single_tool_call/data/example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +limit=10 \
    +num_repeats=2
```

**Output format** (JSONL, one rollout per line):
```text
{
  "reward": 1.0,
  "output": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "22°C, sunny"},
    {"role": "assistant", "content": "The weather in Paris is 22°C and sunny."}
  ],
  "responses_create_params": {...}
}
```

### Prepare Data for Training

Format rollouts for {term}`SFT (Supervised Fine-Tuning)` or {term}`DPO (Direct Preference Optimization)`:

```bash
ng_prepare_data "+config_paths=[your_config.yaml]" \
    +output_dirpath=data/prepared \
    +mode=train_preparation
```

**Full documentation**: {ref}`Offline Training Tutorial <offline-training-w-rollouts>`

---

## Unblocking Requirements

To complete NeMo Customizer integration:

1. **API access**: Coordinate with NeMo Customizer team
2. **Integration code**: Create client wrapper for Customizer API
3. **Example workflow**: End-to-end example from rollouts to deployed model
4. **Authentication docs**: Document credential management

Track progress: [GitHub Issue #550](https://github.com/NVIDIA-NeMo/Gym/issues/550)

---

## Alternative Training Options

Use these available training frameworks instead of NeMo Customizer:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Multi-node {term}`GRPO (Group Relative Policy Optimization)` training for production workloads.
+++
{bdg-primary}`recommended` {bdg-secondary}`production`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth
:link: ../tutorials/unsloth-training
:link-type: doc

Fast single-GPU training for rapid prototyping.
+++
{bdg-secondary}`fast` {bdg-secondary}`prototyping`
:::

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Offline Training
:link: ../tutorials/offline-training-w-rollouts
:link-type: doc

{term}`SFT <SFT (Supervised Fine-Tuning)>` and {term}`DPO <DPO (Direct Preference Optimization)>` training from collected rollouts.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Browse Resource Servers
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Find training environments on GitHub.
+++
{bdg-secondary}`github` {bdg-secondary}`environments`
:::

::::

---

## Related Resources

- {ref}`Rollout Collection Tutorial <gs-collecting-rollouts>` - Generate training data
- {ref}`Offline Training Tutorial <offline-training-w-rollouts>` - SFT and DPO from rollouts
- [NeMo Customizer Documentation](https://docs.nvidia.com/nemo/nemo-microservices/latest/) - Official service docs
- [GitHub Issue #550](https://github.com/NVIDIA-NeMo/Gym/issues/550) - Integration tracking
