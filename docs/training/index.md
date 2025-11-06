(training-overview)=

# Training

Generate high-quality training data at scale with optimized rollout collection, verification, and formatting.

## Why NeMo Gym for Training

NeMo Gym decouples training data generation from your training framework. Your RL framework (VeRL, NeMo-RL, OpenRLHF) sends tasks to NeMo Gym and receives trajectories with reward signals—NeMo Gym handles the complexity of multi-turn agent coordination, tool calling, and async processing so your framework can focus on training.

:::{dropdown} **Architecture: How It Works**
:open:

**Training Framework Responsibility**:
- Sends tasks to NeMo Gym
- Receives completed trajectories with rewards
- Handles backpropagation and model weight updates

**NeMo Gym Responsibility**:
- Multi-turn agent coordination (tool calling loops)
- Model-environment interaction (interleaved calls)
- Async parallel task processing
- Reward computation

**Data Flow**:

```python
# Training framework → NeMo Gym
response = await server_client.post("/run", task)

# NeMo Gym → Training framework  
result = await response.json()  # Contains trajectory + reward
```

**Source**: `nemo_gym/rollout_collection.py:99-102`

:::

:::{dropdown} **Key Architectural Benefits**

**1. Multi-Turn Coordination**

NeMo Gym's agent layer manages complex multi-turn tool-calling independently from your training loop:

```python
while True:
    # Call model
    model_response = await self.server_client.post(
        server_name=self.config.model_server.name,
        url_path="/v1/responses"
    )
    
    # Execute tool calls
    for tool_call in all_fn_calls:
        await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{tool_call.name}"
        )
```

**Source**: `responses_api_agents/simple_agent/app.py:79-125`

---

**2. HTTP-Based Integration**

All components (Models, Resources, Agents) are independent HTTP services, enabling integration with existing agent systems:

```python
class SimpleResponsesAPIAgent(BaseResponsesAPIAgent):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app
```

**Source**: `nemo_gym/base_responses_api_agent.py:34-45`

---

**3. Interleaved Processing**

Model and environment calls happen within each step—no batch waiting:

```python
# 1. Model generates
model_response = await self.server_client.post(...)

# 2. Execute tools immediately  
api_response = await self.server_client.post(...)

# 3. Feed results back to model
new_body = body.model_copy(update={"input": body.input + new_outputs})
```

**Source**: `responses_api_agents/simple_agent/app.py:79-130`

---

**4. Standardized Output Format**

All models return structured `NeMoGymResponse` objects—no model-specific parsing:

```python
# Works across OpenAI, Azure, vLLM
return NeMoGymResponse(
    id=f"resp_{uuid4().hex}",
    model=model_name,
    output=response_output_dicts,
    ...
)
```

**Source**: `responses_api_models/azure_openai_model/app.py:68`, `responses_api_models/vllm_model/app.py:110`

---

**5. Independent Testing**

Test resource servers without full training infrastructure using `ng_test`:

```bash
ng_test +entrypoint=resources_servers/simple_weather
```

Each server includes independent test suite:

```python
class TestApp:
    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(...)
        SimpleWeatherResourcesServer(config=config, ...)
```

**Source**: `resources_servers/simple_weather/tests/test_app.py:23-31`

---

**6. Async Parallelism**

Configurable concurrency with semaphore control for high-throughput generation:

```python
semaphore = Semaphore(config.num_samples_in_parallel)

async def _post_coroutine(row: dict) -> None:
    async with semaphore:
        response = await server_client.post("/run", json=row)
        result = await response.json()

await tqdm.gather(*map(_post_coroutine, rows))
```

**Source**: `nemo_gym/rollout_collection.py:86-105`

**Configuration**: Set `num_samples_in_parallel=10` to control concurrent rollouts

:::

:::{seealso}
For deeper architectural understanding, see {doc}`../about/architecture` and {doc}`../about/concepts/index`.
:::

---

## Training Data Pipeline

Follow the training data pipeline from resource server selection to framework integration:

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Resource Servers
:link: resource-servers/index
:link-type: doc

Choose a resource server that provides tools, datasets, and verification for your training task.
+++
{bdg-secondary}`server-selection` {bdg-secondary}`tasks` {bdg-secondary}`domains`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: rollout-collection/index
:link-type: doc

Generate training rollouts at scale with optimized sampling strategies and parallelization.
+++
{bdg-secondary}`data-generation` {bdg-secondary}`parallelism` {bdg-secondary}`throughput`
:::

:::{grid-item-card} {octicon}`trophy;1.5em;sd-mr-1` Verification
:link: verification/index
:link-type: doc

Validate that verification works correctly and customize reward signals for your training needs.
+++
{bdg-secondary}`validation` {bdg-secondary}`rewards` {bdg-secondary}`custom-patterns`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Data Quality
:link: data-quality/index
:link-type: doc

Filter, curate, and balance rollouts to ensure high-quality training datasets.
+++
{bdg-secondary}`filtering` {bdg-secondary}`curation` {bdg-secondary}`quality-metrics`
:::

:::{grid-item-card} {octicon}`package-dependencies;1.5em;sd-mr-1` Datasets
:link: datasets/index
:link-type: doc

Organize, validate, and prepare datasets in formats for RL training frameworks.
+++
{bdg-secondary}`formats` {bdg-secondary}`validation` {bdg-secondary}`sft-dpo`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Handoff to Training
:link: handoff-to-training
:link-type: doc

Pass your rollouts to RL training frameworks like NeMo-RL, VeRL, OpenRLHF, or TRL.
+++
{bdg-secondary}`training` {bdg-secondary}`frameworks` {bdg-secondary}`handoff`
:::

::::

## Quick Decision Guide

Not sure where to start? Choose based on your current need:

```{list-table}
:header-rows: 1
:widths: 40 60

* - If You Need To...
  - Start Here
* - Choose a resource server
  - {doc}`Resource Servers <resource-servers/index>` (by task type, training algorithm)
* - Generate training data faster
  - {doc}`Rollout Collection / Optimize for Training <rollout-collection/optimize-for-training/index>`
* - Validate verification works
  - {doc}`Verification / Validate <verification/validate-verification>` (check reward signals)
* - Improve training data quality
  - {doc}`Data Quality <data-quality/index>`
* - Prepare data for SFT or DPO
  - {doc}`Datasets / Prepare for Training <datasets/prepare-for-training>`
* - Pass data to your training framework
  - {doc}`Handoff to Training <handoff-to-training>` (NeMo-RL, VeRL, OpenRLHF, TRL)
* - Build custom verification
  - {doc}`Verification / Custom Patterns <verification/custom-patterns-cookbook>` (advanced)
```

## Training Workflow Patterns

Common end-to-end workflows combining data generation, quality filtering, and framework integration for different training objectives.

::::{tab-set}

:::{tab-item} SFT Data Generation

Generate high-quality demonstration data for supervised fine-tuning:

```yaml
# Optimized for consistency
num_samples_in_parallel: 20
responses_create_params:
  temperature: 0.2  # Low for consistent behavior
  
# Then filter for quality
min_reward_threshold: 0.8
```

**Guides**: {doc}`rollout-collection/optimize-for-training/index` → {doc}`data-quality/index` → {doc}`datasets/prepare-for-training`

:::

:::{tab-item} DPO Pair Generation

Generate diverse pairs for preference optimization:

```yaml
# Optimized for diversity
num_repeats: 2  # Multiple samples per task
responses_create_params:
  temperature: 0.7  # Higher for variation
  
# Then create preference pairs
min_quality_difference: 0.1
```

**Guides**: {doc}`rollout-collection/optimize-for-training/index` → {doc}`data-quality/index` → {doc}`datasets/prepare-for-training`

:::

:::{tab-item} PPO Training Data

Generate rollouts with continuous rewards for reinforcement learning:

```yaml
# Balanced approach
num_samples_in_parallel: 10
responses_create_params:
  temperature: 0.5  # Moderate exploration
```

**Guides**: {doc}`rollout-collection/optimize-for-training/index` → {doc}`data-quality/index` → {doc}`handoff-to-training`

:::

::::

## Next Steps

We recommend starting with **Resource Servers** to choose the right task domain and verification, then moving to **Rollout Collection** to generate training data at scale.

:::{button-ref} resource-servers/index
:color: primary
:outline:
:ref-type: doc

Choose a Resource Server →
:::
