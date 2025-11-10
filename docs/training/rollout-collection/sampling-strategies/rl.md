(training-rollout-sampling-rl)=

# RL Sampling Strategy

Configure for reinforcement learning: balancing exploration and exploitation with iterative collection.

:::{card}

**Task**: Generate rollouts for online RL training with balanced exploration and iterative policy improvement.

^^^

**This guide shows you how to**:

1. Configure temperature to balance exploration and exploitation
2. Set up iterative collection as policy improves
3. Organize rollout buffers by training iteration
4. Monitor learning progress across collection cycles

:::

---

## Before You Start

Ensure you have these prerequisites before generating RL data:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers with reward-based verification
* - **Training objective**
  - Understanding of RL (learning from reward signals through exploration)
* - **Task dataset**
  - Input prompts in JSONL format (2K-10K per iteration)
* - **Iteration workflow**
  - Plan for collect → train → update → repeat cycles
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Collect RL Data

Generate rollouts for online reinforcement learning with balanced exploration and exploitation.

```bash
ng_collect_rollouts \
    +agent_name=my_rl_agent \
    +input_jsonl_fpath=rl_tasks.jsonl \
    +output_jsonl_fpath=outputs/rl_iter_001.jsonl \
    +responses_create_params.temperature=<temperature> \
    +num_samples_in_parallel=<parallelism>
```

**Configuration**: For RL, use moderate temperature to balance exploration and exploitation, single samples per iteration, and moderate-high parallelism for fast cycles. Refer to {doc}`parameters` for parameter explanations.

### Expected Output

```
Found 2000 rows!
Querying with 15 concurrent requests
Collecting rollouts: 100%|████████████| 2000/2000 [02:15<00:00, 14.81it/s]
{
    "reward": 0.423
}
```

**Note**: Initial iterations have lower rewards; improves over time.

---

## Iterative RL Pattern

RL requires multiple collection→training cycles:

```bash
#!/bin/bash
# RL training loop

NUM_ITERATIONS=10

for iter in $(seq -f "%03g" 1 $NUM_ITERATIONS); do
    echo "=== RL Iteration $iter ==="
    
    # Step 1: Collect rollouts with current policy
    ng_collect_rollouts \
        +agent_name=my_rl_agent \
        +input_jsonl_fpath=rl_tasks.jsonl \
        +output_jsonl_fpath=outputs/rl_iter_${iter}.jsonl \
        +responses_create_params.temperature=0.5 \
        +num_samples_in_parallel=15
    
    # Step 2: Compute metrics
    avg_reward=$(jq -s 'map(.reward) | add/length' outputs/rl_iter_${iter}.jsonl)
    echo "Iteration $iter: avg_reward=$avg_reward"
    
    # Step 3: Train for N steps
    python train_rl.py \
        --input outputs/rl_iter_${iter}.jsonl \
        --checkpoint models/rl_checkpoint_${iter}.pt \
        --steps 1000
    
    # Step 4: Update agent to use new checkpoint
    # (Agent auto-loads latest checkpoint or update config)
    
    echo "---"
done

# Analyze reward progression
echo "Reward progression:"
for f in outputs/rl_iter_*.jsonl; do
    echo -n "$(basename $f): "
    jq -s 'map(.reward) | add/length' < $f
done
```

Track progress over iterations:
```bash
# Log metrics after each iteration
echo "$iter,$avg_reward" >> rl_progress.csv
```

Watch for steady improvement in average reward—healthy RL shows increasing rewards over iterations.

---

## Advanced Patterns

**Curriculum learning** (start easy, progress to harder tasks):
```bash
# Iterations 1-3: Easy tasks
for iter in {001..003}; do
    ng_collect_rollouts \
        +input_jsonl_fpath=tasks_easy.jsonl \
        +output_jsonl_fpath=outputs/rl_iter_${iter}.jsonl ...
done

# Iterations 4-10: Harder tasks
for iter in {004..010}; do
    ng_collect_rollouts \
        +input_jsonl_fpath=tasks_hard.jsonl \
        +output_jsonl_fpath=outputs/rl_iter_${iter}.jsonl ...
done
```

**Temperature annealing** (reduce exploration over time):
```bash
# Adjust temperature per iteration
ng_collect_rollouts ... +responses_create_params.temperature=0.7  # iter 1
ng_collect_rollouts ... +responses_create_params.temperature=0.5  # iter 5
ng_collect_rollouts ... +responses_create_params.temperature=0.3  # iter 10
```