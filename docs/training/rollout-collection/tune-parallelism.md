(tune-rollout-parallelism)=

# Tune Parallelism

Configure concurrent request limits for optimal throughput.

---

## How It Works

Rollout collection throughput depends on parallelism at multiple levels:

1. **Client-side**: How many requests Gym sends concurrently
2. **Server-side**: How many requests your generation backend handles
3. **Network**: Connection pool limits

---

## Client-Side Parallelism

### num_samples_in_parallel

Control concurrent rollout requests:

```bash
ng_collect_rollouts \
    +agent_name=your_agent \
    +input_jsonl_fpath=prompts.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +num_samples_in_parallel=32
```

**Guidelines:**
- **Start low**: Begin with 8–16 and increase
- **Match backend**: Don't exceed your generation backend's capacity
- **Watch for timeouts**: Too high causes request queuing and timeouts

### Connection Limits

Gym's global connection settings (in config):

```yaml
global_aiohttp_connector_limit_per_host: 16384
global_aiohttp_connector_limit: 65536
```

These are high by default. Reduce if you're overwhelming your backend.

---

## Finding Optimal Settings

### Step 1: Baseline

Start with conservative settings:

```bash
ng_collect_rollouts \
    +agent_name=your_agent \
    +input_jsonl_fpath=prompts.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=100 \
    +num_samples_in_parallel=8
```

Note the throughput (rollouts/second).

### Step 2: Increase Gradually

Double parallelism and measure:

```bash
# Try 16
+num_samples_in_parallel=16

# Try 32
+num_samples_in_parallel=32

# Try 64
+num_samples_in_parallel=64
```

### Step 3: Find the Knee

Plot throughput vs parallelism. You'll typically see:

```text
Parallelism | Throughput (rollouts/sec)
------------|---------------------------
8           | 2.1
16          | 4.0
32          | 7.2
64          | 8.1  ← diminishing returns
128         | 8.3  ← not worth the overhead
```

The "knee" is where throughput gains flatten — that's your optimal setting.

---

## Backend-Specific Tuning

### vLLM

vLLM handles high concurrency well. Safe to use high parallelism:

```bash
+num_samples_in_parallel=64  # or higher
```

Monitor GPU utilization — if it's not saturated, increase parallelism.

### OpenAI API

Rate limits apply. Stay within your tier:

```bash
+num_samples_in_parallel=16  # adjust based on rate limits
```

### SGLang

Similar to vLLM, scales well:

```bash
+num_samples_in_parallel=64
```

---

## Symptoms of Over-Parallelism

| Symptom | Cause | Fix |
|---------|-------|-----|
| Timeouts | Backend queue overflow | Reduce parallelism |
| High latency variance | Request queuing | Reduce parallelism |
| OOM on backend | Too many concurrent generations | Reduce parallelism |
| Low GPU utilization | Parallelism too low | Increase parallelism |

---

## Multi-Node Collection

For production-scale runs, parallelize across nodes:

```bash
# Node 1: Process first half
ng_collect_rollouts \
    +input_jsonl_fpath=prompts.jsonl \
    +output_jsonl_fpath=rollouts_node1.jsonl \
    +start_idx=0 +end_idx=5000

# Node 2: Process second half  
ng_collect_rollouts \
    +input_jsonl_fpath=prompts.jsonl \
    +output_jsonl_fpath=rollouts_node2.jsonl \
    +start_idx=5000 +end_idx=10000
```

Then concatenate results:

```bash
cat rollouts_node1.jsonl rollouts_node2.jsonl > rollouts_combined.jsonl
```

