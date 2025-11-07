(setup-deployment)=

# Setup & Deployment

Configure and deploy NeMo Gym for local development, production, or distributed environments with proper monitoring and operations.

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Topics

Choose the setup and deployment task that matches your current need:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Management
:link: configuration/index
:link-type: doc

Manage NeMo Gym's three-tier configuration system, environments, secrets, and multi-server setups.
+++
{bdg-secondary}`how-to` {bdg-secondary}`configuration`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Deployment Scenarios
:link: deployment/index
:link-type: doc

Deploy locally, on remote servers, in containers, or scale with distributed computing.
+++
{bdg-secondary}`how-to` {bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Operations
:link: operations/index
:link-type: doc

Monitor health, test servers, profile performance, and debug deployment issues.
+++
{bdg-secondary}`how-to` {bdg-secondary}`operations`
:::

::::

---

## Setup Workflow Patterns

Common end-to-end workflows for typical setup and deployment scenarios.

::::{tab-set}

:::{tab-item} Development Environment

**Quick local setup for experimentation and testing**:

```bash
# 1. Clone and install
git clone <repo>
cd Gym
pip install -e ".[dev]"

# 2. Create env.yaml with secrets
cat > env.yaml << EOF
policy_api_key: sk-your-openai-key
EOF

# 3. Test with simple config
ng_run "+config_paths=[responses_api_agents/simple_agent/config.yaml]"
```

**Guides**: {doc}`configuration/index` → {doc}`deployment/local-development` → {doc}`operations/testing`

:::

:::{tab-item} Production Deployment

**Production server deployment with monitoring**:

```bash
# 1. Set up production env.yaml
cat > env.yaml << EOF
policy_api_key: ${PROD_OPENAI_KEY}
judge_api_key: ${PROD_JUDGE_KEY}
EOF

# 2. Deploy with production config
ng_run "+config_paths=[production_config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=11000

# 3. Monitor server availability
watch -n 10 'lsof -i :11000 > /dev/null && echo "✓ Server running" || echo "✗ Server down"'
```

**Guides**: {doc}`configuration/index` → {doc}`deployment/distributed-computing` → {doc}`operations/monitoring`

:::

:::{tab-item} Distributed Training

**Multi-node setup with Ray for high-throughput workloads**:

```bash
# 1. Start Ray cluster
ray start --head --port=6379

# 2. Set up distributed config
cat > distributed_config.yaml << EOF
num_gpus: 4
num_workers: 16
EOF

# 3. Run distributed rollout collection
ng_run "+config_paths=[distributed_config.yaml]" \
    +default_host=0.0.0.0
```

**Guides**: {doc}`deployment/distributed-computing` → {doc}`operations/profiling` → {doc}`operations/monitoring`

:::

::::
