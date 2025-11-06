(setup-deployment)=

# Setup & Deployment

Configure and deploy NeMo Gym for your project. These guides assume you've completed the getting-started tutorials and are ready to set up development, testing, or production environments.

---

## Quick Navigation

Choose based on what you need to do right now:

```{list-table}
:header-rows: 1
:widths: 40 60

* - I Need To...
  - Go Here
* - Set up dev/test/prod environments
  - {doc}`Configuration / Environments <configuration/environments>`
* - Manage API keys and secrets
  - {doc}`Configuration / Secrets <configuration/secrets>`
* - Configure multiple servers
  - {doc}`Configuration / Multi-Server <configuration/multi-server>`
* - Deploy on remote machines
  - {doc}`Deployment / Remote Servers <deployment/remote-servers>`
* - Run in Docker containers
  - {doc}`Deployment / Containers <deployment/containers>`
* - Test my setup
  - {doc}`Operations / Testing <operations/testing>`
* - Debug configuration issues
  - {doc}`Configuration / Troubleshooting <configuration/troubleshooting>`
```

---

## Configuration Management

Master NeMo Gym's three-tier configuration system to handle different environments, secrets, and deployment scenarios.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Environments
:link: configuration/environments
:link-type: doc

Manage separate configurations for development, testing, and production environments.
+++
{bdg-secondary}`how-to` {bdg-secondary}`dev-test-prod`
:::

:::{grid-item-card} {octicon}`shield-lock;1.5em;sd-mr-1` Secrets Management
:link: configuration/secrets
:link-type: doc

Securely handle API keys, credentials, and sensitive configuration values.
+++
{bdg-secondary}`how-to` {bdg-secondary}`security`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Multi-Server Setup
:link: configuration/multi-server
:link-type: doc

Configure multiple models, resource servers, and agents in one deployment.
+++
{bdg-secondary}`how-to` {bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: configuration/troubleshooting
:link-type: doc

Debug common configuration issues and validate your setup.
+++
{bdg-secondary}`reference` {bdg-secondary}`debugging`
:::

::::

:::{seealso}
**Understanding the concepts?** See {doc}`../about/concepts/configuration-system` for conceptual explanation of how the three-tier configuration system works.
:::

---

## Deployment

Deploy NeMo Gym in different environments—local development, remote servers, or containerized infrastructure.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Development
:link: deployment/local
:link-type: doc

Set up NeMo Gym for local development and testing on your laptop.
+++
{bdg-secondary}`how-to` {bdg-secondary}`getting-started`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Remote Servers
:link: deployment/remote-servers
:link-type: doc

Deploy NeMo Gym components on remote machines or cloud infrastructure.
+++
{bdg-secondary}`how-to` {bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containers
:link: deployment/containers
:link-type: doc

Package and deploy NeMo Gym using Docker and container orchestration.
+++
{bdg-secondary}`how-to` {bdg-secondary}`docker`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Scaling
:link: deployment/scaling
:link-type: doc

Scale NeMo Gym components for high-throughput production workloads.
+++
{bdg-secondary}`how-to` {bdg-secondary}`performance`
:::

::::

---

## Operations

Monitor, test, and debug your NeMo Gym deployment to ensure reliable operation.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Monitoring
:link: operations/monitoring
:link-type: doc

Set up health checks and monitoring for your NeMo Gym deployment.
+++
{bdg-secondary}`how-to` {bdg-secondary}`ops`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Testing
:link: operations/testing
:link-type: doc

Test your NeMo Gym setup to validate configuration and functionality.
+++
{bdg-secondary}`how-to` {bdg-secondary}`validation`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: operations/debugging
:link-type: doc

Diagnose and fix common issues with NeMo Gym deployments.
+++
{bdg-secondary}`reference` {bdg-secondary}`troubleshooting`
:::

::::

---

## Common Workflows

End-to-end workflows for typical setup and deployment scenarios:

### Development Environment Setup

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

**Next**: {doc}`configuration/environments` for multiple environment setup

---

### Production Deployment

```bash
# 1. Set up production env.yaml
cat > env.yaml << EOF
policy_api_key: ${PROD_OPENAI_KEY}
judge_api_key: ${PROD_JUDGE_KEY}
EOF

# 2. Deploy with production config
ng_run "+config_paths=[production_config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

**Next**: {doc}`deployment/remote-servers` for distributed deployment

---

## Related Documentation

- {doc}`../get-started/index` - First-time setup and tutorials
- {doc}`../about/concepts/configuration-system` - Configuration system concepts
- {doc}`../about/architecture` - System architecture overview
- {doc}`../how-to-faq` - Additional how-to guides

---

```{toctree}
:hidden:
:maxdepth: 2

configuration/index
deployment/index
operations/index
```

