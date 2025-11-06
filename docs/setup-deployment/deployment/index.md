(setup-deployment-scenarios)=

# Deployment

Deploy NeMo Gym in different environments—local development, remote servers, or containerized infrastructure.

---

## Deployment Guides

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Development
:link: local
:link-type: doc

Set up NeMo Gym for local development and testing on your laptop.
+++
{bdg-secondary}`how-to` {bdg-secondary}`getting-started`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Remote Servers
:link: remote-servers
:link-type: doc

Deploy NeMo Gym components on remote machines or cloud infrastructure.
+++
{bdg-secondary}`how-to` {bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containers
:link: containers
:link-type: doc

Package and deploy NeMo Gym using Docker and container orchestration.
+++
{bdg-secondary}`how-to` {bdg-secondary}`docker`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Scaling
:link: scaling
:link-type: doc

Scale NeMo Gym components for high-throughput production workloads.
+++
{bdg-secondary}`how-to` {bdg-secondary}`performance`
:::

::::

---

```{toctree}
:hidden:
:maxdepth: 1

local
remote-servers
containers
scaling
```

