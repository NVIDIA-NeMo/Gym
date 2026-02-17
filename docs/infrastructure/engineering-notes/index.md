(engineering-notes)=
# Engineering Notes

Technical notes that document infrastructure decisions, performance investigations, and design rationale behind NeMo Gym.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`note;1.5em;sd-mr-1` Responses API Evolution
:link: responses-api-evolution
:link-type: doc
Why NeMo Gym uses the Responses API and how it differs from Chat Completions.
+++
{bdg-secondary}`api-design` {bdg-secondary}`schema`
:::

:::{grid-item-card} {octicon}`note;1.5em;sd-mr-1` SWE RL Infrastructure Case Study
:link: swe-rl-case-study
:link-type: doc
Infrastructure challenges and deployment topology for SWE RL training.
+++
{bdg-secondary}`swe-rl` {bdg-secondary}`case-study`
:::

:::{grid-item-card} {octicon}`note;1.5em;sd-mr-1` aiohttp vs httpx
:link: aiohttp-vs-httpx
:link-type: doc
Why NeMo Gym uses aiohttp instead of httpx for async HTTP.
+++
{bdg-secondary}`server-infra` {bdg-secondary}`performance`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

Responses API <responses-api-evolution>
SWE RL Case Study <swe-rl-case-study>
aiohttp vs httpx <aiohttp-vs-httpx>
```
