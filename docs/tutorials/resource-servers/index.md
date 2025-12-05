(tutorials-resource-servers)=

# Create Resource Servers

Build custom resource servers that provide tools, verification logic, and domain-specific functionality for your AI agents.

---

## Build Servers by Example

Learn by building concrete, working servers. Each tutorial results in a server you can run and adapt.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Weather API Server
:link: simple-tool-calling
:link-type: doc
Build a single-tool weather server with deterministic verification.
+++
{bdg-secondary}`single-tool`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Data Extraction Server
:link: multi-step-interactions
:link-type: doc
Build a multi-step server where agents query multiple data sources.
+++
{bdg-secondary}`multi-step`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Counter Game Server
:link: stateful-sessions
:link-type: doc
Build a stateful server where tools modify persistent state.
+++
{bdg-secondary}`stateful`
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` Math Verifier Server
:link: llm-as-judge
:link-type: doc
Build a math server with LLM-based answer verification.
+++
{bdg-secondary}`llm-judge`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Testing Server
:link: code-execution
:link-type: doc
Build a server that verifies code by executing test cases.
+++
{bdg-secondary}`code-exec`
:::

::::

---

## Patterns Reference

<!-- 
NOTE FOR FUTURE: These patterns could become their own section in training/ or a dedicated 
resource-server/ how-to section. The abstractions are:

- Single-tool calling: One tool, deterministic verification
- Multi-step interactions: Sequential tool calls, aggregation verification  
- Stateful sessions: Persistent state across calls, state-based verification
- LLM-as-judge: Rule-based + LLM fallback verification
- Code execution: Sandbox execution, test-based verification

Each pattern has distinct JTBD dimensions around tool design, state management, 
and verification strategy.
-->

Each tutorial demonstrates a pattern you can apply to your own domains:

| Pattern | Example Server | Key Concept |
|---------|---------------|-------------|
| **Single-tool** | Weather API | One tool call, deterministic verify |
| **Multi-step** | Data Extraction | Sequential calls, aggregate results |
| **Stateful** | Counter Game | Persistent state, state-based verify |
| **LLM-judge** | Math Verifier | Rule-based + LLM fallback |
| **Code-exec** | Code Tester | Sandbox execution, test cases |

---

```{toctree}
:maxdepth: 1
:hidden:

Weather API <simple-tool-calling>
Data Extraction <multi-step-interactions>
Counter Game <stateful-sessions>
LLM Judge Math Verifier <llm-as-judge>
Code Testing <code-execution>
```
