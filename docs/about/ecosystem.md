(nemo-gym-ecosystem)=
# NeMo Gym in the NVIDIA Ecosystem

NeMo Gym standardizes scalable rollout collection for LLM and VLM training with
unified interfaces to heterogeneous reinforcement learning (RL) environments and
integrations with popular RL training frameworks. It provides the
infrastructure to generate agent interaction data and curated environments with
verification logic, making it practical to produce large-scale, high-quality
data for RL workflows using the training framework of your choice.

:::{tip}
For details on NeMo Gym capabilities, refer to the
{ref}`Overview <about-overview>` and {ref}`Key Features <features>`. For internal design and
boundaries, refer to {ref}`Architecture <architecture>`.
:::

---

## Understanding the NVIDIA NeMo Agentic Product Family

Two complementary offerings serve adjacent but distinct needs in agentic AI
workflows.

### NeMo Gym

A toolkit focused on scalable rollout collection and integration with RL
training frameworks.

**Best for**: Teams training models with RL and needing high-throughput data
collection with verification.

**Key characteristics**:
* Unified interfaces to heterogeneous RL environments
* Integrations with multiple RL training frameworks
* Curated environments with verification logic for reward signals
* Parallel and asynchronous rollout collection for scale

---

### NeMo Agent Toolkit (Open Source)

An open-source framework that is interoperable with other ecosystems and
supports end-to-end profiling and optimization of complex agentic systems.

**Best for**: Teams building and operating agent workflows for inference,
needing observability, profiling, and evaluation.

**Key characteristics**:
* Interoperable with other frameworks
* End-to-end profiling and optimization for agent pipelines
* Helps expose bottlenecks and costs to scale agent systems reliably

---

(ecosystem-comparison)=
## Choosing the Right Option

The right choice depends on whether you are optimizing agentic inference
workflows, training models with RL, or combining both.

### Feature Comparison

```{list-table}
::header-rows: 1
::widths: 28 28 28 16

* - Aspect
  - NeMo Gym
  - NeMo Agent Toolkit
  - Both (Integrated)
* - **Target Users**
  - ML engineers focused on RL training and data generation
  - Developers and platform engineers building agentic workflows
  - Teams connecting agent workflows to RL training
* - **Primary Use Case**
  - Scalable rollout collection with verification for RL
  - Build, observe, and optimize agent systems for inference
  - Use agent trajectories as training data for RL
* - **Trajectory Collection**
  - Collects at scale with parallel/async execution
  - Collects trajectories from running agent workflows
  - NAT collects → Gym consumes for RL
* - **Evaluation**
  - Verification logic to derive reward signals for RL
  - Full evaluation harnesses for agent workflows
  - Use NAT eval to score, use Gym for RL rewards
* - **Profiling / Instrumentation**
  - Training-focused metrics for RL efficiency
  - Inference-focused metrics (per-invocation, cost, sizing)
  - Complementary views across training and inference
* - **Observability**
  - RL training metrics (e.g., mixed-reward, truncation rate)
  - Telemetry exporters and observability platform integration
  - End-to-end visibility across the pipeline
* - **Agent Orchestration**
  - Minimal agent for-loop to unblock RL research/training
  - Multiple agent implementations with abstractions and interfaces
  - Use NAT orchestration, send trajectories to Gym
* - **LLM Integration**
  - Via RL frameworks that handle training and generation
  - Interfaces to inference-only LLM endpoints
  - Bridge training and inference paths
* - **Framework Connections**
  - RL frameworks (e.g., VeRL, NeMo RL, OpenRLHF)
  - Agent frameworks (e.g., LangChain, LlamaIndex, LangGraph)
  - Connect agent and RL frameworks through shared data
* - **UI**
  - None
  - Available
  - Use NAT UI while training with Gym
* - **MCP (Model Context Protocol)**
  - Acts as a client to train on tools from MCP servers
  - Acts as client or server to use or publish tools
  - Combine to train on MCP-powered tool use
```

### Decision Guide

**Use NeMo Gym when**:
* You want to train a model with your preferred RL framework
* You need scalable rollout collection with verification for reward signals
* You prefer minimal agent orchestration tailored for RL research and training

**Use NeMo Agent Toolkit when**:
* You are building agent workflows with observability, profiling, and
  evaluation
* You need inference-first instrumentation and cost visibility
* You want to work within an open and interoperable agent framework

**Use NeMo Agent Toolkit + NeMo Gym when**:
* You want to train models using data from agent workflows
* Your agent framework (NAT or another) collects trajectories and scores them,
  then Gym consumes that data for RL with frameworks such as VeRL or NeMo RL

---

## Integration Points

NeMo Gym and NeMo Agent Toolkit can be composed in several ways to support RL
training using agent-generated data.

### NAT Workflow as an Endpoint → Gym for RL

**Workflow**:
1. Serve a NAT agent workflow as an endpoint
2. Gym triggers runs and requests trajectory collection
3. NAT executes the workflow and collects trajectories
4. NAT evaluates and assigns scores/rewards
5. Gym forwards trajectories and rewards to the chosen RL framework for
   backpropagation and checkpoint updates

---

### Other Agent Frameworks → Gym for RL

Applies to users of LangChain, LangGraph, CrewAI, or similar frameworks.

**Workflow**:
1. Serve the existing agent workflow as an endpoint
2. Gym triggers runs and requests trajectory collection
3. The workflow collects trajectories and performs evaluation
4. Scores and trajectories are passed to Gym
5. Gym hands data to the RL framework for training

---

### Model-Only Training Inside Gym

**Workflow**:
1. Run the model within Gym against resource servers (tools and verification)
2. Gym captures trajectories produced during interactions
3. Gym sends trajectories to the RL framework for training

---

## Related NVIDIA Components

* **VeRL**, **NeMo RL**, **OpenRLHF**: RL frameworks commonly used with Gym
* **NVIDIA NIM**: Inference microservices that can be paired with agent
  workflows and training pipelines

