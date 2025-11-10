---
orphan: true
---
(about-features)=
# Key Features of NVIDIA NeMo Gym

This page provides a comprehensive list of NeMo Gym's capabilities. For conceptual understanding of how these features work together, see the {doc}`Overview <index>`.

---

## What Makes NeMo Gym Different

Five core design principles distinguish NeMo Gym from other agent frameworks.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Training-Focused**
  - Optimized for generating large-scale training data, not inference deployment. High throughput over low latency.
* - **Framework Integration**
  - First-class support for popular RL frameworks. Bridges the gap between environments and training infrastructure.
* - **Production-Ready Environments**
  - Curated resource servers with verified accuracy. Not just toys—used for training NVIDIA Nemotron models.
* - **Configuration-First**
  - Swap environments, models, and settings via YAML. Minimal code changes for different training scenarios.
* - **Extensible Foundation**
  - Clear abstractions and base classes make custom environments straightforward. Add new tools without understanding entire system.
```

---

## High-Throughput Rollout Generation

Generate thousands of agent interactions concurrently with async orchestration and configurable parallelism.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Async Orchestration**
  - Built on asyncio with configurable parallelism using semaphores. Process thousands of agent interactions concurrently while managing model API rate limits and resource constraints.
* - **Configurable Concurrency**
  - Control parallel execution with `num_samples_in_parallel` parameter. Scale from single-threaded debugging to high-throughput production workloads by adjusting one setting.
* - **Progress Tracking**
  - Real-time progress bars with `tqdm` integration show rollout collection status, completion estimates, and average metrics across batches.
* - **Batch Processing**
  - Process entire datasets efficiently with automatic batching and progress tracking for long-running collection jobs.
```

---

## Flexible Configuration System

Three-tier configuration with Hydra integration enables environment-specific settings without code changes.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Three-Tier Configuration**
  - Hierarchical configuration with clear precedence: YAML config files (base) < `env.yaml` (secrets) < CLI arguments (overrides). Change deployment without modifying code.
* - **Hydra Integration**
  - Full Hydra framework support enables composition, interpolation, and structured configs. Validate configurations at startup with Pydantic schemas.
* - **Environment Variables**
  - Store sensitive credentials and environment-specific values in `env.yaml` (gitignored by default). Reference with interpolation syntax throughout configs.
* - **Runtime Overrides**
  - Override any configuration parameter via command line using Hydra's `+key=value` syntax. Experiment with settings without editing files.
* - **Configuration Validation**
  - Type-safe configs with Pydantic models catch errors before runtime. Get clear error messages when required fields are missing or types are incorrect.
```

---

## Curated Resource Servers

NeMo Gym includes 12 production-ready resource servers across eight domains:

### Mathematics

* **Library Judge Math**: Symbolic math verification using computer algebra systems (AIME24, DAPO17K datasets)
* **Python Math Exec**: Python code execution sandbox for mathematical reasoning
* **Multiverse Math Hard**: Multi-step mathematical problem solving with tool use

### Coding

* **Comp Coding**: Competitive programming with test case execution (LiveCodeBench integration)

### Agent Workflows

* **Google Search**: Web search tool integration with result verification
* **Workbench**: Business productivity tools (CRM, email, calendar, analytics)
* **Stateful Counter**: Stateful multi-turn interaction examples
* **Simple Weather**: Tutorial environment for getting started

### Knowledge & Reasoning

* **MCQA**: Multiple-choice question answering evaluation
* **Equivalence LLM Judge**: Semantic equivalence verification using LLM-as-judge

### Instruction Following

* **Instruction Following**: General instruction compliance measurement
* **Multineedle**: Long-context information retrieval and synthesis

Each resource server includes:

* Training datasets with license information
* Validation sets for evaluation
* Example data for testing
* Verification logic with reward signals
* Comprehensive test coverage

---

## Model Integration

Connect to OpenAI, Azure OpenAI, or self-hosted models through standardized Responses API interface.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **OpenAI API Support**
  - Direct integration with OpenAI's Responses API. Use GPT-4, GPT-4o, or other OpenAI models with native tool-calling support.
* - **Azure OpenAI**
  - Enterprise deployment support through Azure OpenAI endpoints. Same interface as OpenAI with Azure infrastructure.
* - **vLLM Local Serving**
  - Run open-source models locally using vLLM. Automatic conversion between OpenAI Responses API format and vLLM's chat completions interface.
* - **OpenAI-Compatible Format**
  - Standardized on OpenAI's Responses API schema enables easy swapping between model providers without changing environment code.
* - **Token-Level Data**
  - Capture prompt token IDs, generation token IDs, and logprobs for RL training. Essential for policy gradient algorithms and reward modeling.
```

---

## Dataset Management

Version, share, and prepare training datasets with GitLab integration and license tracking.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Versioned Datasets**
  - Semantic versioning for datasets with GitLab artifact integration. Track dataset lineage and reproduce training runs exactly.
* - **Automated Downloads**
  - CLI commands to download curated datasets from GitLab: `ng_download_dataset_from_gitlab`. Automatically handles authentication and version resolution.
* - **Upload & Sharing**
  - Share custom datasets with team using `ng_upload_dataset_to_gitlab`. Includes license validation and metadata preservation.
* - **Data Preparation Pipeline**
  - `ng_prepare_data` command processes rollout collections into RL training formats. Computes statistics, validates schemas, and exports metrics.
* - **License Tracking**
  - Enforced license declaration for training and validation datasets. Supports Apache 2.0, MIT, Creative Commons variants, ensuring compliance.
* - **Dataset Viewer**
  - Interactive Gradio-based viewer (`ng_viewer`) for exploring rollout collections, comparing responses, and analyzing reward distributions.
```

---

## Developer Tools

Comprehensive CLI utilities, testing framework, and debugging tools accelerate development workflows.

**Comprehensive CLI** — 8 command-line utilities for common workflows:

* `ng_run` - Launch server infrastructure
* `ng_test` - Run server-specific tests
* `ng_test_all` - Test entire system
* `ng_collect_rollouts` - Generate training data
* `ng_prepare_data` - Process rollouts for training
* `ng_viewer` - Interactive data explorer
* `ng_dump_config` - Debug configuration resolution
* `ng_init_resources_server` - Generate new resource server template

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Testing Framework**
  - Built-in test infrastructure for all components. Each resource server includes test suite with example requests and expected responses.
* - **Server Health Checks**
  - Automatic health monitoring for all running servers. Detect and report connection issues, timeout failures, and API errors.
* - **Rapid Iteration**
  - Test resources and agents independently using built-in test commands. Validate changes quickly without deploying full infrastructure.
* - **Configuration Debugging**
  - `ng_dump_config` shows final resolved configuration including all overrides and interpolations. Debug complex config interactions easily.
```

---

## RL Framework Integration

NeMo Gym passes trajectory data to popular RL training frameworks for backpropagation and model weight updates.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **VeRL Compatible**
  - Outputs trajectory data that VeRL (Versatile Reinforcement Learning) consumes for PPO, DPO, and other RL algorithms. Users commonly train with VeRL.
* - **NeMo-RL Integration**
  - Designed as part of the NeMo Framework ecosystem. Passes data directly to NeMo-RL for training orchestration and model updates.
* - **OpenRLHF Compatible**
  - Trajectory format works with OpenRLHF framework. Use same environments across different RL training implementations.
* - **Framework-Agnostic Data**
  - Core rollout format is JSON-based and framework-independent. NeMo Gym handles collection and verification; RL frameworks handle training.
```

---

## Extensibility & Customization

Build custom environments with base classes, templates, and plugin architecture for maximum flexibility.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Base Classes**
  - Abstract base classes for Agents, Models, and Resources with clear interfaces. Inherit and override specific methods for custom behavior.
* - **Template Generation**
  - `ng_init_resources_server` generates complete resource server skeleton with tests, configs, and documentation structure.
* - **Plugin Architecture**
  - Servers communicate via HTTP APIs enabling deployment flexibility. Run components in separate processes, containers, or machines.
* - **Custom Verification Logic**
  - Implement domain-specific reward functions by subclassing `BaseResourcesServer` and overriding the `verify()` method.
* - **Bring Your Own Tools**
  - Add custom tools by defining OpenAI function schemas. NeMo Gym handles orchestration while you focus on tool implementation.
```

---

## Additional Capabilities

Supporting features for stateful environments, metrics computation, error handling, and performance profiling.

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Stateful Environments**
  - Support for multi-turn conversations with session management. Maintain state across agent interactions for complex workflows.
* - **Metrics Computation**
  - Automatic calculation of accuracy, reward distributions, success rates, and domain-specific metrics during rollout collection.
* - **Error Handling**
  - Robust error handling with detailed logging. Failed requests don't crash entire batch, allowing collection to continue and partial results to be saved.
* - **Profiling Support**
  - Built-in profiling with `yappi` integration for performance analysis. Identify bottlenecks in custom resources and agents.
```

---

For hands-on experience with these features, see the {doc}`../tutorials/README`.
