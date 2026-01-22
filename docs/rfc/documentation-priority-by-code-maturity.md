# RFC: Documentation Priority by Code Maturity

**Status**: Implemented  
**Author**: Auto-generated analysis (enhanced with detailed specifications)  
**Date**: 2025-01-13  
**Last Updated**: 2026-01-13  
**Related PR**: Documentation restructure

---

## Executive Summary

| Tier | Articles | Status | Recommended Action |
|------|----------|--------|-------------------|
| 🟢 **Tier 1** | 5 articles | Ready | Complete in this PR |
| 🟡 **Tier 2** | 6 articles | Draft-ready | Draft in this PR, iterate next sprint |
| 🔴 **Tier 3** | 9 articles | Blocked | Create GitHub issues, stub only |

**Key Findings**:
- **75% of resources servers** have comprehensive tests and READMEs
- **Tier 1 articles** have 100% code coverage and clear implementation patterns
- **Tier 3 blockers** are primarily missing reference implementations or external dependencies

**Recommended Priorities**:
1. **Immediate**: Complete `vllm.md` and `multi-step.md` (highest user value)
2. **This sprint**: Draft `llm-as-judge.md` and `ray-distributed.md`
3. **Defer**: All Tier 3 articles until code blockers resolved

---

## Summary

This RFC analyzes the new documentation articles introduced in this PR and prioritizes them based on underlying code maturity. The goal is to identify which articles can be completed with high confidence (grounded in working, tested code) versus which require additional code work before documentation can be finalized.

**This enhanced version includes detailed specifications for each article**, including:
- Specific content requirements
- Code references to use
- Acceptance criteria checklists
- Unblocking requirements for Tier 3 articles

---

## Problem Statement

The documentation restructure introduced 40 new articles across multiple sections. Many are marked as "stub" or "generated (not reviewed)". Without a clear prioritization strategy, reviewers and contributors may:

1. Spend time on articles that lack mature code backing
2. Write documentation that diverges from actual implementation
3. Miss opportunities to document well-established patterns

---

## Analysis Methodology

Each documentation topic was evaluated against:

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Code completeness** | High | Is there working implementation? |
| **Test coverage** | High | Are there unit/integration tests? |
| **README quality** | Medium | Does the component have internal docs? |
| **Example data** | Medium | Are there working examples? |
| **Complexity** | Low | Simpler = easier to document accurately |

---

## Target Personas

To ensure documentation resonance, articles are tailored to specific user personas:

| Persona | Description | Primary Tiers |
|---------|-------------|---------------|
| **Environment Designer** | Researchers building new tasks/verifiers | Tier 1 (`multi-step.md`), Tier 2 (`llm-as-judge.md`) |
| **Infrastructure Engineer** | MLOps/SREs deploying servers | Tier 1 (`vllm.md`), Tier 3 (`deployment-topology.md`) |
| **Data Scientist** | Preparing datasets for training | Tier 1 (`download-huggingface.md`) |
| **RL Practitioner** | Training models using Gym outputs | Tier 2 (`llm-as-judge.md`), Tier 3 (`trl.md`) |

---

## Success Metrics

Documentation is considered "Complete" when it meets the following "4-C" criteria:

1. **Correctness**: 100% of code examples run without modification against the current `main` branch.
2. **Completeness**: All configuration parameters from the source `dataclass` are documented.
3. **Clarity**: Includes at least one visual aid (diagram or table) and one "Quick Start" snippet.
4. **Connectivity**: Links to both the underlying source code and at least one related tutorial.

---

## Code Inventory

### Core Library (`nemo_gym/`)

| File | Lines | Test Coverage | Documentation Value |
|------|-------|---------------|---------------------|
| `cli.py` | 935 | ✅ `test_cli.py` | CLI reference, commands |
| `train_data_utils.py` | 819 | ✅ `test_train_data_utils.py` (45K lines) | Data preparation |
| `server_utils.py` | 599 | ✅ `test_server_utils.py` | Server internals |
| `config_types.py` | 570 | ✅ `test_config_types_help.py` | Configuration reference |
| `global_config.py` | 513 | ✅ `test_global_config.py` (28K lines) | Architecture concepts |
| `openai_utils.py` | 512 | ✅ `test_openai_utils.py` | API compatibility |
| `hf_utils.py` | 185 | ✅ `test_hf_utils.py` | HuggingFace download |
| `base_resources_server.py` | 73 | ✅ `test_base_resources_server.py` | Resources server patterns |
| `base_responses_api_model.py` | 56 | ✅ `test_base_responses_api_model.py` | Model server patterns |
| `base_responses_api_agent.py` | 56 | ✅ `test_base_responses_api_agent.py` | Agent patterns |

### Model Servers (`responses_api_models/`)

| Server | Implementation | Tests | Maturity |
|--------|----------------|-------|----------|
| `vllm_model/` | 641 lines, full feature set | ✅ | **High** - reasoning parser, token IDs, load balancing, error handling |
| `openai_model/` | ~200 lines | ✅ | **Medium** - thin wrapper |
| `azure_openai_model/` | ~150 lines | ✅ | **Medium** - thin wrapper |

### Resources Servers (`resources_servers/`)

| Server | Lines | Tests | README | Best For |
|--------|-------|-------|--------|----------|
| `workplace_assistant/` | 3,733 | ✅ | 31 lines | Complex multi-step, 26 tools |
| `calendar/` | 2,095 | ✅ | 210 lines | Multi-turn conversations |
| `code_gen/` | 1,906 | ✅ | 39 lines | Code execution verification |
| `math_advanced_calculations/` | 1,289 | ✅ | 32 lines | Math with tools |
| `swerl_llm_judge/` | 1,129 | ✅ | 186 lines | LLM judge patterns |
| `math_with_judge/` | 1,125 | ✅ | 56 lines | Judge verification |
| `equivalence_llm_judge/` | 872 | ✅ | 78 lines | LLM-as-judge |
| `mcqa/` | 841 | ✅ | **270 lines** | Multiple choice QA |
| `xlam_fc/` | 827 | ✅ | 27 lines | Function calling |
| `aviary/` | 704 | ✅ | 42 lines | Multi-environment |
| `structured_outputs/` | 646 | ✅ | 69 lines | JSON schema verification |
| `reasoning_gym/` | 564 | ✅ | 95 lines | Reasoning tasks |
| `example_multi_step/` | 449 | ✅ | 12 lines | **Tutorial reference** |
| `instruction_following/` | 347 | ✅ | 93 lines | Instruction verification |
| `math_with_code/` | 339 | ✅ | 141 lines | Code-assisted math |
| `example_session_state_mgmt/` | 318 | ✅ | 14 lines | State management |
| `google_search/` | ~400 | ✅ | **383 lines** | API integration |

---

## Tier Classification

### 🟢 Tier 1: Ready to Document (High Confidence)

These articles have mature, tested code and can be completed with high accuracy:

| Article | Supporting Code | Rationale |
|---------|-----------------|-----------|
| `docs/data/download-huggingface.md` | `hf_utils.py` (185 lines) | Complete implementation, clear API |
| `docs/environment-tutorials/multi-step.md` | `example_multi_step/` (449 lines) | Canonical example, clean implementation |
| `docs/model-server/vllm.md` | `vllm_model/` (641 lines) | Most mature model server, handles edge cases |
| `docs/resources-server/index.md` | `base_resources_server.py` (73 lines) | Stable base class, clear pattern |
| `docs/agent-server/index.md` | `simple_agent/` | Reference implementation |

**Estimated effort**: 2-4 hours each  
**Risk**: Low

---

#### Article Details: Tier 1

##### 📄 `docs/data/download-huggingface.md`

**Status**: Draft exists, needs review  
**Source Code**: `nemo_gym/hf_utils.py`, `nemo_gym/config_types.py`  
**Tests**: `tests/unit_tests/test_hf_utils.py`

**Scope**:
- CLI command reference for `ng_download_dataset_from_hf`
- All parameter options with examples
- Constraints on mutually exclusive parameters
- Private dataset authentication

**Key Content to Include**:

1. **Command Options Table** (from `DownloadJsonlDatasetHuggingFaceConfig`):
   - `repo_id`: HuggingFace repository ID
   - `artifact_fpath`: Path to specific file in repo (for raw JSONL download)
   - `output_dirpath` / `output_fpath`: Output destination
   - `split`: Dataset split (train/validation/test)
   - `hf_token`: Authentication for private repos

2. **Download Methods**:
   - Method 1: `artifact_fpath` → uses `hf_hub_download()` for raw file
   - Method 2: `split` → uses `datasets.load_dataset()` for structured datasets
   - Method 3: No split → downloads all available splits

3. **Constraints** (from `hf_utils.py:57-115`):
   - Cannot use both `output_dirpath` and `output_fpath`
   - Cannot use both `artifact_fpath` and `split`
   - `output_fpath` without `artifact_fpath` requires `split`

4. **Examples**:
   - Download all splits to directory
   - Download specific split to file
   - Download raw JSONL file
   - Private dataset with token

**Acceptance Criteria**:
- [ ] All CLI parameters documented with types
- [ ] Tab-set showing 3+ download methods
- [ ] Private dataset authentication explained
- [ ] Links to NVIDIA HF datasets collection

---

##### 📄 `docs/environment-tutorials/multi-step.md`

**Status**: Draft exists, good structure  
**Source Code**: `resources_servers/example_multi_step/app.py`  
**Tests**: `resources_servers/example_multi_step/tests/`

**Scope**:
- Conceptual explanation of multi-step vs multi-turn
- Complete working example with `example_multi_step`
- Tool registration and state management patterns
- Verification strategies

**Key Content to Include**:

1. **Conceptual Framework**:
   - Multi-step: Sequential tool calls within single turn (Model → Tool₁ → Result₁ → Tool₂ → ...)
   - Distinguish from multi-turn (conversation history)
   - Decision matrix: when to use which pattern

2. **Implementation Pattern** (from `example_multi_step/app.py`):
   - Tool registration with FastAPI `app.post()`
   - Request/response schemas with Pydantic
   - Session state management via `session_state` dict
   - `SESSION_ID_KEY` for per-session isolation

3. **Verification Strategies**:
   - Final state verification: extract from last tool call
   - Partial credit: reward based on progress
   - Code examples for both patterns

4. **Data Format**:
   - Required: `responses_create_params.input`
   - Task-specific: `expected_values`, `expected_synonyms`

5. **Configuration**:
   - `max_steps`: limit tool-calling iterations
   - `done_if_no_tool_calls`: end rollout behavior

**Acceptance Criteria**:
- [ ] Quick-start runnable in <5 minutes
- [ ] State management code example with session IDs
- [ ] Two verification strategy examples
- [ ] Links to `example_session_state_mgmt/` for advanced patterns
- [ ] Links to `workplace_assistant/` for complex real-world example

---

##### 📄 `docs/model-server/vllm.md`

**Status**: Stub exists, significant content added  
**Source Code**: `responses_api_models/vllm_model/app.py` (641 lines)  
**Tests**: `responses_api_models/vllm_model/tests/test_app.py`

**Scope**:
- vLLM server startup patterns
- NeMo Gym configuration reference
- Function calling with tool parsers
- Reasoning model support
- Load balancing and training integration

**Key Content to Include**:

1. **vLLM Server Startup** (tab-set):
   - Basic: `vllm serve <model> --host --port --api-key`
   - With Tool Calling: `--enable-auto-tool-choice --tool-call-parser hermes`
   - Multi-GPU: `--tensor-parallel-size N`
   - Reasoning Models: **Do NOT use `--reasoning-parser`** (NeMo Gym handles internally)

2. **Configuration Reference** (from `VLLMModelConfig`):
   | Parameter | Type | Default | Description |
   |-----------|------|---------|-------------|
   | `base_url` | `str \| list[str]` | Required | Endpoint(s) for load balancing |
   | `api_key` | `str` | Required | Auth key |
   | `model` | `str` | Required | Model name |
   | `return_token_id_information` | `bool` | `false` | Enable for training |
   | `uses_reasoning_parser` | `bool` | `false` | Parse `<think>` tags |
   | `replace_developer_role_with_system` | `bool` | `false` | Role compatibility |
   | `chat_template_kwargs` | `dict` | `null` | Template overrides |
   | `extra_body` | `dict` | `null` | vLLM-specific params |

3. **Function Calling**:
   - Tool parser selection by model family
   - Conversion flow: Responses API → Chat Completions → vLLM → back
   - Supported parsers: hermes (Qwen3), llama3_json (Llama), mistral

4. **Training Integration**:
   - `return_token_id_information: true` enables:
     - `prompt_token_ids`
     - `generation_token_ids`
     - `generation_log_probs`
   - Required for GRPO and policy gradient methods

5. **Troubleshooting** (dropdowns):
   - Context length errors
   - Connection errors
   - Tool calling not working
   - Chat template issues

**Acceptance Criteria**:
- [ ] 4+ vLLM startup command examples
- [ ] Complete configuration reference table
- [ ] Tool parser table by model family
- [ ] Reasoning model warning clearly visible
- [ ] Training integration section with `vllm_model_for_training.yaml`

---

##### 📄 `docs/resources-server/index.md`

**Status**: Draft exists, needs expansion  
**Source Code**: `nemo_gym/base_resources_server.py` (73 lines)  
**Tests**: `tests/unit_tests/test_base_resources_server.py`

**Scope**:
- Core concepts: tools, verification, session management
- `SimpleResourcesServer` pattern
- Configuration reference
- Links to how-to guides

**Key Content to Include**:

1. **Core Concepts**:
   - Tools = FastAPI POST endpoints callable by models
   - Verification = `/verify` endpoint returns reward signal
   - Session = Per-rollout state via `/seed_session`

2. **Base Classes** (from `base_resources_server.py`):
   - `BaseResourcesServerConfig`: Server configuration
   - `BaseRunRequest`: Contains `responses_create_params`
   - `BaseVerifyRequest`: Adds `response` to run request
   - `BaseVerifyResponse`: Adds `reward` field
   - `BaseSeedSessionRequest/Response`: Session initialization

3. **SimpleResourcesServer Pattern**:
   ```python
   class MyServer(SimpleResourcesServer):
       def setup_webserver(self) -> FastAPI:
           app = super().setup_webserver()
           app.post("/my_tool")(self.my_tool)
           return app
       
       async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
           reward = compute_reward(body.response)
           return BaseVerifyResponse(**body.model_dump(), reward=reward)
   ```

4. **Configuration**:
   - `domain` field (required): math, coding, agent, knowledge
   - `entrypoint`: Python file with server class
   - Tool-specific config fields

**Acceptance Criteria**:
- [ ] Class hierarchy diagram or explanation
- [ ] Minimal working server example
- [ ] All base request/response types documented
- [ ] Grid cards linking to all how-to guides

---

##### 📄 `docs/agent-server/index.md`

**Status**: Draft exists, needs expansion  
**Source Code**: `responses_api_agents/simple_agent/`  
**Tests**: Agent integration tests

**Scope**:
- Agent role in the rollout lifecycle
- `SimpleResponsesAPIAgent` pattern
- Configuration reference
- Integration with external agents

**Key Content to Include**:

1. **Rollout Lifecycle**:
   1. Receive task via `/run`
   2. Initialize session → `/seed_session` on resources server
   3. Call model → `/v1/responses` on model server
   4. Execute tools → tool endpoints on resources server
   5. Repeat steps 3-4 until done or max_steps
   6. Verify → `/verify` on resources server

2. **Agent Endpoints**:
   - `/v1/responses`: Direct model passthrough
   - `/run`: Complete rollout orchestration

3. **Configuration**:
   ```yaml
   my_agent:
     responses_api_agents:
       simple_agent:
         entrypoint: app.py
         resources_server:
           type: resources_servers
           name: my_resources
         model_server:
           type: responses_api_models
           name: policy_model
         max_steps: 10
   ```

4. **Custom Agent Implementation**:
   - Extending `SimpleResponsesAPIAgent`
   - Override `responses()` for custom model handling
   - Override `run()` for custom orchestration

**Acceptance Criteria**:
- [ ] Lifecycle diagram or numbered steps
- [ ] Configuration example with all options
- [ ] Custom agent skeleton code
- [ ] Grid cards for external agent integrations

### 🟡 Tier 2: Good Foundation (Medium Confidence)

Working code exists but may need verification or has higher complexity:

| Article | Supporting Code | Gap |
|---------|-----------------|-----|
| `docs/environment-tutorials/llm-as-judge.md` | `equivalence_llm_judge/` (872 lines) | Document configuration options |
| `docs/environment-tutorials/multi-turn.md` | `calendar/` (2,095 lines) | High complexity, needs simplification |
| `docs/resources-server/integrate-apis.md` | `google_search/` (383-line README) | Best reference available |
| `docs/infrastructure/ray-distributed.md` | `server_utils.py` Ray usage | Needs concrete examples |
| `docs/model-server/openai.md` | `openai_model/` | Simple but undifferentiated |
| `docs/environment-tutorials/rlhf-reward-models.md` | No implementation yet | Stubbed - awaiting implementation |

**Estimated effort**: 4-8 hours each  
**Risk**: Medium

---

#### Article Details: Tier 2

##### 📄 `docs/environment-tutorials/llm-as-judge.md`

**Status**: Stub  
**Source Code**: `resources_servers/equivalence_llm_judge/` (872 lines)  
**README Quality**: 78 lines (good)

**Scope**:
- When to use LLM-as-judge vs deterministic verification
- Configuration of judge prompts and labels
- Advanced features: swap checking, per-record regex
- Judge model selection guidance

**Key Content to Include**:

1. **When to Use LLM-as-Judge**:
   - Open-ended responses without single correct answer
   - Semantic equivalence checking
   - Tasks where regex/exact match fails
   - Quality/style evaluation

2. **Configuration Reference** (from README):
   | Parameter | Type | Default | Description |
   |-----------|------|---------|-------------|
   | `judge_system_message` | `str` | `null` | Optional system prompt |
   | `judge_prompt_template` | `str` | Required | Placeholders: `{question}`, `{expected_answer}`, `{generated_answer}` |
   | `judge_equal_label` | `str` | `[[A=B]]` | Label for equivalence |
   | `judge_not_equal_label` | `str` | `[[A!=B]]` | Label for non-equivalence |
   | `check_twice_swap` | `bool` | `false` | Reduce position bias |
   | `reward_if_swap_fails` | `float` | `0.0` | Reward when swap disagrees |
   | `use_per_record_regex` | `bool` | `true` | Per-record answer extraction |
   | `check_full_generation_on_fail` | `bool` | `true` | Fallback on regex failure |

3. **Example Configuration**:
   ```yaml
   equivalence_llm_judge:
     resources_servers:
       equivalence_llm_judge:
         entrypoint: app.py
         judge_prompt_template: |
           <|Problem|>
           {question}
           <|Gold|>
           {expected_answer}
           <|Prediction|>
           {generated_answer}
   ```

4. **Swap Checking** (bias mitigation):
   - First pass: compare generated vs expected
   - If equal, second pass: swap positions
   - Only award reward if both passes agree
   - Reduces LLM position bias

5. **Judge Model Guidance**:
   - Tested with Gemma3-27B-it
   - Check license compliance for your use case
   - Larger judges generally more reliable

**Acceptance Criteria**:
- [ ] Decision tree: when to use LLM judge
- [ ] Complete configuration table
- [ ] Working example config
- [ ] Swap checking explanation with diagram
- [ ] Dataset example from SciQ

---

##### 📄 `docs/environment-tutorials/multi-turn.md`

**Status**: Stub  
**Source Code**: `resources_servers/calendar/` (2,095 lines)  
**README Quality**: 210 lines (excellent)

**Scope**:
- Multi-turn conversation patterns
- Session state across turns
- Verification with conversation history
- Data generation pipeline

**Key Content to Include**:

1. **Multi-Turn vs Multi-Step**:
   - Multi-turn: User↔Assistant exchanges over time
   - Multi-step: Tool calls within single assistant turn
   - Calendar example: 7 events scheduled across conversation

2. **Conversation State Management**:
   - Track calendar state across turns
   - Verify intermediate states (not just final)
   - Handle constraint validation per turn

3. **Verification Logic** (simplified from `calendar/app.py`):
   - Parse JSON calendar from response
   - Check event count
   - Validate time constraints: before, after, between, at
   - Detect time conflicts
   - Return binary reward (0 or 1)

4. **Data Format**:
   ```json
   {
     "responses_create_params": {
       "input": [
         {"role": "system", "content": "You are a scheduling assistant..."},
         {"role": "user", "content": "Schedule a meeting at 10am"}
       ]
     },
     "exp_cal_state": {
       "1": {"event_id": 1, "event_name": "Meeting", "duration": 60, ...}
     }
   }
   ```

5. **Data Generation Pipeline** (advanced section):
   - Step 1: `create_synth_conversations.py` with personas
   - Step 2: `generate_rollouts.py` to get model responses
   - Step 3: `dataset_preprocess.py` for training format

**Acceptance Criteria**:
- [ ] Clear multi-turn vs multi-step distinction
- [ ] Simplified calendar server example (not full 2K lines)
- [ ] Verification logic walkthrough
- [ ] Link to full `calendar/README.md` for data generation

---

##### 📄 `docs/resources-server/integrate-apis.md`

**Status**: Stub  
**Source Code**: `resources_servers/google_search/` (400 lines)  
**README Quality**: 383 lines (excellent - best available)

**Scope**:
- Integrating external REST APIs as tools
- API key management via `env.yaml`
- Error handling and rate limiting
- Inheritance pattern for reuse

**Key Content to Include**:

1. **Pattern: API as Tool**:
   - Wrap external API in FastAPI endpoint
   - Define Pydantic request/response schemas
   - Handle authentication via config

2. **Example: Google Search Integration**:
   ```python
   async def search(self, body: SearchRequest) -> SearchResponse:
       """Search Google and return results."""
       response = await self.google_client.search(body.query)
       return SearchResponse(results=response.items)
   ```

3. **Environment Configuration** (from `google_search/README.md`):
   ```yaml
   # env.yaml
   google_search:
     resources_servers:
       google_search:
         google_api_key: <your_api_key>
         google_cx: <your_cx_engine>
   ```

4. **Tool Definitions for Inheritance**:
   - `search`: Query external API, return structured results
   - `browse`: Fetch and parse webpage content
   - Tools can be inherited by other environments

5. **Error Handling Patterns**:
   - API rate limiting
   - Timeout handling
   - Graceful degradation

**Acceptance Criteria**:
- [ ] Complete API integration example
- [ ] Environment variable configuration
- [ ] Tool definition schema for inheritance
- [ ] Link to Google Search README for full details

---

##### 📄 `docs/infrastructure/ray-distributed.md`

**Status**: Draft exists, good structure  
**Source Code**: `nemo_gym/server_utils.py`, `nemo_gym/global_config.py`  
**Tests**: Various integration tests

**Scope**:
- Ray's role in NeMo Gym (CPU parallelization, NOT rollout collection)
- Automatic vs manual cluster connection
- `@ray.remote` patterns for verification
- Version constraints and troubleshooting

**Key Content to Include**:

1. **When Ray is Used**:
   - ✅ CPU-intensive verification tasks
   - ✅ Batch processing of independent items
   - ❌ NOT for rollout collection (uses async HTTP)
   - ❌ NOT for HTTP requests (use asyncio)

2. **Initialization Flow**:
   - Main process: Ray starts in `RunHelper.start()`
   - Server processes: Call `initialize_ray()`, connect to cluster
   - Address shared via global config

3. **Connecting to Existing Cluster**:
   ```yaml
   ray_head_node_address: "ray://your-cluster:10001"
   ```

4. **@ray.remote Pattern**:
   ```python
   @ray.remote(scheduling_strategy="SPREAD")
   def verify_single(answer: str, expected: str) -> bool:
       return compute_match(answer, expected)
   
   # Submit tasks
   futures = [verify_single.remote(a, e) for a, e in pairs]
   results = ray.get(futures)
   ```

5. **Version Constraints**:
   - Ray versions must match exactly
   - Child servers receive `ray[default]=={ray_version}`

**Acceptance Criteria**:
- [ ] Clear "when to use" vs "when NOT to use" section
- [ ] Working `@ray.remote` example
- [ ] Cluster connection config
- [ ] Troubleshooting dropdowns (version mismatch, sandbox errors)

---

##### 📄 `docs/model-server/openai.md`

**Status**: Stub  
**Source Code**: `responses_api_models/openai_model/` (~200 lines)  
**Tests**: `responses_api_models/openai_model/tests/test_app.py`

**Scope**:
- Configuration for OpenAI API access
- When to use OpenAI vs vLLM
- Model selection guidance

**Key Content to Include**:

1. **When to Use OpenAI**:
   - Quick prototyping without GPU setup
   - Access to latest models (GPT-4, o1, etc.)
   - Production inference without self-hosting
   - Baseline comparisons

2. **Configuration**:
   ```yaml
   policy_model:
     responses_api_models:
       openai_model:
         entrypoint: app.py
         api_key: ${OPENAI_API_KEY}
         model: gpt-4.1-2025-04-14
   ```

3. **Environment Setup**:
   ```yaml
   # env.yaml
   OPENAI_API_KEY: sk-...
   ```

4. **Limitations vs vLLM**:
   - No token ID information (training incompatible)
   - Rate limits
   - Cost considerations
   - Data privacy

**Acceptance Criteria**:
- [ ] Decision matrix: OpenAI vs vLLM
- [ ] Configuration example
- [ ] Environment variable setup
- [ ] Limitations clearly documented

### 🔴 Tier 3: Needs Code Work (Low Confidence)

Documentation is blocked on code development or external dependencies:

| Article | Blocker |
|---------|---------|
| `docs/infrastructure/deployment-topology.md` | Conceptual; no concrete tooling |
| `docs/resources-server/containerize.md` | No Dockerfile templates in repo |
| `docs/resources-server/profile.md` | No profiling utilities implemented |
| `docs/model-server/azure-openai.md` | Thin wrapper, low differentiation from OpenAI |
| `docs/model-server/responses-native.md` | Unclear scope vs vLLM |
| `docs/training-tutorials/trl.md` | External framework, needs integration code |
| `docs/training-tutorials/verl.md` | External framework, partial integration |
| `docs/training-tutorials/nemo-customizer.md` | External service, API documentation |
| `docs/environment-tutorials/user-modeling.md` | No reference implementation |

**Estimated effort**: Unknown (blocked)  
**Risk**: High

---

#### Article Details: Tier 3

##### 📄 `docs/infrastructure/deployment-topology.md`

**Status**: Conceptual  
**Source Code**: None  
**Blocker**: No deployment scripts, Terraform, or Kubernetes manifests exist

**Scope** (if unblocked):
- Production deployment architectures
- Single-node vs multi-node topologies
- Model server placement strategies
- Resources server scaling patterns

**Unblocking Requirements**:
1. Create example deployment configurations (Kubernetes, Docker Compose, or Terraform)
2. Document at least one production deployment from real users
3. Add deployment scripts to repo

**Recommended Action**: Mark as stub, link to [GitHub Issue #XXX] for tracking. Consider gathering deployment patterns from internal teams using NeMo Gym.

---

##### 📄 `docs/resources-server/containerize.md`

**Status**: Stub  
**Source Code**: None  
**Blocker**: No Dockerfile templates in repository

**Scope** (if unblocked):
- Creating Dockerfiles for resources servers
- Multi-stage builds for efficiency
- GPU support configuration
- Docker Compose for local development

**Unblocking Requirements**:
1. Create `resources/Dockerfile.template` with best practices
2. Add example Dockerfile to `example_multi_step/`
3. Create Docker Compose example for common setups

**Recommended Action**: Create GitHub issue. Low effort to unblock—estimate 4 hours to create templates.

**Proposed Template Content**:
```dockerfile
# resources/Dockerfile.template
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# Run server
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

##### 📄 `docs/resources-server/profile.md`

**Status**: Stub  
**Source Code**: None  
**Blocker**: No profiling utilities implemented

**Scope** (if unblocked):
- Measuring verification throughput
- Identifying bottlenecks
- Resource utilization monitoring
- Performance benchmarking

**Unblocking Requirements**:
1. Add profiling decorator or utility in `nemo_gym/`
2. Create example profiling script
3. Document key metrics to track

**Recommended Action**: Create GitHub issue. Medium effort—estimate 8 hours to implement profiling utilities.

**Proposed Profiling Approach**:
```python
# nemo_gym/profiling.py (proposed)
import time
from functools import wraps

def profile_endpoint(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper
```

---

##### 📄 `docs/model-server/azure-openai.md`

**Status**: Stub  
**Source Code**: `responses_api_models/azure_openai_model/` (~150 lines)  
**Blocker**: Thin wrapper, almost identical to OpenAI

**Scope** (if worth documenting):
- Azure-specific configuration
- Deployment name vs model name
- Regional endpoint setup

**Differentiation from OpenAI**:
- `azure_endpoint` instead of `base_url`
- `api_version` required
- `deployment_name` instead of `model`

**Recommended Action**: Consider merging into `openai.md` as a subsection rather than separate article. Low value as standalone.

**If Kept Separate**:
```yaml
# Azure OpenAI Configuration
azure_model:
  responses_api_models:
    azure_openai_model:
      entrypoint: app.py
      azure_endpoint: https://your-resource.openai.azure.com
      api_version: "2024-02-15-preview"
      deployment_name: gpt-4
      api_key: ${AZURE_OPENAI_API_KEY}
```

---

##### 📄 `docs/model-server/responses-native.md`

**Status**: Stub  
**Source Code**: Unclear  
**Blocker**: Scope overlaps with vLLM, unclear differentiation

**Questions to Resolve**:
1. What is "Responses Native" vs vLLM?
2. Is this for models with native Responses API support?
3. What models would use this instead of vLLM?

**Recommended Action**: Clarify scope with team. May be deprecated or merged into vLLM article.

---

##### 📄 `docs/training-tutorials/trl.md`

**Status**: Stub exists with TODO markers  
**Source Code**: None in NeMo Gym  
**External**: [Hugging Face TRL](https://huggingface.co/docs/trl)  
**Blocker**: No integration code exists

**Scope** (if unblocked):
- Using NeMo Gym verifiers as TRL reward functions
- PPO/DPO training with NeMo Gym environments
- Single-step task training

**Unblocking Requirements**:
1. Create `examples/trl_integration/` with working PPO example
2. Implement reward function wrapper:
   ```python
   # Proposed: nemo_gym/integrations/trl.py
   class NeMoGymRewardFunction:
       def __init__(self, resources_server_url: str):
           self.client = ResourcesServerClient(resources_server_url)
       
       def __call__(self, outputs: List[str]) -> List[float]:
           rewards = []
           for output in outputs:
               verify_response = self.client.verify(output)
               rewards.append(verify_response.reward)
           return rewards
   ```

**Recommended Action**: Create GitHub issue. High value if implemented—estimate 16 hours for working example.

---

##### 📄 `docs/training-tutorials/verl.md`

**Status**: Stub  
**Source Code**: Partial integration exists  
**External**: [veRL](https://github.com/volcengine/verl)  
**Blocker**: Partial integration, needs documentation

**Scope** (if unblocked):
- veRL architecture overview
- NeMo Gym as environment for veRL
- Multi-GPU training setup

**Unblocking Requirements**:
1. Document existing integration patterns
2. Create end-to-end example
3. Test with current veRL version

**Recommended Action**: Medium priority—partial code exists. Estimate 8 hours to document existing integration.

---

##### 📄 `docs/training-tutorials/nemo-customizer.md`

**Status**: Stub  
**Source Code**: None in NeMo Gym  
**External**: NVIDIA NeMo Customizer (cloud service)  
**Blocker**: External service, requires separate API documentation

**Scope** (if unblocked):
- NeMo Customizer service overview
- Uploading NeMo Gym datasets
- Launching training jobs
- Downloading fine-tuned models

**Unblocking Requirements**:
1. Coordinate with NeMo Customizer team for API access
2. Create example workflow
3. Document authentication and quotas

**Recommended Action**: Low priority for this PR—external dependency. Link to NeMo Customizer docs when available.

---

##### 📄 `docs/environment-tutorials/user-modeling.md`

**Status**: Stub  
**Source Code**: None  
**Blocker**: No reference implementation

**Scope** (if unblocked):
- Simulating user behavior for training
- Multi-agent scenarios
- Persona-based interactions
- Evaluation with synthetic users

**Unblocking Requirements**:
1. Create `resources_servers/example_user_modeling/`
2. Implement simple user simulator
3. Add dataset with persona-based interactions

**Related Work**:
- `calendar/` uses Nemotron-Personas-USA for synthetic conversations
- Could extract user modeling patterns from `calendar/create_synth_conversations.py`

**Recommended Action**: Create GitHub issue. High complexity—estimate 24+ hours. Consider extracting patterns from existing `calendar/` implementation as starting point.

---

## Recommendations

### Immediate Actions (This PR)

1. **Prioritize Tier 1 articles for review** - These can be validated against working code
2. **Mark Tier 3 articles as stubs** - Avoid investing review time until blockers are resolved
3. **Use existing READMEs as source material** - `google_search/` and `mcqa/` have excellent internal docs

### Short-Term (Next Sprint)

1. **Create reference implementations** for undocumented patterns:
   - Dockerfile template for containerization
   - Profiling example script
   - User modeling example server

2. **Extract patterns from complex servers** into documentation:
   - `workplace_assistant/` → Advanced multi-step patterns
   - `calendar/` → Multi-turn conversation patterns

### Long-Term

1. **External framework tutorials** should link to external docs rather than duplicate
2. **Deployment topology** needs production deployment examples from real users
3. **Consider deprecating** low-value articles (Azure OpenAI, Responses Native) or merging into parent pages

---

## Visual & Formatting Standards

To maintain a professional and consistent look across the documentation site:

### 1. Diagramming
- **Tool**: Use [Mermaid.js](https://mermaid.js.org/) for all flowcharts and sequence diagrams.
- **Theme**: Use the `neutral` theme to ensure readability in both light and dark modes.
- **Lifecycle Diagrams**: Required for all "Agent" and "Orchestration" articles.

### 2. Code Blocks
- **Language Tags**: Always specify the language (e.g., `bash`, `python`, `yaml`).
- **Filename Headers**: Use the format `::: {code-block} python :caption app.py` (if supported by the renderer) or a bold filename above the block.
- **Tab-sets**: Use `::::{tab-set}` for multi-platform instructions (e.g., local vs. Ray) or model selection.

### 3. Callouts
- **Important**: Use `:::{important}` for critical warnings (e.g., reasoning parser issues).
- **Tip**: Use `:::{tip}` for performance optimizations or shorthand CLI commands.
- **Note**: Use `:::{note}` for version constraints or external links.

---

## Documentation Maintenance Strategy

As NeMo Gym is in active development, documentation must evolve with the code:

1. **Automated Validation**: Integrate `pytest --doctest-modules` or a similar tool to verify snippets in Tier 1 articles.
2. **Maturity Re-evaluation**: Every major release (0.x.0), re-evaluate Tier 2 and Tier 3 articles. If code blockers are resolved, promote them.
3. **API Reference Sync**: Use `mkdocstrings` or `sphinx-autodoc` to ensure `BaseResourcesServer` and `VLLMModelConfig` parameters stay synced with source code.
4. **Stale Content Warning**: Any article not updated for 2+ major releases should be marked with a "Legacy" warning or reviewed for deprecation.

---

## Documentation Completion Checklist

### Tier 1 (Target: Complete)

#### `docs/data/download-huggingface.md`

- [ ] All CLI parameters documented with types
- [ ] Tab-set showing 3+ download methods
- [ ] Private dataset authentication explained
- [ ] Links to NVIDIA HF datasets collection
- [ ] Verify against `hf_utils.py` implementation

#### `docs/environment-tutorials/multi-step.md`

- [ ] Quick-start runnable in <5 minutes
- [ ] State management code example with session IDs
- [ ] Two verification strategy examples (final state, partial credit)
- [ ] Links to `example_session_state_mgmt/` for advanced patterns
- [ ] Links to `workplace_assistant/` for complex real-world example

#### `docs/model-server/vllm.md`

- [ ] 4+ vLLM startup command examples (tab-set)
- [ ] Complete configuration reference table
- [ ] Tool parser table by model family
- [ ] Reasoning model warning clearly visible (use `:::{important}`)
- [ ] Training integration section with `vllm_model_for_training.yaml`
- [ ] Troubleshooting dropdowns for common issues

#### `docs/resources-server/index.md`

- [ ] Class hierarchy explanation (BaseResourcesServer → SimpleResourcesServer)
- [ ] Minimal working server example
- [ ] All base request/response types documented
- [ ] Grid cards linking to all how-to guides

#### `docs/agent-server/index.md`

- [ ] Lifecycle diagram or numbered steps
- [ ] Configuration example with all options
- [ ] Custom agent skeleton code
- [ ] Grid cards for external agent integrations

### Tier 2 (Target: Draft)

#### `docs/environment-tutorials/llm-as-judge.md`

- [ ] Decision tree: when to use LLM judge
- [ ] Complete configuration table from `equivalence_llm_judge/README.md`
- [ ] Working example config
- [ ] Swap checking explanation
- [ ] Dataset example from SciQ

#### `docs/environment-tutorials/multi-turn.md`

- [ ] Clear multi-turn vs multi-step distinction
- [ ] Simplified calendar server example
- [ ] Verification logic walkthrough
- [ ] Link to full `calendar/README.md` for data generation

#### `docs/resources-server/integrate-apis.md`

- [ ] Complete API integration example
- [ ] Environment variable configuration via `env.yaml`
- [ ] Tool definition schema for inheritance
- [ ] Link to Google Search README for full details

#### `docs/infrastructure/ray-distributed.md`

- [ ] Clear "when to use" vs "when NOT to use" section
- [ ] Working `@ray.remote` example
- [ ] Cluster connection config
- [ ] Troubleshooting dropdowns

#### `docs/model-server/openai.md`

- [ ] Decision matrix: OpenAI vs vLLM
- [ ] Configuration example
- [ ] Environment variable setup
- [ ] Limitations clearly documented

### Tier 3 (Target: Stub with tracking issue)

- [ ] `deployment-topology.md` → Create issue for deployment patterns
- [ ] `containerize.md` → Create issue + Dockerfile template
- [ ] `profile.md` → Create issue for profiling utilities
- [ ] `azure-openai.md` → Consider merging into `openai.md`
- [ ] `responses-native.md` → Clarify scope or deprecate
- [ ] `trl.md` → Create issue for TRL integration code
- [ ] `verl.md` → Create issue for veRL documentation
- [ ] `nemo-customizer.md` → Link to external docs when available
- [ ] `user-modeling.md` → Create issue, reference `calendar/` patterns

---

## Appendix: Code Quality Metrics

### Test Coverage by Component

```bash
tests/unit_tests/
├── test_train_data_utils.py     45,156 lines (comprehensive)
├── test_global_config.py        28,284 lines (comprehensive)
├── test_dataset_viewer.py       22,245 lines
├── test_server_utils.py          9,323 lines
├── test_server_status.py         8,223 lines
├── test_config_types_help.py     7,000 lines
├── test_cli.py                   5,080 lines
└── ... (others < 2,000 lines)
```

### Resources Servers with Tests

✅ 20/23 resources servers have test directories  
❌ Missing: `comp_coding/`, `example_simple_weather/`, `library_judge_math/`

### README Quality Distribution

- **Excellent (>200 lines)**: `google_search/`, `mcqa/`, `calendar/`, `swerl_gen/`
- **Good (50-200 lines)**: `swerl_llm_judge/`, `math_with_code/`, `reasoning_gym/`, `instruction_following/`
- **Minimal (<50 lines)**: Most example servers, `workplace_assistant/`

---

## Decision

**Proposed**: Adopt tiered documentation approach based on code maturity.

**Alternatives Considered**:

1. Document all articles equally - Rejected (wastes effort on blocked topics)
2. Wait for all code to mature - Rejected (delays value delivery)
3. External-only for framework tutorials - Partially adopted

---

## References

- [Existing RFC: Training Tutorials Restructure](./training-tutorials-restructure.md)
- [NeMo Gym Contributing Guide](../contribute/index.md)
- [Resources Server Template](../../resources/resources_server_template.py)
