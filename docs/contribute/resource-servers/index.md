(contribute-resource-servers)=

# Contribute Resource Servers

This section covers requirements for contributing new resource servers (training environments) to NeMo Gym. Resource servers are high-priority contributions that expand Gym's training capabilities.

:::{note}
For step-by-step implementation instructions, refer to {doc}`/tutorials/creating-resource-server`. This guide focuses on contribution requirements and quality standards for pull requests.
:::

## Prerequisites

Before contributing a resource server, ensure you have:

- Completed the {doc}`/tutorials/creating-resource-server` tutorial
- A working resource server with tests passing locally
- Example data and rollouts generated
- Familiarity with the PR review process

## Contribution Workflow

Contributing a resource server follows this sequence:

```{list-table}
:header-rows: 1
:widths: 10 25 65

* - Step
  - Phase
  - Description
* - 1
  - Implementation
  - Build your resource server following the tutorial and required structure
* - 2
  - Testing
  - Write unit tests and generate example rollouts
* - 3
  - Validation
  - Run reward profiling and training-based validation
* - 4
  - Documentation
  - Complete README with licensing and usage information
* - 5
  - PR submission
  - Submit PR with all required artifacts and information
* - 6
  - Review
  - Address reviewer feedback and verify reproducibility
```

## Required Artifacts

Your resource server must include these files:

```{list-table}
:header-rows: 1
:widths: 30 70

* - File
  - Description
* - `app.py`
  - Main server implementation with `verify` function
* - `configs/*.yaml`
  - Configuration with valid `domain` field
* - `tests/test_app.py`
  - At least one unit test
* - `data/example.jsonl`
  - At least five example inputs
* - `data/example_rollouts.jsonl`
  - Pre-generated rollouts from example data (generate before submitting PR)
* - `requirements.txt`
  - Python dependencies
* - `README.md`
  - Documentation with licensing information
```

## Domain Categories

Set the `domain` field in your configuration to categorize your resource server:

```{list-table}
:header-rows: 1
:widths: 25 75

* - Domain
  - Use Case
* - `math`
  - Mathematical problem-solving
* - `coding`
  - Code generation and programming
* - `agent`
  - Agent-based interactions and tool calling
* - `knowledge`
  - Knowledge-based question answering
* - `instruction_following`
  - Instruction following benchmarks
* - `other`
  - General purpose or uncategorized
```

## Technical Requirements

Your resource server implementation must follow these design patterns to ensure compatibility with NeMo Gym's training infrastructure.

### Async-First Design

All endpoint handlers must be asynchronous. This is critical for handling concurrent requests during large-scale training.

```python
# Correct: async function
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)

# Incorrect: synchronous function will block other requests
def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

Avoid spawning additional threads or processes unless necessary. A single Gym instance can handle tens of thousands of concurrent requests when properly implemented.

### Use NeMo Gym OpenAI Utilities

Use the NeMo Gym OpenAI client and data models from `nemo_gym.openai_utils` rather than external clients such as LiteLLM or the standard OpenAI SDK:

```python
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
```

The NeMo Gym client is optimized for scale and provides consistent behavior. External clients often preprocess or postprocess inputs and outputs in ways that interfere with training data collection.

### Pydantic Models for Validation

Use Pydantic models for request and response validation. Extend the base classes from `nemo_gym.base_resources_server`:

```python
from pydantic import BaseModel
from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse

class MyVerifyRequest(BaseVerifyRequest):
    expected_result: str
    difficulty: int

class MyVerifyResponse(BaseVerifyResponse):
    reasoning: str
```

### Error Handling

Tool execution errors must be propagated back to the model rather than crashing the server. This enables the model to learn from and correct its mistakes:

```python
async def execute_tool(self, path: str, body: ToolRequest) -> ToolResponse:
    try:
        result = self.tool_functions[path](**body.model_dump())
        return ToolResponse(output=result)
    except Exception as e:
        # Return error to model so it can correct itself
        return ToolResponse(output=f"Error executing tool '{path}': {str(e)}")
```

Only raise `HTTPException` for invalid session states or malformed requests—not for tool execution failures that the model should learn from.

### Configuration via Config Files

Pass configuration through NeMo Gym config files rather than environment variables. This ensures reproducibility and makes debugging easier:

```yaml
# configs/my_server.yaml
host: 0.0.0.0
port: 8000
domain: agent
# Add server-specific configuration here
```

### Multi-Step Rollout Support

For multi-step or multi-turn scenarios, the model returns additional training information on response messages:

- `prompt_token_ids`
- `generation_token_ids`
- `generation_log_probs`

When constructing messages for subsequent model calls, propagate this information from previous responses to maintain the training data chain.

## Quality Requirements

### PR Description Requirements

Include the following in your pull request description:

:::{dropdown} Required PR Information
:icon: checklist

- **Dataset reference**: Link to corresponding dataset (if applicable)
- **Prompt description**: Source and domain coverage of prompts
- **Environment description**: What the environment simulates
- **Verifier description**: How verification works and correctness validation
- **Legal approval status**: Note if synthetically generated with open models
- **Command used**: The exact `ng_run` command to start the server
- **Five example rollouts**: Demonstrate correct reward signals
- **Additional notes**: Any special configuration or setup requirements
:::

### Reward Profiling

Run inference to validate your reward distribution:

:::{dropdown} Reward Profiling Requirements
:icon: graph

1. Use a ~500 sample subset (minimum)
2. Use Qwen3 30B A3B or equivalent model
3. Generate 16 responses per prompt
4. Report reward distribution statistics

**Expected reward distribution**

Environments should meaningfully separate model capabilities:

- Weaker models (30B-class): Target pass rate below 30%
- Stronger models (frontier-class): Target pass rate around 70-75%

**For tool-calling environments**

Provide tool call metrics and correlation analysis:

- Average tool calls per trajectory (target: more than two to three calls)
- Correlation between tool call count and reward
- Distribution of tool call types
:::

### Training-Based Validation

Demonstrate meaningful training signal:

:::{dropdown} Training Validation Requirements
:icon: rocket

1. Train with GRPO on Qwen 30B A3B Instruct (or equivalent)
2. Use NeMo RL + Gym
3. Include training accuracy curve
4. Include test benchmark accuracy curve (if applicable)

**Acceptance criteria**

Your environment should demonstrate:

- **Reward separation**: Environment meaningfully separates strong and weak models
- **Reward fidelity**: Similar-capability models perform consistently across model families
- **Tool use correlation**: For agent environments, higher tool call counts should correlate with improved rewards
- **Training improvement**: Model performance should improve during training
:::

:::{tip}
Review 10 to 20 collected trajectories manually before submission. Verify that the reward signal correctly identifies successful completions and that tool calls follow a logical progression toward the goal.
:::

## README Template

Your `README.md` must include at minimum:

:::{dropdown} README Template
:icon: file

```markdown
# Description

[Brief description of the environment, domain, and use case]

[Optional: Commands to run the server and collect rollouts]

# Licensing Information

Code: [License, typically Apache 2.0]

Data: [License for any data included]

Dependencies
- nemo_gym: Apache 2.0
- [Other dependencies with licenses]
```

:::

:::{tip}
Refer to `resources_servers/workplace_assistant/README.md` for a good example that includes server startup commands and trajectory collection examples.
:::

:::{important}
Your PR will not be merged without complete and accurate licensing information.
:::

## CI/CD Requirements

All contributions must pass these automated checks:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Check
  - Requirement
* - Unit Tests
  - All existing tests must pass
* - Build Docs
  - Documentation must build without errors
* - Copyright Check
  - All files must have proper copyright headers
* - Commit Signing
  - All commits must be cryptographically signed (GPG or SSH)
* - Pre-commit Hooks
  - Code formatting and linting
```

:::{dropdown} Copyright Header Template
:icon: law

Add this header to all new Python files:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

:::

## PR Review Process

After submitting your PR:

```{list-table}
:header-rows: 1
:widths: 10 30 60

* - Step
  - Action
  - Description
* - 1
  - Assignment
  - Team member assigned for reproduction and review
* - 2
  - Verification
  - Reviewer verifies all checklist items from PR description
* - 3
  - Correctness
  - Reviewer checks correctness of five example rollouts
* - 4
  - Reproduction
  - Reviewer re-runs procedure to ensure reproducibility
* - 5
  - Approval
  - Final approval from maintainers
```

## Reference Implementations

Review these existing resource servers for guidance:

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Server
  - Domain
  - Key Features
* - `example_single_tool_call`
  - `agent`
  - Basic tool calling pattern
* - `example_multi_step`
  - `instruction_following`
  - Multi-step task verification
* - `math_with_judge`
  - `math`
  - LLM-as-judge verification
* - `code_gen`
  - `coding`
  - Unit test-based verification
* - `workplace_assistant`
  - `agent`
  - Multi-tool sandbox environment
```

## Special Considerations

### Environment Behavior Changes

:::{warning}
Changing existing environment behavior requires careful consideration:

- Requires releasing a new version
- Makes results hard to compare across versions
- Discuss major changes in issues before implementing
:::

### Breaking Changes

- Discuss API changes in issues before implementing
- Provide migration guides for breaking changes
- Consider backward compatibility when possible

## Related Topics

- {doc}`/tutorials/creating-resource-server` - Step-by-step implementation guide
- {doc}`/about/concepts/task-verification` - Verification and reward concepts
- {doc}`/about/concepts/core-components` - Resource server architecture
