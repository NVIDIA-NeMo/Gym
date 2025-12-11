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
2. Use Qwen 3 30B A3B or equivalent model
3. Generate 16 responses per prompt
4. Report reward distribution statistics
5. **For tool calling**: Provide tool call metrics and correlation with rewards
:::

### Training-Based Validation

Demonstrate meaningful training signal:

:::{dropdown} Training Validation Requirements
:icon: rocket

1. Train with GRPO on Qwen 30B A3B Instruct (or equivalent)
2. Use VeRL or NeMo RL + Gym
3. Include training accuracy curve
4. Include test benchmark accuracy curve (if applicable)
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
