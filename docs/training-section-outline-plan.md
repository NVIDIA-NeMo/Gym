# Training Section Documentation Plan

**Date**: 2025-11-04  
**Status**: Planning Draft  
**Context**: Creating new top-level "Training" section for NeMo Gym documentation

---

## Executive Summary

The new **Training** section will focus on How-To guides and Jobs-to-be-done (JTBD) for users who have completed Get Started and are ready to scale up training data generation, optimize their training workflows, and integrate with RL training frameworks.

**Key Insight**: The outline mentions "Verifying Agent Results" and "Rollout Collection Fundamentals" not as topics to duplicate, but as indicators that we need to capture all the **dimensional aspects** that make up training actions:

- **Data generation dimensions**: input datasets, rollout strategies, parallelization, sampling
- **Verification dimensions**: reward shaping, quality thresholds, verification patterns
- **Data quality dimensions**: filtering, curation, validation, statistics
- **Format dimensions**: SFT, DPO, preference pairs, framework compatibility
- **Configuration dimensions**: temperature, model selection, multi-environment setups
- **Scale dimensions**: throughput optimization, parallel processing, batch management
- **Integration dimensions**: RL frameworks, training pipelines, deployment

---

## Current State Analysis

### What Exists Today

**About/Concepts** (Explanation-oriented):
- Rollout Collection Fundamentals (what rollouts are, why they matter)
- Verifying Agent Results (what verification is, why it matters)

**Get Started** (Tutorial-oriented):
- Collecting Rollouts (first batch generation experience)
- Verifying Agent Results (building verification logic)

**Tutorials** (Learning-oriented):
- Offline Training with Rollouts (SFT/DPO)
- Separate Policy and Judge Models

### Gaps for Training Users

**Missing How-To guides**:
- No guidance on optimizing rollout collection for scale
- No guidance on advanced verification strategies
- No guidance on dataset management and quality assurance
- No guidance on integrating with specific RL frameworks
- No configuration guidance for production training pipelines

**Missing Reference content**:
- No comprehensive configuration reference
- No rollout format specification
- No verification patterns catalog
- No performance optimization reference

**Target audience gap**:
- Current docs serve beginners (Get Started) and advanced learners (Tutorials)
- Missing content for practitioners who know basics but need task-specific guidance

---

## Target Audiences for Training Section

### Primary: Practitioners (ML Engineers, Data Scientists)

**They have**:
- Completed Get Started tutorials
- Basic understanding of rollouts and verification
- Access to compute resources for scale

**They need**:
- Task-specific guidance ("How do I...?")
- Configuration examples for common scenarios
- Performance optimization guidance
- Integration patterns for their stack

**Jobs to be done**:
1. Scale up rollout collection for training
2. Optimize data quality and efficiency
3. Integrate with RL training frameworks
4. Debug and troubleshoot production issues
5. Configure multi-environment setups

### Secondary: Platform Engineers

**They have**:
- Responsibility for deploying training infrastructure
- Need for production-ready configurations
- Multi-user or multi-project requirements

**They need**:
- Deployment patterns and best practices
- Resource management guidance
- Monitoring and observability setup
- Security and access control patterns

### Tertiary: Researchers

**They have**:
- Novel verification or reward shaping ideas
- Custom environment requirements
- Experimental workflow needs

**They need**:
- Extension points and customization patterns
- Advanced verification techniques
- Research-oriented configuration examples

---

## Training Section Outline (Diataxis-Aligned)

### Structure Overview

```
training/
├── index.md                           # Overview and navigation
├── how-to/                            # Task-oriented guides (PRIMARY)
│   ├── optimize-rollout-collection.md
│   ├── manage-training-datasets.md
│   ├── configure-verification-strategies.md
│   ├── scale-parallel-generation.md
│   ├── integrate-with-rl-frameworks.md
│   ├── filter-and-curate-data.md
│   ├── monitor-data-quality.md
│   ├── configure-multi-model-setups.md
│   └── troubleshoot-common-issues.md
└── reference/                         # Information-oriented (SECONDARY)
    ├── configuration-options.md
    ├── rollout-format-spec.md
    ├── verification-patterns-catalog.md
    └── performance-tuning.md
```

---

## Detailed Content Plan

### 1. training/index.md

**Purpose**: Orientation and navigation hub for training workflows

**Diataxis Type**: Explanation + Navigation

**Content**:
- Brief overview: "You've learned the basics—now scale up your training workflows"
- When to use this section vs. Tutorials vs. Concepts
- Quick decision guide: "Which guide should I read?"
- Grid navigation to How-To guides organized by JTBD
- Links to reference content

**Progressive Disclosure**:
- Start with most common tasks (optimize collection, manage datasets)
- Advanced tasks revealed through categories
- Reference section mentioned but not overwhelming

---

### 2. How-To Guides (JTBD-Focused)

#### 2.1 `how-to/optimize-rollout-collection.md`

**Job to be done**: "I need to generate training data faster and more efficiently"

**Dimensional aspects covered**:
- **Scale**: Parallelization settings, throughput optimization
- **Configuration**: Sampling parameters, temperature tuning
- **Integration**: Model server optimization

**Content structure**:
```markdown
## Before You Start
Prerequisites, expected knowledge

## Understanding Collection Performance
Brief context on what affects speed

## Optimize Parallelization
- How to set num_samples_in_parallel
- Resource allocation considerations
- Bottleneck identification

## Tune Sampling Parameters
- Temperature for diversity vs consistency
- Top-p, top-k configuration
- When to use different settings

## Monitor and Profile
- Tracking throughput
- Identifying slow components
- Using built-in profiling

## Production Configuration Examples
- High-throughput SFT data generation
- Diverse DPO pair generation
- Balanced evaluation benchmarking

## Troubleshooting
Common performance issues and solutions
```

**Links to**:
- Concepts: Rollout Collection Fundamentals (why these parameters matter)
- Reference: Configuration Options (complete parameter list)
- Tutorial: Offline Training (using collected data)

---

#### 2.2 `how-to/manage-training-datasets.md`

**Job to be done**: "I need to organize, version, and prepare datasets for training"

**Dimensional aspects covered**:
- **Data Quality**: Validation, statistics, format checking
- **Integration**: Training framework compatibility
- **Configuration**: Dataset organization patterns

**Content structure**:
```markdown
## Before You Start

## Organize Dataset Files
- Directory structure patterns
- Naming conventions
- Version control strategies

## Validate Dataset Format
- Using ng_prepare_data command
- Format requirements by framework
- Common format issues

## Compute Dataset Statistics
- Distribution of rewards
- Length statistics
- Task diversity metrics
- Using built-in statistics tools

## Prepare for Training Frameworks
- NeMo-RL format requirements
- VeRL integration patterns
- OpenRLHF compatibility

## Version and Track Datasets
- Dataset versioning strategies
- Metadata management
- Reproducibility patterns

## Troubleshooting
Dataset validation errors and solutions
```

---

#### 2.3 `how-to/configure-verification-strategies.md`

**Job to be done**: "I need to customize how my agents are scored for training"

**Dimensional aspects covered**:
- **Verification**: Reward shaping, quality thresholds
- **Configuration**: Verification parameters
- **Integration**: Custom verifiers

**Content structure**:
```markdown
## Before You Start

## Understanding Verification Impact
How verification affects training outcomes

## Choose Verification Pattern
- Binary rewards (0 or 1)
- Continuous rewards (0.0-1.0)
- Partial credit strategies
- Composite scoring

## Implement Custom Verification
- Extending BaseVerifyRequest/Response
- Domain-specific scoring logic
- Tool usage verification
- Response quality verification

## Configure Reward Shaping
- Scaling and normalization
- Reward clipping
- Multi-objective rewards

## Test Verification Logic
- Verification testing patterns
- Edge case handling
- Debugging verification failures

## Production Examples
- Math problem verification
- Code execution scoring
- Instruction following evaluation
- LLM-as-judge patterns

## Troubleshooting
```

---

#### 2.4 `how-to/scale-parallel-generation.md`

**Job to be done**: "I need to generate millions of rollouts efficiently"

**Dimensional aspects covered**:
- **Scale**: Multi-node, distributed generation
- **Configuration**: Parallelization limits, batch management
- **Integration**: Resource allocation

**Content structure**:
```markdown
## Before You Start

## Understand Parallelization Architecture
Brief explanation of how parallel generation works

## Configure Parallel Generation
- num_samples_in_parallel parameter
- Finding optimal parallelism
- Resource constraints

## Batch Processing Strategies
- Chunking large datasets
- Resume interrupted collections
- Progress tracking

## Distributed Generation (Multi-Node)
- When to use distributed generation
- Configuration patterns
- Load balancing

## Monitor and Debug
- Tracking progress across parallel jobs
- Identifying stragglers
- Failure handling and retry

## Production Patterns
- Large-scale SFT data generation
- Multi-task curriculum generation
- Continuous data generation pipelines

## Troubleshooting
```

---

#### 2.5 `how-to/integrate-with-rl-frameworks.md`

**Job to be done**: "I need to connect my rollouts to my RL training pipeline"

**Dimensional aspects covered**:
- **Integration**: Framework-specific patterns
- **Format**: Data format requirements
- **Configuration**: Training pipeline setup

**Content structure**:
```markdown
## Before You Start

## Choose Your RL Framework
- NeMo-RL (NVIDIA)
- VeRL
- OpenRLHF
- TRL (Hugging Face)
- Custom frameworks

## Data Format Requirements by Framework
Detailed format specs for each

## NeMo-RL Integration
- Configuration pattern
- Data preparation workflow
- Training launch example

## VeRL Integration
- Configuration pattern
- Data preparation workflow
- Training launch example

## OpenRLHF Integration
[Similar structure]

## Custom Framework Integration
- Generic requirements
- Adaptation patterns
- Conversion utilities

## End-to-End Pipeline Examples
- Rollout collection → Training → Evaluation
- Continuous improvement loops

## Troubleshooting
```

---

#### 2.6 `how-to/filter-and-curate-data.md`

**Job to be done**: "I need to ensure my training data is high quality"

**Dimensional aspects covered**:
- **Data Quality**: Filtering, curation strategies
- **Verification**: Quality thresholds
- **Configuration**: Filter parameters

**Content structure**:
```markdown
## Before You Start

## Why Data Quality Matters
Impact on training outcomes

## Automatic Filtering Strategies
- Reward-based filtering
- Length-based filtering
- Tool usage filtering
- Success rate filtering
- Combining multiple criteria

## Manual Curation
- When manual review is needed
- Sampling strategies for review
- Annotation patterns
- Quality review workflows

## Analyze Data Distributions
- Reward distributions
- Task diversity
- Length distributions
- Identifying outliers

## Balance Datasets
- Task type balancing
- Difficulty balancing
- Diversity vs quality tradeoffs

## Iterative Curation
- Multi-stage filtering
- Progressive quality improvement
- Feedback loops

## Production Examples
- SFT data curation (high threshold)
- DPO pair curation (quality differences)
- Curriculum learning datasets

## Troubleshooting
```

---

#### 2.7 `how-to/monitor-data-quality.md`

**Job to be done**: "I need to track quality metrics during data generation"

**Dimensional aspects covered**:
- **Data Quality**: Metrics, monitoring
- **Scale**: Real-time tracking at scale
- **Integration**: Monitoring tools

**Content structure**:
```markdown
## Before You Start

## Key Quality Metrics
- Average reward
- Success rate
- Distribution of scores
- Length statistics
- Tool usage patterns

## Real-Time Monitoring During Collection
- Using ng_collect_rollouts progress
- Custom monitoring scripts
- Alert thresholds

## Post-Collection Analysis
- Using ng_viewer for interactive analysis
- Computing aggregate statistics
- Comparing across collection runs

## Identify Quality Issues
- Low reward causes
- Verification failures
- Model degradation signals
- Data drift detection

## Quality Dashboards
- Building custom dashboards
- Integration with monitoring tools
- Visualization patterns

## Production Patterns
- Continuous quality monitoring
- Automated quality gates
- Quality trend tracking

## Troubleshooting
```

---

#### 2.8 `how-to/configure-multi-model-setups.md`

**Job to be done**: "I need different models for policy and verification"

**Dimensional aspects covered**:
- **Configuration**: Multi-model patterns
- **Integration**: Model server coordination
- **Scale**: Resource optimization

**Content structure**:
```markdown
## Before You Start

## Why Use Multiple Models
Use cases for multi-model setups

## Configuration Patterns
- Policy model + Judge model
- Multiple policy models (A/B testing)
- Specialized models by task type
- Model cascades

## Configure Separate Policy and Judge
- Server configuration
- Agent configuration
- Verification routing

## Resource Management
- GPU allocation per model
- Memory considerations
- Throughput optimization

## Testing Multi-Model Configs
- Verification testing
- End-to-end validation
- Performance profiling

## Production Examples
- Efficient policy + expensive judge
- Multiple domain-specific models
- Model version comparison

## Troubleshooting
```

**Note**: This builds on existing Tutorial: "Separate Policy and Judge Models" but focuses on production configuration patterns rather than learning exercise.

---

#### 2.9 `how-to/troubleshoot-common-issues.md`

**Job to be done**: "I'm experiencing problems with my training workflows"

**Dimensional aspects covered**:
- All dimensions through troubleshooting lens

**Content structure**:
```markdown
## Before You Start

## Performance Issues
- Slow rollout collection
- Low throughput
- Resource bottlenecks

## Data Quality Issues
- Low average rewards
- Verification failures
- Format errors

## Configuration Issues
- Server connection failures
- Incorrect parameter settings
- Environment setup problems

## Integration Issues
- Framework compatibility errors
- Data format mismatches
- Training pipeline failures

## Scale Issues
- Parallel generation failures
- Out of memory errors
- Distributed coordination problems

## Debugging Strategies
- Systematic debugging approach
- Logging and diagnostics
- Isolation testing

## Common Error Messages
Comprehensive list with solutions
```

---

### 3. Reference Content

#### 3.1 `reference/configuration-options.md`

**Purpose**: Comprehensive configuration parameter reference

**Content structure**:
```markdown
## Overview

## Rollout Collection Configuration
Complete parameter list for ng_collect_rollouts:
- agent_name
- input_jsonl_fpath
- output_jsonl_fpath
- num_samples_in_parallel
- limit
- num_repeats
- responses_create_params.*
- [all parameters with descriptions, types, defaults, examples]

## Agent Configuration
Parameters for agent server configs

## Model Configuration
Parameters for model server configs

## Resource Server Configuration
Parameters for resource server configs

## Data Preparation Configuration
Parameters for ng_prepare_data

## Environment Variables
Complete list of env.yaml options

## Configuration Inheritance
How YAML → env.yaml → CLI works

## Example Configurations by Use Case
- Development/testing
- Production SFT data generation
- Production DPO data generation
- Large-scale distributed
```

---

#### 3.2 `reference/rollout-format-spec.md`

**Purpose**: Formal specification of rollout data format

**Content structure**:
```markdown
## Overview

## Core Rollout Structure
JSON schema with descriptions

## responses_create_params
- input field
- tools field
- model parameters
- [complete spec]

## output Field
- Message sequence format
- Tool call format
- Tool result format

## reward Field
- Type and range
- Interpretation

## Optional Metadata Fields
Resource-specific extensions

## Compatibility Notes
Framework-specific requirements

## Validation
- Format checking utilities
- Common validation errors

## Examples
Complete rollout examples for different scenarios
```

---

#### 3.3 `reference/verification-patterns-catalog.md`

**Purpose**: Catalog of verification approaches with examples

**Content structure**:
```markdown
## Overview

## Pattern 1: Binary Verification
- When to use
- Implementation example
- Production examples (mcqa, etc.)

## Pattern 2: Continuous Scoring
- When to use
- Implementation example
- Production examples (code execution, etc.)

## Pattern 3: Partial Credit
- When to use
- Implementation example
- Production examples

## Pattern 4: Composite Scoring
- When to use
- Implementation example
- Production examples

## Pattern 5: LLM-as-Judge
- When to use
- Implementation example
- Bias mitigation patterns
- Production examples

## Pattern 6: Hybrid Verification
- When to use
- Implementation example
- Production examples (math, etc.)

## Custom Verification
Guidelines for creating new patterns

## Performance Considerations
Speed vs accuracy tradeoffs
```

---

#### 3.4 `reference/performance-tuning.md`

**Purpose**: Comprehensive performance optimization reference

**Content structure**:
```markdown
## Overview

## Collection Throughput Optimization
- Parallelization tuning
- Batch size optimization
- Resource allocation
- Network optimization

## Model Server Optimization
- vLLM configuration
- Batch inference settings
- GPU utilization
- Memory optimization

## Resource Server Optimization
- Tool execution optimization
- Verification speed
- Caching strategies

## Data I/O Optimization
- JSONL streaming
- Buffering strategies
- Disk I/O optimization

## Profiling and Diagnostics
- Built-in profiling tools
- Custom profiling
- Bottleneck identification

## Benchmarking
- Setting baselines
- Measuring improvements
- Comparing configurations

## Production Configurations
Optimized configs for different scales
```

---

## Progressive Disclosure Strategy

### Entry Points by User Journey

**Beginner Practitioners** (just finished Get Started):
1. Start: `training/index.md` → "I just finished Get Started, what's next?"
2. First guide: "Optimize Rollout Collection" (scale up what they learned)
3. Second guide: "Filter and Curate Data" (improve quality)
4. Third guide: "Integrate with RL Frameworks" (use the data)

**Intermediate Practitioners** (have generated some data):
1. Start: `training/index.md` → "I have basic workflows, need to optimize"
2. Relevant guides: "Scale Parallel Generation", "Monitor Data Quality"
3. Reference: Configuration Options, Performance Tuning

**Advanced Practitioners** (production deployments):
1. Start: Direct to specific how-to guides or reference
2. Focus: "Configure Multi-Model Setups", "Troubleshoot Common Issues"
3. Reference: All reference docs

### Information Layering

**Level 1: Index Page**
- What this section is for
- Decision guide: which guide for which JTBD
- Quick links to most common tasks

**Level 2: How-To Guides**
- Task-focused, practical guidance
- Minimal theory (link to Concepts for "why")
- Multiple examples with variations
- Clear success criteria

**Level 3: Reference**
- Comprehensive, exhaustive information
- Linked from how-to guides when needed
- Structured for lookup, not reading cover-to-cover

### Cross-References

**From Training → Other Sections**:
- Link to Concepts for deep explanations of "why"
- Link to Tutorials for learning-oriented experiences
- Link to About for product positioning
- Link to Models for model-specific setup

**From Other Sections → Training**:
- Get Started: "Ready to scale up? See Training section"
- Tutorials: "For production patterns, see Training how-to guides"
- Concepts: "To apply this, see Training how-to guides"

---

## Content Dependencies and Sequencing

### Phase 1: Core How-To Guides (Highest Priority)

**Week 1-2**:
1. `training/index.md` - Navigation hub
2. `how-to/optimize-rollout-collection.md` - Most requested
3. `how-to/manage-training-datasets.md` - Critical workflow

**Week 3-4**:
4. `how-to/filter-and-curate-data.md` - Data quality essential
5. `how-to/integrate-with-rl-frameworks.md` - Training integration

### Phase 2: Advanced How-To Guides

**Week 5-6**:
6. `how-to/scale-parallel-generation.md` - Scaling needs
7. `how-to/configure-verification-strategies.md` - Customization
8. `how-to/monitor-data-quality.md` - Production observability

**Week 7-8**:
9. `how-to/configure-multi-model-setups.md` - Advanced patterns
10. `how-to/troubleshoot-common-issues.md` - Support reduction

### Phase 3: Reference Content

**Week 9-10**:
11. `reference/configuration-options.md` - Comprehensive reference
12. `reference/rollout-format-spec.md` - Format specification

**Week 11-12**:
13. `reference/verification-patterns-catalog.md` - Pattern library
14. `reference/performance-tuning.md` - Optimization reference

---

## Success Metrics

### Documentation Quality

- [ ] Every how-to guide has clear JTBD stated upfront
- [ ] Every how-to guide includes working examples
- [ ] Every how-to guide includes success criteria
- [ ] Every how-to guide includes troubleshooting
- [ ] Reference content is comprehensive and accurate
- [ ] Cross-references enable easy navigation

### User Experience

- [ ] Users can find the right guide for their task within 1 minute
- [ ] Users can complete common tasks using only how-to guides (without reading Concepts)
- [ ] Advanced users can find optimization details in Reference
- [ ] Users report reduced time-to-production

### Coverage

- [ ] All dimensional aspects of training covered:
  - ✓ Data generation
  - ✓ Verification
  - ✓ Data quality
  - ✓ Formats
  - ✓ Configuration
  - ✓ Scale
  - ✓ Integration

---

## Open Questions

1. **Framework Integration Depth**: How much detail for each RL framework? Should we have separate pages per framework or sections within one page?

2. **Code Examples**: Should we provide complete runnable scripts or focus on configuration snippets? Balance between copy-paste and understanding.

3. **Version Specificity**: Should we version-tag content based on NeMo Gym versions? How to handle changes in configuration schema?

4. **Performance Numbers**: Should we include benchmark numbers? Risk of becoming outdated vs. value of concrete targets.

5. **Multi-Environment**: Should we have dedicated content for containerized/Kubernetes deployments vs. bare metal?

---

## Next Steps

1. **Review and approve this outline** with stakeholders
2. **Prioritize Phase 1 deliverables** based on user feedback and support tickets
3. **Draft first how-to guide** as template for others
4. **Set up content review process** with SMEs
5. **Plan integration** with existing Tutorials and Concepts

---

## Appendix: Dimensional Coverage Matrix

| Dimension | Index | How-To Guides | Reference |
|-----------|-------|---------------|-----------|
| **Data Generation** | ✓ | optimize-rollout-collection, scale-parallel-generation | configuration-options, performance-tuning |
| **Verification** | ✓ | configure-verification-strategies | verification-patterns-catalog |
| **Data Quality** | ✓ | filter-and-curate-data, monitor-data-quality | rollout-format-spec |
| **Formats** | ✓ | integrate-with-rl-frameworks, manage-training-datasets | rollout-format-spec |
| **Configuration** | ✓ | All how-to guides | configuration-options |
| **Scale** | ✓ | scale-parallel-generation, optimize-rollout-collection | performance-tuning |
| **Integration** | ✓ | integrate-with-rl-frameworks, configure-multi-model-setups | configuration-options |

Every dimensional aspect is covered by multiple pieces of content at different levels of detail.

