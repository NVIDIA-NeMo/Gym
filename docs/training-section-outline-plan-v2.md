# Training Section Documentation Plan (v2)

**Date**: 2025-11-04  
**Status**: Planning Draft v2  
**Context**: Topic-organized structure with content type tracking

---

## Executive Summary

The new **Training** section organizes content by **natural topic areas** (not Diataxis categories) while tracking content types internally to identify gaps. Topics large enough to warrant multiple pages become directories with subtopic pages.

**Organizational Principle**: Group by subject matter → Track content type per page → Identify gaps in coverage

---

## Revised Structure: Topic-Based Organization

```
training/
├── index.md                                    [Overview + Navigation Hub]
│
├── rollout-collection/                         [DIRECTORY: Core training data generation]
│   ├── index.md                                (Overview + navigation)
│   ├── optimize-throughput.md                  (How-to: Speed & efficiency)
│   ├── parallel-generation.md                  (How-to: Scale with parallelism)
│   ├── sampling-strategies.md                  (How-to: Temperature, diversity, repeats)
│   ├── collection-patterns.md                  (Reference: Common patterns by use case)
│   └── configuration-reference.md              (Reference: All ng_collect_rollouts params)
│
├── verification/                               [DIRECTORY: Scoring and reward signals]
│   ├── index.md                                (Overview + navigation)
│   ├── custom-verification.md                  (How-to: Build custom verify() logic)
│   ├── reward-shaping.md                       (How-to: Design reward signals)
│   ├── multi-objective-scoring.md              (How-to: Combine multiple signals)
│   ├── verification-patterns.md                (Reference: Pattern catalog with examples)
│   └── testing-verifiers.md                    (How-to: Test and debug verification)
│
├── data-quality/                               [DIRECTORY: Curation and monitoring]
│   ├── index.md                                (Overview + navigation)
│   ├── filtering-strategies.md                 (How-to: Filter rollouts for quality)
│   ├── manual-curation.md                      (How-to: Review and annotate)
│   ├── quality-metrics.md                      (How-to: Track and monitor quality)
│   ├── dataset-balancing.md                    (How-to: Balance task distributions)
│   └── statistics-reference.md                 (Reference: All quality metrics explained)
│
├── datasets/                                   [DIRECTORY: Dataset management]
│   ├── index.md                                (Overview + navigation)
│   ├── organize-datasets.md                    (How-to: Directory structure, versioning)
│   ├── validate-format.md                      (How-to: Use ng_prepare_data)
│   ├── prepare-for-training.md                 (How-to: Format for SFT/DPO/RL)
│   ├── format-specification.md                 (Reference: Rollout JSON schema)
│   └── conversion-utilities.md                 (Reference: Format conversion tools)
│
├── integration/                                [DIRECTORY: Connect to training frameworks]
│   ├── index.md                                (Overview + navigation)
│   ├── nemo-rl.md                              (How-to: NeMo-RL integration)
│   ├── verl.md                                 (How-to: VeRL integration)
│   ├── openrlhf.md                             (How-to: OpenRLHF integration)
│   ├── trl.md                                  (How-to: TRL/HuggingFace integration)
│   ├── custom-frameworks.md                    (How-to: Custom integration patterns)
│   └── framework-comparison.md                 (Reference: Framework requirements matrix)
│
├── configuration/                              [DIRECTORY: System configuration]
│   ├── index.md                                (Overview + navigation)
│   ├── multi-model-setups.md                   (How-to: Policy + judge patterns)
│   ├── environment-configs.md                  (How-to: Dev/staging/prod configs)
│   ├── resource-allocation.md                  (How-to: GPU/memory management)
│   ├── configuration-reference.md              (Reference: All config parameters)
│   └── configuration-patterns.md               (Reference: Common config combinations)
│
├── performance/                                [DIRECTORY: Optimization and scaling]
│   ├── index.md                                (Overview + navigation)
│   ├── throughput-optimization.md              (How-to: Maximize rollouts/hour)
│   ├── bottleneck-diagnosis.md                 (How-to: Profile and identify issues)
│   ├── model-server-tuning.md                  (How-to: Optimize vLLM/NIM)
│   ├── distributed-generation.md               (How-to: Multi-node setups)
│   └── performance-benchmarks.md               (Reference: Expected throughput by config)
│
└── troubleshooting/                            [DIRECTORY: Debugging and fixes]
    ├── index.md                                (Overview + navigation)
    ├── collection-failures.md                  (How-to: Debug rollout generation issues)
    ├── verification-errors.md                  (How-to: Debug scoring problems)
    ├── data-format-issues.md                   (How-to: Fix format validation errors)
    ├── performance-problems.md                 (How-to: Fix slow generation)
    └── error-reference.md                      (Reference: Common errors and solutions)
```

---

## Content Type Matrix

Track content type per page to identify gaps:

| Topic Directory | Page | Content Type | JTBD |
|----------------|------|--------------|------|
| **rollout-collection/** | | | |
| | index.md | Explanation | Understand rollout generation for training |
| | optimize-throughput.md | How-to | Speed up data generation |
| | parallel-generation.md | How-to | Scale to millions of rollouts |
| | sampling-strategies.md | How-to | Control diversity and quality |
| | collection-patterns.md | Reference | Choose right approach for use case |
| | configuration-reference.md | Reference | Lookup all parameters |
| **verification/** | | | |
| | index.md | Explanation | Understand verification for training |
| | custom-verification.md | How-to | Build domain-specific scoring |
| | reward-shaping.md | How-to | Design effective rewards |
| | multi-objective-scoring.md | How-to | Combine multiple metrics |
| | verification-patterns.md | Reference | Choose verification approach |
| | testing-verifiers.md | How-to | Validate verification logic |
| **data-quality/** | | | |
| | index.md | Explanation | Understand quality impact on training |
| | filtering-strategies.md | How-to | Remove low-quality rollouts |
| | manual-curation.md | How-to | Review and improve datasets |
| | quality-metrics.md | How-to | Track quality during generation |
| | dataset-balancing.md | How-to | Balance task distributions |
| | statistics-reference.md | Reference | Interpret quality metrics |
| **datasets/** | | | |
| | index.md | Explanation | Understand dataset management |
| | organize-datasets.md | How-to | Structure and version datasets |
| | validate-format.md | How-to | Check format compliance |
| | prepare-for-training.md | How-to | Convert to training formats |
| | format-specification.md | Reference | Rollout JSON schema |
| | conversion-utilities.md | Reference | Format conversion tools |
| **integration/** | | | |
| | index.md | Explanation | Understand RL framework integration |
| | nemo-rl.md | How-to | Integrate with NeMo-RL |
| | verl.md | How-to | Integrate with VeRL |
| | openrlhf.md | How-to | Integrate with OpenRLHF |
| | trl.md | How-to | Integrate with TRL |
| | custom-frameworks.md | How-to | Integrate with custom frameworks |
| | framework-comparison.md | Reference | Compare framework requirements |
| **configuration/** | | | |
| | index.md | Explanation | Understand configuration system |
| | multi-model-setups.md | How-to | Configure multiple models |
| | environment-configs.md | How-to | Configure for different environments |
| | resource-allocation.md | How-to | Manage GPU/memory resources |
| | configuration-reference.md | Reference | All config parameters |
| | configuration-patterns.md | Reference | Common config combinations |
| **performance/** | | | |
| | index.md | Explanation | Understand performance factors |
| | throughput-optimization.md | How-to | Maximize generation speed |
| | bottleneck-diagnosis.md | How-to | Find performance bottlenecks |
| | model-server-tuning.md | How-to | Optimize model servers |
| | distributed-generation.md | How-to | Scale across nodes |
| | performance-benchmarks.md | Reference | Expected performance metrics |
| **troubleshooting/** | | | |
| | index.md | Explanation | Understand common issues |
| | collection-failures.md | How-to | Fix generation failures |
| | verification-errors.md | How-to | Fix scoring errors |
| | data-format-issues.md | How-to | Fix format problems |
| | performance-problems.md | How-to | Fix slow performance |
| | error-reference.md | Reference | Common error messages |

---

## Gap Analysis by Content Type

### By Content Type

| Content Type | Count | Pages |
|--------------|-------|-------|
| **Explanation** | 8 | All index.md pages (topic overviews) |
| **How-to** | 27 | Task-oriented guides |
| **Reference** | 13 | Parameter lists, specs, catalogs |
| **Tutorial** | 0 | (Covered in Get Started - no gap) |

**Total pages**: 48 (8 directories × ~6 pages each)

### Content Type Distribution

```
Explanation: 17% (8/48)  → Topic overviews
How-to:      56% (27/48) → Task guidance ✓ GOOD (matches "How-To/JTBD" focus)
Reference:   27% (13/48) → Lookup information
Tutorial:    0% (0/48)   → Already in Get Started
```

**Assessment**: Good balance - heavy on How-to (as intended), sufficient Reference, minimal Explanation (just topic overviews).

### Coverage Gaps Identified

**None** - All dimensional aspects covered:

| Dimension | Covered in Directory |
|-----------|---------------------|
| Data generation | rollout-collection/ |
| Verification | verification/ |
| Data quality | data-quality/ |
| Formats | datasets/ |
| Configuration | configuration/ |
| Scale | performance/ |
| Integration | integration/ |

---

## Directory Rationale

### Why These Became Directories

**rollout-collection/** (6 pages):
- Core training workflow with multiple dimensions
- Throughput, parallelism, sampling, patterns, config
- Large enough topic to warrant directory

**verification/** (6 pages):
- Critical for training quality
- Multiple approaches and patterns
- Custom implementation guidance needed

**data-quality/** (6 pages):
- Essential for good training outcomes
- Multiple techniques: filtering, curation, monitoring, balancing
- Metrics and statistics warrant reference content

**datasets/** (6 pages):
- Organizational complexity
- Format specifications needed
- Multiple training framework formats

**integration/** (6 pages):
- Multiple frameworks (5+) each need guidance
- Each framework has different requirements
- Comparison/selection guidance needed

**configuration/** (6 pages):
- Complex multi-model setups
- Environment-specific patterns
- Comprehensive parameter reference needed

**performance/** (6 pages):
- Multi-faceted optimization
- Diagnosis, tuning, distributed scaling
- Benchmarks and metrics reference

**troubleshooting/** (6 pages):
- Multiple failure modes
- Organized by problem category
- Error reference needed

---

## Dimensional Coverage per Directory

| Directory | Dimensions Covered |
|-----------|-------------------|
| **rollout-collection/** | Data generation, Scale, Configuration |
| **verification/** | Verification, Configuration |
| **data-quality/** | Data quality, Data generation |
| **datasets/** | Formats, Data quality, Integration |
| **integration/** | Integration, Formats, Configuration |
| **configuration/** | Configuration, Scale, Integration |
| **performance/** | Scale, Configuration, Data generation |
| **troubleshooting/** | All dimensions (debugging lens) |

**Every dimension covered by multiple directories** ✓

---

## Progressive Disclosure Strategy

### Entry Points by User Journey

**Beginner Practitioners** (just finished Get Started):
1. `training/index.md` → "What next after Get Started?"
2. `rollout-collection/index.md` → Understand scaling basics
3. `rollout-collection/optimize-throughput.md` → First optimization
4. `data-quality/filtering-strategies.md` → Improve quality
5. `datasets/prepare-for-training.md` → Use the data

**Intermediate Practitioners**:
1. Direct to topic directories (rollout-collection, data-quality)
2. How-to guides for specific tasks
3. Reference content for parameters

**Advanced Practitioners**:
1. Direct to specific pages via search/navigation
2. Reference content heavily used
3. Performance and troubleshooting directories

### Navigation Hierarchy

```
Level 1: training/index.md
    ├─ "Scale up rollout collection" → rollout-collection/
    ├─ "Customize verification" → verification/
    ├─ "Ensure data quality" → data-quality/
    ├─ "Manage datasets" → datasets/
    ├─ "Integrate with training" → integration/
    ├─ "Configure system" → configuration/
    ├─ "Optimize performance" → performance/
    └─ "Troubleshoot issues" → troubleshooting/

Level 2: {topic}/index.md
    ├─ Brief explanation of topic
    ├─ When you need this
    ├─ Quick links to common tasks (how-to)
    └─ Reference materials

Level 3: Specific pages (how-to or reference)
```

---

## Implementation Priority

### Phase 1: Essential Workflows (Weeks 1-4)

**Goal**: Enable basic scaling and integration

```
training/
├── index.md                                    Week 1
├── rollout-collection/
│   ├── index.md                                Week 1
│   ├── optimize-throughput.md                  Week 2
│   ├── parallel-generation.md                  Week 2
│   └── configuration-reference.md              Week 3
├── datasets/
│   ├── index.md                                Week 3
│   ├── prepare-for-training.md                 Week 3
│   └── format-specification.md                 Week 4
└── integration/
    ├── index.md                                Week 4
    ├── nemo-rl.md                              Week 4
    └── framework-comparison.md                 Week 4
```

**Deliverables**: 11 pages covering core workflow

---

### Phase 2: Quality and Configuration (Weeks 5-8)

**Goal**: Enable production-quality workflows

```
├── data-quality/
│   ├── index.md                                Week 5
│   ├── filtering-strategies.md                 Week 5
│   ├── quality-metrics.md                      Week 6
│   └── statistics-reference.md                 Week 6
├── verification/
│   ├── index.md                                Week 6
│   ├── custom-verification.md                  Week 7
│   ├── reward-shaping.md                       Week 7
│   └── verification-patterns.md                Week 8
└── configuration/
    ├── index.md                                Week 8
    ├── multi-model-setups.md                   Week 8
    └── configuration-reference.md              Week 8
```

**Deliverables**: 11 pages covering quality and config

---

### Phase 3: Scaling and All Frameworks (Weeks 9-11)

**Goal**: Complete integration coverage and scaling

```
├── integration/
│   ├── verl.md                                 Week 9
│   ├── openrlhf.md                             Week 9
│   ├── trl.md                                  Week 9
│   └── custom-frameworks.md                    Week 10
├── performance/
│   ├── index.md                                Week 10
│   ├── throughput-optimization.md              Week 10
│   ├── bottleneck-diagnosis.md                 Week 11
│   └── performance-benchmarks.md               Week 11
└── rollout-collection/
    ├── sampling-strategies.md                  Week 11
    └── collection-patterns.md                  Week 11
```

**Deliverables**: 11 pages completing core directories

---

### Phase 4: Advanced and Polish (Weeks 12-15)

**Goal**: Complete all directories, troubleshooting, advanced topics

```
├── troubleshooting/
│   ├── index.md                                Week 12
│   ├── collection-failures.md                  Week 12
│   ├── verification-errors.md                  Week 12
│   ├── data-format-issues.md                   Week 13
│   ├── performance-problems.md                 Week 13
│   └── error-reference.md                      Week 13
├── [Complete remaining pages]                  Weeks 14-15
│   ├── verification/multi-objective-scoring.md
│   ├── verification/testing-verifiers.md
│   ├── data-quality/manual-curation.md
│   ├── data-quality/dataset-balancing.md
│   ├── datasets/organize-datasets.md
│   ├── datasets/validate-format.md
│   ├── datasets/conversion-utilities.md
│   ├── configuration/environment-configs.md
│   ├── configuration/resource-allocation.md
│   ├── configuration/configuration-patterns.md
│   ├── performance/model-server-tuning.md
│   └── performance/distributed-generation.md
```

**Deliverables**: 15 pages completing all directories

**Total Timeline**: 15 weeks for 48 pages (all 8 directories complete)

---

## Sample Directory: rollout-collection/

### rollout-collection/index.md

**Content Type**: Explanation  
**Purpose**: Orient users to rollout collection for training

```markdown
# Rollout Collection for Training

Brief overview of scaling rollout generation for training data.

## When You Need This

- You've completed Get Started
- Need to generate large training datasets
- Want to optimize generation speed
- Need specific sampling strategies

## Quick Tasks

Most common tasks in this section:

- [Optimize throughput](optimize-throughput.md) - Speed up generation
- [Scale with parallelism](parallel-generation.md) - Generate millions of rollouts
- [Control sampling](sampling-strategies.md) - Tune diversity and quality

## Topics

### Generation Optimization
- [Optimize Throughput](optimize-throughput.md) - How-to guide
- [Parallel Generation](parallel-generation.md) - How-to guide

### Strategies
- [Sampling Strategies](sampling-strategies.md) - How-to guide
- [Collection Patterns](collection-patterns.md) - Reference

### Configuration
- [Configuration Reference](configuration-reference.md) - Complete parameter list

## Related

- **Concepts**: [Rollout Collection Fundamentals](../about/concepts/rollout-collection-fundamentals.md) - Understand why
- **Get Started**: [Collecting Rollouts](../get-started/collecting-rollouts.md) - First experience
- **Performance**: [Throughput Optimization](../training/performance/throughput-optimization.md) - Advanced optimization
```

---

### rollout-collection/optimize-throughput.md

**Content Type**: How-to  
**JTBD**: Speed up rollout generation

```markdown
# Optimize Rollout Generation Throughput

Learn to maximize the speed of rollout generation for training data.

## Before You Start

**Prerequisites**:
- Completed Get Started tutorials
- Have working rollout collection setup
- Understand basic ng_collect_rollouts command

**Expected outcome**: 2-5x improvement in rollouts/hour

---

## Understand Current Performance

[Brief explanation of what affects speed]

### Measure Current Throughput

```bash
# Run collection with timing
time ng_collect_rollouts +agent_name=my_agent \
    +input_jsonl_fpath=sample.jsonl \
    +output_jsonl_fpath=output.jsonl \
    +limit=100
```

Calculate: rollouts per hour = (100 / seconds) * 3600

---

## Optimize Parallelization

[How-to content for tuning num_samples_in_parallel]

### Find Optimal Parallelism

[Step-by-step guide with examples]

**Example configurations**:
```yaml
# Conservative (GPU memory limited)
num_samples_in_parallel: 4

# Balanced (typical)
num_samples_in_parallel: 10

# Aggressive (high-memory GPUs)
num_samples_in_parallel: 20
```

---

## Tune Model Server

[How-to content for vLLM/NIM optimization]

---

## Reduce Verification Overhead

[How-to content for faster verification]

---

## Monitor and Profile

[How-to content for identifying bottlenecks]

---

## Production Examples

[Complete working examples for common scenarios]

---

## Troubleshooting

[Common issues and solutions]

---

## Related

- [Parallel Generation](parallel-generation.md) - Scale to millions
- [Performance Benchmarks](../performance/performance-benchmarks.md) - Expected throughput
- [Bottleneck Diagnosis](../performance/bottleneck-diagnosis.md) - Find slow components
```

---

### rollout-collection/configuration-reference.md

**Content Type**: Reference  
**Purpose**: Complete ng_collect_rollouts parameter list

```markdown
# Rollout Collection Configuration Reference

Complete parameter reference for `ng_collect_rollouts` command.

## Overview

All parameters that control rollout collection behavior.

---

## Core Parameters

### agent_name

- **Type**: `str` (required)
- **Description**: Name of agent server configuration to use
- **Example**: `+agent_name=simple_agent`
- **Related**: Agent configuration in YAML files

### input_jsonl_fpath

- **Type**: `str` (required)
- **Description**: Path to input dataset JSONL file
- **Format**: One task per line in Responses API format
- **Example**: `+input_jsonl_fpath=data/tasks.jsonl`

[... complete parameter list with descriptions, types, defaults, examples ...]

---

## Sampling Parameters

[Detailed reference for all responses_create_params options]

---

## Parallelization Parameters

[Detailed reference for parallel generation options]

---

## Output Parameters

[Detailed reference for output control]

---

## Example Configurations

### High-Throughput SFT Data

```yaml
# Optimized for fast generation
agent_name: my_agent
input_jsonl_fpath: full_dataset.jsonl
output_jsonl_fpath: sft_data.jsonl
num_samples_in_parallel: 20
responses_create_params:
  temperature: 0.2
  top_p: 0.9
```

### Diverse DPO Pairs

```yaml
# Optimized for diversity
agent_name: my_agent
input_jsonl_fpath: tasks.jsonl
output_jsonl_fpath: dpo_data.jsonl
num_repeats: 2
num_samples_in_parallel: 10
responses_create_params:
  temperature: 0.7
  top_p: 0.95
```

[More examples...]

---

## Related

- [Optimize Throughput](optimize-throughput.md) - How-to guide
- [Sampling Strategies](sampling-strategies.md) - Choose right parameters
- [Collection Patterns](collection-patterns.md) - Common patterns
```

---

## Success Criteria

### Documentation Quality
- [ ] Each directory has clear topic scope
- [ ] Each index.md orients users to topic
- [ ] Each how-to has clear JTBD
- [ ] Each reference is comprehensive
- [ ] Content type tracked for every page

### Coverage
- [ ] All 7 dimensions covered
- [ ] No content type gaps (sufficient how-to, reference)
- [ ] Progressive disclosure enables multiple entry points
- [ ] Topic organization feels natural (not forced)

### User Experience
- [ ] Users find right directory for their task quickly
- [ ] Directory index pages guide to specific pages
- [ ] How-to guides enable task completion
- [ ] Reference content supports advanced use

---

## Key Advantages of This Structure

### vs. v1 (Diataxis-organized)

**Better**:
- ✓ Natural topic grouping (easier to find content)
- ✓ Multiple related pages per topic (more depth)
- ✓ Directory indexes provide topic overview
- ✓ Scales better (can add pages to directories)

**Same**:
- = Content type tracking (we still know what each is)
- = Coverage completeness (all dimensions)
- = Progressive disclosure (same navigation patterns)

### Organization Benefits

**Topic-based directories**:
- Users think in topics, not content types
- Related content grouped together
- Natural place to add new content
- Directory index provides orientation

**Internal content type tracking**:
- We know gaps in how-to vs reference
- Can balance content types per topic
- Ensure sufficient task guidance
- Maintain Diataxis principles without forcing structure

---

## Next Steps

1. **Review** this topic-organized structure
2. **Validate** directory groupings make sense
3. **Confirm** content type distribution is appropriate
4. **Approve** to proceed with Phase 1 implementation
5. **Start drafting** training/index.md + first directory

---

## Deliverables Summary

**Total**: 48 pages organized in 8 topic directories

**By Phase**:
- Phase 1: 11 pages (essential workflows)
- Phase 2: 11 pages (quality and config)
- Phase 3: 11 pages (scaling and frameworks)
- Phase 4: 15 pages (advanced and polish)

**By Content Type**:
- Explanation: 8 pages (directory indexes)
- How-to: 27 pages (task guidance)
- Reference: 13 pages (lookup information)

**Timeline**: 15 weeks for complete Training section

