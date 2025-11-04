# Training Section: Final Approved Plan

**Date**: 2025-11-04  
**Status**: ✅ Approved  
**Scope**: Training data pipeline (generation → quality → format → integrate)

---

## Executive Summary

The **Training** section covers the training data pipeline from rollout collection through RL framework integration. System configuration, performance optimization, and custom resource building are delegated to other sections.

**Structure**: 5 topic directories, 24 pages total  
**Timeline**: 8 weeks to complete  
**Focus**: How-To guides (JTBD) for practitioners scaling up training workflows

---

## Final Structure

```
training/
├── index.md                                    [Navigation hub]
│
├── rollout-collection/                         [4 pages]
│   ├── index.md                                [Explanation]
│   ├── optimize-for-training.md                [How-to]
│   ├── sampling-strategies.md                  [How-to]
│   └── collection-patterns.md                  [Reference]
│
├── verification/                               [4 pages]
│   ├── index.md                                [Explanation]
│   ├── reward-shaping.md                       [How-to]
│   ├── verification-patterns.md                [Reference]
│   └── multi-objective-scoring.md              [How-to]
│
├── data-quality/                               [4 pages]
│   ├── index.md                                [Explanation]
│   ├── filtering-strategies.md                 [How-to]
│   ├── quality-metrics.md                      [How-to]
│   └── dataset-balancing.md                    [How-to]
│
├── datasets/                                   [4 pages]
│   ├── index.md                                [Explanation]
│   ├── prepare-for-training.md                 [How-to]
│   ├── validate-format.md                      [How-to]
│   └── format-specification.md                 [Reference]
│
└── integration/                                [7 pages]
    ├── index.md                                [Explanation]
    ├── nemo-rl.md                              [How-to]
    ├── verl.md                                 [How-to]
    ├── openrlhf.md                             [How-to]
    ├── trl.md                                  [How-to]
    ├── custom-frameworks.md                    [How-to]
    └── framework-comparison.md                 [Reference]
```

**Total**: 24 pages across 5 directories

---

## Content Type Distribution

| Content Type | Count | % | Coverage |
|--------------|-------|---|----------|
| **Explanation** | 6 | 25% | Directory overviews (index.md) |
| **How-to** | 14 | 58% | Task-oriented guides ✅ |
| **Reference** | 4 | 17% | Lookup information |
| **Total** | **24** | **100%** | |

**Assessment**: ✅ How-to heavy (58%) matches "JTBD" focus

---

## Dimensional Coverage

| Dimension | Coverage in training/ | Notes |
|-----------|---------------------|-------|
| **Data Generation** | rollout-collection/ | Training data collection strategies |
| **Verification** | verification/ | Reward shaping for training |
| **Data Quality** | data-quality/ | Curation and filtering |
| **Formats** | datasets/ | SFT, DPO, RL formats |
| **Configuration** | Inline examples only | System config → Configuration Management |
| **Scale** | Brief optimization tips | System scaling → Performance & Scaling |
| **Integration** | integration/ | RL framework connections |

---

## What's Delegated to Other Sections

### To: Setup and Deployment / Configuration Management

**Receives**:
- Multi-model setups (policy + judge)
- Environment configs (dev/staging/prod)
- Resource allocation (GPU/memory)
- Configuration reference (all parameters)
- Three-tier config hierarchy

**training/ keeps**: Inline config examples for training scenarios

---

### To: Setup and Deployment / Performance & Scaling

**Receives**:
- System-level throughput optimization
- Bottleneck diagnosis and profiling
- Model server tuning (vLLM, NIM)
- Distributed generation (multi-node)
- Performance benchmarks

**training/ keeps**: Training data collection optimization tips

---

### To: Environments / Building Custom Resource Servers

**Receives**:
- Building custom verifiers from scratch
- Custom tools and environments
- MCP integration
- External service integration

**training/ keeps**: Reward shaping patterns using existing verifiers

---

### To: Agents / Debugging Agent Behaviors

**Receives**:
- General agent behavior debugging
- Multi-step reasoning issues
- Conversation context problems

**training/ keeps**: Training data-specific validation errors

---

## Directory Responsibilities

### rollout-collection/ (4 pages)
**Mission**: Generate training datasets at scale  
**JTBD**: "I need to collect training data efficiently"

**Pages**:
- `index.md` - Orientation to rollout collection for training
- `optimize-for-training.md` - Training-specific optimization (parallelism, sampling for SFT vs DPO)
- `sampling-strategies.md` - Temperature, diversity, repeats by training type
- `collection-patterns.md` - Reference of patterns (high-throughput SFT, diverse DPO, etc.)

**Delegates**: System performance tuning → Performance & Scaling

---

### verification/ (4 pages)
**Mission**: Design reward signals for training  
**JTBD**: "I need to shape rewards for my training objective"

**Pages**:
- `index.md` - Orientation to verification for training
- `reward-shaping.md` - Design effective reward signals (binary, continuous, sparse, dense)
- `verification-patterns.md` - Reference catalog of verification approaches
- `multi-objective-scoring.md` - Combine multiple reward signals

**Delegates**: Building custom verifiers → Building Custom Resource Servers

---

### data-quality/ (4 pages)
**Mission**: Curate high-quality training data  
**JTBD**: "I need to ensure my training data is high quality"

**Pages**:
- `index.md` - Orientation to data quality for training
- `filtering-strategies.md` - Filter rollouts by quality thresholds
- `quality-metrics.md` - Track quality during collection
- `dataset-balancing.md` - Balance task distributions and difficulty

**Delegates**: None (all training-specific)

---

### datasets/ (4 pages)
**Mission**: Manage and format training datasets  
**JTBD**: "I need to prepare datasets for training frameworks"

**Pages**:
- `index.md` - Orientation to dataset management
- `prepare-for-training.md` - Convert to SFT/DPO/RL formats
- `validate-format.md` - Validate with ng_prepare_data
- `format-specification.md` - Rollout JSON schema reference

**Delegates**: None (all training-specific)

---

### integration/ (7 pages)
**Mission**: Connect to RL training frameworks  
**JTBD**: "I need to integrate with my training framework"

**Pages**:
- `index.md` - Orientation to RL framework integration
- `nemo-rl.md` - NeMo-RL integration guide
- `verl.md` - VeRL integration guide
- `openrlhf.md` - OpenRLHF integration guide
- `trl.md` - TRL/HuggingFace integration guide
- `custom-frameworks.md` - Custom integration patterns
- `framework-comparison.md` - Framework requirements matrix

**Delegates**: None (all training-specific)

---

## Implementation Timeline

### Phase 1: Core Pipeline (Weeks 1-3)

**Goal**: Enable basic training data generation and formatting

**Deliverables**:
```
Week 1:
- training/index.md
- rollout-collection/index.md
- rollout-collection/optimize-for-training.md

Week 2:
- rollout-collection/sampling-strategies.md
- rollout-collection/collection-patterns.md
- datasets/index.md

Week 3:
- datasets/prepare-for-training.md
- datasets/validate-format.md
- datasets/format-specification.md
- data-quality/index.md
```

**Output**: 10 pages covering core data generation and formatting

---

### Phase 2: Quality & Verification (Weeks 4-5)

**Goal**: Enable data quality assurance and reward design

**Deliverables**:
```
Week 4:
- data-quality/filtering-strategies.md
- data-quality/quality-metrics.md
- data-quality/dataset-balancing.md

Week 5:
- verification/index.md
- verification/reward-shaping.md
- verification/verification-patterns.md
- verification/multi-objective-scoring.md
```

**Output**: 7 pages covering quality and verification

---

### Phase 3: Framework Integration (Weeks 6-7)

**Goal**: Complete RL framework integration guides

**Deliverables**:
```
Week 6:
- integration/index.md
- integration/nemo-rl.md
- integration/framework-comparison.md

Week 7:
- integration/verl.md
- integration/openrlhf.md
- integration/trl.md
- integration/custom-frameworks.md
```

**Output**: 7 pages covering all RL frameworks

---

### Phase 4: Polish & Cross-References (Week 8)

**Goal**: Finalize content and cross-references

**Tasks**:
- Review all 24 pages for consistency
- Add cross-references to other sections (Configuration Management, Performance & Scaling, etc.)
- Add inline config examples where delegating to Configuration Management
- Add inline optimization tips where delegating to Performance & Scaling
- Finalize troubleshooting tips (training data-specific only)
- Final technical review with SMEs

**Output**: Complete, polished Training section

---

## Cross-Reference Patterns

### From training/ → Other Sections

**Pattern**: Reference other section + provide training-specific inline guidance

**Examples**:

```markdown
<!-- In rollout-collection/optimize-for-training.md -->

## Configure Parallelization

For system-level parallelization configuration, see 
[Configuration Management](../setup-deployment/configuration-management.md).

**For training data generation specifically**:

```yaml
# SFT data (consistency over diversity)
num_samples_in_parallel: 20
responses_create_params:
  temperature: 0.2
  
# DPO pairs (diversity for comparison)
num_samples_in_parallel: 10
responses_create_params:
  temperature: 0.7
```
```

**References to add**:
- Configuration Management: 5-7 references with inline training examples
- Performance & Scaling: 3-5 references with inline optimization tips
- Building Custom Resource Servers: 2-3 references from verification/
- Debugging Agent Behaviors: 1-2 references with training data error tips

---

### From Other Sections → training/

**References expected**:

- **Get Started** → "Ready to scale up? See [Training](../training/)"
- **Tutorials** → "For production workflows, see [Training](../training/)"
- **Configuration Management** → "For training-specific configs, see [Training / Rollout Collection](../training/rollout-collection/)"
- **Performance & Scaling** → "For training data optimization, see [Training / Rollout Collection](../training/rollout-collection/optimize-for-training.md)"
- **Building Custom Resource Servers** → "For reward shaping, see [Training / Verification](../training/verification/reward-shaping.md)"

---

## Success Criteria

### Documentation Quality
- [x] Reduced scope (24 vs 48 pages)
- [ ] Every directory has clear mission statement
- [ ] Every how-to has clear JTBD
- [ ] Every reference is comprehensive
- [ ] Content type distribution supports practitioner needs (58% how-to)
- [ ] All cross-references to other sections in place

### Coverage Completeness
- [ ] Training data pipeline fully covered (generation → quality → format → integrate)
- [ ] All 5 RL frameworks documented
- [ ] Data quality patterns comprehensive
- [ ] Verification approaches cataloged
- [ ] Format specifications accurate

### User Experience
- [ ] Users find right directory for task within 1 minute
- [ ] How-to guides enable task completion without reading Concepts
- [ ] Reference content supports advanced users
- [ ] Clear delegation to other sections (no confusion about where to look)
- [ ] Inline examples sufficient for training-specific needs

### Maintainability
- [ ] No duplication with other sections
- [ ] Clear boundaries between training/ and other sections
- [ ] Manageable scope (24 pages vs 48)
- [ ] Topic-based organization scales well

---

## Page-Level Specifications

### training/index.md

**Content Type**: Explanation + Navigation  
**Length**: ~200 lines  
**Sections**:
- Brief overview: What this section covers
- When to use Training vs. other sections
- Quick decision guide: "Which directory should I start with?"
- Grid navigation to 5 directories
- Cross-references to related sections

**Key Message**: "You've learned the basics in Get Started—now scale up your training data pipeline"

---

### rollout-collection/optimize-for-training.md

**Content Type**: How-to  
**JTBD**: "I need to generate training data faster"  
**Length**: ~250-300 lines  
**Sections**:
- Before You Start (prerequisites)
- Understand Collection Performance (brief context)
- Optimize for SFT Data (low temperature, consistency)
- Optimize for DPO Pairs (higher temperature, diversity)
- Optimize for RL Data (balanced approach)
- Monitor Generation (throughput tracking)
- Production Examples (complete configs)
- Troubleshooting (common issues)
- Related Resources (cross-references)

**References**:
- Performance & Scaling for system optimization
- Configuration Management for parameter details

---

### verification/reward-shaping.md

**Content Type**: How-to  
**JTBD**: "I need to design effective rewards for training"  
**Length**: ~300-350 lines  
**Sections**:
- Before You Start
- Understand Reward Impact (brief theory)
- Choose Reward Type (binary, continuous, sparse, dense)
- Design for SFT (reward for correct behavior)
- Design for DPO (reward differences for comparison)
- Design for RL (shaped rewards for exploration)
- Test Reward Signals (validation approaches)
- Production Examples (common patterns)
- Troubleshooting (reward issues)
- Related Resources

**References**:
- Building Custom Resource Servers for implementing verifiers
- Concepts/Verifying Agent Results for deep theory

---

### integration/nemo-rl.md

**Content Type**: How-to  
**JTBD**: "I need to integrate with NeMo-RL"  
**Length**: ~300-350 lines  
**Sections**:
- Before You Start
- NeMo-RL Requirements (format, dependencies)
- Prepare Data (from rollouts to NeMo-RL format)
- Configure Training (NeMo-RL config)
- Launch Training (command and workflow)
- Monitor Training (progress tracking)
- Complete Example (end-to-end)
- Troubleshooting (common issues)
- Related Resources

**Similar structure** for verl.md, openrlhf.md, trl.md

---

## Next Steps

1. **Begin Phase 1** (Week 1)
   - Draft training/index.md
   - Draft rollout-collection/index.md
   - Draft rollout-collection/optimize-for-training.md

2. **Coordination** with other section owners
   - Confirm Configuration Management will cover multi-model setups
   - Confirm Performance & Scaling will cover system optimization
   - Confirm Building Custom Resource Servers will cover verifier implementation
   - Establish cross-reference conventions

3. **Technical Review** setup
   - Identify SMEs for each directory
   - Schedule review cycles
   - Set up validation workflow

4. **Progress Tracking**
   - Weekly check-ins on page completion
   - Content review after each phase
   - User feedback collection (if applicable)

---

## Appendix: Comparison with Original Plan

| Aspect | Original Plan | Final Plan | Change |
|--------|--------------|------------|--------|
| **Directories** | 8 | 5 | -3 |
| **Total Pages** | 48 | 24 | -24 (50%) |
| **Timeline** | 15 weeks | 8 weeks | -7 weeks |
| **How-to %** | 56% | 58% | +2% |
| **Scope** | Training + config + perf + debug | Training data pipeline only | Focused |
| **Delegation** | None | 4 sections | Clear boundaries |

**Result**: More manageable, realistic, and maintainable scope ✅

