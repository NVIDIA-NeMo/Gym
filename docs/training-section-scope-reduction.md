# Training Section: Scope Reduction Analysis

**Context**: Other sections will cover configuration, performance, custom resource servers, and debugging  
**Question**: What should training/ focus on vs. delegate to other sections?

---

## Overlap Analysis

### Clear Delegations to Other Sections

#### 1. configuration/ → **Setup and Deployment / Configuration Management**

**Current training/configuration/ scope**:
- multi-model-setups.md
- environment-configs.md
- resource-allocation.md
- configuration-reference.md
- configuration-patterns.md

**Why move**: 
- Configuration Management section specifically covers "three-tier configuration hierarchy and practical patterns for development, testing, and production"
- This is system configuration, not training-specific
- Multi-model, environment, and resource configs apply to all workflows, not just training

**Training/ should only have**: Training-specific config examples within other guides (e.g., "config for SFT data generation" in rollout-collection/)

---

#### 2. performance/ → **Setup and Deployment / Performance & Scaling**

**Current training/performance/ scope**:
- throughput-optimization.md
- bottleneck-diagnosis.md
- model-server-tuning.md
- distributed-generation.md
- performance-benchmarks.md

**Why move**:
- "Performance & Scaling" section specifically covers "Profile, optimize, and scale your agent systems for high-throughput scenarios"
- System-level performance optimization, not training data-specific
- Model server tuning, distributed setups apply to all use cases

**Training/ should only have**: Training data generation optimization tips within rollout-collection/ (e.g., "optimize parallel collection for training datasets")

---

#### 3. verification/custom-verification.md → **Environments / Building Custom Resource Servers**

**Current training/verification/ includes**:
- custom-verification.md (building new verifiers from scratch)

**Why move**:
- "Building Custom Resource Servers" section specifically covers "create your own tools and verification systems"
- Building custom resource servers is broader than training
- Includes tools, verification, MCP integration

**Training/ should keep**: 
- reward-shaping.md (how to design rewards for training)
- verification-patterns.md (reference for choosing approaches)
- training-specific verification configuration

---

#### 4. troubleshooting/ → **Split between Agents and Setup sections**

**Current training/troubleshooting/ scope**:
- collection-failures.md
- verification-errors.md
- data-format-issues.md
- performance-problems.md
- error-reference.md

**Why split**:
- General agent debugging → **Agents / Debugging Agent Behaviors**
- System/deployment issues → **Setup and Deployment** (implied in deployment patterns)
- Training data-specific issues → Keep in training/

**Training/ should keep only**:
- Data format validation errors (training datasets)
- Data quality issues
- RL framework integration errors

---

## Revised Training Section Scope

### What Training/ Should Focus On

**Core mission**: Training data pipeline - from rollout generation to RL framework integration

```
training/
├── index.md                                    [Overview + Navigation]
│
├── rollout-collection/                         [KEEP - Core training data generation]
│   ├── index.md
│   ├── optimize-for-training.md               (Training-specific optimization)
│   ├── sampling-strategies.md                 (Diversity, temperature for SFT/DPO)
│   └── collection-patterns.md                 (Patterns by training type)
│
├── verification/                               [KEEP - Reward design for training]
│   ├── index.md
│   ├── reward-shaping.md                      (Design rewards for RL)
│   ├── verification-patterns.md               (Choose approach)
│   └── multi-objective-scoring.md             (Combine signals)
│
├── data-quality/                               [KEEP - Training data curation]
│   ├── index.md
│   ├── filtering-strategies.md
│   ├── quality-metrics.md
│   └── dataset-balancing.md
│
├── datasets/                                   [KEEP - Training dataset management]
│   ├── index.md
│   ├── prepare-for-training.md                (SFT/DPO/RL formats)
│   ├── validate-format.md
│   └── format-specification.md
│
└── integration/                                [KEEP - RL framework integration]
    ├── index.md
    ├── nemo-rl.md
    ├── verl.md
    ├── openrlhf.md
    ├── trl.md
    └── framework-comparison.md
```

**Reduced from**: 8 directories (48 pages)  
**Reduced to**: 5 directories (~25 pages)  
**Reduction**: ~48% fewer pages

---

## What Moves to Other Sections

### Setup and Deployment / Configuration Management

**Takes from training/configuration/**:
- Multi-model setups (policy + judge)
- Environment configs (dev/staging/prod)
- Resource allocation (GPU/memory)
- Configuration reference (all parameters)
- Configuration patterns (common combinations)

**Plus adds** (as originally planned):
- Three-tier configuration hierarchy
- Secrets management
- YAML → env.yaml → CLI patterns

**Training/ can reference**: "See Configuration Management for multi-model setup" with training-specific examples inline

---

### Setup and Deployment / Performance & Scaling

**Takes from training/performance/**:
- Throughput optimization (system-level)
- Bottleneck diagnosis (profiling)
- Model server tuning (vLLM, NIM optimization)
- Distributed generation (multi-node)
- Performance benchmarks

**Plus adds** (as originally planned):
- Production scaling strategies
- High-throughput scenarios

**Training/ can reference**: "See Performance & Scaling for throughput optimization" with training data-specific tips inline

---

### Environments / Building Custom Resource Servers

**Takes from training/verification/**:
- custom-verification.md (building new verifiers)

**Plus adds** (as originally planned):
- Building custom tools
- MCP integration
- External service integration
- Dynamic prompts

**Training/ keeps**: 
- reward-shaping.md (how to design rewards)
- verification-patterns.md (reference of approaches)
- But references "Building Custom Resource Servers" for implementation

---

### Agents / Debugging Agent Behaviors

**Takes from training/troubleshooting/**:
- General agent behavior debugging
- Multi-step reasoning issues
- Conversation context problems

**Training/ keeps**:
- Training data format errors
- Data quality validation
- RL framework integration errors

---

## Revised Content Type Distribution

| Section | Explanation | How-to | Reference | Total |
|---------|-------------|--------|-----------|-------|
| **rollout-collection** | 1 | 2 | 1 | 4 |
| **verification** | 1 | 2 | 1 | 4 |
| **data-quality** | 1 | 3 | 0 | 4 |
| **datasets** | 1 | 2 | 1 | 4 |
| **integration** | 1 | 5 | 1 | 7 |
| **training/index.md** | 1 | 0 | 0 | 1 |
| **Total** | **6** | **14** | **4** | **24** |

**Reduced from**: 48 pages → **24 pages** (50% reduction)

---

## Revised Dimensional Coverage

| Dimension | Training/ Coverage | Delegated To | Notes |
|-----------|-------------------|--------------|-------|
| **Data Generation** | rollout-collection/ | Performance & Scaling (system optimization) | Training keeps collection strategies |
| **Verification** | verification/ (reward design) | Environments (custom building) | Training keeps reward shaping |
| **Data Quality** | data-quality/ | - | Training keeps all |
| **Formats** | datasets/ | - | Training keeps all |
| **Configuration** | Inline examples only | Configuration Management | Delegate system config |
| **Scale** | Brief tips only | Performance & Scaling | Delegate system performance |
| **Integration** | integration/ | - | Training keeps RL frameworks |

**Training/ focus**: Training data pipeline (collection → quality → format → integrate)

---

## Benefits of Scope Reduction

### 1. Clearer Section Purpose

**Before**: Training covers everything from data generation to system configuration to performance tuning  
**After**: Training focuses on training data pipeline specifically

### 2. Avoid Duplication

**Before**: Configuration covered in both training/ and Configuration Management  
**After**: Single source of truth for each topic

### 3. Easier Maintenance

**Before**: 48 pages across 8 directories  
**After**: 24 pages across 5 directories (50% less content)

### 4. Better User Navigation

**Before**: User unsure whether to look in training/ or Setup and Deployment/  
**After**: Clear boundaries - training data → training/, system setup → Setup and Deployment/

### 5. Faster Implementation

**Before**: 15 weeks for 48 pages  
**After**: 7-8 weeks for 24 pages

---

## Revised Phased Implementation

### Phase 1: Core Training Data Pipeline (Weeks 1-3)

```
Week 1: training/index.md
        rollout-collection/index.md, optimize-for-training.md
        
Week 2: rollout-collection/sampling-strategies.md, collection-patterns.md
        datasets/index.md, prepare-for-training.md
        
Week 3: datasets/validate-format.md, format-specification.md
        data-quality/index.md, filtering-strategies.md
```

**Deliverable**: 10 pages - Core data generation and formatting

---

### Phase 2: Quality and Verification (Weeks 4-5)

```
Week 4: data-quality/quality-metrics.md, dataset-balancing.md
        verification/index.md, reward-shaping.md
        
Week 5: verification/verification-patterns.md, multi-objective-scoring.md
```

**Deliverable**: 6 pages - Quality assurance and reward design

---

### Phase 3: Framework Integration (Weeks 6-7)

```
Week 6: integration/index.md, nemo-rl.md, framework-comparison.md
        
Week 7: integration/verl.md, openrlhf.md, trl.md
```

**Deliverable**: 7 pages - All RL framework integrations

---

### Phase 4: Polish (Week 8)

```
Week 8: Review all content
        Add cross-references to other sections
        Finalize examples and troubleshooting tips
```

**Total Timeline**: 8 weeks for complete Training section (vs. 15 weeks before)

---

## Cross-Section References

### From training/ → Other Sections

**To Configuration Management**:
- "For multi-model setup details, see [Configuration Management](../setup-deployment/configuration-management.md)"
- Training/ includes: Inline config examples for training scenarios

**To Performance & Scaling**:
- "For system-level optimization, see [Performance & Scaling](../setup-deployment/performance-scaling.md)"
- Training/ includes: Training data collection optimization tips

**To Building Custom Resource Servers**:
- "To build custom verification logic, see [Building Custom Resource Servers](../environments/custom-resource-servers.md)"
- Training/ includes: Reward shaping patterns using existing verifiers

**To Debugging Agent Behaviors**:
- "For agent behavior debugging, see [Debugging Agent Behaviors](../agents/debugging-agent-behaviors.md)"
- Training/ includes: Training data-specific validation errors only

---

### From Other Sections → training/

**From Configuration Management**:
- "For training-specific configurations, see [Training / Rollout Collection](../training/rollout-collection/)"

**From Performance & Scaling**:
- "For training data generation optimization, see [Training / Rollout Collection](../training/rollout-collection/optimize-for-training.md)"

**From Building Custom Resource Servers**:
- "For reward shaping patterns, see [Training / Verification](../training/verification/reward-shaping.md)"

**From Get Started**:
- "Ready to scale up training data generation? See [Training](../training/)"

---

## Revised Directory Details

### rollout-collection/ (4 pages)

**Focus**: Generate training datasets at scale

**Pages**:
1. `index.md` - Overview of rollout collection for training
2. `optimize-for-training.md` - Training-specific optimization (not system-level)
3. `sampling-strategies.md` - Temperature, diversity, repeats for SFT/DPO
4. `collection-patterns.md` - Patterns by training type (SFT vs DPO vs RL)

**Removed**: Configuration reference (→ Configuration Management), Performance tuning (→ Performance & Scaling)

---

### verification/ (4 pages)

**Focus**: Design reward signals for training

**Pages**:
1. `index.md` - Overview of verification for training
2. `reward-shaping.md` - Design effective reward signals
3. `verification-patterns.md` - Reference of verification approaches
4. `multi-objective-scoring.md` - Combine multiple reward signals

**Removed**: custom-verification.md (→ Building Custom Resource Servers), testing-verifiers.md (→ Building Custom Resource Servers)

---

### data-quality/ (4 pages)

**Focus**: Curate high-quality training data

**Pages**:
1. `index.md` - Overview of data quality for training
2. `filtering-strategies.md` - Filter rollouts by quality
3. `quality-metrics.md` - Track quality during collection
4. `dataset-balancing.md` - Balance task distributions

**Removed**: manual-curation.md (merged into filtering-strategies.md), statistics-reference.md (→ Configuration Management or inline)

---

### datasets/ (4 pages)

**Focus**: Manage and format training datasets

**Pages**:
1. `index.md` - Overview of dataset management
2. `prepare-for-training.md` - Convert to SFT/DPO/RL formats
3. `validate-format.md` - Format validation with ng_prepare_data
4. `format-specification.md` - Rollout JSON schema reference

**Removed**: organize-datasets.md (basic enough to be in index.md), conversion-utilities.md (merged into prepare-for-training.md)

---

### integration/ (7 pages)

**Focus**: Connect to RL training frameworks

**Pages**:
1. `index.md` - Overview of RL framework integration
2. `nemo-rl.md` - NeMo-RL integration guide
3. `verl.md` - VeRL integration guide
4. `openrlhf.md` - OpenRLHF integration guide
5. `trl.md` - TRL/HuggingFace integration guide
6. `custom-frameworks.md` - Custom integration patterns (brief)
7. `framework-comparison.md` - Framework requirements matrix

**No changes**: This is training-specific and doesn't overlap with other sections

---

## Final Recommendations

### Training/ Should Be

**Scope**: Training data pipeline from rollout generation to RL framework integration

**Size**: 5 directories, 24 pages total

**Timeline**: 8 weeks to complete

**Focus**: 
- ✓ Rollout collection strategies for training
- ✓ Reward shaping for RL
- ✓ Data quality curation
- ✓ Dataset format management
- ✓ RL framework integration

**Delegates**:
- System configuration → Configuration Management
- Performance optimization → Performance & Scaling
- Custom verifier building → Building Custom Resource Servers
- General debugging → Debugging Agent Behaviors

### Benefits

1. **50% less content** (24 vs 48 pages)
2. **Clear boundaries** between sections
3. **No duplication** with other sections
4. **Faster to implement** (8 vs 15 weeks)
5. **Easier to maintain** (fewer pages, clearer scope)
6. **Better user navigation** (clear section purposes)

---

## Questions for Validation

1. **Scope agreement**: Does this reduced scope for training/ feel right?
2. **Delegation boundaries**: Are the boundaries between training/ and other sections clear?
3. **Integration section**: Should integration/ (7 pages) stay in training/ or move somewhere else?
4. **Cross-references**: Are cross-reference patterns clear enough?
5. **Implementation order**: Should training/ wait for Configuration Management and Performance & Scaling to be drafted first?

---

## Next Steps

1. **Approve reduced scope** (5 directories, 24 pages)
2. **Confirm delegation plan** with other section owners
3. **Update navigation** in index.md to reference other sections
4. **Begin Phase 1** (3 weeks, 10 pages: rollout-collection + datasets + data-quality basics)

