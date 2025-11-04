# Training Section: Structure & Content Type Tracking (v2)

**Organizational Approach**: Topic-based directories (not Diataxis categories)  
**Content Type Tracking**: Internal tracking to identify gaps  
**Total Deliverables**: 48 pages in 8 topic directories

---

## Structure Overview

```
training/
├── index.md                                    [Explanation]
│
├── rollout-collection/                         [6 pages]
│   ├── index.md                                [Explanation]
│   ├── optimize-throughput.md                  [How-to]
│   ├── parallel-generation.md                  [How-to]
│   ├── sampling-strategies.md                  [How-to]
│   ├── collection-patterns.md                  [Reference]
│   └── configuration-reference.md              [Reference]
│
├── verification/                               [6 pages]
│   ├── index.md                                [Explanation]
│   ├── custom-verification.md                  [How-to]
│   ├── reward-shaping.md                       [How-to]
│   ├── multi-objective-scoring.md              [How-to]
│   ├── verification-patterns.md                [Reference]
│   └── testing-verifiers.md                    [How-to]
│
├── data-quality/                               [6 pages]
│   ├── index.md                                [Explanation]
│   ├── filtering-strategies.md                 [How-to]
│   ├── manual-curation.md                      [How-to]
│   ├── quality-metrics.md                      [How-to]
│   ├── dataset-balancing.md                    [How-to]
│   └── statistics-reference.md                 [Reference]
│
├── datasets/                                   [6 pages]
│   ├── index.md                                [Explanation]
│   ├── organize-datasets.md                    [How-to]
│   ├── validate-format.md                      [How-to]
│   ├── prepare-for-training.md                 [How-to]
│   ├── format-specification.md                 [Reference]
│   └── conversion-utilities.md                 [Reference]
│
├── integration/                                [6 pages]
│   ├── index.md                                [Explanation]
│   ├── nemo-rl.md                              [How-to]
│   ├── verl.md                                 [How-to]
│   ├── openrlhf.md                             [How-to]
│   ├── trl.md                                  [How-to]
│   ├── custom-frameworks.md                    [How-to]
│   └── framework-comparison.md                 [Reference]
│
├── configuration/                              [6 pages]
│   ├── index.md                                [Explanation]
│   ├── multi-model-setups.md                   [How-to]
│   ├── environment-configs.md                  [How-to]
│   ├── resource-allocation.md                  [How-to]
│   ├── configuration-reference.md              [Reference]
│   └── configuration-patterns.md               [Reference]
│
├── performance/                                [6 pages]
│   ├── index.md                                [Explanation]
│   ├── throughput-optimization.md              [How-to]
│   ├── bottleneck-diagnosis.md                 [How-to]
│   ├── model-server-tuning.md                  [How-to]
│   ├── distributed-generation.md               [How-to]
│   └── performance-benchmarks.md               [Reference]
│
└── troubleshooting/                            [6 pages]
    ├── index.md                                [Explanation]
    ├── collection-failures.md                  [How-to]
    ├── verification-errors.md                  [How-to]
    ├── data-format-issues.md                   [How-to]
    ├── performance-problems.md                 [How-to]
    └── error-reference.md                      [Reference]
```

---

## Content Type Distribution

| Content Type | Count | Percentage | Pages |
|--------------|-------|------------|-------|
| **Explanation** | 8 | 17% | All `index.md` (directory overviews) |
| **How-to** | 27 | 56% | Task-oriented guides |
| **Reference** | 13 | 27% | Lookup information, specs, catalogs |
| **Tutorial** | 0 | 0% | Already covered in Get Started |
| **Total** | **48** | **100%** | |

**Assessment**: ✅ Good balance - Heavy on How-to (matches "JTBD" focus), sufficient Reference, minimal Explanation

---

## Content Type by Directory

| Directory | Explanation | How-to | Reference | Total |
|-----------|-------------|--------|-----------|-------|
| **rollout-collection** | 1 | 3 | 2 | 6 |
| **verification** | 1 | 4 | 1 | 6 |
| **data-quality** | 1 | 4 | 1 | 6 |
| **datasets** | 1 | 3 | 2 | 6 |
| **integration** | 1 | 5 | 1 | 7* |
| **configuration** | 1 | 3 | 2 | 6 |
| **performance** | 1 | 4 | 1 | 6 |
| **troubleshooting** | 1 | 5 | 1 | 7* |
| **Total** | **8** | **31** | **11** | **50** |

*Note: integration and troubleshooting have 1 extra how-to page (5 frameworks, 5 problem types)

---

## Dimensional Coverage Matrix

| Dimension | Directories Covering It | How-to Pages | Reference Pages |
|-----------|------------------------|--------------|-----------------|
| **Data Generation** | rollout-collection, performance | 7 | 3 |
| **Verification** | verification, troubleshooting | 5 | 1 |
| **Data Quality** | data-quality, datasets | 7 | 3 |
| **Formats** | datasets, integration | 6 | 3 |
| **Configuration** | configuration, rollout-collection | 6 | 4 |
| **Scale** | performance, rollout-collection | 7 | 2 |
| **Integration** | integration, datasets | 8 | 2 |

**Every dimension has**: ✓ Multiple how-to guides ✓ Reference material ✓ Coverage in 2+ directories

---

## Gap Analysis

### Content Type Gaps

**How-to guides**: ✅ No gaps (27 guides covering all major tasks)
- Data generation: 7 guides
- Quality/curation: 7 guides  
- Integration: 8 guides
- Configuration: 6 guides
- Troubleshooting: 5 guides

**Reference content**: ✅ No gaps (13 references covering all lookups)
- Configuration parameters: 3 references
- Formats/specs: 3 references
- Patterns/catalogs: 2 references
- Performance/benchmarks: 2 references
- Error codes: 1 reference
- Framework comparison: 1 reference

**Explanation content**: ✅ Minimal (8 directory overviews only)
- Deep explanations already in About/Concepts
- Directory indexes provide orientation
- No duplication

**Tutorial content**: ✅ No gap
- Already covered in Get Started section
- Training section focuses on scaling/production (post-tutorial)

### Topic Coverage Gaps

**No gaps identified** - All aspects of training workflows covered:

- ✅ Rollout generation (collection, optimization, parallelization)
- ✅ Verification (custom logic, reward shaping, patterns)
- ✅ Data quality (filtering, curation, monitoring, balancing)
- ✅ Dataset management (organization, validation, format conversion)
- ✅ Framework integration (5 frameworks + custom)
- ✅ Configuration (multi-model, environments, resources)
- ✅ Performance (optimization, profiling, distributed)
- ✅ Troubleshooting (all failure modes)

---

## Phased Implementation

### Phase 1: Essential Workflows (Weeks 1-4)
**Goal**: Enable basic scaling and integration  
**Pages**: 11 (training index + 2.5 directories)

```
Week 1: training/index.md
        rollout-collection/index.md, optimize-throughput.md
        
Week 2: rollout-collection/parallel-generation.md, configuration-reference.md
        
Week 3: datasets/index.md, prepare-for-training.md, format-specification.md
        
Week 4: integration/index.md, nemo-rl.md, framework-comparison.md
```

**Deliverable**: Core workflow documented (collection → datasets → training)

---

### Phase 2: Quality and Configuration (Weeks 5-8)
**Goal**: Enable production-quality workflows  
**Pages**: 11 (3 directories)

```
Week 5: data-quality/index.md, filtering-strategies.md
        
Week 6: data-quality/quality-metrics.md, statistics-reference.md
        verification/index.md, custom-verification.md
        
Week 7: verification/reward-shaping.md, verification-patterns.md
        
Week 8: configuration/index.md, multi-model-setups.md, configuration-reference.md
```

**Deliverable**: Quality assurance and configuration patterns documented

---

### Phase 3: Scaling and All Frameworks (Weeks 9-11)
**Goal**: Complete integration coverage and scaling  
**Pages**: 11 (complete integration + performance)

```
Week 9:  integration/verl.md, openrlhf.md, trl.md
         
Week 10: integration/custom-frameworks.md
         performance/index.md, throughput-optimization.md
         
Week 11: performance/bottleneck-diagnosis.md, performance-benchmarks.md
         rollout-collection/sampling-strategies.md, collection-patterns.md
```

**Deliverable**: All frameworks + performance optimization documented

---

### Phase 4: Advanced and Polish (Weeks 12-15)
**Goal**: Complete all directories  
**Pages**: 15 (complete remaining 4+ directories)

```
Week 12: troubleshooting/index.md, collection-failures.md, verification-errors.md
         
Week 13: troubleshooting/data-format-issues.md, performance-problems.md, error-reference.md
         
Weeks 14-15: Complete remaining pages:
         - verification/multi-objective-scoring.md, testing-verifiers.md
         - data-quality/manual-curation.md, dataset-balancing.md
         - datasets/organize-datasets.md, validate-format.md, conversion-utilities.md
         - configuration/environment-configs.md, resource-allocation.md, configuration-patterns.md
         - performance/model-server-tuning.md, distributed-generation.md
```

**Deliverable**: Complete Training section (all 48 pages)

---

## Directory Responsibilities

### rollout-collection/
**Focus**: Generating training rollouts at scale  
**Covers**: Throughput, parallelism, sampling, patterns, configuration  
**Content types**: 1 explanation + 3 how-to + 2 reference

### verification/
**Focus**: Scoring agent performance for training  
**Covers**: Custom logic, reward shaping, multi-objective, patterns, testing  
**Content types**: 1 explanation + 4 how-to + 1 reference

### data-quality/
**Focus**: Ensuring training data quality  
**Covers**: Filtering, curation, metrics, balancing, statistics  
**Content types**: 1 explanation + 4 how-to + 1 reference

### datasets/
**Focus**: Managing and preparing datasets  
**Covers**: Organization, validation, format conversion, specifications  
**Content types**: 1 explanation + 3 how-to + 2 reference

### integration/
**Focus**: Connecting to RL training frameworks  
**Covers**: 5 frameworks + custom patterns + comparison  
**Content types**: 1 explanation + 5 how-to + 1 reference

### configuration/
**Focus**: System configuration for training  
**Covers**: Multi-model, environments, resources, parameters, patterns  
**Content types**: 1 explanation + 3 how-to + 2 reference

### performance/
**Focus**: Optimizing training data generation  
**Covers**: Throughput, profiling, model tuning, distributed, benchmarks  
**Content types**: 1 explanation + 4 how-to + 1 reference

### troubleshooting/
**Focus**: Debugging training workflows  
**Covers**: Collection, verification, format, performance errors + reference  
**Content types**: 1 explanation + 5 how-to + 1 reference

---

## Success Metrics

### Quantitative

- **Total pages**: 48
- **How-to ratio**: 56% (27/48) - ✅ Matches JTBD focus
- **Reference ratio**: 27% (13/48) - ✅ Sufficient for lookup
- **Explanation ratio**: 17% (8/48) - ✅ Minimal (just overviews)
- **Average pages per directory**: 6
- **Directories**: 8 (all topics covered)

### Qualitative

- ✅ Topic-based organization (natural groupings)
- ✅ Content type tracked (no gaps)
- ✅ All dimensions covered
- ✅ Progressive disclosure enabled
- ✅ Scalable structure (can add pages to directories)

---

## Key Advantages

### vs. Single-File Approach
- ✓ Multiple pages per topic (more depth)
- ✓ Directory indexes provide orientation
- ✓ Related content grouped together
- ✓ Easy to add new pages

### vs. Diataxis-Organized Structure
- ✓ Natural topic grouping (users think in topics)
- ✓ Still track content types internally
- ✓ Better navigation (topic → specific page)
- ✓ Avoid artificial content type boundaries

### Content Type Tracking Benefits
- ✓ Identify gaps (e.g., "missing how-to for X")
- ✓ Ensure balance (not all reference, not all explanation)
- ✓ Maintain Diataxis principles without forced structure
- ✓ Guide content creation (what type does this topic need?)

---

## Next Actions

1. **Review** structure and content type distribution
2. **Validate** directory groupings make sense
3. **Confirm** no gaps in coverage
4. **Approve** Phase 1 scope (11 pages, 4 weeks)
5. **Begin** drafting training/index.md

---

## Questions for Discussion

1. **Directory names**: Are these topic names clear and intuitive?
2. **Page distribution**: 6 pages per directory feel balanced, or should some have more/fewer?
3. **Phase 1 scope**: Is 11 pages in 4 weeks reasonable for first phase?
4. **Content type balance**: 56% how-to feel right, or need more/less?
5. **Missing topics**: Any training aspects not covered by these 8 directories?

