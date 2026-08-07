---
name: Environment Integration
about: Propose integrating an existing environment or benchmark into NeMo Gym
title: '[Environment] '
labels: 'env-integration'
assignees: ''

---

### Environment Overview

- Name:
- Source repo:
- Paper/reference (if applicable):
- License:
- Brief description: What does this environment evaluate? (e.g. web navigation, code generation, tool use)

### How does the agent interact with the environment?

Describe what a typical task looks like from the agent's perspective. For example:
- Does the agent receive a natural language prompt and return an answer?
- Does the model use tools (function calling, code execution, web browsing)?
- Is it single-turn or multi-turn (does the model get feedback and retry)?

### Verifier Shape

Describe the reward signal — what constitutes a successful completion? Is it binary pass/fail, a score, or multiple metrics? How is correctness determined (exact match, test cases, judge model, human eval)?

### External Dependencies

Does this environment require external tools, specific runtimes, or sandboxes (e.g. compilers, browsers, Docker, VMs)?
If so, list them and note whether they can be auto-installed on server startup.

### Data

- Dataset source (e.g. HuggingFace, custom):
- Approximate size (number of tasks):
- Splits available (train/validation/test):

### Known Results

Are there published or known results to use as a reference? Link to leaderboards, papers, or repos with reported numbers.

### Constraints & Requirements

Note anything an engineer should know about running this environment:
- Does it need specific hardware (GPUs, large memory)?
- Does it require network access, Docker, or a VM?
- Are there known limitations on parallelism or throughput?
- Any OS or platform restrictions?

### Implementation Request
- [ ] I plan to implement this myself
- [ ] I'm requesting help to implement this

### Definition of Done

- [ ] Environment can be launched with `ng_run`
- [ ] Rollouts can be collected end-to-end with `ng_collect_rollouts`
- [ ] Reward scores reproduce known/expected results
- [ ] Example data committed for smoke testing
- [ ] Train/validation datasets uploaded to dataset registry
- [ ] Tests passing
- [ ] Documentation in environment README
- [ ] Benchmark config defined if applicable (e.g. pinned agent harness, dataset subset, num_repeats)

### Additional Context

Add any other context, links, or screenshots here.
