# NeMo Gym Agent Skills

Agent skills for NeMo Gym development. Each skill follows the [agentskills.io](https://agentskills.io/specification) specification and can be used standalone or composed into multi-step chains.

## Skills

| Skill | What it does | When to use it |
|-------|-------------|----------------|
| [gym-review](gym-review/) | Deterministic anti-pattern checker + judgment-based review | Reviewing PRs, auditing servers before merge |
| [gym-debug](gym-debug/) | Diagnose server failures, rollout errors, unexpected rewards | Servers won't start, rollouts hang, rewards look wrong |
| [gym-run](gym-run/) | Run benchmarks — env.yaml setup, server launch, rollout collection | First run, smoke testing, full rollout collection |
| [gym-profile](gym-profile/) | Analyze rollout results, reward distributions, pass rates | Baselining benchmarks, comparing models, investigating variance |
| [gym-config](gym-config/) | Compose and validate Hydra YAML configurations | Setting up server configs, debugging composition errors |
| [gym-data](gym-data/) | Prepare, validate, and register JSONL datasets | Converting data, uploading to GitLab registry, validating schemas |
| [gym-scaffold-agent](gym-scaffold-agent/) | Create custom agent servers | Multi-turn interaction, external library wrapping, tool orchestration |
| [add-benchmark](add-benchmark/) | End-to-end benchmark creation guide | Adding a new resources server + agent + data + config |

## Chains

Chains compose skills into multi-step workflows. Defined in [`chains.yaml`](chains.yaml).

| Chain | Steps | Use case |
|-------|-------|----------|
| **run** | gym-config > gym-run > gym-profile | Executing a configured benchmark end-to-end |
| **new-benchmark** | add-benchmark > gym-data > gym-config > gym-run > gym-profile > gym-review | Building a benchmark from scratch |
| **validate** | gym-config > gym-data > gym-run > gym-profile | Checking an existing benchmark works correctly |
| **diagnose** | gym-debug > gym-review | Debugging a failing benchmark |
| **external-integration** | gym-scaffold-agent > gym-data > gym-config > gym-run > gym-profile > gym-review | Wrapping a 3rd-party benchmark library |
| **pre-merge** | gym-review > gym-config > gym-data | Final checks before merging a PR |

## Skill structure

Each skill follows a consistent layout:

```
skill-name/
  SKILL.md             # Skill definition (YAML frontmatter + instructions)
  evals/
    evals.json         # Assertion-based evaluations
    files/             # Self-contained test fixtures (if applicable)
  references/          # Portable reference docs (if applicable)
  scripts/             # Deterministic tooling (if applicable)
```

**gym-review** is the reference implementation: it includes a standalone Python checker (`scripts/review.py`), self-contained reference docs, and eval fixtures that work without the NeMo Gym repo.

## Evaluating skills

Each skill has 3 evals in `evals/evals.json`. Evals follow the [agentskills.io evaluation spec](https://agentskills.io/skill-creation/evaluating-skills).

### Running evals

Compare agent performance **with-skill** vs **without-skill** (baseline):

1. **With-skill**: Load the SKILL.md, give the agent the eval prompt, grade the response against assertions.
2. **Without-skill (baseline)**: Give the agent the same prompt with no skill loaded, grade against the same assertions.
3. **Compute delta**: The percentage-point improvement from loading the skill.

Each eval in `evals.json` has:

```json
{
  "id": 1,
  "prompt": "The task the agent must perform",
  "expected_output": "What a good response looks like",
  "files": ["evals/files/fixture.py"],
  "assertions": [
    "Specific claim that must be true in the response",
    "Another required element"
  ]
}
```

### Grading

For each assertion, score 1 (present in response) or 0 (missing). The skill's score is the average across all assertions and evals. A skill is useful when its with-skill score meaningfully exceeds the baseline.

### Example: gym-review

```bash
# The review script can also be tested directly
python .claude/skills/gym-review/scripts/review.py .claude/skills/gym-review/evals/files/

# Expected: 9 BLOCK, 4 WARN across the fixture files
# sample_clean_server.py should produce 0 findings
```

## Portability

Skills are designed to work when pulled standalone. Key design principles:

- **References are self-contained** -- no links to repo-internal paths that won't exist for external users
- **Scripts have zero dependencies** -- `review.py` uses only the Python standard library
- **Eval fixtures are bundled** -- test files live in `evals/files/`, not scattered across the repo
- **SKILL.md frontmatter** includes `license`, `compatibility`, and `allowed-tools` per the spec
