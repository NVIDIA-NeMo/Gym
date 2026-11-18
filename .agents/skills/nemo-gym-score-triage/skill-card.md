## Description: <br>
Decide whether two NeMo Gym scores are comparable before any deeper analysis: a score that differs from a reference number, two runs or harnesses that disagree, repeat-to-repeat variance, or a baseline versus candidate difference such as BF16 against a quantized checkpoint. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NeMo Gym maintainers (NVIDIA-NeMo/Gym) <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Engineers and researchers who compare NeMo Gym results with a reference or with another run, and the coding agents that work for them, use this skill to rule out noise, code, benchmark, method, failure accounting, health, scoring and serving differences, in order, and to report which one explains the gap. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

The skill ships no credentials. Credentials a run needs are the user's own and are passed through environment variables. Do not include secrets in prompts, logs or output. <br>

## Known Risks and Mitigations: <br>
Risk: The commands the skill recommends can start jobs or change state on the user's own cluster or account. <br>
Mitigation: They are the user's own `gym` commands, run with the user's own access, and the skill has the agent report every operator action with its reason. <br>

## Reference(s): <br>
- [Comparability checklist](references/comparability-checklist.md) <br>
- [Variance](references/variance.md) <br>
- [Benchmark notes](references/benchmark-notes.md) <br>
- [Compare results](https://docs.nvidia.com/nemo/gym/latest/evaluation/compare-results) <br>
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) <br>

## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline shell and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>
