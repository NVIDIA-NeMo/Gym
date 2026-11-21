## Description: <br>
Analyze NeMo Gym benchmark rollouts into an evidence-backed report: aggregate metrics, task outcome buckets, failure modes tied to rollouts, root-cause labels and improvement recommendations, and check whether a benchmark package has its BLADE deliverables. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NeMo Gym maintainers (NVIDIA-NeMo/Gym) <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Engineers and researchers who need to know which tasks fail and why in finished NeMo Gym runs, and the coding agents that work for them, use this skill to turn rollouts and metrics into a report with a failure taxonomy and concrete next actions. <br>

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
- [BLADE benchmark build guide](references/blade-benchmark-build-guide.md) <br>
- [Diagnose results](https://docs.nvidia.com/nemo/gym/latest/evaluation/diagnose-results) <br>
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) <br>

## Skill Output: <br>
**Output Type(s):** [Analysis, Markdown report, Shell commands] <br>
**Output Format:** [Markdown with inline shell and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>
