## Description: <br>
Take a NeMo Gym evaluation from a request to a trusted result: compose the config when none exists, submit a resumable Slurm run, recover it through walltime limits, preemption and node loss, check run health and the acceptance gates, and decide when to stop a chain or start a fresh run. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NeMo Gym maintainers (NVIDIA-NeMo/Gym) <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Engineers and researchers who run evaluations with NeMo Gym on a Slurm cluster, and the coding agents that work for them, use this skill to submit a benchmark run, read the recovery inspect, apply the one action it names, and report a score that is complete and healthy. <br>

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
- [Fresh run, and stopping a chain](references/fresh-run-and-stop.md) <br>
- [Benchmark notes](references/benchmark-notes.md) <br>
- [Resumable runs](https://docs.nvidia.com/nemo/gym/latest/orchestration/how-to/resumable-runs) <br>
- [Resume](https://docs.nvidia.com/nemo/gym/latest/evaluation/resume) <br>
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline shell and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>
