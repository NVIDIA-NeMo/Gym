# Validation

The implementation was checked against Gym `86e2252f5` using NousResearch
Hermes `v2026.8.31` (`29112bef099274229cadff79cdff7bf7b99c4b77`).
The [recorded results](validation.json) include the dataset/evaluator pins,
selected instance, container checksum and CPU outcomes.

## Local checks

149 targeted tests passed, covering the agent, Pro resources server, image
provenance and Apptainer provider. Ruff and shell syntax checks passed.

```bash
pytest responses_api_agents/hermes_sandboxed_agent/tests \
  resources_servers/swebench_pro/tests tests/unit_tests/test_apptainer_provider.py
```

## CPU container checks

CPU job `1918309` completed in 2m55s with 8 CPUs, 32 GiB and zero GPUs.
It used the existing [Ansible example](../../resources_servers/swebench_pro/data/example.jsonl#L4)
and real Gym HTTP services, the pinned Hermes runtime, and the Pro verifier.

The intentionally unreachable model endpoint caused a harness failure. The
result retained `verifier_reward=0`, omitted the benchmark reward, and was
routed to the failures file. Metrics recorded one attempted, zero scored,
one excluded, and `accuracy=null`. The smoke command exited 1 as intended;
the enclosing validation job asserted this behavior and completed successfully.
The candidate's empty patch produced 38 passing tests and 4 failing tests.

The same job then used a fresh container to grade the reference patch. All
42 tests passed and `resolved=true`. This was a reference-patch control;
Hermes did not generate the solution.

The agent container and both verification containers recorded the same
registry digest and SIF checksum. Writable overlays used a job-specific path
under `/var/tmp`, on a filesystem reporting 14 TiB capacity. An earlier CPU
probe confirmed `/tmp` was tmpfs. The portable [Slurm example](examples/slurm/run_pro.sh)
uses this launch sequence with site paths supplied through environment variables;
the example itself has been checked for shell syntax.

Full raw logs, trajectories and JUnit output are retained in the author's
development workspace. This report summarizes those outputs.

## Remaining validation

A live model rollout is still required to inspect inference, tool activity,
the generated patch and grading. Cross-benchmark swappability is unproven:
Verified and multilingual work remain outside the current scope.
