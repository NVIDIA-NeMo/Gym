# Evaluation intent

Measure whether an assistant can prepare a reproducible Gym Slurm evaluation, choose the permitted recovery action,
and report only the score supported by the saved evidence. The dataset contains 13 distinct cases: 11 evaluation
requests and two negative triggers (authoring a benchmark and integrating a harness).

These are CPU-only, offline operator-review tasks. The fixtures are independently authored, synthetic projections of
Gym configuration and inspection output; they are not receipts from actual model or Slurm runs. Fields irrelevant to
each decision are omitted. The public Gym revision named in configuration fixtures is a real reference, not a claim
that this dataset executed it. No external code, third-party task data, credentials or production run artifacts are
included. Assertions are authored from intended outcomes rather than generated from observed agent answers.

## Questions

- Prepare a bounded smoke and acceptance checks around a supplied deployment without re-deriving its settings.
- Compose a model's complete config, retain IFBench's protocol, and stop after the requested config-only deliverable.
- Wait for a live continuation instead of duplicating it.
- Extend an exhausted chain walltime while preserving pin and score-selection flags.
- Re-grade saved answers on an available CPU placement without re-sampling the policy.
- Refuse an unapproved semantic change to an existing evaluation.
- Distinguish complete but unobserved grading evidence from verified acceptance.
- Reconcile a final score, denominator and tracking state against saved artifacts instead of an old chat number.
- Report a limited, lower-repeat smoke accurately without promoting it to a full benchmark score.
- Separate settled model faults from judge infrastructure failures and their different recovery actions.
- Replace a moving Gym branch with a complete public commit before validation and submission.

## Behaviors

- Preserve the supplied model deployment and benchmark protocol unless the request explicitly authorizes changes.
- Choose actions from the current inspection and retained evidence; live work must not be queued again.
- Keep pin and gate options when deriving recovery commands; extend only the budget that was exhausted.
- Reuse saved answers after judge failures; do not erase artifacts or turn infrastructure failures into model zeros.
- Report uncertainty and missing grading evidence explicitly; complete accounting and process success are insufficient.
- Use the saved aggregate metric and its actual selected denominator; distinguish smoke results from benchmark scores.

## Negative Cases

- Add an entirely new benchmark, including its dataset converter and verifier/server, before running any evaluation.
- Integrate an unsupported CLI agent runtime through agent sessions, with adapter design and conformance checks.

## Notes

Every entry explicitly selects its own fixture directory with `files: ["files/<case-id>"]`. The evaluator stages its
`evidence.json` or `evidence.yaml` under `/workspace/input/<case-id>/`. No case can read another case's evidence or the
judge's expected answer.
The with-skill and without-skill arms receive identical requests, input files, runtime policy, expected outcomes and
assertions. Outcomes never require reading a particular instruction file or invoking a bundled script. The two
negative cases use JSON `null` for `expected_skill`; routing is measured separately from outcome correctness.

Run the official structural preflight before live evaluation:

```bash
skillevaluator tier3 validate skills/nemo-gym-run-and-babysit-eval --strict
```

Run the entire dataset on both supported agents, retaining each agent's paired baseline. Keep provider/agent selection
in the operator's CLI invocation rather than baking a model or credentials into this portable dataset. The committed
policy uses one attempt per case, four concurrent trials, and a 0.3 timeout multiplier. CI CPU and memory
allocations come from the selected sandbox template.
The target is under one hour for the small offline cases; measure actual elapsed time rather than claiming this target
was met in advance. Do not cap the case count or skip the baseline for publication evidence.

A manual Claude Code/Codex login-based comparison is useful supplementary evidence, but it is not SkillEvaluator
Tier 3 certification. Its raw outputs, command/agent versions, independent grading and limitations must remain
identifiable. Do not create a PASS benchmark report, signature or generated skill card from these fixture checks.

The fixture suite tests operator decisions and config preparation. Publishing the broader customer workflow also needs
actual representative model rollouts, Slurm recovery and final artifact reconciliation. Local pip/endpoint/task-folder
paths require their own end-to-end evidence and a released Gym version that implements them. Record unavailable
compute, provider or signing prerequisites as outstanding work rather than silently substituting these offline cases.
