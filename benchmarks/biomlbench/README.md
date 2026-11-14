# BioML-bench v0.1-alpha

[BioML-bench](https://github.com/science-machine/biomlbench) (MIT,
[paper](https://www.biorxiv.org/content/10.1101/2025.09.01.673319v2)) asks an agent to build a model
end to end for a biomedical task and submit predictions, which the task's own grader scores against
a public leaderboard. Gym runs the 20 CPU tasks of its v0.1a release: 6 ProteinGym DMS, 5
OpenProblems single-cell and 9 Polaris/TDC drug-discovery tasks.

Each task is a Harbor task folder written by `prepare.py`. Terminus-2 runs beside Gym's model server
(`harbor_agent_general`, entrypoint `biomlbench_app.py`), and its commands run in BioML-bench's own
task image on OpenSandbox.

## Measurement contract

- **Source.** Task descriptions and configs from upstream commit `d5b4488d`, the tree the published
  image `millerh1/biomlbench-env:v0.1a` was built from (pinned by digest in `prepare.py`). The task
  list is `experiments/biomlbench_v0.1a.txt` from `673f3d8d`, which also adds the MIT license.
- **Prompt.** BioML-bench's `instructions.txt`, the agent notes it gives AIDE (without AIDE's step
  budget), and the task's `description.md`, in the order of upstream's agent start scripts.
- **Data.** At sandbox start, `environment.py` runs upstream's own preparer as root and keeps only the
  public split, at `/home/data`, read-only for the `nonroot` agent. A submission checker answers on
  `localhost:5000`. Leaderboards are the snapshots baked into the image.
- **Isolation.** A process in an OpenSandbox sandbox can run commands as root through the sandbox's
  exec daemon on `localhost`, so a root-only folder does not hide the answers from the agent. The
  private split therefore never exists in the agent's sandbox. The verifier runs in a separate, fresh
  sandbox (`environment_mode = "separate"`), which receives only `/home/submission`, prepares the task
  again from the same pins, and grades. A symlinked submission is not graded.
- **Budget.** The paper's CPU budget: 16 CPUs, 64 GB, no GPU, 8 hours per task, with internet.
- **Scoring.** `tests/test.sh` grades `/home/submission/submission.{csv,h5ad}` with upstream's grader
  and writes `reward.json`. `reward` is 1 when the score reaches a leaderboard medal (MLE-bench
  thresholds), so `mean/reward` is the any-medal rate. The file also holds `score`,
  `leaderboard_percentile` (0 without a valid submission, the paper's penalized setting),
  `gold_medal`, `silver_medal`, `bronze_medal`, `above_median` and `valid_submission`; they are in each
  trial's `verifier/` folder under the Harbor jobs folder. The paper averages 4 replicates per task.
  A grader error other than an invalid submission fails the verifier, so it is never scored as 0.

## Data checksums

The image's preparers run with checksum verification off: the `checksums.yaml` files in the image
do not match what its preparers write today, for ProteinGym and OpenProblems tasks. `checksums.json`
pins the files the image prepares today. Against upstream's
`checksums.yaml` at `d5b4488d`, 3 tasks differ:

- `open-problems-predict-modality`: the answers and the `.h5ad` inputs.
- `open-problems-single-cell-perturbations`: `de_train.parquet`.
- `open-problems-spatially-variable-genes`: the sample submission.

Two preparations gave identical files except the sample submissions of `open-problems-predict-modality`
and `open-problems-spatially-variable-genes`, which upstream fills with unseeded random values. Their
pins are `null`: the file must exist. A task whose prepared files differ from the pins fails at sandbox
start, before the agent runs, and in the verifier, before grading.

## Deviations

| Setting | Upstream | Here | Why |
|---|---|---|---|
| Agent | AIDE, Biomni, STELLA, MLAgentBench | Terminus-2 | Gym measures a model under one harness; Terminus-2 summarizes long contexts |
| Tasks | 24 | 20 | The 4 Kaggle imaging tasks need a Kaggle account that accepted each competition's rules, and a GPU |
| AWS CLI | Host tool | `awscli==1.35.0` installed in the image's `biomlb` env for OpenProblems tasks | Their preparers call `aws s3 cp --no-sign-request`; the image has no AWS CLI |
| Data checksums | `checksums.yaml` | `checksums.json` | See above |
| Submission checker | Runs the grader on the private answers | Checks columns, row count, ids and missing values against the public sample submission | The agent's sandbox holds no answers (see Isolation) |
| Verification | In the agent's container | In a separate sandbox that prepares the task again | See Isolation |

Agents have internet access, as upstream. Several source datasets are public, so a determined agent
could find test labels; BioML-bench relies on its instructions forbidding this.

## Run

```bash
gym eval prepare --benchmark biomlbench/harbor_opensandbox
gym eval run --benchmark biomlbench/harbor_opensandbox --model-type openai_model \
  --model "$POLICY_MODEL" --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY"
```

Preparation fetches each task's description and config from GitHub at the pinned commit; it writes
`data/`, which Git ignores. Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the driver. The
first sandbox on a node pulls the 12 GB image, which took 6 minutes. Data preparation takes up to 75
seconds per task. `harbor_agent_general.concurrency` caps concurrent trials (default 20).

To check the lifecycle without a model, run Harbor's oracle agent: each task's `solution/solve.sh`
copies the sample submission, BioML-bench's dummy agent. It scores a valid submission and no medal.
