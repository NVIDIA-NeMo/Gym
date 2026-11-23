# Example review comments

These come from review comments on NVIDIA-NeMo/Gym PRs that touch `resources_servers/`, `environment_servers/`,
`benchmarks/`, `environments/` or `nemo_gym/data/tasks/harbor/`, opened from 2026-09-01 onward:

- 94 PRs scanned, 69 with a kept comment;
- 532 comments kept, from 34 reviewers;
- The most active reviewers are `yaoyu-33` (201) and `ananthsub` (134). The other 32 reviewers wrote 197.

Each line paraphrases one comment. They are examples of the bar, not rules.

| Item | PR comment | What it caught |
|---|---|---|
| 1 legacy | [#4209](https://github.com/NVIDIA-NeMo/Gym/pull/4209#discussion_r4526242364) | OSWorld drives its own `/run`; move it to an Environment Server. |
| 1 legacy | [#5480](https://github.com/NVIDIA-NeMo/Gym/pull/5480#pullrequestreview-5760779722) | `/run` seed code is copied from PR to PR instead of using the session path. |
| 2 seed | [#5480](https://github.com/NVIDIA-NeMo/Gym/pull/5480#discussion_r4477749666) | A re-seed leaks the previous run's state. |
| 2 seed | [#5487](https://github.com/NVIDIA-NeMo/Gym/pull/5487#discussion_r4462901117) | A typed-seed guard rejects the seeds shipped configs send. |
| 2 seed | [#5486](https://github.com/NVIDIA-NeMo/Gym/pull/5486#discussion_r4459898678) | Malformed typed bodies return 200 instead of a refusal. |
| 3 sandbox | [#5348](https://github.com/NVIDIA-NeMo/Gym/pull/5348#discussion_r4515788041) | A failed stop is never retried, so the sandbox lives until its TTL. |
| 3 sandbox | [#5989](https://github.com/NVIDIA-NeMo/Gym/pull/5989#discussion_r4534031835) | The lent sandbox can reach the judge and spend its key. |
| 3 sandbox | [#5653](https://github.com/NVIDIA-NeMo/Gym/pull/5653#discussion_r4530796449) | Model-written code runs on the host, not in a sandbox. |
| 4 pairing | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4490739972) | A harness was added to `allowed_agents` before its qualification runs. |
| 5 reward | [#5620](https://github.com/NVIDIA-NeMo/Gym/pull/5620#discussion_r4485371412) | Unresolved model text is penalized twice. |
| 5 reward | [#3348](https://github.com/NVIDIA-NeMo/Gym/pull/3348#issuecomment-5884368423) | The judge receives the instruction line instead of the source sentence. |
| 5 reward | [#3348](https://github.com/NVIDIA-NeMo/Gym/pull/3348#discussion_r4001096613) | Unparsable judge scores are skipped, which inflates the mean. |
| 5 reward | [#5031](https://github.com/NVIDIA-NeMo/Gym/pull/5031#issuecomment-6772743746) | With all-positive data, always answering yes scores 1. |
| 6 data | [#5839](https://github.com/NVIDIA-NeMo/Gym/pull/5839#discussion_r4521442457) | Data should come from a pinned public source with a checksum. |
| 6 data | [#5703](https://github.com/NVIDIA-NeMo/Gym/pull/5703#discussion_r4516919312) | Re-verify sends `agent_context` in a different shape from the original episode. |
| 7 harbor | [#5131](https://github.com/NVIDIA-NeMo/Gym/pull/5131#pullrequestreview-5676522495) | The Harbor docs contradict existing pages and give no `harbor_agent` migration path. |
| 8 codes | [#5675](https://github.com/NVIDIA-NeMo/Gym/pull/5675#discussion_r4491520874) | Any nonzero exit becomes a retryable 503. |
| 8 codes | [#5989](https://github.com/NVIDIA-NeMo/Gym/pull/5989#discussion_r4534031808) | A missing sandbox extra makes every seed return 500, which reruns forever. |
| 9 retry | [#5540](https://github.com/NVIDIA-NeMo/Gym/pull/5540#discussion_r4487543967) | A short backoff ends episodes on a brief 429 burst. |
| 10 mask | [#5485](https://github.com/NVIDIA-NeMo/Gym/pull/5485#issuecomment-6618756920) | A judge 429 or a missing key is scored as a wrong answer instead of masked. |
| 10 mask | [#5546](https://github.com/NVIDIA-NeMo/Gym/pull/5546#discussion_r4487547937) | A deadline with no model output is scored as the model's zero. |
| 10 mask | [#5488](https://github.com/NVIDIA-NeMo/Gym/pull/5488#discussion_r4462806949) | A forced re-verify masks saved rollouts. |
| 11 async | [#5780](https://github.com/NVIDIA-NeMo/Gym/pull/5780#discussion_r4519656612) | A multi-GB environment is built on the event loop. |
| 11 async | [#5686](https://github.com/NVIDIA-NeMo/Gym/pull/5686#discussion_r4487387128) | A metrics read shares the saturated global aiohttp pool. |
| 13 config | [#5397](https://github.com/NVIDIA-NeMo/Gym/pull/5397#discussion_r4483349741) | The v2 config silently drops v1's timeout default. |
| 13 config | [#5463](https://github.com/NVIDIA-NeMo/Gym/pull/5463#discussion_r4526706776) | A misspelled `required_model_servers` name silently turns enforcement off. |
| 14 evidence | [#5820](https://github.com/NVIDIA-NeMo/Gym/pull/5820#pullrequestreview-5828498518) | A scripted smoke proves routing only; attach a real policy and judge run. |
| 14 evidence | [#5768](https://github.com/NVIDIA-NeMo/Gym/pull/5768#pullrequestreview-5807945611) | The validation section is empty: no rollout evidence and no `N/A`. |
| 15 tests | [#5386](https://github.com/NVIDIA-NeMo/Gym/pull/5386#discussion_r4513247936) | The sort-order test still passes with the sort removed. |
| 15 tests | [#5780](https://github.com/NVIDIA-NeMo/Gym/pull/5780#discussion_r4519656636) | Scorer tests always skip in CI, so the real worker never runs. |
| 15 tests | [#5928](https://github.com/NVIDIA-NeMo/Gym/pull/5928#pullrequestreview-5848816809) | The regression test sits where CI does not collect it. |
| 16 claims | [#5989](https://github.com/NVIDIA-NeMo/Gym/pull/5989#discussion_r4534031870) | The README gives a false reason for single-worker mode. |
| 16 claims | [#5480](https://github.com/NVIDIA-NeMo/Gym/pull/5480#discussion_r4461098404) | The tutorial keeps state at verify; the real server releases it. |
| 16 scope | [#5480](https://github.com/NVIDIA-NeMo/Gym/pull/5480#discussion_r4477749719) | A live bug fix on main should land separately first. |
| 16 scope | [#5386](https://github.com/NVIDIA-NeMo/Gym/pull/5386#discussion_r4513247918) | An unrelated `num_repeats` change alters benchmark cost and results. |
| 16 compat | [#5149](https://github.com/NVIDIA-NeMo/Gym/pull/5149#discussion_r4409203811) | The default search provider is switched silently. |
| 16 reuse | [#4658](https://github.com/NVIDIA-NeMo/Gym/pull/4658#discussion_r4505561924) | Benchmark-specific columns are added to shared request models. |
| 16 reuse | [#5620](https://github.com/NVIDIA-NeMo/Gym/pull/5620#discussion_r4485294742) | An upstream SHA is copied instead of using the shared constant. |
| 17 public | [#5135](https://github.com/NVIDIA-NeMo/Gym/pull/5135#discussion_r4498804821) | The example rollouts contain an internal hostname. |
| 17 provenance | [#2917](https://github.com/NVIDIA-NeMo/Gym/pull/2917#issuecomment-6764747576) | Adapted evaluator files carry no source URL or license. |
| 17 provenance | [#5619](https://github.com/NVIDIA-NeMo/Gym/pull/5619#discussion_r4484519216) | A judge prompt is copied from an unlicensed repository. |

**What reviewers ask most, by PRs:** docs and PR body match the code (about 43 PRs); tests that fail without the
change (45); infrastructure failures separated from model failures (29); a real-model smoke or baseline from the
current head (29); unsupported config or input refused up front, not as a 500 mid-rollout (about 21); existing
defaults kept and new behaviour opt-in (about 21).

**Rarely raised so far:** legacy-path (13 comments), pairing (6) and Harbor (3). These items are new in this guide
because the session contract and the Harbor format are new, not because they matter less.
