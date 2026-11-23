# Example review comments

These come from human review comments on agent-harness PRs in NVIDIA-NeMo/Gym, opened from 2026-10-01 onward:

- 87 PRs scanned;
- 185 comments kept;
- 11 reviewers.

Each line paraphrases one comment. They are examples of the bar, not rules.

| Item | PR comment | What it caught |
|---|---|---|
| 12 claims | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313638) | The README said foreground tools are stopped; they actually escape. |
| 12 claims | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804380) | The guide still offers the install paths being removed. |
| 12 claims | [#5540](https://github.com/NVIDIA-NeMo/Gym/pull/5540#pullrequestreview-5793906857) | The description says the Agent Server; the code uses the sandbox. |
| 11 callers | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4261625246) | The supervisor change reaches SWE-bench Pro and Hermes; add owners or split. |
| 11 callers | [#5455](https://github.com/NVIDIA-NeMo/Gym/pull/5455#discussion_r4516578528) | Published aggregates change for other benchmarks; add a compatibility note. |
| 13 evidence | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4490739972) | Hold `allowed_agents` until full qualification runs exist. |
| 13 evidence | [#5820](https://github.com/NVIDIA-NeMo/Gym/pull/5820#pullrequestreview-5828498518) | A scripted smoke is not enough; attach a real policy and judge run. |
| 2 typed endings | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4487544201) | The classifier parses error-text substrings; there is no context-limit kind. |
| 2 typed endings | [#5543](https://github.com/NVIDIA-NeMo/Gym/pull/5543#discussion_r4491842869) | A fallback invents HTTP 400 from the SDK error type. |
| 2 refusals | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4487544214) | A registry 404 becomes a retryable 500; it should be a seed 422. |
| 2 refusals | [#5152](https://github.com/NVIDIA-NeMo/Gym/pull/5152#pullrequestreview-5757735361) | Unsupported platforms answer 500, and the build reruns every seed. |
| 3 retry | [#5543](https://github.com/NVIDIA-NeMo/Gym/pull/5543#discussion_r4487547536) | A deadline with no events gives an untyped 500 that is retried forever. |
| 3 retry | [#5540](https://github.com/NVIDIA-NeMo/Gym/pull/5540#discussion_r4487543967) | Two retries with sub-second backoff end the episode on a 429 burst. |
| 3 retry | [#3961](https://github.com/NVIDIA-NeMo/Gym/pull/3961#discussion_r4188733943) | A diagnostic validation failure gives a 500 and reruns the episode. |
| 4 mask | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4487544210) | Exit 0 without a completion event is scored as the model's zero. |
| 4 mask | [#5546](https://github.com/NVIDIA-NeMo/Gym/pull/5546#discussion_r4487547937) | A hung model server at the deadline is scored as incomplete. |
| 4 mask | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#pullrequestreview-5507166508) | Verify infrastructure failures return 0 without `mask_sample`. |
| 5 conformance | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4490739978) | The checker rejects `incomplete` tool calls; extend the checker. |
| 7 evidence | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4491841558) | The failed response loses the native reason; keep it, bounded and redacted. |
| 7 evidence | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4491844098) | A cleanup exception masks the original connect failure. |
| 8 cleanup | [#3961](https://github.com/NVIDIA-NeMo/Gym/pull/3961#discussion_r4199066746) | An owned sandbox has no `ttl_s`, so it leaks if the server crashes. |
| 8 cleanup | [#5152](https://github.com/NVIDIA-NeMo/Gym/pull/5152#discussion_r4459029872) | The setup directory and its archives leak when the installer exits early. |
| 8 process tree | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313614) | `setsid` tool commands escape the process-group kill. |
| 8 process tree | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804214) | The receipt says confirmed while descendants still run. |
| 9 task image | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804354) | Hermes' uv install reads the task's `uv.toml` and writes into its home. |
| 9 task image | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4261625265) | The private `HOME` leaks into tool shells and could lower scores. |
| 9 probing | [#5152](https://github.com/NVIDIA-NeMo/Gym/pull/5152#discussion_r4418367281) | Some glibc images have a musl loader; probe glibc first. |
| 9 network | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#issuecomment-6703159153) | There is no `network_access`, the download needs egress, and web fetch is on. |
| 14 tests | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4261625274) | The test still passes with the timeout removed. |
| 14 tests | [#5153](https://github.com/NVIDIA-NeMo/Gym/pull/5153#discussion_r4496923410) | `conftest` only warns, so CI silently skips the tests. |
| 14 waits | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313675) | An unbounded await makes the test hang instead of fail. |
| 15 fail closed | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313646) | A plugin load failure is ignored; add a sentinel that the runner checks. |
| 15 sizes | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804240) | A quadratic CR strip takes seconds per 1 MB line. |
| 15 paths | [#5703](https://github.com/NVIDIA-NeMo/Gym/pull/5703#discussion_r4516919326) | `realpath(strict=False)` lets a symlink escape the root. |
| 16 naming | [#3961](https://github.com/NVIDIA-NeMo/Gym/pull/3961#discussion_r4198769598) | Four names for one concept; settle on one each. |
| 1 scope | [#5703](https://github.com/NVIDIA-NeMo/Gym/pull/5703#discussion_r4516919353) | On legacy `/run`, Simple Agent forwards `agent_context` to `/verify`. |
