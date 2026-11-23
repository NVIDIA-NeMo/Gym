# Example review comments

These published reviews illustrate how to tie a finding to a concrete failure and its affected caller.

| Topic | PR comment | What it caught |
|---|---|---|
| claims | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313638) | The README said foreground tools are stopped; they actually escape. |
| claims | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804380) | The guide still offers the install paths being removed. |
| claims | [#5540](https://github.com/NVIDIA-NeMo/Gym/pull/5540#pullrequestreview-5793906857) | The description says the agent server; the code uses the sandbox. |
| callers | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4261625246) | The supervisor change reaches SWE-bench Pro and Hermes; add owners or split. |
| callers | [#5455](https://github.com/NVIDIA-NeMo/Gym/pull/5455#discussion_r4516578528) | Published aggregates change for other benchmarks; add a compatibility note. |
| evidence | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4490739972) | Hold `allowed_agents` until full qualification runs exist. |
| evidence | [#5820](https://github.com/NVIDIA-NeMo/Gym/pull/5820#pullrequestreview-5828498518) | A scripted smoke is not enough; attach a real policy and judge run. |
| failure reporting | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4487544201) | The classifier parses error-text substrings; there is no context-limit kind. |
| failure reporting | [#5543](https://github.com/NVIDIA-NeMo/Gym/pull/5543#discussion_r4491842869) | A fallback invents HTTP 400 from the SDK error type. |
| refusals | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4487544214) | A registry 404 becomes a retryable 500; it should be a seed 422. |
| refusals | [#5152](https://github.com/NVIDIA-NeMo/Gym/pull/5152#pullrequestreview-5757735361) | Unsupported platforms answer 500, and the build reruns every seed. |
| retry | [#5543](https://github.com/NVIDIA-NeMo/Gym/pull/5543#discussion_r4487547536) | A deadline with no events gives an untyped 500 that is retried forever. |
| retry | [#5540](https://github.com/NVIDIA-NeMo/Gym/pull/5540#discussion_r4487543967) | Two retries with sub-second backoff end the episode on a 429 burst. |
| retry | [#3961](https://github.com/NVIDIA-NeMo/Gym/pull/3961#discussion_r4188733943) | A diagnostic validation failure gives a 500 and reruns the episode. |
| mask | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4487544210) | Exit 0 without a completion event is scored as the model's zero. |
| mask | [#5546](https://github.com/NVIDIA-NeMo/Gym/pull/5546#discussion_r4487547937) | A hung model server at the deadline is scored as incomplete. |
| mask | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#pullrequestreview-5507166508) | Verify infrastructure failures return 0 without `mask_sample`. |
| conformance | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4490739978) | The checker rejects `incomplete` tool calls; extend the checker. |
| evidence | [#5537](https://github.com/NVIDIA-NeMo/Gym/pull/5537#discussion_r4491841558) | The failed response loses the harness error; keep it, bounded and redacted. |
| evidence | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#discussion_r4491844098) | A cleanup exception masks the original connect failure. |
| cleanup | [#3961](https://github.com/NVIDIA-NeMo/Gym/pull/3961#discussion_r4199066746) | An owned sandbox has no `ttl_s`, so it leaks if the server crashes. |
| cleanup | [#5152](https://github.com/NVIDIA-NeMo/Gym/pull/5152#discussion_r4459029872) | The setup directory and its archives leak when the installer exits early. |
| process tree | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313614) | `setsid` tool commands escape the process-group kill. |
| process tree | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804214) | The receipt says confirmed while descendants still run. |
| task image | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804354) | Hermes' uv install reads the task's `uv.toml` and writes into its home. |
| task image | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4261625265) | The private `HOME` leaks into tool shells and could lower scores. |
| probing | [#5152](https://github.com/NVIDIA-NeMo/Gym/pull/5152#discussion_r4418367281) | Some glibc images have a musl loader; probe glibc first. |
| network | [#5082](https://github.com/NVIDIA-NeMo/Gym/pull/5082#issuecomment-6703159153) | There is no `network_access`, the download needs egress, and web fetch is on. |
| tests | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4261625274) | The test still passes with the timeout removed. |
| tests | [#5153](https://github.com/NVIDIA-NeMo/Gym/pull/5153#discussion_r4496923410) | `conftest` only warns, so CI silently skips the tests. |
| waits | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313675) | An unbounded await makes the test hang instead of fail. |
| fail closed | [#3902](https://github.com/NVIDIA-NeMo/Gym/pull/3902#discussion_r4265313646) | A plugin load failure is ignored; add a sentinel that the runner checks. |
| sizes | [#5094](https://github.com/NVIDIA-NeMo/Gym/pull/5094#discussion_r4389804240) | A quadratic CR strip takes seconds per 1 MB line. |
| paths | [#5703](https://github.com/NVIDIA-NeMo/Gym/pull/5703#discussion_r4516919326) | `realpath(strict=False)` lets a symlink escape the root. |
| naming | [#3961](https://github.com/NVIDIA-NeMo/Gym/pull/3961#discussion_r4198769598) | Four names for one concept; settle on one each. |
| scope | [#5703](https://github.com/NVIDIA-NeMo/Gym/pull/5703#discussion_r4516919353) | On legacy `/run`, Simple Agent forwards `agent_context` to `/verify`. |
