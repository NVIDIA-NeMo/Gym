# Evaluation Notes

`evals.json` has 13 cases: 9 that need this skill and 4 near misses that belong to other skills. Every case is a full agent session in both arms (with and without the skill), so the set stays close to the size of other catalog skills to fit the CI time limit. Expected outputs and assertions describe results that an agent without the skill could also reach.

## Questions

- Choosing RL training tasks from a partial run, and checking whether its headline pass rate can be quoted.
- What `missing_count` in the per-repeat metrics does and does not count.
- Two tasks with the same mean reward but different spread, and a task with a single saved rollout.
- The `gym eval profile` command, its output files, and its error for missing rollouts.
- Checking a new environment before RL from a pip install, and how many repeats hard tasks need.

## Negative Cases

Near misses that mention NeMo Gym but belong elsewhere: explaining a score gap against a model card, a crashed run, planning repeats to compare checkpoints, and latency profiling of a server.

## Fixtures

`files/` holds two small synthetic runs in the format that `gym eval run` writes. Both were profiled with `gym eval profile` from nemo-gym 0.7.0, which leaves tasks with no saved rollout out of the profile file. Cases stage them into `/workspace/input/`.

- `math_run/`: 16 math tasks with 8 planned repeats, scored 0 or 1 by an exact string match of the final answer; the job stopped at 98 of 128 rollouts and was profiled with `++allow_partial_rollouts=True`. Tasks 0 to 3 always pass. Task 4 never passes, with different wrong answers. Task 5 never passes because the exact match rejects `\frac{1}{6}` for the label `1/6`. Tasks 6 to 10 vary. Task 11 varies with 3 of 8 repeats saved. Task 13 has one saved rollout. Task 15 never passes with 6 of 8 saved. Tasks 12 and 14 have no saved rollout.
- `if_run/`: 8 instruction-following tasks with 4 repeats, rewarded by the fraction of 4 instructions followed; complete. Task 1 scores 0.5 every time and task 2 alternates between 1 and 0.

The example in `SKILL.md` uses a different run, so the skill text does not contain these answers.
