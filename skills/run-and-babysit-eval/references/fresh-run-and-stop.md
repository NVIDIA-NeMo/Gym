# Fresh run, and stopping a chain

Moved from section 8 of [SKILL.md](../SKILL.md), with semicolons replaced by commas or periods and no other change.

A segment that ended before the task inventory was written has nothing to resume: recovery answers
`fresh-restart` and names the command, which restarts in the same directory with fresh budgets (saved
files kept, plan journaled):

```bash
gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover \
  --max-walltime HH:MM:SS --walltime-reason "<why the longer walltime>"
```

`--max-walltime` must be longer than the recorded walltime. To evaluate something different, submit a new
run to a new location: a fresh `gym eval run` at an existing output path replaces it.

To stop a chain: `scancel` the running job, its successor (in `.gym_followup_queued.<job id>`) and the
sentinels (`.gym_followup_sentinel.<job id>`, and the one its predecessor queued for it, which becomes
eligible when you cancel the job. `squeue --me -o '%i %j %T %E'` shows it by its dependency). A successor
left to start sees `CANCELLED`, writes the stop to `.gym_chain_incomplete` and exits (it ends `FAILED`,
exit 1, by design: that is the stop, not a fault), after it got an
allocation. Cancelling the successor and the sentinels saves that allocation and leaves no stop record
(the inspection reads the state from the scheduler either way). Then inspect (section 3) before deciding
anything else, and keep the run directory.

A segment whose driver exits 76 (nothing is eligible for another attempt) and left no failure kind the
next segment's guard handles stops the chain itself: it writes that stop to `.gym_chain_incomplete`, cancels the successor and
sentinel it queued, and journals the end in `chain-end-journal.jsonl`, so no allocation starts just to
write the stop. A typed stop (credentials rejected, resume refused, spin-up timeout, install failed) is
still written by the next segment's guard, with its own remedy.
