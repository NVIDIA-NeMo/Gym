# VS-Bench integration (experimental)

This adapter runs the VS-Bench authors' reasoning, perception, and gameplay
interfaces through NeMo Gym. The benchmark source and its Python environment
are separate runtime dependencies; the evaluation factory contains only
configuration. The upstream source is MIT-licensed and is pinned to
`2c3531cdf91d1b620b4a9015126a242eb305a4d0`. The released dataset is pinned
to `30ef045024e1f4a778166f274767f6c8d354c676`.

## Source and scoring

`reference_source_manifest.json` identifies each expected upstream file.
`reference_identity.py` accepts only that source or the exact KAZ patch whose
hash is recorded there. Both must be labeled correctly in evaluation results.
The patch changes KAZ perception scores as well as fixing a crash: nested
NumPy labels become Python lists before the authors' matching logic runs.
Other game scoring logic is unchanged. New offline rollout records carry
`author_source_variant` (`upstream` or `kaz_numpy_labels_v1`), and aggregation
rejects missing or mixed variants. Agent and perception scorer variants must
also agree. Earlier prototype rollout records do not contain this field and
must not be silently mixed with records from this revision.

To produce the patched source, first obtain a clean export of the pinned
upstream revision. From the Gym checkout, run:

```bash
python -m benchmarks.vs_bench.prepare_patched_source CLEAN_EXPORT NEW_EXPORT
```

The command checks the entire pinned manifest, refuses an existing destination,
copies the files to a separate staging directory, applies the checked-in patch,
verifies all resulting file hashes, and writes a provenance record. It does
not fetch source or modify the input. The resulting export needs an explicitly
provisioned Python 3.10 environment with the benchmark dependencies.
Dependency building and ROM installation are supplied by the
[companion provisioning change](https://github.com/aroshanghias-nvd/Gym/tree/vs-bench-runtime-provisioning-20261110/benchmarks/vs_bench/runtime).
After applying both changes to the same checkout, follow the
[runtime instructions](runtime/README.md). Provision that runtime before
starting Gym; the adapter does not install packages or download game assets.

## Isolation and configuration

Some upstream perception scorers evaluate model output as Python. They must
run inside the provided Bubblewrap isolation boundary, with a clean environment,
no network, no credentials, read-only source/runtime, and a single read-only
dataset file. `score_in_bwrap.sh` and the resource server require explicit
paths for the executable, source, dataset, virtual environment, and Python
installation. The serving model and credential-bearing driver stay outside
the scorer sandbox. A supported, executable Bubblewrap installation and user
namespace support are deployment prerequisites; this integration does not
install or automatically qualify them.

## Coverage and metrics

The released reasoning track contains 4,000 examples (400 per game); its
headline is equal-game mean accuracy and is emitted only at full coverage.
Perception has 3,900 examples (300 KAZ, 400 for each other game), with an
equal-game mean score at full coverage. Gameplay runs separate paper-scheduled episodes for Poker (1,200 hands,
ten balanced blocks) and Breakthrough (20 balanced games), and upstream suites
for the other games. It reports raw per-game returns and approximate per-game normalized
scores using rounded published anchors. Pong uses the model seat points in the
terminal returns and the episode step count, weighted 90% and 10% respectively.
The approximate equal-game mean is emitted only after all ten paper-scheduled
games are complete. For mixed-motive games, the adapter reports both model-seat
means and their mean; this is an explicit reporting convention, since the
authors' aggregation of those seats has not been established. Earlier
source-default runs used ten Poker hands and single-seat competitive games;
do not mix their records with the paper schedule. Rounded anchors are especially
sensitive for Poker and cannot reproduce an exact leaderboard. Normalized scores
can exceed 100 or be negative. Report hand count and individual game scores.

## Qualification and limitations

The adapters have been exercised through real Gym and factory model calls.
The full reasoning and patched perception tracks have completed for Super
models. Paper-scheduled gameplay has completed for a baseline and a trained
checkpoint using disjoint shards; source, checkpoint, coverage, and shard
identities were checked before combining compact metrics. These runs qualify
the tested snapshots and runtime, not every deployment or this checkout's
latest changes. A separate full-ledger reconciliation also completed for
those historical runs; it does not qualify subsequent adapter changes.
The original KAZ scorer is known to
raise on released labels; failures remain failures and must not be silently
converted to zero. A timed-out gameplay suite has to be replayed as a suite;
completed suite records can be reused by Gym's verified resume path. The exact
paper leaderboard normalization anchors have not been recovered.

Gameplay's `response` is an explicit suite summary, with usage summed over
completed policy calls. If any completed call omits usage, aggregate usage is
unknown (`null`), rather than a misleading partial total.
Use Gym's model-call capture for individual prompts and completions. Transport
errors recovered by the author's retry loop do not invalidate a completed
suite; a transport error that contributes to an exhausted-retry fallback does.

The integration remains experimental pending review of dependency provisioning,
isolation, source distribution, and the factory configuration.
