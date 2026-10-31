# Data-build scripts

Scripts that clean, mix and check training rows for `agentic_turn_simulation`. Rows follow
[`../docs/data.md`](../docs/data.md); the row id is `verifier_metadata.row_id`. The rules, their levels
and the full rebuild recipe are in [Data rules](../docs/data.md#data-rules) and
[Rebuilding from new raw data](../docs/data.md#rebuilding-from-new-raw-data). No script calls a model.

| script | Python | what it does |
|---|---|---|
| `clean_rows.py` | env | Applies the named data rules to the rows of one source, in order: normalise, then validity drops, then the run-level session cap. `--levels` picks the levels; `--plan` writes the ids each rule changed or dropped. |
| `mix_rows.py` | stdlib | Concatenates part files in the order given, then shuffles with `random.Random(--seed)`. Fails on a repeated row id. |
| `validate_rows.py` | env | Loads every row through the env's own parsing (run request, harness, history, tool list, workspace replay) and counts a dry obstacle draw. Writes no rows. |

"env" means the script imports this environment. Run it with a Python that has the server's requirements (for
example the server's `.venv`) and with the Gym root on `PYTHONPATH`.

- The scripts that write rows (`clean_rows.py`, `mix_rows.py`) write a new file with mode 0600 through a temporary
  file and refuse an existing output unless `--overwrite`. Side reports (`--plan`, `--report`) are overwritten.
- A line passes through byte for byte unless a normalise rule changed it.
- `clean_rows.py` and `mix_rows.py` print one JSON line of counts with the sha256 of the rows file they wrote;
  `validate_rows.py` prints one `<file> <json>` line per file.

`clean_rows.py` reads three optional data-side files. They describe one dataset, so they live with the data, not in
this repository:

| option | format | used by |
|---|---|---|
| `--reviewed` | `{"absent_input": {"<row id>": "<reason code>"}}` from a review list | `absent_input` |
| `--placeholders` | `{"tags": ["NAME", ...], "numbered_tags": ["ID", ...]}`: the redaction step's placeholders, `[TAG]`, and `[TAG_<digits>]` for a numbered tag | `last_user_placeholder` |
| `--patterns` | `{"strip": [regex, ...], "cut": [regex, ...]}`: dataset text to remove before messages are compared | the duplicate rules, the session cap's effective sessions, `absent_input`'s same-text match, `--holdout` |

```bash
S=resources_servers/agentic_turn_simulation/scripts
PY=resources_servers/agentic_turn_simulation/.venv/bin/python
export PYTHONPATH=$PWD   # the Gym root

# a source's train split: normalise and drop invalid rows; report validation rows that leak into it
$PY $S/clean_rows.py train.jsonl train.clean.jsonl --levels normalise,validity --placeholders tags.json \
    --plan train.plan.json --holdout val.jsonl
# a run, per source: every level in one call (the cap needs the copies the duplicate rules drop); then pick the
# run's rows into a.picked.jsonl (../docs/data.md#picking-rl-rows), mix the sources and check
$PY $S/clean_rows.py train.jsonl a.capped.jsonl --placeholders tags.json --cap 5 --plan a.plan.json
python3 $S/mix_rows.py --seed 0 train_mix.jsonl a.picked.jsonl b.picked.jsonl
$PY $S/validate_rows.py train_mix.jsonl
```
