# Example pivot pool: 2 games, 5 pivots

A small pivot pool, committed so that `data/example.jsonl` runs with no download. It is the server's default `pool_dir`.

| Game | Pivots (request into level 1) | Oracle distance at the pivot (`po_d0`) |
|---|---|---|
| `cn04-2fe56bfb` (keyboard and click) | c4, c8 | 13, 14 |
| `ls20-9607627b` (keyboard) | c3, c6, c9 | 14, 13, 14 |

The attempts are neutral recordings: the Kaggle kit's harness (source commit `30d2cc8b`) played each game for 12
model calls against the scripted model of `responses_api_agents/arc_agi_3_agent/tests/stub_model.py` (one action per
call, alternating keys and clicks; no learning, no level cleared). No real model output is in them.

| File | What |
|---|---|
| `example.parquet` | the pool split: one row per pivot (`pivot_json`, harness revision, harness env file, `po_d0`, `restore_ok`) |
| `games.parquet` | the two games' code, with the oracle that measures the distance to completing a level |
| `env/kit-harness_env.toml` | the harness env file the attempts were recorded under: the Kaggle kit's `harness_env.toml` settings |
| `snap-raw/*.snap.jsonl` | one per attempt: every model request the harness sent, the state it was sent from and each reply's `usage` |

## Rebuild

From the Gym root, with the `arc_agi_3_pivot` server's venv (Linux; the harness and the Python 3.12 worker venv are
installed under `.runtime/` on first use). Each recording needs a fresh stub server (its replies count from 0):

```bash
PY=resources_servers/arc_agi_3_pivot/.venv/bin/python
POOL=resources_servers/arc_agi_3_pivot/data/example_pool
rm -f $POOL/example.parquet
record() {  # game port pivots
  $PY responses_api_agents/arc_agi_3_agent/tests/stub_model.py --port $2 --log /tmp/stub-$1.jsonl & STUB=$!
  sleep 2
  $PY -m resources_servers.arc_agi_3_pivot.record $POOL --game $1 --harness-env env/kit-harness_env.toml \
      --model-url http://127.0.0.1:$2/v1 --model stub --attempt stub-$1 --calls 12 --pivots $3
  kill $STUB
}
record cn04-2fe56bfb 8771 4,8
record ls20-9607627b 8772 3,6,9
$PY resources_servers/arc_agi_3_pivot/create_dataset.py $POOL/example.parquet \
    --output resources_servers/arc_agi_3_pivot/data/example.jsonl
```

`record.py` restores every pivot by replay before it writes its row: all five restore exactly in about 2 s each, and
continuing with the recording's own next replies reproduces its next requests (`tests/test_worker_live.py`).
