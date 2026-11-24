# BEAVER SQL Resource Server

Execution-based text-to-SQL verification on [BEAVER](https://beaverbench.github.io/)'s enterprise data warehouses,
scored with BEAVER's official execution accuracy (`ex_acc`). The benchmark that uses it is
[`benchmarks/beaver`](../../benchmarks/beaver/README.md).

Binary reward, no LLM judge. The grader runs the gold query and the model's query on MySQL and scores them as BEAVER's
official evaluator ([`eval/utils/ex_acc.py`](https://github.com/beaverbench/beaver/blob/81a2a88a1a2e236281c5d09b3171edd7580f8ec9/eval/utils/ex_acc.py))
does (`ex_acc.py` here):

- Each query gets its own connection, with no session settings, and 10 seconds, from connecting to the last row read.
  A slower query is a timeout; it is then stopped on the server with `KILL QUERY`, its rows stop being read, and its
  connection is killed rather than read to the end.
- A query that fails or times out is an **empty result**, as is a missing prediction (no ```` ```sql ```` block in the
  answer). A gold query that times out is an empty result too, so its task stays in the score. A gold query that
  *errors* within the 10 seconds masks the row instead: every gold query runs on a correctly set-up server (see the
  benchmark's gold check), so that means the setup is wrong. (The official evaluator would score it as an empty result.)
- A result over 5,000,000 rows or about 1 GiB is not compared and **never matches**, not even an empty gold result
  (officially it would be read whole and compared). Reading stops at the limit and the query's connection is killed.
  The largest BEAVER gold result has 622,224 rows.
- Two results match when both are empty, or when their rows, built into a `pandas.DataFrame` and converted with
  `astype(str)` and stripped, are equal sets with the same number of columns. Row order, duplicate rows and column
  names don't matter. `150.250` and `150.25` differ; `1` and `'1'` don't.

The `DataFrame` step is deliberate: pandas types a column from all of its values, so an integer column with a NULL
prints `5.0`, and a DATETIME column of midnights prints as dates. pandas is pinned below 3 because pandas 3 keeps NULL
as missing where pandas 2 prints `None`; the official evaluator is set up for Python 3.10, which pandas 3 doesn't
support. `benchmarks/beaver/official_check.py` checks the grader against the official file on the same predictions.

Each row records `gold_result` (`empty_gold` or `nonempty_gold`). The metrics report accuracy over all tasks
(`pass@1[avg-of-k]/accuracy`) and over tasks whose gold returns rows (`nonempty_gold/pass@1[avg-of-k]/accuracy`): an
empty gold result is matched by any query that returns nothing, so BEAVER publishes both.

**Grading path:** the resources server grades (path A, `observe_programmatic_grading`). **Failures:** when the grader
can't reach the data (it can't connect within the 10 seconds, the connection is lost during a query, or the password is
wrong), when the grader runs out of memory, or when a gold query errors within the 10 seconds, the row is masked with
`failure_kind` `verifier_error` and `failure_reason` `database_unavailable`; a crash in the grader is masked as
`unknown_error`. Any other error in the model's query is the query's and scores as an empty result. That includes
access denied, and the server running short of memory, threads, disk or temporary space (MySQL errors 1021, 1030,
1037, 1038, 1041, 1114, 1135), which a runaway query causes itself. A request cancelled while its query runs stops the
query first. A stateless reverify grades the saved answer once the cause is fixed. **Time
contract:** 10 seconds per query (`query_timeout_s`), connecting included, timed on the query's own thread
(`MySqlRunner.run`). A query that ends after the limit is a timeout, an empty result, whether it returned rows, failed
or lost its connection; one that never connected masks the row. There is no episode deadline in the server; the
agent's token cap applies.

## MySQL server

BEAVER needs MySQL 8.0 with **case-insensitive table names**:

```bash
docker run -d --name beaver-mysql -p 127.0.0.1:3306:3306 \
    -e MYSQL_ROOT_PASSWORD=<admin password> mysql:8.0 --lower-case-table-names=1
```

`--lower-case-table-names=1` can only be set when the data directory is created. BEAVER's dumps were made by MySQL
9.3 on macOS, which compares table names case-insensitively, and many gold queries spell a table or an alias in
another case. With Linux's default (case-sensitive), 950 of neutron's 1,017 gold queries and 15 of dw_real's 121 fail;
with it, none does. The loader and the startup check refuse a case-sensitive server.

## Data

The databases are BEAVER's anonymized MySQL dumps, `beaver_db.zip` on the gated Hugging Face dataset
[`beaverbench/beaver-table`](https://huggingface.co/datasets/beaverbench/beaver-table) (MIT). Accept the terms on the
dataset page, then `hf auth login`. The zip is pinned to revision `86f2c4ec` and each dump to its SHA-256
(`setup_beaver_mysql.DUMPS`); nothing is committed here.

| Database | Dump | Tables | Questions (`beaver-query`) |
|---|---|---|---|
| `dw` | 93 MB | 97 | `dw` 5,787 and `dw_real` 121 |
| `neutron` | 6 MB | 175 | 1,017 |
| `nova` | 2.1 GB | 109 | 1,053 |

With `mysql_admin_dsn` set, startup downloads the zip into `cache_dir`, loads each database in `databases` that isn't
loaded yet (about 15 s for dw), records it with its dump hash in `_beaver_meta.loaded`, and creates the grader's user
from `mysql_dsn` with `SELECT` on those databases only. The loader never drops or replaces a database it didn't record
itself. Without `mysql_admin_dsn`, startup only checks that every database is loaded from the pinned dump and that the
grader's user sees all its tables.

The grader's user can only read: a query that writes or creates a temporary table fails, and is scored as an empty
result; `LOAD_FILE()` returns NULL. (The official evaluator connects as `MYSQL_USER`, which BEAVER's README example sets
to root.) Startup reads every table as the grader, and
refuses a grader user that holds anything beyond `SELECT` on the BEAVER databases and `_beaver_meta`: another
privilege, a role, or `SELECT` elsewhere. The loader creates that user if it doesn't exist and never changes an
existing user's password; it refuses a grader DSN that names its own (admin) user, runs one loader at a time per server
(`GET_LOCK`), and creates a database only if no one else has, even between its check and its load.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `mysql_dsn` | `mysql://beaver_reader@127.0.0.1:3306` | The grader's read-only user, without a password |
| `mysql_password` | `${oc.env:BEAVER_MYSQL_PASSWORD,null}` | The grader's password |
| `mysql_admin_dsn` | `null` | Set to load data and create the grader's user at startup |
| `mysql_admin_password` | `${oc.env:BEAVER_MYSQL_ADMIN_PASSWORD,null}` | The admin user's password |
| `databases` | `[dw]` | Databases checked (and loaded) at startup; add `neutron` or `nova` for those questions |
| `cache_dir` | `resources_servers/beaver_sql/.beaver` | Download cache for `beaver_db.zip` (git-ignored) |
| `max_concurrency` | `8` | Queries at a time |
| `query_timeout_s` | `10.0` | BEAVER's limit; change it only on purpose |

**Passwords never go in a DSN.** A run records its configuration, DSNs included, beside its rollouts (the identity file
next to `rollouts_manifest.json`), so a DSN that holds a password, or any query or fragment, is refused at startup,
with an error that doesn't quote it. The passwords come from the environment, and Gym redacts credential-named
settings such as `mysql_password` in everything a run saves:

```bash
export BEAVER_MYSQL_PASSWORD='<grader password>'
export BEAVER_MYSQL_ADMIN_PASSWORD='<admin password>'
export BEAVER_MYSQL_ADMIN_DSN=mysql://root@127.0.0.1:3306
gym env start --model-type vllm_model --resources-server beaver_sql \
    '+beaver_sql_resources_server.resources_servers.beaver_sql.mysql_admin_dsn=${oc.env:BEAVER_MYSQL_ADMIN_DSN}'
```

**Concurrency changes outcomes near the limit.** The gold queries take 0.02 s at the median. Run one at a time on a
laptop (MySQL 8.0.46 in Docker), none of the 6,925 `dw`, `dw_real` and `neutron` golds times out, but five `dw` golds
take 8.9 to 9.5 s (`dw_4036`, `dw_4213`, `dw_4242`, `dw_4480`, `dw_4826`). With 8 at a time, those five and `dw_3814`
time out, and a gold query that times out is an empty result, so those tasks then score any failing answer as correct.
None of them is in the default 300-question sample, whose slowest gold takes 3.2 s alone. The official evaluator runs
one query at a time; lower `max_concurrency` on a slower server, or when you score all of `dw`.

**Driver.** `mysql-connector-python` 26.10.0 with its C extension, as the official evaluator's default. It runs a
string with several statements and returns the first statement's result (`SELECT 1; SELECT 2` gives `1`), so an
answer whose first statement is a `SET` returns no rows. Statements after the first would run on after the result is
read, so when the driver reports more results pending, the grader kills that connection (`KILL CONNECTION`). On 12
such probes (`SELECT 1; SELECT 2`, a trailing comment, `SET @x = 3; SELECT @x`, ...) the official `ex_acc.py` and this
grader returned the same result.

**Known limits.** All grader connections share one MySQL user, so a model answer can `KILL QUERY` another rollout's
query, which then scores as a timeout; nothing stops that short of a user per connection. A gold query runs again for
every rollout, so with `--num-repeats` above 1 a gold near the time limit can time out on some rollouts and not others
(and its task's `gold_result` subset follows the first rollout); report `gold-sql-timeout_count` with the headline.

**Where this departs from the official evaluator,** on purpose: a gold query that *errors* within the time limit masks
the row instead of counting as an empty result (none of the 6,925 `dw`, `dw_real` and `neutron` golds errors on a correctly set-up
server; `nova` isn't checked); a failure to connect or a lost connection masks the row; a result over the size limit
never matches; and statements after the first are stopped rather than left running.

## Tests

`gym env test --resources-server beaver_sql` runs the unit tests and the verifier fixture with a fake driver. With
`BEAVER_MYSQL_TEST_ADMIN_DSN` and `BEAVER_MYSQL_TEST_DSN` set (DSNs without passwords, on a MySQL 8 server with
`--lower-case-table-names=1`) and their passwords in `BEAVER_MYSQL_TEST_ADMIN_PASSWORD` and
`BEAVER_MYSQL_TEST_PASSWORD`, a live test also loads a one-table dump and checks that the grader's user can't write. With `BEAVER_OFFICIAL_EX_ACC`
pointing at a copy of the official `ex_acc.py`, the comparison cases also run against it.

## Licensing information

- Code: Apache 2.0. The grading rule follows BEAVER's official evaluator; that repository has no license file, so no
  code is copied from it. Loader and prompt adapted from
  [auto-ontology-eval](https://github.com/NVIDIA/auto-ontology-eval) (Apache 2.0).
- Data: [BEAVER](https://huggingface.co/collections/beaverbench/beaver-dataset), MIT, gated on Hugging Face; downloaded
  at run time, never committed. `data/example.jsonl` holds our own literal-answer questions, not BEAVER's.

Dependencies
- nemo_gym: Apache 2.0
- mysql-connector-python: GPL-2.0 with the Universal FOSS Exception
- pandas: BSD-3-Clause
