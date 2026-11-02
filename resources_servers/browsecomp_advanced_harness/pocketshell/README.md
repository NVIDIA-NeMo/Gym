# pocketshell

An in-process, workspace-confined, read-only shell for LLM agent harnesses.

Drop-in replacement for `bash -c` where the only thing a model needs to do is
inspect files it has already retrieved. Nothing is spawned, so **there is no
sandbox to build or maintain**, and because every path argument is resolved
against a workspace root, filesystem escape is structurally impossible rather
than blocked by an enumerated deny-list.

```python
from pocketshell import run

r = run('grep -i "pemantle" pages/*.txt | head -5', workspace="/path/to/sample")
r.stdout, r.stderr, r.exit_code
```

## Why this exists

The BrowseComp harness ran model-authored shell commands as a real subprocess,
protected by a regex deny/allow list and `ulimit -f 0`. That guard filters
command **names** and never inspects path **arguments**, so all of these passed:

```
ALLOWED  cat /home/<user>/.bashrc      ALLOWED  cat /proc/self/environ
ALLOWED  cat /etc/passwd               ALLOWED  find / -name '*.key'
ALLOWED  grep -r API_KEY /home/<user>
```

Its own source comment called it "NOT a security boundary". Running that on a
shared cluster means either investing in a real sandbox, or not spawning
processes at all. This is the second option.

## Scope is measurement-driven, not aspirational

Sized against **1.65M real agent-written `bash_command` calls** sampled from
BrowseComp evaluation and synthetic-data rollouts.

**Grammar** — cumulative coverage of real traffic:

| Tier | Coverage |
|---|---|
| single command | 39.9 – 49.5 % |
| + linear pipeline | 86.9 – 91.0 % |
| + `;` `&&` `\|\|` | 97.5 – 98.6 % |
| + variables / `$(( ))` | 98.6 – 98.7 % |
| + `for` / `while` | **100.00 %** |
| + command substitution, heredocs, backgrounding | 0 calls / 22 of 1.55M |

The last row is the important one: the upstream guard already rejected the hard
parts of bash, so they never appear in executed traffic. Those constructs raise
`ParseError` and fail **closed**.

**Commands** — 12 programs cover 98.7–99.1 % of all command-position
occurrences (`grep head cat sed echo ls wc tail tr printf sort cut`); 7 more
(`nl uniq strings cd find file diff`) pass 99.9 %. grep needs 12 flags for 99 %
coverage, sed needs 2.

## The subtle part: BRE

GNU grep's default dialect is POSIX BRE, where the escaping of `| ( ) { } + ?`
is **inverted** relative to Python:

```
BRE:     \|  = alternation        |  = literal pipe
Python:   |  = alternation       \| = literal pipe
```

So `re.compile(pattern)` does not fail loudly — it silently matches different
text. Measured against real GNU grep on 2,993 real patterns:

| Strategy | Distinct-pattern agreement | Usage-weighted |
|---|---|---|
| no translation | 71.87 % | 96.91 % |
| escape swap only (~15 lines) | 99.90 % | 99.84 % |
| `regex_xlate.bre_to_python` (context-aware) | **99.97 %** | **99.96 %** |

Note the trap in row 1: 96.9 % usage-weighted looks passable because the
heavily-reused patterns are plain literals, but 28 % of *distinct* patterns
break — precisely the multi-term `grep "david\|phd\|doctoral"` lookups that hard
questions turn on.

Malformed patterns (which GNU tolerates and Python rejects — unbalanced `\)`,
`13,**`, `[2013-2014]`) degrade to a literal substring search rather than raising.

## Correctness gate

Unit tests cover semantics we thought of. The real gate is
`tools/difftest.py`, which replays **real agent commands** through both
pocketshell and real GNU bash against an identical workspace and diffs stdout:

```bash
.venv/bin/python tools/difftest.py --calls-glob '/path/to/*.calls.txt' --n 12000
```

It filters the corpus through the *old* harness guard first, so only commands
that actually executed in production are replayed. Results:
`DIFFTEST_RESULTS.txt`.

Known, accepted divergences:
- **`grep -r` file order** — pocketshell sorts; GNU walks readdir order.
  Deterministic is preferable, and in a real workspace (`0001_`, `0002_` …)
  sorted order matches creation order anyway.
- **Workspace confinement** — bash can read `/etc`; pocketshell cannot. By design.
- **`ls -l` size-column padding** in mixed listings.

## Development

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

## Layout

| File | Role |
|---|---|
| `shell.py` | executor: expansion, globbing, brace expansion, pipelines, `run()` |
| `syntax.py` | lexer + parser for the pocket-shell grammar |
| `commands.py` | the read-only command implementations |
| `regex_xlate.py` | BRE / ERE → Python `re` translation |
| `fsview.py` | workspace confinement (replaces the sandbox) |
