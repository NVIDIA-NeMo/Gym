# 52 loghub: real-log incident triage over LogHub BGL (family F)

## Task
The world is `n` windows of 50 consecutive lines of the real Blue Gene/L RAS log. For each window the agent lists
the alert categories that occur in it (most windows have none). Unit = one window. Answer: one JSON object
`{window_id: [categories]}` for the windows with alerts. Score: F1 over (window, category) pairs; also reports
window-level detection F1.

## Source and license
LogHub BGL, Zenodo record 10.5281/zenodo.8196385 (`BGL.zip`, md5 4452953c470f2d95fcb32d5f6e733f7a, sha256
d67fd82a711aea0157a9b83175892c6ee60e384a2ddf5bc51f39118453816da8), license CC-BY-4.0. Downloaded on the build
host only. 4,747,963 lines with per-line alert labels (Oliner and Stearley 2007).

## Gold is the dataset's own labels
`bgl_rules.py` gives 40 categories as (component, level, message regex). `../_build/bgl_verify.py` checks them
against every labeled line of BGL: 2 lines disagree (one `MONILL` line whose text also appears 73 times labeled
normal, so that category was dropped, and one `ciodb exited normally` line labeled `MASNORM`). Blocks containing a
disagreeing line are excluded from the pool, so on every shipped window the rulebook reproduces the dataset label
exactly; `generate()` asserts it.

## Pool and sampling
`../_build/bgl_pool.py` cuts BGL into non-overlapping 50-line blocks (94,957 blocks; 8,794 contain alerts) and
keeps 7,001: 2,501 alert blocks (1,500 with at most 25 alert lines, up to 60 per category so rare categories are
present, 300 alert-dense) and 4,500 normal blocks (3,500 with at least 5 non-INFO lines, the decoys: register
dumps, failed program loads, `LOGIN chdir ... No such file or directory`, node-card discovery errors, 1,000 plain).
`data/bgl_pool.jsonl.gz` sha256 3753c673382c3f9f3849f31d6024963448317466b38ef93af771609b1bf47ee5. An instance has
25% alert windows drawn category-first (uniform over categories with at least 8 pool windows, so the common ones
do not dominate) and 75% normal windows (80% decoy-rich). The label column is never shown.

Traps that need careful reading: FATAL is the level of 506k normal lines; `data storage interrupt` (alert) versus
`data store interrupt caused by dcbf` (normal); `program interrupt` exactly (alert) versus `program interrupt:
illegal instruction` (normal); the CioStream errors split three ways by their tail (severed / reset / timed out);
`LOGIN chdir` is an alert only with `Input/output error`; node-map errors split four ways by errno text.

## Tools
`list_windows(offset, limit<=200)`, `read_window(window_id, start_line, max_lines)` (paged under 8k chars),
`get_rulebook()`. No search across windows, so there is no grep shortcut.

## Tiers (n windows / budget s)
easy 64 / 1800, medium 256 / 1200, hard 2048 / 600. About 7.5k chars per window: 0.5M, 2M and 15M chars to read. The first hard setting, 1024 / 1200, gave Sol solo 0.69-0.78 and was replaced.

## Files
`type.py`, `bgl_rules.py`, `solutions.py` (oracle; FATAL-any, FATAL-by-component, keyword and all-windows
baselines; units), `data/bgl_pool.jsonl.gz`, `tests/test_type.py`, `checks.json`.

## Provenance
All task-facing prose (prompt, rules, tool descriptions, phrase banks, unit prompts) was written by Kimi K3 (`nvidia/moonshotai/kimi-k3` on the NVIDIA inference API, served as `kimi-k3-high`) through `../../k3gen.py`, with the system prompt below, and frozen in `text/`. Claude wrote only code, schemas, verifiers and tests. Code fills ids and numbers into K3's `{placeholders}`.

### `text/k3.json` (sha256 bd7e56da98377753..., created 2026-10-14 06:38:31 UTC, served model kimi-k3-high, 1 attempt(s))

System prompt:

```
You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and nothing else.
```

User prompt:

```
Write the task-facing text for a log-triage task over REAL logs from the Blue Gene/L supercomputer (the public LogHub BGL dataset). The logs are cut into windows of 50 consecutive lines. The agent must find every window that contains at least one alert line and name the alert categories present in it. A line is an alert of category X exactly when its component and level match and its message matches the category's signature; every other line is normal, including the many normal FATAL, FAILURE, ERROR and SEVERE lines (register dumps, failed program loads, missing directories, node card discovery messages). Code shows the exact signature for each category next to your description; your description only helps the reader. Return one JSON object with these keys:

"intro": the opening paragraph. Placeholders {n} (number of windows) and {lines} (lines per window). Explain the setting (real system logs from a large supercomputer, cut into {n} windows of {lines} consecutive lines each, one log line per row with its timestamp, node location, component and level), the job (for every window, list the alert categories that occur in it; most windows contain no alert), and that the alert rules below are exact: a line is an alert only if its component, level and message all match one category's rule, so a line's severity alone (FATAL, FAILURE, ERROR, SEVERE) does not make it an alert.
"rules_intro": one or two sentences introducing the rulebook table that follows, saying that each rule gives the category id, the component, the level and the message signature, and that a message signature is a regular expression searched in the message text (^ anchors the start, $ the end, .* matches anything).
"categories": an object with one key per category id below and a one-sentence plain description of what that alert means on the machine (hardware or software failure, what part). Do not restate the signature.
"tools": an object with one-sentence descriptions for: list_windows (window ids with their first and last timestamp, paged, at most 200 per call), read_window (the numbered lines of one window, with optional start line and line count for paging; output is capped at about 8,000 characters), get_rulebook (returns the alert rulebook again).
"answer": one short paragraph defining the final answer: one JSON object whose keys are the ids of windows that contain at least one alert and whose values are lists of the alert category ids present in that window, for example {example}; windows that are left out count as containing no alert.
"score": one sentence: the score is the F1 score over all (window, category) pairs in the answer compared with the true pairs.
"unit_prompt": instructions for a single worker who must triage exactly one window, {window}, with the same tools and the rulebook placed at {rules}; the worker must end with exactly one final line of the literal form UNIT | {window2} | <comma-separated category ids, or NONE>. ({window2} is the same id as {window}.)

Categories:
- KERNDTLB: component KERNEL, level FATAL, message signature /^data TLB error interrupt/, example message: data TLB error interrupt
- KERNTLBE: component KERNEL, level FATAL, message signature /^instruction TLB error interrupt/, example message: instruction TLB error interrupt
- KERNSTOR: component KERNEL, level FATAL, message signature /^data storage interrupt/, example message: data storage interrupt
- KERNMNTF: component KERNEL, level FATAL, message signature /^Lustre mount FAILED/, example message: Lustre mount FAILED : bglio11 : block_id : location
- KERNMNT: component KERNEL, level FATAL, message signature /^Error: unable to mount filesystem/, example message: Error: unable to mount filesystem
- KERNTERM: component KERNEL, level FATAL, message signature /^rts: kernel terminated for reason/, example message: rts: kernel terminated for reason 1004rts: bad message header: expecting type 1, type=3, cpu=0, index=0, total=1
- KERNRTSP: component KERNEL, level FATAL, message signature /^rts panic! - stopping execution/, example message: rts panic! - stopping execution
- KERNRTSA: component KERNEL, level FATAL, message signature /^rts assertion failed/, example message: rts assertion failed: `personality->version == BGLPERSONALITY_VERSION' in `void start()' at start.cc:131
- KERNREC: component KERNEL, level FATAL, message signature /^Error receiving packet on tree network/, example message: Error receiving packet on tree network, expecting type 57 instead of type 3 (softheader=0064588e 8aff0003 00000002 00000000) PSR0=00001f01 PSR1=00000000 PRXF=00
- KERNMICRO: component KERNEL, level FATAL, message signature /^Microloader Assertion/, example message: Microloader Assertion
- KERNMC: component KERNEL, level FATAL, message signature /^(machine check interrupt|DDR machine check register)/, example message: machine check interrupt
- KERNSOCK: component KERNEL, level FATAL, message signature /socket closed$/, example message: MailboxMonitor::serviceMailboxes() lib_ido_error: -1019 socket closed
- KERNCON: component KERNEL, level FATAL, message signature /BGLERR_IDO_PKT_TIMEOUT connection lost/, example message: MailboxMonitor::serviceMailboxes() lib_ido_error: -1033 BGLERR_IDO_PKT_TIMEOUT connection lost to node/link/service card
- KERNPOW: component KERNEL, level FATAL, message signature /^Power deactivated:/, example message: Power deactivated: R05-M0-N4
- KERNSERV: component KERNEL, level FATAL, message signature /^Power Good signal deactivated:/, example message: Power Good signal deactivated: R73-M1-N5. A service action may be required.
- KERNPAN: component KERNEL, level FATAL, message signature /^kernel panic$/, example message: kernel panic
- KERNNOETH: component KERNEL, level FATAL, message signature /^no ethernet link$/, example message: no ethernet link
- KERNPROG: component KERNEL, level FATAL, message signature /^program interrupt$/, example message: program interrupt
- KERNFLOAT: component KERNEL, level FATAL, message signature /^floating point unavailable interrupt/, example message: floating point unavailable interrupt
- KERNEXT: component KERNEL, level FATAL, message signature /tree header with no target waiting/, example message: external input interrupt (unit=0x03 bit=0x01): tree header with no target waiting
- KERNBIT: component KERNEL, level FATAL, message signature /redundant bit steering failed/, example message: ddr: redundant bit steering failed, sequencer timeout
- APPTORUS: component APP, level FATAL, message signature /uncorrectable torus error/, example message: external input interrupt (unit=0x02 bit=0x00): uncorrectable torus error
- APPSEV: component APP, level FATAL, message signature /^ciod: Error reading message prefix.*Link has been severed/, example message: ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to 172.16.96.116:56214: Link has been severed
- APPRES: component APP, level FATAL, message signature /^ciod: Error reading message prefix.*Connection reset by peer/, example message: ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to 172.16.96.116:52783: Connection reset by peer
- APPTO: component APP, level FATAL, message signature /^ciod: Error reading message prefix.*Connection timed out/, example message: ciod: Error reading message prefix on CioStream socket to 172.16.96.116:41739, Connection timed out
- APPREAD: component APP, level FATAL, message signature /^ciod: failed to read message prefix on control stream/, example message: ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.116:33399
- APPOUT: component APP, level FATAL, message signature /^ciod: LOGIN chdir\(.*\) failed: Input/output error/, example message: ciod: LOGIN chdir(/p/gb1/stella/RAPTOR/2183) failed: Input/output error
- APPUNAV: component APP, level FATAL, message signature /^ciod: Error creating node map from file .*: Resource temporarily unavailable/, example message: ciod: Error creating node map from file /home/auselton/bgl/64mps.sequential.mapfile: Resource temporarily unavailable
- APPBUSY: component APP, level FATAL, message signature /^ciod: Error creating node map from file .*: Device or resource busy/, example message: ciod: Error creating node map from file /p/gb2/pakin1/sweep3d-5x5x400-10mk-3mmi-1024pes-sweep/sweep.map: Device or resource busy
- APPCHILD: component APP, level FATAL, message signature /^ciod: Error creating node map from file .*: No child processes/, example message: ciod: Error creating node map from file /p/gb2/cabot/miranda/newmaps/8k_128x64x1_8x4x4.map: No child processes
- APPALLOC: component APP, level FATAL, message signature /^ciod: Error creating node map from file .*: Cannot allocate memory/, example message: ciod: Error creating node map from file /p/gb2/draeger/benchmark/dat16k_062205/map16k_bipartyz: Cannot allocate memory
- LINKIAP: component LINKCARD, level FATAL, message signature /iap failed/, example message: MidplaneSwitchController::receiveTrain() iap failed: R72-M1-L1-U18-A, status=beeaabff ec000000
- LINKDISC: component LINKCARD, level FATAL, message signature /port disconnected/, example message: MidplaneSwitchController::sendTrain() port disconnected: R07-M1-L1-U19-E
- LINKPAP: component LINKCARD, level FATAL, message signature /pap failed/, example message: MidplaneSwitchController::parityAlignment() pap failed: R22-M0-L0-U22-D, status=00000000 00000000
- LINKBLL: component LINKCARD, level FATAL, message signature /bll_clear_port failed/, example message: MidplaneSwitchController::clearPort() bll_clear_port failed: R63-M0-L0-U19-A
- MASABNORM: component BGLMASTER, level FAILURE, message signature /exited abnormally due to signal/, example message: mmcs_server exited abnormally due to signal: Aborted
- MASNORM: component BGLMASTER, level FAILURE, message signature /^mmcs_server exited normally/, example message: mmcs_server exited normally with exit code 13
- MONPOW: component MONITOR, level FAILURE, message signature /power module/, example message: monitor caught java.lang.UnsupportedOperationException: power module U69 not present and is stopping
- MONNULL: component MONITOR, level FAILURE, message signature /NullPointerException/, example message: While inserting monitor info into DB caught java.lang.NullPointerException
- MMCS: component MMCS, level FATAL, message signature /major internal error/, example message: L3 major internal error
```

