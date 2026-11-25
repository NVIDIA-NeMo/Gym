# Turn labeling guide (annotator edition)

This file is the single source of truth for how turns are labeled. Give it, unabridged, to every labeling agent, review
agent and judge-process variant. HANDOFF.md §7 only summarises it. When a rule here changes, log it in LOG.md.

The user's standing instruction (2026-11-13): **"these are the things we need to be extra careful about, and reason a lot
about."** A label is a causal claim about one turn. A wrong label does not just add noise: it misattributes the
outcome (e.g. blaming a correct read, or blaming re-checking). Precision matters more than coverage; when in doubt,
leave a turn unlabeled and say why in a note.

---

## 1. Unit, classes, file format

**Unit**: one model call (its reasoning + its tool call(s) or message).
In `response.output` a model call is **a reasoning item plus every item up to the next reasoning item**: parallel tool
calls interleave inside one call (`R F o F o`), so a tool call right after a tool output is not a new call (AGREEMENT A5,
LOG 010, user ruling LOG 014). In label files `idx` = the call's first item (its reasoning item); metrics normalise to the call
ordinal.

**Classes**
- `good`: the turn where a decision was made that produced what the tests need (an inference, a design choice, catching
  a regression).
- `harmful`: the turn where a decision was made that caused the failure (wrong inference, wrong edit, a self-confirming
  check, an answer-leak attempt).
- `minor` (waste with no effect on the outcome), `note` (context for humans), `luck` (analysis-only marker): context only, no decision label.
- Everything else: unlabeled = neutral. Expect most turns to be unlabeled; typically 1-3 labels per rollout.

**Group-level flag `ill_posed`** (user ruling, 2026-11-13): if a group's pass/fail split comes down to a tested detail
that nothing available determines (the tests are not well written for the task), set `"ill_posed": true` with evidence,
and put **no decision labels on any rollout of the group**: no `good` on the lucky guess, no `harmful` on the miss.
The group gets no decision labels.

**Per-rollout field `cause`** (failing rollouts): `wrong_action` | `missed_cue` | `premature_commit` |
`uncued_omission` | `ill_posed` | `other:<text>` (see §2 rule 3): the cause of the decisive failure. Report the share of
each cause. **Per-label `cause`** (optional, LOG 042): a `harmful` label may carry its own `cause` (`wrong_action` |
`missed_cue` | `premature_commit`) when a rollout's labels differ (e.g. a premature first commitment, then missed cues).

**Label file** (viewer format, `$PPO/tools/trajectory-viewer/labels/*.json`; extra fields are ignored by the viewer):
```json
{"labeler": "name", "rules": "LABELING.md @ <date or LOG entry>",
 "entries": [{"file": "/abs/path/date_x.jsonl", "line": 34, "verdict": "PASS|FAIL",
              "group_ill_posed": false, "ill_posed_evidence": "",
              "cause": "missed_cue", "summary": "…",
              "labels": [{"idx": 40, "label": "harmful",
                          "note": "what the turn did (quote the agent if giving its reason)",
                          "evidence": "#31 output: 'def parse_date(s, tz=UTC)'; #40: \"I'll assume naive datetimes\"",
                          "info_state_test": "what the agent had seen by this turn, and why a careful engineer would have acted differently",
                          "contrary_evidence_checked": "what was searched for cues pointing the other way, and what was found",
                          "decision_turn_check": "why this is the turn where the decision was made, not a restatement or a later write"}]}]}
```
The last three fields are **required on every decision label** (`good` / `harmful`) and are checked mechanically for being non-empty and
for quotes that match the trajectory. They exist to force the reasoning to be written down.

---

## 2. Rules

1. **Never label a turn for what it did not do.** A turn is harmful only for its own content. Reading two of three needed
   files is not harmful. If something was missed, the label (if any) goes on the turn that should have gone further
   (rule 3).
2. **Judge from the agent's information state at that turn, not from hindsight.** Test: would a careful engineer who had
   seen exactly what the agent had seen up to and including this turn have acted differently? Cite the observation that
   should have changed the action (turn `#idx` + quote). No citation → not harmful.
3. **Classify the failure before labeling.**
   - `wrong_action`: a turn did something wrong with the information it had → label that turn. The task text is part
     of that information: a write that contradicts what the task or spec states (a coding slip, e.g. an escaper that
     emits a raw tab the spec says to escape) is `wrong_action` at the write that introduced it; cite the task text
     (LOG 042).
   - `missed_cue`: the needed information was visible (grep hit, import, field list, stack trace, error text) and the
     agent moved on or explained it away → label the turn that saw the cue and moved on, citing the cue.
   - `premature_commit`: no cue, but the agent committed to a plan or edit while stating an unverified assumption →
     label the commitment turn `harmful` (user ruling 2026-11-13: "i don't think
     premature commit should be neutral"). The note must quote the stated assumption
     and say what cheap check would have verified it.
   - `uncued_omission`: nothing visible pointed to the missing information → **no turn labels** on the failing rollout
     ; label the **passing sibling's** turn that found it `good`.
   - `ill_posed`: the deciding detail is not determinable → group flag, no labels (§1).
4. **Mechanical help is diagnostic only.** The script pre-pass compares files touched by the golden patch / hidden tests
   with files the rollout ever read (tool args and outputs), and for each missed file or symbol finds the first output
   where its name appeared before the first edit (present → candidate `missed_cue`; absent → candidate
   `uncued_omission`). They are hints, never evidence on their own.
5. **Confirm every label on the full turn text.** Never label from a digest or keyword excerpts: they cut exactly the
   reasoning and tool outputs where cues and decisions sit, and they scramble order.
6. **Label only the turn where the decision was made** (user ruling). One decision = one label (a re-decision under
   new contrary evidence counts as a new decision, see below). Restatements, re-checks
   that keep the decision, and the later write that carries out an earlier decision get no label; otherwise the weight
   grows with verbosity and re-checking gets penalised. If the decision is made silently inside a write (no earlier
   reasoning chose it), the write is the decision turn. Same for `good`: label the inference, not the file read or the
   restatement that follows it. Two genuinely different decisions (e.g. a wrong inference, then separately a check that
   stubs out the very code under test) get two labels; a check that only re-encodes a decision already labeled is part
   of that decision (rule 10). **Re-decision under new evidence** (user rulings P3, P8): a turn that keeps an earlier
   decision is a restatement only if no new evidence bearing on it arrived since. A turn that saw new evidence that on
   its own should have changed a wrong decision and explained it away is a decision turn (`harmful`, `missed_cue`); a
   turn that saw new contrary evidence, weighed it and correctly kept (or switched to) the right choice is a decision
   turn (`good`). **Each such turn gets its own label, and so does the first commitment if it qualifies on its own
   information state** (user, 2026-11-13, LOG 035: "both" for `good`, mirrored for `harmful`; this replaces "still one
   label, latest turn" of LOG 016). So the first commitment of a wrong choice is `harmful` only if, at that turn, it was
   itself a `missed_cue`, `premature_commit` or `wrong_action`; otherwise it is unlabeled and only the later turn(s) that
   explained away sufficient new evidence are `harmful`. Later restatements after weaker evidence stay unlabeled (e.g. a
   turn that sees the field list naming the right object and explains it away is a decision turn; a later turn that only
   sees the shape of the data is not). A `good` inference later confirmed by evidence stays on the inference turn, not
   the confirming read (rev 4 C1: 233 #18): confirming evidence does not create a second label, contrary evidence does.
   **What counts** (user rulings P15, P16, LOG 042): the new evidence must on its own point to the other choice for a
   careful engineer; the agent's own doubts or weak, ambiguous hints create no extra label (same bar for `good` and
   `harmful`). An early lean is a commitment when the agent acts on it (designs or codes with it); a lean framed as an
   open question that the agent then checks itself is not a commitment (no label).
   A call that contains the rollout's fatal mistake in any form (first commitment, restatement, or the write that
   executes it) gets no `good` (P7, P17).
7. **A turn that contains the mistake is harmful even if most of it is right** (user: "95% right and 5% wrong is still
   wrong"). No softening for mixed turns.
8. **Don't invent reasons.** A note says what the turn did; it states why only by quoting the agent.
9. **Count contrary evidence before calling a detail inferable.** Search what the agent could see (issue, requirements,
   repo files it opened or grepped, outputs) for cues pointing the other way. One-sided evidence → inferable, label
   normally (inferring conventions is a legitimate skill, user ruling). Conflicting evidence or none → `ill_posed`.
   General case (user ruling, LOG 016): hidden tests that compare **exact message strings** the requirements describe
   only qualitatively ("descriptive", "explicit", "distinguishable"), where the only cue is one existing message, are
   `ill_posed` (C2 flipt).
   **Exception, a free hedge the agent found and dropped** (user ruling P13, LOG 039): if the spec leaves a detail open
   but the agent itself identified an option that satisfies every reading at no cost (e.g. a default export next to the
   named one) and then dropped it, that failure does not make the group `ill_posed`; the dropping call is `harmful`
   (`missed_cue`). A rollout that never saw the hedge falls back to the rule above.
10. **Verification turns**: a check is harmful only if a reasonable check at that point would have caught the bug (e.g.
    a reproduce script that stubs out the very function under test, when calling the real function would have exposed
    the bug). A check that could not have revealed a hidden-test detail is not harmful. A self-confirming check that only
    re-encodes a belief already labeled (e.g. hand-made mocks shaped by the same wrong assumption) is a `note`, not a
    second harmful label, unless it stubs out the code under test or a cheap check using the real structures would have
    failed (user ruling, LOG 016; C1 34 #66).
11. **Answer leaks** (git history mining, upstream fetch, hunting hidden tests) are harmful regardless of outcome; a turn
    that considers leaking and declines is a `note`; an attempt the harness blocked, so that nothing was revealed, is a
    `note` (user ruling P11, LOG 035). Searching the environment for hidden tests or grader files is "hunting hidden
    tests"; when nothing was found it is a `note` too (LOG 042). Show the attempt from the trajectory (the tool call and what its
    output revealed). The Gym behaviour flags (`behavior.json`, `rollout_info.behavior`: `gold_seen_external`,
    `official_test_edited`, …) are obsolete (user, 2026-11-13): never use them as evidence or as a filter.
12. **`good` may not rest on narration** ("I verified …"): narration is not evidence and a trajectory-reading judge can
    be fooled by it. Judge what the turn did and what it changed.
13. **Mistakes the agent fixed before the end are not harmful** (user ruling P1). `harmful` means the decision caused the
    failure; a mistake absent from the final state caused nothing. Leave it unlabeled (a `note` is fine) and give the
    fix `good` if it was a decision a test needs (catching a regression). Applies to passing and failing rollouts.
14. **Requirement-dictated decisions that every sibling makes stay unlabeled** in the first labeled set (user ruling
    P2): edits spelled out by the task that all rollouts of the group make (e.g. registering a new CLI flag that the
    issue names verbatim). Label decisions that separate siblings. A `note` may record them; a later measurement on a val
    subset decides whether to label them.
15. **Old tests vs the requirements: decide with the hidden tests** (user ruling P4, 2026-11-13). The repo the agent
    works in has the base commit's tests; at grading the hidden test patch (the PR's own test changes) is applied, then
    `fail_to_pass` and `pass_to_pass` run. The prompt's "I've already taken care of all changes to any of the test
    files" does not hold for the repo the agent sees. The pre-pass marks each failing test of each rollout as unchanged
    by the hidden patch (`pass_to_pass`) or added / modified by it.
    - **Unchanged (`pass_to_pass`)**: a real regression; the golden solution passes it together with the requirements,
      so a change satisfying both existed. If the agent saw it fail and dismissed it (deselected it, called it stale or
      "to be updated by the grader", ignored it), the dismissing call is `harmful` (rule 6: the call that decided, not
      the test run that revealed the failure); its `cause` is `missed_cue` (the failing output was the cue; user ruling,
      LOG 016). If the agent never saw it fail, rules 2-3 apply to the edit that broke it.
    - **Added or modified by the hidden patch**: the hidden tests define the intended behaviour. **As long as they agree
      with the requirements** (the requirements, read carefully, determine what they assert; user: "as long as hidden
      test agree with requirement, it's fine"), label from them: letting the conflicting old copy in the repo fail is
      correct (no label); keeping the old behaviour against a clear requirement is `harmful` if it causes the failure.
      If the hidden tests assert something the requirements do not determine, the group is `ill_posed` (§1).
16. **A correct decision that separates siblings is `good` in any rollout, passing or failing** (user ruling P5,
    2026-11-13: "let's label it good for now"; provisional). Separating = a test needs it and at least one sibling
    decided it differently (or missed it) and fails a test because of that. The label judges the decision, not the
    rollout's outcome: the same inference gets the same label in a passing and in a failing sibling. Same
    behaviour, same label across the group (M6). Decisions every sibling makes stay unlabeled (rule 14). Majority defaults and
    silent spec-following count too (P9), but **copying text the spec gives verbatim** (a regex, an offset, an escape
    table) is not a decision: no `good`, even if a sibling slipped while copying it; the slip itself is `harmful`
    (`wrong_action`) (user ruling P14, LOG 042).
17. **Ending the session mid-plan is `harmful`** (user ruling P12, 2026-11-13: "ok harmful"). In the opencode harness a
    reply with no tool call ends the session. A final call that announces a next step (or only reasons) and emits no tool
    call, while the work is unfinished and no context limit or timeout forced the end, is `harmful` (`wrong_action`) if
    the rollout fails. Confirm from the run record: not context-limited, no timeout, last reply without a tool call.

---

## 3. Procedure for one group

1. **Read the facts first** (script pre-pass): per-test status, failing assertions from the hidden test source, fatal
   difference between each failing patch and the golden / passing patches, edit replay (which turn introduced each
   final hunk), compaction boundaries, the rule-4 missed-file/cue table.
2. **Ill-posed check** (rule 9): does the pass/fail split hinge on something the task, requirements and code the agents
   could see do not determine? Look actively for contrary evidence (other places in the repo doing the same operation,
   the agent's own search results). If ill-posed: set the flag with evidence and stop.
3. **For each failing rollout**: from the failing assertion → the hunk → the turn that wrote it → the reasoning that
   chose it → the observation behind that choice. Find the **decision turn**; run the information-state test on it
   (rule 2) and on the turns before it where a cue was visible. Classify the cause (rule 3). Look for separate decisions
   that also contributed (e.g. a self-confirming check, rule 10).
4. **For each passing rollout**: find the turn where it made the inference or choice the failing siblings missed (rule 6:
   the inference, not the read). Also look for a recovery (caught a regression).
5. **Write labels** with all required fields (§1). Re-read the full text of every labeled turn and of the turns just
   before it.
6. **Self-check against §4** (the mistake catalog) before handing over.

---

## 4. Mistake catalog (the review checklist)

Mistakes actually made in the pilot (rev 2 → rev 3, LOG 001-004). Every reviewer checks every label against every item.

| # | Mistake | Pilot example |
|---|---|---|
| M1 | Turn charged for what it did not do (an omission) | openlibrary 34 #61 "never read content_info.py"; the real decision was #52 |
| M2 | Story built from a digest / excerpts: wrong order, wrong turn | flipt 1249 #25 said "wrote the right wording, then switched"; it was the reverse, and the decisive turn was the silent reversal in the write at #34; openlibrary 34 "#67 used real SDK objects, #73 switched to mocks": #67 already mocked (LOG 007) |
| M3 | Contrary evidence the agent saw was ignored (hindsight) | flipt 1224's grep found `checking directory: %w` in a structurally identical `ensureDir`; rev 2 called it "unrelated" |
| M4 | Reason invented for the agent | flipt 1083 #15 note "follows the existing message"; the agent never said so and never searched |
| M5 | Execution labeled instead of the inference | openlibrary 233 #37 (read) / #39 (restate) labeled; the inference was #30. Mirror: 34's earlier missed cue at #43 went unnoticed |
| M6 | Same behaviour, different labels | flipt 1224 #27 harmful vs #71 note, both re-checks that kept the draft |
| M7 | One decision labeled several times | flipt 1224 #21/#27/#33; openlibrary 34 #52/#63/#81 |
| M8 | Ill-posed test treated as the agent's mistake | flipt: the hidden test asserts exact wording that nothing determines |

---

## 5. Case catalog

Worked cases with the user's rulings. Add new ones as you find them (§6). Status: `ruled` (user decided), `proposed`
(awaiting the user), `pending-check` (needs full-turn confirmation).

**C1 openlibrary `…2fe532a3…`, eval run 20261111-130351 step 60 Pro, lines 233 pass / 34, 96 fail** (`ruled`: final =
**rev 4**, LOG 015-016: 34 #52 harmful, 34 #22 good; 96 #101 harmful, 96 #36 good; 233 #18 good, 233 #79 good; the
bullets below are the history, superseded where they differ).
Task: keep Amazon languages; serialize() must read `ItemInfo.ContentInfo.Languages`. 34 read `item_info.languages`.
- 34: #41 printed the resource constants incl. `ITEMINFO_CONTENTINFO` (14th, inside its `head -50`); #43 moved on;
  #50 read `item_info.py`, whose `swagger_types` lists `'content_info': 'ContentInfo'`; #52 noted ItemInfo has no
  `languages` and concluded it "might be … added dynamically" → **#52 harmful (`missed_cue`)**. #73 reproduce script
  switched from real SDK objects (#67) to mocks that accept any attribute although #52 knew ItemInfo lacks the field →
  **#73 harmful** (separate decision, rule 10). #85 adds `languages` to `conforming_fields` → **good**. #61, #63, #81
  unlabeled (reads / restatement / write carrying out #52). #43: earlier cue, unlabeled per rule 6 (note).
- 233: **#30 good** ("no ITEMINFO_CONTENTINFO_LANGUAGES … so languages might be part of ITEMINFO_CONTENTINFO"),
  **#69 good** (`conforming_fields`), **#77 good** (re-ran tests, caught `'languages': []` breaking a test),
  **#87 good** (made the key conditional). #37, #39 unlabeled. #61 to check: it may contain its own decision
  (dedup, skipping "Original Language"); label it only if that decision is what a test needs.
- Evidence so far: digest + keyword excerpts + a PyPI copy of the same SDK. Confirm #41, #50, #52, #61 on full text.
- **Checked on full turns (LOG 007, rev 3 = `labels/claude-pilot-20261113.json`; idx = first item of the model call).**
  Confirmed: #42 (output of #41) lists `ITEMINFO_CONTENTINFO` (12th, not 14th); #51 (output of #50) shows
  `'content_info': 'ContentInfo'` and the `ContentInfo` import; #52 says the field "might not be part of the standard
  PAAPI response model, or it's added dynamically" and chooses `getattr` on item_info; 233 #30 inference verbatim.
  Corrections to the bullets above:
  - 34: the reproduce script used dataclass mocks from the start (#66/#67 call, `languages: Languages = None` on
    ItemInfo); "#67 used real SDK objects" was wrong (an M2 from excerpts). #70/#73 only patch missing mock attributes.
    The harmful check label moves to **#66**; #73 unlabeled.
  - 34: #22 already read `item_info.languages` off the PR snippet but deferred ("Let me look at the paapi5 SDK to
    understand the structure"), so #52 stays the decision turn (it is the first turn holding the contrary evidence).
  - 34 #85 / 233 #69 `good` (conforming_fields) were writes carrying out earlier plans (34 #19, 233 #18): rule 6. The
    decision is requirement-dictated and both siblings make it → `note` pending **P2**.
  - 233 #61 is the same model call as #59 (reasoning #59, message #60, edit #61). #59 designs the extraction and adds
    the `languages` key unconditionally, breaking the existing test `test_serialize_does_not_load_translators_as_authors`
    (#78); it is fixed at #79 ("If I only include `languages` when the list is non-empty, the test would pass") and
    written at #87 → **#79 good** (catching a regression), #77 and #87 unlabeled (rev 2's #77 + #87 = M7); #59 → `note`
    pending **P1**.
  - **Third rollout, line 96 (FAIL; AGREEMENT A2)**: reads the correct `edition_info.languages` but adds the `languages`
    key unconditionally and fails the pass-to-pass `test_serialize_does_not_load_translators_as_authors`; it saw that
    test fail (#99) and deselected it (#101 call) → rule 15: the #101 call is harmful (the test is `pass_to_pass`). Not yet labeled (dry run).
  - The only fatal difference between 34 and 233 is item_info vs content_info: 34's patch already has the conditional
    key and conforming_fields and passes the other 3 F2P tests; it fails only `test_serialize_sample_record`.
  - Rev 3 decision labels: 34 #52 harmful (`missed_cue`), 34 #66 harmful; 233 #30 good, 233 #79 good. Notes: 34 #19,
    #43, #91; 233 #18, #39, #59; 233 #19 minor (a `find /` that hit the 120 s tool timeout).

**C2 flipt `…72d06db1…`, same eval file, lines 1083 pass / 1224, 1249 fail → `ill_posed`** (`ruled`).
All three write the same code; the hidden `TestNewSink_Error` asserts exact strings (`checking log directory: …`,
`creating log directory: …`, `opening log file: …`). The issue/requirements only ask for "explicit" / "distinguishable,
descriptive" messages. In the repo at base commit `b6edc5e4`: no "log directory" string anywhere; `cmd/flipt/main.go`
`ensureDir` does the same operation (Stat → MkdirAll) with `checking directory: %w`; the only pointer toward "log" is
the existing `opening log file` in the same function. User: "the tests are not well written … let's not encourage or
penalize any turns." No labels on 1083, 1224, 1249. Note: 1249 also renamed the pre-existing,
tested `opening log file`; unlabeled under this ruling (user confirmed, LOG 016: a determinable mistake inside an
`ill_posed` group gets no decision label, here also not the but-for cause, since 1249 fails the ill-posed subtests anyway;
a `note` records it).

**P1 Mistake made and later fixed by the agent** (`ruled` 2026-11-13, LOG 008: option (a); rule 13). C1 233: the #59 call adds the
`languages` key unconditionally, breaking an existing (pass-to-pass) test; the agent's own test run (#77) shows it and
#79 fixes it. The final patch does not contain the mistake and the rollout passes.
- Options: (a) #59 unlabeled, #79 `good` (the recovery is the labeled decision); (b) #59 `harmful` too (rule 7 read
  literally: the turn contained a mistake); (c) #59 `minor` (cost turns, no effect).
- Recommendation: (a). `harmful` is defined as causing the failure, and there is none; (b) would push down the same
  rollout that the outcome pushes up and penalise exploring-then-fixing. Same rule for failing rollouts: a mistake the
  agent fixed before the end gets no harmful label; the fix gets `good` only if a test needs it.

**P2 Requirement-dictated decisions that every sibling makes** (`ruled` 2026-11-13, LOG 008: option (b) for the first labeled set; rule 14). C1: adding `languages` to
`conforming_fields` (34 #19, 233 #18; needed by 3 of 4 F2P tests) and skipping "Original Language" + dedup (needed by
`test_serialize_sample_record`) are spelled out in the requirements; both rollouts do them. Rev 2 labeled the
conforming_fields write `good` in both.
- Options: (a) `good` on the decision turn in every rollout that makes it (passing and failing) when a hidden test
  needs it; (b) unlabeled (`note`), the outcome already treats them alike within the group; (c) `good` only in
  failing rollouts (offsets the negative push on a correct step) or only in passing ones.
- Trade-off: (a) is correct by the class definition and reduces the wrong lesson in failing rollouts (34 is right on
  3 of 4 tests), but almost every rollout has several such decisions, so labels per rollout, judge cost and the room
  for placement mistakes all grow; with zero-sum shaping a `good` in a failing rollout also makes its other turns more
  negative. (b) keeps labels on the decisions that separate siblings, which is where the pilot found the signal.
- Recommendation: (b) for the first labeled set (precision first), measured later: label a val subset both ways and
  see if the judge can do (a) reliably. Rev 3 writes these as notes.

**P3 Rule 3 vs rule 6: an earlier decision kept after new contrary evidence** (`ruled` 2026-11-13, LOG 008: the later turn; rule 6 amended). Rule 6 leaves
"re-checks that keep the decision" unlabeled; rule 3 (`missed_cue`) labels "the turn that saw the cue and moved on".
They conflict when a decision is made early (no cue yet) and a later turn sees contrary evidence and keeps it. In C1
it does not bite (34 #22 explicitly deferred to an SDK check, so #52 is the first real decision), but it will.
- Recommendation: a turn that keeps a decision is a restatement only if no new evidence bearing on it arrived since
  the decision; if the turn saw new evidence that should have changed the decision and explained it away, it is the
  decision turn (`missed_cue`), and the early turn is unlabeled (or `premature_commit` if it committed with a stated
  unverified assumption). One label either way.


**P4 An existing test broken by the agent's own change, then dismissed** (`ruled` 2026-11-13, LOG 011 → rule 15;
proposed by the local agent, AGREEMENT A2). SWE-bench Pro prompts say "I've already taken care of all changes to any of the test files … you DON'T have to
modify the testing logic", while the repo's tests are unchanged. C1 line 96 (FAIL): #97 predicts that adding the key
breaks `test_serialize_does_not_load_translators_as_authors`; #99 runs the suite and it fails on `{'languages': []}`;
the #101 call decides "the grading test file has been updated to include the `languages` field … the test failure is
expected" and deselects the test (#103). That test is pass-to-pass, i.e. unchanged by the hidden patch, and 96 fails
exactly it. 233 met the same failure, weighed the same reading at #79, and made the key conditional instead.
- Local agent's recommendation: a pre-existing test that the agent's own change broke, then dismissed / deselected /
  explained away, is `harmful` at the turn that dismissed it, not `ill_posed`.
- AGA agent: agree, with one condition. Harmful when a change satisfying both the requirements and the existing test was
  available (here: the requirements never say the key must exist when empty, so a conditional key satisfies both; 233
  found it). When the requirements explicitly change the behaviour the existing test asserts, keeping that test
  failing is correct and gets no label. Placement: the call that decided to dismiss (96: the #101 call), not the test
  run that revealed the failure (#99) and not the deselecting command (#103, same call).
- **User ruling**: decide with the hidden tests: unchanged (`pass_to_pass`) → dismissing it is harmful; modified by the
  hidden patch → follow the requirement as long as the hidden tests agree with it, else `ill_posed` (rule 15).

**P5 A correct, separating decision inside a failing rollout** (`ruled` 2026-11-13, LOG 013 → rule 16, provisional:
"let's label it good for now"). Raised by the protocol dry run (AGREEMENT A6): C1 line 96 #36 makes the same inference
as 233 #18 (languages under ContentInfo, from the SDK grep at #26) but fails on a dismissed regression; 34 #22 chooses
the conditional key that 96 got wrong ("If I add `languages` to the returned dict, and the test expects it to be absent,
the test would fail"), but fails on the wrong object. Both labelers had left these as notes. Ruling: `good`.

**P6 A group with one ill-posed failure mode and one determinable one** (`ruled` 2026-11-13, LOG 022: **whole group
`ill_posed` for now**). Round 1, privacy-data-governance (TB): 3 rollouts fail only an exact case-sensitive
'Validation complete' check the task describes qualitatively; 5 fail a determinable check. Proposed: apply `ill_posed`
per failure mode (label the determinable failures). User: "let's make it whole group ill posed for now". So §1 stands:
if any failure mode of the group is ill-posed, the group is `ill_posed` and no rollout gets a decision label.

**P7 `good` on a call that also commits the rollout's fatal mistake** (`ruled` 2026-11-13, LOG 034: **no `good`, note
only**). Round 1c: a call holds a correct separating decision (rule 16) and the first commitment to the decision that
later fails the rollout, whose `harmful` label rule 6 / P3 put on a later turn (vault R5 #215, R12 #12; home-assistant
R4 #22, R7 #18). The vault checker gave a note, the home-assistant checker gave `good`. Ruling: the call gets no
decision label; record the correct decision in a note (rule 7: a turn containing the mistake does not get `good`).

**P8 `good` when a later turn keeps the right choice against new contrary evidence** (`ruled` 2026-11-13, LOG 034:
**both turns `good`**). Round 1c home-assistant R14: #47 leans "make the service handler async and use aiohttp"; #55
output shows a `requests_mock` fixture; #61 weighs it: "requests calls them `RequestException`. So aiohttp it is."
The home-assistant checker moved the label to #61 (symmetric P3), the vault checker kept R13's on the first inference
(#39 vs #47). User: "both": the first inference turn and each later turn that re-decides correctly under new contrary
evidence get `good`. Later turns that keep it without new contrary evidence (restatements, confirming reads) stay
unlabeled (rule 6). Mirrored for `harmful` (user, LOG 035): the first commitment is `harmful` if it qualifies on its
own information state, and each later turn that explained away sufficient new evidence is `harmful` too.

**P9 Rule 16 scope: majority defaults and spec-following** (`ruled` 2026-11-13, LOG 034: **as written**). Round 1c
strict-versioning: 13 rollouts keep the restored API running, 3 fail by not doing so; facebook-react: 8 rollouts keep
the directly portaled child ambiguous, 3 break it. Ruling: rule 16 applies wherever the separating decision is made,
including a majority default and a decision taken silently inside a write (rule 6: then the write is the decision
turn). Same behaviour, same label across the group (M6). Rule 14 still exempts decisions every sibling makes.

**P10 Three whole-group ill-posed exposures in round 1c: all determinable** (`ruled` 2026-11-13, LOG 035).
home-assistant-core-42129: (a) HTTP client: the interface names `ClientError` (aiohttp's class; requests has
`RequestException`), HA components mostly use `async_get_clientsession`, the repo ships `aioclient_mock`; the hidden
test mocks only that session. Codex labeler A had called the group `ill_posed` ("requests is a core dependency");
overruled. (b) Allowlist matching: the requirement says "present in hass.config.allowlist_external_urls", the hidden
tests allowlist prefixes ("http://example.com/images/", "/opt/"); the repo's `is_allowed_external_url` /
`is_allowed_path` implement HA's allowlist semantics and the task says "respect Home Assistant's security
configuration". strict-versioning-cascade: (c) leaving the restored API running: "restore access to all old claims so
adjusters can process them"; the grader starts postgres/redis but not the API and curls port 8000 first. General
reading: a library or mechanism named or implied by the spec plus the repo's own convention is determinable even when
another library could work; a literal phrase is read in the sense the repo's own helpers give it; "restore access"
means the service is left serving.

**P11 Blocked leak attempts** (`ruled` 2026-11-13, LOG 035: **note**). home-assistant R9 #192 (`git log --all -S
is_allowed_external_url`, aimed at a helper's future history after calling it "borderline") and R13 #146 (`git log
--oneline --all -- */manifest.json | head -3`): the harness blocked both, nothing was revealed. Rule 11 now: harmful when
the leak attempt revealed something; a blocked attempt is a `note`.

**P12 The model ends the session mid-plan** (`ruled` 2026-11-13, LOG 036: **harmful**, rule 17).
strict-versioning-cascade R12: the last reply (#60 reasoning, #61 text '**4. Make the normalizer enforce v2 compliance
actively** instead of a no-op (so schema v2 means "verify every event"):') has no tool call; finish "stop" after 89
tokens, run time 87 s of 3000 s, not context-limited, opencode exit 0. The API was never restarted and the tests fail.
Earlier steps of the same rollout put the announcement and the tool call in one reply. In the 2026-11-13 collection
about 28 of 1,679 rollouts stop like this (6 on a colon, 22 on a bare reasoning item), 27 of them failing.
---

**P13 A free hedge the agent found and dropped** (`ruled` 2026-11-13, LOG 039: **harmful**, rule 9 exception).
comet-ml-opik-5440 (swemer_oml, 12 pass / 4 fail). The task lists "`LinkifyText` (function)" under "expose the following
public API exactly as named" with no import form; the hidden test (shared/LinkifyText/LinkifyText.test.tsx) does
`import LinkifyText from "./LinkifyText"`. 15/16 rollouts exported LinkifyText both named and default. R7 (named only,
fails all 41 tests): #1602 "Components use named exports sometimes, default others."; #1786 "Compromise: named export
`LinkifyText` AND `export default LinkifyText`. That covers both import styles."; #3153 "what if grader expects default
export? I could add `export default LinkifyText` at bottom — harmless, increases compatibility"; #3156 "Decision: keep it
clean — only named exports LinkifyText & health, plus types." Codex's three agents called the group `ill_posed` (export
form not determined; P6). User: "seems harmful in this case": the #3156 call is `harmful` (`missed_cue`, like P4: a
choice satisfying both readings was available, the agent saw it and dropped it); the group is labelable. The other
failures are determinable: R1 placed the file at shared/LinkifyText.tsx (repo convention shared/X/X.tsx, 15/16 did so),
R4/R15 linkify text without a standard web protocol (spec: "using standard web protocols").

**P14-P17 Re-sweep follow-ups** (`ruled` 2026-11-13, LOG 042; raised by the round 1c re-sweep agents).
- P14 (vault): applying P9 literally would add ~44 `good` labels for copying the spec's `Bearer[ \t]+` regex, zstd
  payload offset and escape table, separating only because R8 / R12 / R5 slipped while copying. Ruling: verbatim
  copying is not a decision (rule 16 amended); the slips are `wrong_action`.
- P15 (strict-versioning R3 #61, R9 #74, R13 #62, R1 #76): second `good` labels for turns where the agent itself read
  weak hints (a seed file, "units enabled but stopped") as pointing to stopping the API. Ruling: contrary evidence
  must point away on its own; these lose the second `good`.
- P16 (home-assistant R10 #47 "Let me design with requests.get" vs R9 #17 "That might refer to
  requests.exceptions.RequestException or aiohttp.ClientError"; vault R1 #0, R2 #0): acted-on leans are commitments
  (R10 #47 `harmful` premature_commit; vault R1 #0 / R2 #0 stay `harmful`); an open question the agent then checks is
  not (R9 #17 no label).
- P17 (strict-versioning R3 #61, whose call contains the write #71 that fails test_check_6): a call that contains the
  fatal mistake in any form gets no `good`.

## 6. Discovering new cases (required)

The pilot had 5 rollouts and produced 8 mistake types and 2 rulings. A labeled set of hundreds of groups will contain
patterns nobody has seen yet. Finding them is part of the job, not a side effect.
- Every labeler and reviewer reports **anything that does not fit the rules cleanly**: a failure with no clear decision
  turn, a rule that gives an answer that feels wrong, two rules that conflict, an outcome caused by the environment.
- Add each to §5 as `proposed`, with the group, the turns, what is tricky, the options and your recommendation. If it
  changes how labels are assigned, **ask the user for a ruling** before applying it at scale; then mark it `ruled`,
  add a rule or catalog row, and **re-sweep all existing labels** against it (log how many changed).
- Seeds to watch for (not yet ruled; propose a ruling when you meet one):
  - ~~a mistake made and later fixed by the agent~~ (ruled P1, rule 13);
  - a mistake in a passing rollout that did not matter (`minor`?); ~~a good decision in a failing rollout~~ (ruled P5, rule 16);
  - information lost at a compaction boundary (is forgetting a cue harmful?);
  - environment / infrastructure failures: tool timeouts, sandbox errors, 3 h cap kills, flaky tests;
  - several independent mistakes in one rollout, each sufficient to fail;
  - Terminal-Bench tasks (no golden patch; judge from the test log and final state);
  - subagent turns, if a harness uses them;
  - all-fail and all-pass groups (no sibling contrast).

---

## 7. Labeling protocol (simplified)

User, 2026-11-13: "the protocol, no need to be so complicated, and goal is to make sure labels are correct." The earlier
multi-round adversarial design (4 reviewer roles, adjudicator, up to 3 rounds) is replaced by this. Agents run Opus 5.5
at xhigh effort, read full turns (the group workspace from `tools/group_workspace.py`, or the raw JSONL), and get this
file.

1. **Two independent labelers** per group, different starting points: one walks back from the failing tests (fatal
   difference → hunk → turn → reasoning → observation), one reads each rollout forward and marks where the agent's
   beliefs or plan changed. They do not see each other's output. Record their agreement.
2. **One checker** per group, whose only job is correctness. It takes the union of both label sets and, for every label
   and every disagreement, re-reads the full text of the turn and the turns before it, and decides keep / move / drop
   against §2 and the §4 checklist, with the reason written down. It also checks the decisions only one labeler saw.
   Every kept label is confirmed on the full turn text (rule 5).
3. **Mechanical checks** (`tools/check_labels.py`): idx exists, quotes match the cited item, required fields non-empty,
   ≤ one decision label per model call, no decision labels in `ill_posed` groups (notes, `minor`, `luck` allowed;
   user ruling, LOG 016).
4. Groups where the checker made a substantive change or could not resolve a disagreement go **to the user**, on top
   of the one consolidated set. No further rounds by default. User rulings go into §5 and, if general, §2.

---

## 8. Using the catalog in judge development (Phase 2)

- Score every judge variant not only on agreement with gold but **per mistake type**: how often it charges omissions
  (M1), labels executions instead of inferences (M5), labels a decision several times (M7), mislabels ill-posed groups
  (M8), invents reasons (M4). These rates say which variant reasons the way we want.
- Read a sample of each variant's errors every iteration and look for new mistake types; add them to §4 with an
  example, then check earlier variants for the same type.
- Build the judge's prompt from §2-§3 and give it §4-§5 as worked examples; never include test-split groups as
  examples.
