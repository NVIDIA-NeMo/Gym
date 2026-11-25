# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Instructions for the judge sessions (OpenCode in the judge sandbox). The rules are /judge/LABELING.md (the
turn_judge annotator edition, unabridged); these texts say what to do and where the files are.

Variants (turn_judge LOG 061): "v0" = the first pair prompt; "A" = pair prompt with the precision fixes of the sbx-v0
error analysis and a mandatory self-check; "B" = A + a group decision list written beforehand by a group analyst
(GROUP_INSTRUCTIONS, separate session); "Bp" = A + group facts in every pair (all siblings' patches / test results)
from which the pair agent writes the decision list itself; "C" = B + the round-2 B error-analysis fixes (decision list
with label policy and membership, no placement claims; per-decision checklist; operational P16; P7 guard; hedges).
"D" = C + the round-4 C error-analysis fixes (P16 lean test narrowed, first_mention.py, list membership / policy fixes,
cascade guard). "E" = D + the round-6 D error-analysis fixes (stage-1 ill-posed cue rules, `but_for` on harmful labels, lean judged by
how the call ends, too-late placement rules, P8 evidence bar). Variants C / D / E fall back to Bp for a group whose
stage 1 fails twice."""

INTRO = """You are annotating the turns of coding-agent rollouts of one task: for each rollout, find the turns where the
decisions that determined its outcome were made. The annotations are used to analyze agent behaviour and as a reference
set for evaluating automated judges. A wrong annotation misattributes the outcome, so precision matters more than
coverage. Reason carefully.
"""

FILES = """
## Files (everything is under /judge; work only there, never use the network)
- `/judge/LABELING.md`: the authoritative rules (unit, classes, file format, rules 1-17, procedure, mistake checklist
  M1-M8, worked cases). Read it in full first and follow it exactly.
- `/judge/ws/TASK.md`: the exact prompt the agents received. Judge every turn against it and what the agent observed.
- `/judge/ws/answer_key/` (the agents never saw it): SWE: `golden.patch` (reference fix), `hidden_tests.patch` (tests
  the grader adds or changes), `tests.json` (`fail_to_pass`; `pass_to_pass` = existing tests the hidden patch leaves
  unchanged, rule 15), `task_fields.json`. Terminal-Bench: the task folder (`solution/`, `tests/` with the hidden
  tests and exact expected values, `environment/`, designer notes where present); the grader checks the final
  container state, there is no patch.
- `/judge/ws/R<k>/trajectory.txt`: the full rollout, every item as `===== #<idx> [call <c>] <type> =====` followed by
  its full text. A model call = one reasoning item plus every item up to the next reasoning item; `[call c]` is its
  ordinal. Very long lines are wrapped, not cut: a line starting with `↪ ` continues the line above (never quote
  across a wrap). Items can be very long: read each relevant item to its end.
- `/judge/ws/R<k>/calls_index.txt`: one line per model call (call number, item range, file line, start of its
  reasoning, its tool calls). For navigation only, never a basis for a label.
- `/judge/ws/R<k>/tests.json` (per-test status), `test_output.txt` (grader log), `failing_assertions.txt` (failure
  report of each failing test), `final.patch` (SWE).
- `/judge/ws/PREPASS.md`: script facts per rollout (failing tests, context-limit / cap flags, ...). Diagnostic only.
- `/judge/ws/group.json`: the `file` and `line` to put in the label file, and each rollout's outcome (1 = pass).
"""

PRECISION = """
## Precision rules for this job (they apply LABELING; follow them exactly)
1. One decision, one label, on its decision turn: the FIRST model call whose plan or code commits to the choice (an
   acted-on lean counts, P16). A later call that restates it ("Final decision: ...") is a restatement, unless new
   evidence arrived in between that on its own points the other way (P8 / P15); a second label on the same choice must
   name that new tool output by its #idx in `decision_turn_check`. The agent's own re-derivation, a re-read or a
   re-run of something it already knew is not new evidence.
2. Information state is the rollout's OWN: never use what the other rollout saw or did as the cue a turn missed
   (rule 2). `premature_commit` requires quoting the agent's own stated, unverified assumption (rule 3); if nothing
   the agent saw pointed to the missing information, the cause is `uncued_omission` and there is no harmful label.
3. Every `harmful` label names, in its `note`, the failing hidden test(s) it causes and how; trace every failing test
   of the failing rollout to the hunk / final state and the turn that decided it. A mistake absent from the final
   state is not harmful (rule 13).
4. `good` = a decision a hidden test needs that at least one sibling decided differently or missed and failed a test
   because of it (rule 16; not verbatim spec copying, P14). A fix of the agent's own regression is `good` only if
   that regression would have failed a hidden test. No `good` for process (testing, careful reading) or for "not
   making the other rollout's mistake" as such: credit the decision itself. No `good` on a call that contains the
   fatal mistake (P7 / P17).
5. Ill-posed check: decide from what the AGENTS could see (task text, repo, their tool outputs), never from the answer
   key (they never saw it). List the cues for each reading in a two-sided table in your notes before deciding. P13
   applies only when the agent itself states that the option satisfies both readings. When most failing rollouts fail
   the same single check while everything else passes, examine that check's determinability closely.
"""

SELF_CHECK = """
## Before you finish (required)
Run `python3 /judge/bin/check_pair.py`. It checks /judge/out/pair.json mechanically (format, required fields, idx = the
first item of its call, every label has a verbatim quote from the cited item or its call). If it prints FAIL, fix
every problem it lists and run it again; finish only when it prints PASS. Then write `/judge/out/notes.md` and stop.
"""

OUTPUT = """
## Output: `/judge/out/pair.json` (valid JSON, exactly this shape)
{{"ill_posed": true|false, "ill_posed_evidence": "...", "fatal_difference": "...",
 "fail": {{"rollout": "R{fail}", "line": {fail_line}, "verdict": "FAIL", "cause": "wrong_action|missed_cue|premature_commit|
  uncued_omission|ill_posed|other:...", "summary": "...", "labels": [...]}},
 "pass": {{"rollout": "R{passing}", "line": {pass_line}, "verdict": "PASS", "summary": "...", "labels": [...]}}}}
Labels follow LABELING §1: `idx` = the first item of the model call (its reasoning item); every `good` / `harmful` label
fills `label`, `note`, `evidence` (verbatim quotes as `#N: "exact text"` / `#N output: "..."` / `#N args: "..."`, the
quote must appear in item N or another item of its call), `info_state_test`, `contrary_evidence_checked`,
`decision_turn_check`{decision_field}; `note` / `minor` labels for context only.
`/judge/out/notes.md`: the fatal difference with evidence, the turns you considered and rejected, and anything that does
not fit the rules (LABELING §6).
"""

PAIR = """
## Your pair
`R{fail}` FAILED its hidden tests (line {fail_line}); `R{passing}` PASSED them (line {pass_line}).{others}

## How to work (use your shell, read and search tools; read in chunks, never judge from a truncated view)
1. Read LABELING.md in full.
2. Facts first: TASK.md; the answer key (what the hidden tests assert; the reference solution); for R{fail}: tests.json,
   failing_assertions.txt, test_output.txt, final.patch; for R{passing}: tests.json, final.patch; their PREPASS.md
   sections. Pin down the fatal difference: what R{fail}'s final state does that fails the test(s), and what R{passing}
   did instead.{decision_step}
3. Ill-posed check for this contrast (LABELING §3 step 2; rule 9 with its exception; §5 P6, P10, P13; precision rule
   5). If ill-posed, set `ill_posed` true with the evidence and put no decision labels.
4. Otherwise find the decision turns with calls_index.txt, then read the relevant calls and the calls before them in
   full in trajectory.txt:
   - R{fail}: the turn(s) where the decision that caused the failure was made (rules 3, 6, 17), with the
     information-state test (rule 2) and contrary evidence (rule 9); separate contributing decisions (rules 10, 15).
     No label for what a turn did not do (rule 1). It may also hold `good` labels for correct separating decisions it
     made (rule 16; precision rule 4).
   - R{passing}: the turn(s) where it made {pass_scope}, labeled `good`.
5. Re-read the full text of every labeled call; quotes must be verbatim; check §4 (M1-M8) and the precision rules.
"""

V0_PASS_SCOPE = "the decision R{fail} got wrong (and other separating decisions you can see)"
LIST_PASS_SCOPE = "each applicable decision of the list correctly"

DECISION_STEP_B = """
   `/judge/decisions.json` lists the group's separating decisions (written by a group analyst who read every
   rollout's patch and test results; verify what you use, you may disagree with evidence). For EACH decision in it
   that applies to R{fail} or R{passing}, label the turn where that rollout made it: `good` if it decided correctly,
   `harmful` if it decided wrongly and that caused a failing test (cued, rules 2-3). Put the decision's id in a
   `decision` field of the label. Decisions not in the list may be labeled too if you find them."""

DECISION_STEP_BP = """
   `/judge/ws/group/` holds the group-level facts: `MATRIX.md` (which rollout fails which hidden test) and every
   rollout's `R<k>.final.patch` / `R<k>.tests.json` / `R<k>.failing_assertions.txt` (no trajectories). From them FIRST
   write `/judge/out/decisions.json`: {{"decisions": [{{"id": "D1", "what": "...", "correct": "...", "wrong": "...",
   "tests": [...], "made_correctly_by": ["R.."], "got_wrong_by": ["R.."], "how_to_recognize": "file / symbol / command
   / phrase to look for"}}]}}: every decision a hidden test needs on which the rollouts differ. Then, for EACH decision
   that applies to R{fail} or R{passing}, label the turn where that rollout made it: `good` if decided correctly,
   `harmful` if decided wrongly and that caused a failing test (cued, rules 2-3); put its id in a `decision` field."""

GROUP_INSTRUCTIONS = (
    INTRO
    + FILES.replace(
        "- `/judge/ws/R<k>/trajectory.txt`",
        "- (trajectories are not in this workspace; `/judge/ws/R<k>/calls_index.txt`"
        " gives each rollout's calls)\n- unused: `/judge/ws/R<k>/trajectory.txt`",
    )
    + """
## Your job: list the group's separating decisions (stage 1 of 2; no turn labels)
Annotators will label each pass/fail pair of this group against your list. From TASK.md, the answer key, PREPASS.md and
every rollout's tests.json, failing_assertions.txt and final.patch (SWE) / final-state evidence (TB):
1. Read LABELING.md in full (rules 9, 14, 15, 16 and P6, P9, P13, P14 matter most here).
2. Build the test-failure matrix (which rollout fails which hidden test) and group the failing rollouts by failure.
3. For each failure, find the decision behind it: what the failing rollouts chose versus the passing ones (compare the
   patches / final states), which hidden test needs it, and how to recognize the decision in a trajectory.
4. Ill-posed check per decision (rule 9 with its exception; P6, P10, P13; decide from what the agents could see, never
   from the answer key; two-sided cue table). One ill-posed decision makes the whole group ill_posed (P6).
Write `/judge/out/decisions.json`: {"group_ill_posed": true|false, "ill_posed_evidence": "...", "decisions": [{"id":
"D1", "what": "...", "correct": "...", "wrong": "...", "tests": [...], "made_correctly_by": ["R.."], "got_wrong_by":
["R.."], "how_to_recognize": "file / symbol / command / phrase to look for", "determinable_evidence": "..."}]}
and `/judge/out/notes.md` (the matrix and your reasoning). Then stop.
"""
)


# ---- Variant C (turn_judge LOG 065): B + the fixes of the round-2 B error analysis (062-judge-r2-B/ERROR_ANALYSIS.md)
PRECISION_C = (
    PRECISION.replace(
        """1. One decision, one label, on its decision turn: the FIRST model call whose plan or code commits to the choice (an
   acted-on lean counts, P16). A later call""",
        """1. One decision, one label, on its decision turn: the FIRST model call that commits to the choice (rule 6, P16). A
   call commits when its plan or code fixes the choice AND its tool calls act on it (write it, design the next step on
   it, or move on to a different sub-question). If the same call's tool calls check that very question, it is a lean;
   the commitment is the first call after the check that settles it. Wording ("Decision:", "FINAL") neither makes nor
   breaks a commitment, and code is not required. Reasoning items can hold 30k-140k characters and the commitment often
   sits near the end: for every call you consider, run `python3 /judge/bin/decision_lines.py R<k> <call>` (it lists the
   decision statements of each item of the call with their position) and read the item to its end. A later call""",
    )
    .replace(
        """   (rule 2). `premature_commit`""",
        """   (rule 2). A tool output arrives AFTER the reasoning of its own call: it is not part of that call's information
   state (it is for the next call). `premature_commit`""",
    )
    .replace(
        """No `good` on a call that contains the
   fatal mistake (P7 / P17).""",
        """No `good` on a call that contains the
   fatal mistake (P7 / P17). P7 guard: before a `good` in a failing rollout, find the FIRST call that holds that
   decision; if that call also holds the fatal mistake in any form, the decision gets no `good` anywhere in the rollout
   (a later restatement or the write that carries it out is not its decision turn).""",
    )
    .replace(
        """When most failing rollouts fail
   the same single check while everything else passes, examine that check's determinability closely.""",
        """When most failing rollouts fail
   the same single check while everything else passes, examine that check's determinability closely. Never use outcomes
   as evidence (how many siblings got it right says how often a guess was right, not whether it was cued).
   `/judge/ws/HEDGES.md` lists where each rollout said it was guessing or that something is unknowable / ambiguous: check
   those that concern the failing check.""",
    )
)

DECISION_STEP_C = """
   `/judge/decisions.json` lists the group's separating decisions, written by a group analyst who read every
   rollout's patch and test results but not the trajectories (verify what you use; you may disagree with evidence).
   Each decision has a `label_policy`: `good+harmful` = label the turn where a rollout made it, `good` if right,
   `harmful` if wrong and that caused a failing test (cued, rules 2-3); `harmful_only` = only the wrong side is
   labeled (a slip, verbatim copying, a process failure): no `good` for doing it right; `none` = requirement-dictated,
   no labels. `membership` says which rollouts got it right / wrong: check it for your two rollouts against their final
   state and last tool calls, and say so in notes.md if it is wrong. The list gives no turn placement: find the turn
   yourself (step 4). Put the decision's id in a `decision` field of each label, and fill `decisions_addressed`
   (Output) for EVERY decision of the list and BOTH rollouts: the label's idx, or `none` with the reason (not
   applicable; right but `harmful_only`; policy `none`; the rollout never made it, rule 1; ...). For each applicable
   decision also scan that rollout's LATER tool outputs that bear on it (its own test runs, errors, failing existing
   tests): a call that saw output which on its own points the other way and explained it away is its own `harmful` (P8
   / P15); a call that then re-decided correctly is its own `good`; the agent's own doubts are not new evidence.
   Separating decisions missing from the list may be labeled too (ids "X1", "X2", ...)."""

OUTPUT_C = OUTPUT.replace(
    """ "pass": {{"rollout": "R{passing}", "line": {pass_line}, "verdict": "PASS", "summary": "...", "labels": [...]}}}}""",
    """ "pass": {{"rollout": "R{passing}", "line": {pass_line}, "verdict": "PASS", "summary": "...", "labels": [...]}},
 "decisions_addressed": [{{"id": "D1", "rollout": "R{fail}", "label_idx": <idx of its label>}},
                         {{"id": "D1", "rollout": "R{passing}", "none": "<why no label>"}}, ...]}}""",
)

GROUP_INSTRUCTIONS_C = (
    INTRO
    + FILES.replace(
        "- `/judge/ws/R<k>/trajectory.txt`: the full rollout,",
        "- `/judge/ws/R<k>/trajectory.txt`: for SEARCH only in this stage (`grep -n` with a few lines of context; never read "
        "one in full, they are huge), to check what an agent saw or said. The full rollout,",
    ).replace(
        "- `/judge/ws/PREPASS.md`:",
        "- `/judge/ws/MATRIX.md`: which rollout fails which hidden test. `/judge/ws/HEDGES.md`: where each rollout said it was "
        "guessing or that something is unknowable / ambiguous (script grep; check the hits that concern a failing check).\n"
        "- `/judge/ws/PREPASS.md`:",
    )
    + """
## Your job: list the group's separating decisions (stage 1 of 2; no turn labels)
Annotators will label each pass/fail pair of this group against your list. Work from TASK.md, the answer key,
MATRIX.md, PREPASS.md, HEDGES.md and every rollout's tests.json, failing_assertions.txt and final.patch (SWE) /
final-state evidence (TB); search the trajectories when you need to know what an agent saw or did.
1. Read LABELING.md in full (rules 9, 13, 14, 15, 16 and P6, P9, P13, P14 matter most here).
2. Group the failing rollouts by failure (MATRIX.md).
3. For each failure, find the decision behind it: what the failing rollouts chose versus the passing ones (compare the
   patches / final states), which hidden test needs it, and how to recognize it in a trajectory. A decision is a choice
   visible in the final state. Restoring a state lost by accident (relaunching a killed server, re-applying an
   overwritten file) is not the decision, unless a sibling failed because it lost that state and did not recover.
   Before you attribute a failure to a decision, check the rollout's last tool calls (calls_index.txt, then search its
   trajectory): a final state broken by the rollout's own last commands (it killed its server, overwrote a file) is that
   slip, not the design decision.
4. Ill-posed check per decision (rule 9 with its exception; P6, P10, P13). Decide from what the AGENTS could see,
   never from the answer key: the task text, the repository / container they started from (it is this sandbox's own
   filesystem; find its path from the agents' first commands, e.g. `grep -n -m5 'cd \\|pwd\\|ls ' /judge/ws/R0/trajectory.txt`)
   and their tool outputs (search the trajectories). Write a two-sided cue table in notes.md; every claim that a file or
   name exists or does not exist must cite the command you ran and its output, or a trajectory item #idx. Never use
   outcomes as evidence: how many rollouts guessed right says nothing about whether it was cued. An agent saying it is
   guessing or that the answer is unknowable (HEDGES.md) is strong evidence when it concerns the failing check. P13 (a
   free hedge found and dropped) applies only when an agent itself states that the option satisfies both readings. One
   ill-posed decision makes the whole group ill_posed (P6).
Write `/judge/out/decisions.json`:
{"group_ill_posed": true|false, "ill_posed_evidence": "...", "decisions": [{"id": "D1", "what": "...", "correct": "...",
"wrong": "...", "tests": [...], "label_policy": "good+harmful|harmful_only|none", "policy_reason": "...",
"membership": {"R0": "right|wrong|n/a|unknown", "R1": "...", ... every rollout}, "how_to_recognize": "files / symbols /
commands / phrases to look for", "determinable_evidence": "..."}]}
`label_policy`: `good+harmful` = a design or inference choice on which rollouts differ (rule 16: `good` for those who
made it right, `harmful` for those whose wrong choice failed a test); `harmful_only` = only a slip, a process failure
or a verbatim-copying error separates them (P14, rule 13): the others get no `good`; `none` = requirement-dictated,
every rollout does it (rule 14). Give recognition cues only: NO call numbers, item numbers or claims about which turn
decided (you have not read the trajectories; the annotators place the labels).
Also write `/judge/out/notes.md` (matrix, cue tables, reasoning). Run `python3 /judge/bin/check_pair.py --decisions` and
fix every problem until it prints PASS. Then stop.
"""
)


# ---- Variant D (turn_judge LOG 067): C + the fixes of the round-4 C error analysis (065-judge-r4-C/ERROR_ANALYSIS.md §6)
_P16_C = """A
   call commits when its plan or code fixes the choice AND its tool calls act on it (write it, design the next step on
   it, or move on to a different sub-question). If the same call's tool calls check that very question, it is a lean;
   the commitment is the first call after the check that settles it. Wording ("Decision:", "FINAL") neither makes nor
   breaks a commitment, and code is not required. Reasoning items can hold 30k-140k characters and the commitment often
   sits near the end: for every call you consider, run `python3 /judge/bin/decision_lines.py R<k> <call>` (it lists the
   decision statements of each item of the call with their position) and read the item to its end."""
_P16_D = """A
   call commits when its reasoning, plan or code fixes the choice and the call goes on to build on it (writes it,
   designs the next step on it, or moves to another sub-question). A call is a lean only if its reasoning leaves the
   choice open (it asks the question, or says it will decide after a check) AND its own tool calls test that open
   question, i.e. their output could make the agent pick the other option. These do NOT make a call a lean: reading a
   helper's name or signature, looking up how to implement the chosen option, a confirming read of something the agent
   already concluded, the task's required reading of call sites and tests. A plan line, a design list or a code sketch
   that fixes the choice is a commitment even if the agent never weighs the alternative and uses no decision words
   (P9: silent spec-following counts). A later call that explicitly weighs and keeps it without new contrary evidence
   is a restatement (P15), however emphatic. Wording ("Decision:", "FINAL") neither makes nor breaks a commitment, and
   code is not required. To find the first commitment, run `python3 /judge/bin/first_mention.py R<k> '<regex>'` with
   the decision's symbol, file or phrase (it lists every call whose reasoning or tool arguments mention it, in order)
   and start from its first hits; before you write that no earlier reasoning chose X, or label a write as a silent
   decision, cite its first hit and say why it is not the commitment. Reasoning items can hold 30k-140k characters and
   the commitment often sits deep inside: `python3 /judge/bin/decision_lines.py R<k> <call>` lists the decision
   statements of each item with their offsets and prints the end of long items; read every item you consider to its
   end (`sed -n` over its whole line range), never only its start."""
assert _P16_C in PRECISION_C
PRECISION_D = PRECISION_C.replace(_P16_C, _P16_D).replace(
    """   (a later restatement or the write that carries it out is not its decision turn).""",
    """   (a later restatement or the write that carries it out is not its decision turn). If your only reason for no
   `good` on a call is a `harmful` you placed on that same call, re-check that harmful against rules 2-3 first: a wrong
   harmful must not also remove a correct good (M16).""",
)
assert PRECISION_D.count("first_mention.py") == 1 and "M16" in PRECISION_D

DECISION_STEP_D = DECISION_STEP_C.replace(
    """   Separating decisions missing from the list may be labeled too (ids "X1", "X2", ...).""",
    """   Label every applicable list decision in BOTH rollouts, even when both made it the same way (whether it separates
   THIS pair does not matter). For a wrong decision, label the first commitment if it qualifies on its own
   information state (P8 mirror; P16: an acted-on plan, even in call 0, counts), in addition to any later re-decision.
   When two list decisions are first made in the same call, label the one this pair's failing test needs and give
   that call's idx for the other in `decisions_addressed`. Membership `wrong-uncued` means the rollout never saw the
   deciding fact: no `harmful` for it unless you find the cue yourself (quote it).
   Separating decisions missing from the list may be labeled too (ids "X1", "X2", ...).""",
)
assert DECISION_STEP_D != DECISION_STEP_C

_G3_C = """   slip, not the design decision.
4. Ill-posed check per decision"""
_G3_D = """   slip, not the design decision. Membership describes the final state of the decision's own code: a rollout whose
   final patch contains the right code is `right` even if a later slip broke the file or the run. Use `wrong-uncued`
   for a wrong rollout whose tool outputs never show the deciding fact (search its trajectory for it). When two hidden
   tests need different parts of one choice, list them as separate decisions.
   Policy: `harmful_only` when every failing rollout stated the correct intent and failed only by a slip in carrying it
   out, or when the wrong side is a single rollout's extra guard or change on top of what the task describes (rule 14,
   P14). `policy_reason` explains the policy only; never classify individual rollouts' causes (cued / uncued / rule 10)
   there: the annotators do that.
4. Ill-posed check per decision"""
assert _G3_C in GROUP_INSTRUCTIONS_C
GROUP_INSTRUCTIONS_D = (
    GROUP_INSTRUCTIONS_C.replace(_G3_C, _G3_D)
    .replace(
        """"membership": {"R0": "right|wrong|n/a|unknown", "R1": "...", ... every rollout}""",
        """"membership": {"R0": "right|wrong|wrong-uncued|n/a|unknown", "R1": "...", ... every rollout}""",
    )
    .replace(
        """Give recognition cues only: NO call numbers, item numbers or claims about which turn
decided (you have not read the trajectories; the annotators place the labels).""",
        """Give recognition cues only: NO call numbers, item numbers or claims about which turn
decided (you have not read the trajectories; the annotators place the labels). Phrases you found in trajectories go in
`how_to_recognize` (a phrase marks where the topic is discussed, not the decision call), never in `wrong` or
`policy_reason`.""",
    )
)
assert GROUP_INSTRUCTIONS_D.count("wrong-uncued") == 2 and "how_to_recognize` (a phrase" in GROUP_INSTRUCTIONS_D


# ---- Variant E (turn_judge LOG 072): D + the fixes of the round-6 D error analysis (068-judge-r6-D/ERROR_ANALYSIS.md §5)
_LEAN_D = _P16_D[
    _P16_D.index("A call is a lean only if") : _P16_D.index("however emphatic.") + len("however emphatic.")
]
_LEAN_E = """A call is a lean if its reasoning, at its END, leaves the choice open (a question,
   "wait", "let me reconsider", "must check", "this is decisive", "I'll decide after ...") and its own tool calls test
   that open point; an earlier "FINAL" or "Decision:" in the same item does not change that. Before labeling a call,
   quote its last 300 characters in `decision_turn_check` and say whether it ends open. An early statement that nothing
   was built on (no design, code sketch or write uses it) is a lean if the agent itself reopens the question in a
   later call before building on it; then the call that closes the reopened question is the commitment, not a
   restatement. A belief about the environment ("the checker probably starts the API") is not a plan. These do NOT
   make a call a lean: reading a helper's name or signature, looking up how to implement the chosen option, a
   confirming read, the task's required reading of call sites and tests. A plan line, a design list or a code sketch
   that fixes the choice and is built on is a commitment even if the agent never weighs the alternative and uses no
   decision words (P9: silent spec-following counts). A later call that explicitly weighs and keeps it without new
   contrary evidence is a restatement (P15), however emphatic."""
assert _LEAN_D in PRECISION_D
_FM_D = """To find the first commitment, run `python3 /judge/bin/first_mention.py R<k> '<regex>'` with
   the decision's symbol, file or phrase (it lists every call whose reasoning or tool arguments mention it, in order)
   and start from its first hits;"""
_FM_E = """To find the first commitment, run `python3 /judge/bin/first_mention.py R<k> '<regex>' ['<regex>' ...]`
   with the decision's code symbols and the sub-question's topic words (e.g. a function, flag, path or service name),
   not decision phrases: paraphrases defeat phrase regexes. It lists every call whose reasoning or tool arguments
   mention them, with every match offset; start from its first hits;"""
assert _FM_D in PRECISION_D, "fm"
_R3_D = """   state is not harmful (rule 13).
4. `good` ="""
_R3_E = """   state is not harmful (rule 13). Before any `harmful` label, fill `but_for`: name the failing assertion, the input
   or shape it uses, and say what this call decided about THAT input. If this call chose a mechanism or scope and never
   weighed the failing input (a consequence the agent did not consider at this call), this call is not the decision
   (rule 1): label the first call that weighs that input and decides it wrongly; if no call ever weighs it, the
   failure is `uncued_omission` unless a cue about that input was visible. A rewrite of the agent's own expectation
   for a different input, or a wrong debugging hypothesis, is not the decision unless it dismisses a cue about the
   failing input.
4. `good` ="""
assert _R3_D in PRECISION_D, "r3"
_P8_D = """The agent's own re-derivation, a re-read or a
   re-run of something it already knew is not new evidence."""
_P8_E = """The agent's own re-derivation, a re-read or a
   re-run of something it already knew is not new evidence. New
   evidence for a second `harmful` must be a tool output about the decision's own subject that on its own reads as
   contrary (a failing test, an error, a value that contradicts the choice); a directory listing, a file name, or an
   output the agent had already explained is not."""
assert PRECISION_D.count(_P8_D) == 1, "p8"
PRECISION_E = PRECISION_D.replace(_LEAN_D, _LEAN_E).replace(_FM_D, _FM_E).replace(_R3_D, _R3_E).replace(_P8_D, _P8_E)
assert (
    "quote its last 300 characters" in PRECISION_E
    and "but_for" in PRECISION_E
    and "on its own reads as" in PRECISION_E
)

_UNCUED_D = """Membership `wrong-uncued` means the rollout never saw the
   deciding fact: no `harmful` for it unless you find the cue yourself (quote it)."""
_UNCUED_E = """Membership `wrong-uncued` means the rollout never saw the
   deciding fact: no `missed_cue` harmful unless you find the cue yourself (quote it); a `premature_commit` (an acted-on
   plan that rests on an assumption the agent states as unverified, quoted) is still `harmful`.
   For `missed_cue`, the decision turn is the first call that had the cue in its information state and planned or
   acted on that sub-question without it ("saw the cue and moved on", rule 3), even if it states no position; a later
   explicit dismissal without new evidence is a restatement. For a P8 re-decision, label the first call after the new
   output that moved on or explained it away, not a later call that weighs it more explicitly. A slip whose wrong
   output is derived in reasoning (a worked example ticked as correct) is decided there, not at the write. When two
   list decisions are first made in the same call, never move one of them to a later call: give that call's idx for
   it in `decisions_addressed`."""
assert _UNCUED_D in DECISION_STEP_D, "uncued"
DECISION_STEP_E = DECISION_STEP_D.replace(_UNCUED_D, _UNCUED_E)

_CUE_D = """Write a two-sided cue table in notes.md; every claim that a file or
   name exists or does not exist must cite the command you ran and its output, or a trajectory item #idx."""
_CUE_E = """Write a two-sided cue table in notes.md; every claim that a file or
   name exists or does not exist must cite the command you ran and its output, or a trajectory item #idx. Put every
   task sentence that mentions any of the options (each directory name, file name, verb or format) in the table, on
   the side it supports on its plain reading, before you argue that it is about something else. P13 needs a
   rollout's own sentence that one option satisfies BOTH readings; "either is fine", "arguably", or choosing one of two
   allowed options is not P13, and P13 only exempts the rollout that said it (P6 still applies to the others). P10
   needs a convention visible in the starting repository or container (existing files that do the same thing); an
   installed tool, an empty directory or "the standard way" is not one. If a hidden test asserts a property (a
   location, a name, an order, an exact string) that no task sentence and no visible file states, and each side rests
   only on reading a general sentence, the detail is not determined: `group_ill_posed: true`."""
assert _CUE_D in GROUP_INSTRUCTIONS_D, "cue"
_WHAT_D = """3. For each failure, find the decision behind it:"""
_WHAT_E = """3. For each failure, find the decision behind it (phrase `what` as the underlying choice: the convention, mechanism or
   semantics, not the deliverable field or the end-of-session state):"""
assert _WHAT_D in GROUP_INSTRUCTIONS_D, "what"
_MEM_D = """"membership": {"R0": "right|wrong|wrong-uncued|n/a|unknown", "R1": "...", ... every rollout}"""
_MEM_E = """"membership": {"R0": "right|wrong|wrong-uncued|n/a|unknown", "R1": "...", ... every rollout},
"membership_notes": {"R<k>": "why n/a or unknown", ...}, "visible_alternative": "only for good+harmful with a single
wrong rollout: the alternative the other rollouts visibly weighed\""""
assert _MEM_D in GROUP_INSTRUCTIONS_D, "mem"
GROUP_INSTRUCTIONS_E = GROUP_INSTRUCTIONS_D.replace(_CUE_D, _CUE_E).replace(_WHAT_D, _WHAT_E).replace(_MEM_D, _MEM_E)

OUTPUT_E = OUTPUT_C.replace(
    """`info_state_test`, `contrary_evidence_checked`,
`decision_turn_check`{decision_field};""",
    """`info_state_test`, `contrary_evidence_checked`,
`decision_turn_check`{decision_field}; every `harmful` label also fills `but_for` (precision rule 3);""",
)
assert OUTPUT_E != OUTPUT_C, "out"


def build_instructions(
    fail: int, fail_line: int, passing: int, pass_line: int, variant: str = "A", template: str | None = None
) -> str:
    if template:
        return template.format(fail=fail, fail_line=fail_line, passing=passing, pass_line=pass_line)
    v = variant.upper()
    if v == "V0":
        raise ValueError("variant v0 is the committed df6bc56b2 prompt; use that commit to reproduce it")
    decision_step = {
        "A": "",
        "B": DECISION_STEP_B,
        "BP": DECISION_STEP_BP,
        "C": DECISION_STEP_C,
        "D": DECISION_STEP_D,
        "E": DECISION_STEP_E,
    }[v]
    pass_scope = V0_PASS_SCOPE if v == "A" else LIST_PASS_SCOPE
    others = (
        " The other rollouts' trajectories are not in this workspace."
        if v == "A"
        else " Other rollouts' trajectories are not in this workspace."
    )
    text = (
        INTRO
        + FILES
        + {"C": PRECISION_C, "D": PRECISION_D, "E": PRECISION_E}.get(v, PRECISION)
        + PAIR.format(
            fail=fail,
            fail_line=fail_line,
            passing=passing,
            pass_line=pass_line,
            others=others,
            decision_step=decision_step.format(fail=fail, passing=passing),
            pass_scope=pass_scope.format(fail=fail),
        )
        + ({"C": OUTPUT_C, "D": OUTPUT_C, "E": OUTPUT_E}.get(v, OUTPUT)).format(
            fail=fail,
            fail_line=fail_line,
            passing=passing,
            pass_line=pass_line,
            decision_field=", and `decision` (the decision id)" if v != "A" else "",
        )
        + SELF_CHECK
    )
    return text
