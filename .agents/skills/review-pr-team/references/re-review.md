# Re-review after the author pushes

Use this when the reviewer asks to re-review a PR, or asks whether the author addressed the review.
It is a smaller pass than a full review. The lead checks the threads itself. It spawns the review
agents only when the new commits add new files or change more than a few hundred lines. Any new
finding still goes through the adversarial pass, the follow-up passes, and the audit, as in a full
review. Confirm with the reviewer before every GitHub write, one action per question.

## 1. Collect the state

- The review record from the first review, `$HOME/.cache/gym-pr-review/pr-<PR>/review.json`. It
  holds the reviewed head and base commits and whether the summary was posted. If it is missing,
  use the review's `commit_id` from GitHub as the reviewed head.
- If the record shows a summary that was written but never posted, say so first, because the author
  never saw it. Ask whether it is still needed.
- The PR head, commits, and checks:
  `gh pr view <PR> --repo NVIDIA-NeMo/Gym --json headRefOid,baseRefName,commits,mergeStateStatus` and
  `gh pr checks <PR> --repo NVIDIA-NeMo/Gym`.
- Your review threads, through GraphQL, because REST under-reports pending and threaded comments.
  Keep the unresolved threads whose first comment is yours (`gh api user --jq .login`):

  ```bash
  gh api graphql -F pr=<PR> -f query='
    query($pr: Int!) { repository(owner: "NVIDIA-NeMo", name: "Gym") { pullRequest(number: $pr) {
      reviewThreads(first: 100) { nodes { id isResolved isOutdated path line
        comments(first: 50) { nodes { databaseId author { login } body url } } } } } } }'
  ```
- Any pending review of yours, read from GitHub. Its comments are the reviewer's to change.
- Set up a fresh worktree and venv at the new head, as in SKILL.md §1.1 and §1.6.

## 2. Read the author's replies first

A reply usually names the fix the author chose, where it lives, and what they skipped and why. Use it
as a map, not as evidence.

## 3. Check each thread against the problem

- Re-read your comment and every fix it offered.
- Check the problem, not the flagged line. A fix can land somewhere else, and an unchanged line can
  still be fixed by a guard above it.
- Re-run the failing case the comment described: the test, the reproduction, or the mutation. The
  check runs at the new head, in a fresh worktree and venv (SKILL.md §1.6).
- If the comment's wording is met but the underlying need is not, the thread is partly fixed.

## 4. Separate the author's change from changes on main

If the author rebased or merged `main`, compare the PR's own added and removed lines before and after:
`git diff <old-base> <old-head>` against `git diff <new-base> <new-head>`, with the old commits from
the review record. Changes on `main` are not the author's.

For an applied suggestion block, read the removed lines of the applied hunk. An applied block can
delete more than intended.

## 5. Review the new commits

Review what the new commits change with the same checks as a full review: the rubric, compatibility,
related PRs, and tests that fail on the old code.

## 6. Show the reviewer, top-down

1. The diagram, if the design changed.
2. The recommended verdict.
3. Each thread as fixed, partly fixed, or not fixed, with the evidence.
4. Any new findings.

## 7. Post, after confirmation

- **Fixed:** reply `fixed in <short SHA>` as plain text, without backticks, so GitHub links the
  commit. Then resolve the thread with its GraphQL `id`:
  `gh api graphql -f query='mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) { thread { isResolved } } }' -f id=<thread-id>`.
- **Partly fixed:** reply with what the commit did and what remains, linking the remaining line.
  Leave the thread unresolved.
- **Not fixed:** post nothing on the thread. Tell the reviewer.
- **New findings:** stage them as a new pending review, after the audit in `posting.md`.
